from functools import partial
from clu import parameter_overview
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from sharpened_cosine_similarity import SCS, MaxAbsPool
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

config = ml_collections.ConfigDict()
config.learning_rate = 0.03
config.batch_size = 64
config.num_epochs = 100
config.warmup_epochs = 3


class SCSNN(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        x = jnp.pad(
            x,
            # pad to avoid the pooling to leave an uneven image shape
            [[0,0], [1,1], [1,1], [0,0]],
            mode='constant',
            constant_values=0)
        # x = nn.Dropout(rate=0.07)(x, deterministic=not train)
        x = SCS(channels_in=1, features=10, kernel_size=3)(x)
        x = SCS(channels_in=10, features=10, kernel_size=1)(x)
        x = SCS(channels_in=10, features=12, kernel_size=1)(x)
        # x = nn.Dropout(rate=0.07)(x, deterministic=not train)
        x = MaxAbsPool()(x)
        x = SCS(channels_in=12, features=2, kernel_size=3, depthwise=True)(x)
        x = SCS(channels_in=24, features=8, kernel_size=1)(x)
        # x = nn.Dropout(rate=0.07)(x, deterministic=not train)
        x = MaxAbsPool()(x)
        x = SCS(
            channels_in=8,
            features=32,
            kernel_size=3,
            depthwise=True,
            shared_weights=False)(x)
        x = SCS(channels_in=32, features=10, kernel_size=1)(x)
        # x = nn.Dropout(rate=0.07)(x, deterministic=not train)
        x = MaxAbsPool(window_shape=(4, 4), strides=(4, 4))(x)
        x = x.reshape((x.shape[0], -1))   # flatten
        x = nn.Dense(features=10)(x)
        return x


@partial(jax.jit, static_argnums=3)
def apply_model(state, images, labels, train, dropout_rng):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = SCSNN().apply(
            {'params': params},
            images,
            train=train,
            rngs={'dropout': dropout_rng})
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng, dropout_rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(
            state,
            batch_images,
            batch_labels,
            train=True,
            dropout_rng=dropout_rng)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def val_epoch(state, test_ds, batch_size, dropout_rng):
    """Validate."""
    # TODO: last incomplete batch not considered.
    ds_size = len(test_ds['image'])
    steps_per_epoch = ds_size // batch_size

    epoch_loss = []
    epoch_accuracy = []

    for step in range(steps_per_epoch):
        batch_images = test_ds['image'][
            step * batch_size : (step + 1) * batch_size]
        batch_labels = test_ds['label'][
            step * batch_size : (step + 1) * batch_size]
        grads, loss, accuracy = apply_model(
            state,
            batch_images,
            batch_labels,
            train=False,
            dropout_rng=dropout_rng)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return train_loss, train_accuracy


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image'])
    test_ds['image'] = jnp.float32(test_ds['image'])

    return train_ds, test_ds


def create_learning_rate_fn(config):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=config.learning_rate,
        transition_steps=config.warmup_epochs * config.steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=cosine_epochs * config.steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * config.steps_per_epoch])
    return schedule_fn


def create_train_state(config, rng, dropout_rng):
    """Creates initial `TrainState`."""
    scsnn = SCSNN()
    params = scsnn.init(
        {'params': rng, 'dropout': dropout_rng},
        jnp.ones([1, 28, 28, 1]),
        train=True)['params']

    print("\nParamter overview:")
    print(parameter_overview.get_parameter_overview(
        params, include_stats=True) + "\n")

    lr_schedule = create_learning_rate_fn(config)

    optimizer = optax.chain(
        optax.add_decayed_weights(1e-6),
        optax.adaptive_grad_clip(0.01, eps=0.001),
        optax.scale_by_belief(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    return train_state.TrainState.create(
        apply_fn=scsnn.apply, params=params, tx=optimizer)


def train_and_evaluate(config: ml_collections.ConfigDict
    ) -> train_state.TrainState:
    """Execute model training and evaluation loop.
        Args:
        config: Hyperparameter configuration for training and evaluation.
        Returns:
            The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()

    ds_size = len(train_ds['image'])
    config.steps_per_epoch = ds_size // config.batch_size

    rng = jax.random.PRNGKey(0)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    state = create_train_state(config, init_rng, dropout_rng)
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    for epoch in range(1, config.num_epochs + 1):

        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config.batch_size, input_rng, dropout_rng)
        test_loss, test_accuracy = val_epoch(
            state, test_ds, config.batch_size, dropout_rng)

        print(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
                test_accuracy * 100))

    return state

final_state = train_and_evaluate(config)
