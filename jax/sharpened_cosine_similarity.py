"""
Based entirely on Raphael Pisoni's original implementation
https://colab.research.google.com/drive/1KUKFEMneQMS3OzPYnWZGkEnry3PdzCfn#scrollTo=sqs64Hv2HtPZ&line=37&uniqifier=1
"""
from flax import linen as nn
import jax
import jax.numpy as jnp


class SharpCosSim2d(nn.Module):
    channels_in: int
    features: int
    kernel_size: int
    stride: int = 1
    padding: str = "VALID"
    groups: int = 1
    shared_weights: bool = True
    shuffle: bool = False
    p_init: float = 1.
    q_init: float = 10.
    p_scale: float = 5.
    q_scale: float = .3
    eps: float = 1e-6

    def setup(self):
        assert self.groups == 1 or self.groups == self.channels_in, " ".join([
            "'groups' needs to be 1 or 'channels_in'",
            f"({self.channels_in})."])
        assert self.features % self.groups == 0, " ".join([
            "    When not using shared weights, the number of",
            "features needs to be a multiple of the number",
            "of input channels.\n    Here there are",
            f"{self.features} features and {self.channels_in}",
            "input channels."])

        if self.groups == self.channels_in:
            self.sharing = self.shared_weights
        else:
            self.sharing = False

        # Scaling weights in this way generates kernels that have
        # an l2-norm of about 1. Since they get normalized to 1 during
        # the forward pass anyway, this prevents any numerical
        # or gradient weirdness that might result from large amounts of
        # rescaling.
        self.channels_per_kernel = self.channels_in // self.groups
        weights_per_kernel = self.channels_per_kernel * self.kernel_size ** 2
        if self.shared_weights:
            self.n_kernels = self.features // self.groups
        else:
            self.n_kernels = self.features
        initialization_scale = (3 / weights_per_kernel) ** .5
        self.w = self.param(
            'w',
            nn.initializers.uniform(scale=2 * initialization_scale),
            (self.n_kernels,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size))
        # The end result here should be uniformly distributed weights between
        # -initialiation_scale and initialization_scale.
        self.w = self.w - initialization_scale

        self.p = self.param(
            'p',
            (lambda k, s: jnp.full(s, self.p_init)),
            (1, self.n_kernels, 1, 1))
        self.q = self.param(
            'q',
            (lambda k, s: jnp.full(s, self.q_init)),
            (1, 1, 1, 1))

    def __call__(self, inputs):
        x = jnp.transpose(inputs, [0,3,1,2])
        p = jnp.exp(self.p / self.p_scale)
        q = jnp.exp(-self.q / self.q_scale)

        if self.sharing:
            w = jnp.tile(self.w, (self.groups, 1, 1, 1))
            p = jnp.tile(p, (1, self.groups, 1, 1))
        else:
            w = self.w

        y = self.scs(x, w, q, p)
        y = jnp.transpose(y, [0,2,3,1])
        return y

    def scs(self, x, weight, q, p):
        weight_norm = jnp.sqrt(
            jnp.sum(weight**2, axis=(1, 2, 3), keepdims=True) + self.eps)
        weight = weight / weight_norm

        cos_sim = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=[self.stride, self.stride],
            padding=self.padding,
            feature_group_count=self.groups
        ) / self.input_norm(x, q)

        # Raise the result to the power p, keeping the sign of the original.
        y = jnp.sign(cos_sim) * (jnp.abs(cos_sim) + self.eps) ** p

        if self.shuffle:
            y = jnp.reshape(
                y,
                (-1, self.groups, self.features, y.shape[-2], y.shape[-1]))
            y = jnp.transpose(y, axes=(0, 2, 1, 3, 4))
            y = jnp.reshape(
                y,
                (-1, self.groups * self.features, y.shape[-2], y.shape[-1]))

        return y

    def input_norm(self, x, q):
        xsqsum = jax.lax.conv_general_dilated(
            x**2,
            jnp.ones([
                self.groups,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size], dtype=x.dtype),
            window_strides=[self.stride, self.stride],
            padding=self.padding,
            feature_group_count=self.groups,
        )
        # Has the shape [batch, group, input_height, input_width]

        xnorm = jnp.sqrt(xsqsum + self.eps) + q
        outputs_per_group =  self.features // self.groups
        return jnp.repeat(xnorm, outputs_per_group, axis=1)


# Aliases for the class name
SharpCosSim2D = SharpCosSim2d
SharpCosSim = SharpCosSim2d
SCS = SharpCosSim2d
SharpenedCosineSimilarity = SharpCosSim2d
sharp_cos_sim = SharpCosSim2d


class MaxAbsPool(nn.Module):
    window_shape: tuple = (2, 2)
    strides: tuple = (2, 2)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, inputs):
        high = nn.max_pool(inputs, self.window_shape, self.strides, self.padding)
        low = nn.max_pool(-inputs, self.window_shape, self.strides, self.padding)
        pooled = jnp.where(
            high > low,
            high,
            -low
        )
        return pooled
