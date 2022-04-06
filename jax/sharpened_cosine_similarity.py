from flax import linen as nn
import jax
import jax.numpy as jnp


class SharpCosSim2d(nn.Module):
    channels_in: int
    features: int
    kernel_size: int
    stride: int = 1
    padding: str = "VALID"
    # dilation: int = 1
    # groups: int = 1
    depthwise: bool = False
    shared_weights: bool = True
    shuffle: bool = False
    p_init: float = 1.
    q_init: float = 10.
    p_scale: float = 5.
    q_scale: float = .3
    eps: float = 1e-6

    def setup(self):
        if self.depthwise:
            self.groups = self.channels_in
            self.sharing = self.shared_weights
            if not self.sharing:
                assert self.features % self.groups == 0, " ".join([
                    "    When not using shared weights, the number of",
                    "features needs to be a multiple of the number",
                    "of input channels.\n    Here there are",
                    f"{self.features} features and {self.channels_in}",
                    "input channels."])
        else:
            self.groups = 1
            self.sharing = False

        # Scaling weights in this way generates kernels that have
        # an l2-norm of about 1. Since they get normalized to 1 during
        # the forward pass anyway, this prevents any numerical
        # or gradient weirdness that might result from large amounts of
        # rescaling.
        weights_per_kernel = (
            (self.channels_in // self.groups) * self.kernel_size ** 2)
        initialization_scale = (3 / weights_per_kernel) ** .5
        self.w = self.param(
            'w',
            nn.initializers.uniform(scale=2 * initialization_scale),
            (self.features,
                self.channels_in // self.groups,
                self.kernel_size,
                self.kernel_size))
        # The end result here should be uniformly distributed weights between
        # -initialiation_scale and initialization_scale.
        self.w = self.w - initialization_scale

        self.p = self.param(
            'p',
            (lambda k, s: jnp.full(s, self.p_init)),
            (1, self.features, 1, 1))
            # (1, self.features * self.groups, 1, 1))
        self.q = self.param(
            'q',
            (lambda k, s: jnp.full(s, self.q_init)),
            (1, 1, 1, 1))

    def sharp_normalize(self, n, q):
        square_sum = jnp.sum(n**2, axis=(0, 2, 3), keepdims=True)
        norm = jnp.sqrt(square_sum + self.eps)
        norm = norm + q
        return n / norm

    def scs(self, x, w, q, p):
        w = self.sharp_normalize(w, q)
        o, i, kh, kw = w.shape

        if self.sharing:
            xsqsum = jax.lax.conv_general_dilated(
                x**2,
                jnp.ones(
                    [self.groups, self.channels_in // self.groups, kh, kw],
                    dtype=x.dtype),
                window_strides=[self.stride, self.stride],
                padding=self.padding,
                feature_group_count=self.groups,
            ) # [b g h_x w_x]
        else:
            xsqsum = jax.lax.conv_general_dilated(
                x**2,
                jnp.ones([1, self.channels_in, kh, kw], dtype=x.dtype),
                window_strides=[self.stride, self.stride],
                padding=self.padding,
                # feature_group_count=self.groups,
            ) # [b g h_x w_x]

        xnorm = jnp.sqrt(xsqsum + self.eps)
        xnorm = xnorm + q
        xnorm = jnp.repeat(xnorm, self.features, axis=1)

        y = jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=[self.stride, self.stride],
            padding=self.padding,
            feature_group_count=self.groups
        ) # [b o h_x w_x]

        y = y / xnorm
        sign = jnp.sign(y)
        y = jnp.abs(y) + self.eps
        y = sign * y ** p

        if self.shuffle:
            y = jnp.reshape(
                y,
                (-1, self.groups, self.features, y.shape[-2], y.shape[-1]))
            y = jnp.transpose(y, axes=(0, 2, 1, 3, 4))
            y = jnp.reshape(
                y,
                (-1, self.groups * self.features, y.shape[-2], y.shape[-1]))

        return y

    def __call__(self, inputs):
        x = jnp.transpose(inputs, [0,3,1,2])
        q = jnp.exp(-self.q / self.q_scale)
        p = jnp.exp(self.p / self.p_scale)

        if self.sharing:
            w = jnp.tile(self.w, (self.groups, 1, 1, 1))
            p = jnp.tile(p, (1, self.groups, 1, 1))
        else:
            w = self.w

        y = self.scs(x, w, q, p)
        y = jnp.transpose(y, [0,2,3,1])
        return y


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
