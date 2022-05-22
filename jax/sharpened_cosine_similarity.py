"""
Based entirely on Raphael Pisoni's original implementation
https://colab.research.google.com/drive/1KUKFEMneQMS3OzPYnWZGkEnry3PdzCfn#scrollTo=sqs64Hv2HtPZ&line=37&uniqifier=1
"""
from flax import linen as nn
import jax
import jax.numpy as jnp


class SharpCosSim2d(nn.Module):
    lhs: int
    rhs: int
    kernel_size: int
    stride: int=1
    padding: str="VALID"
    groups: int=1
    shared_weights: bool=False
    shuffle: bool=False
    w_max: float=1.
    p_min: int=.1
    q_init: float=1e-3
    eps: float=1e-6

    def setup(self):
        assert self.groups == 1 or self.groups == self.lhs, " ".join([
            "'groups' needs to be 1 or 'lhs'",
            f"({self.lhs})."])
        assert self.rhs % self.groups == 0, " ".join([
            "    When not using shared weights, the number of",
            "rhs needs to be a multiple of the number",
            "of input channels.\n    Here there are",
            f"{self.rhs} rhs and {self.lhs}",
            "input channels."])

        self.sharing = self.shared_weights
        if self.groups == 1:
            self.sharing = False

        # The end result here should be uniformly distributed weights between
        # -w_max and w_max.
        self.channels_per_kernel = self.lhs // self.groups
        if self.shared_weights:
            self.n_kernels = self.rhs // self.groups
        else:
            self.n_kernels = self.rhs
        self.w = self.param(
            'w',
            nn.initializers.uniform(scale=2 * self.w_max),
            (self.n_kernels,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size))
        self.w -= self.w_max

        # Initialize p values on a uniform interval from 1 to 3.
        # The final values of p are pretty insensitive to this,
        # so I don't think it's worth it to expose this is a parameter.
        p_max = 3
        p_min = 1
        self.p = self.param(
            'p',
            nn.initializers.uniform(scale=p_max - p_min),
            (1, self.n_kernels, 1, 1))
        self.p += p_min
        self.log_q = self.param(
            'q',
            (lambda k, s: jnp.full(s, jnp.log(self.q_init))),
            (1, 1, 1, 1))

    def __call__(self, inputs):
        # Enforce limits on parameters
        w = jax.lax.clamp(-self.w_max, self.w, self.w_max)
        # Set the maximum value of p unreasonably high. It's not really needed.
        p_max = 100.0
        p = jax.lax.clamp(self.p_min, self.p, p_max)

        x = jnp.transpose(inputs, [0,3,1,2])
        q = jnp.exp(self.log_q)

        if self.sharing:
            w = jnp.tile(w, (self.groups, 1, 1, 1))
            p = jnp.tile(p, (1, self.groups, 1, 1))

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
                (-1, self.groups, self.rhs, y.shape[-2], y.shape[-1]))
            y = jnp.transpose(y, axes=(0, 2, 1, 3, 4))
            y = jnp.reshape(
                y,
                (-1, self.groups * self.rhs, y.shape[-2], y.shape[-1]))

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
        outputs_per_group =  self.rhs // self.groups
        return jnp.repeat(xnorm, outputs_per_group, axis=1)


# Aliases for the class name
SharpCosSim2D = SharpCosSim2d
SharpCosSim = SharpCosSim2d
SCS = SharpCosSim2d
SharpenedCosineSimilarity = SharpCosSim2d
sharp_cos_sim = SharpCosSim2d


class MaxAbsPool(nn.Module):
    window_shape: tuple=(2, 2)
    strides: tuple=(2, 2)
    padding: str='VALID'

    @nn.compact
    def __call__(self, inputs):
        high = nn.max_pool(
            inputs,
            self.window_shape,
            self.strides,
            self.padding)
        low = nn.max_pool(
            -inputs,
            self.window_shape,
            self.strides,
            self.padding)
        pooled = jnp.where(high > low, high, -low)
        return pooled
