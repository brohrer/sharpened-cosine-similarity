import numpy as np
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class SharpCosSim2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        padding: int=0,
        stride: int=1,
        groups: int=1,
        shared_weights: bool = False,
        w_max: float=1.,
        p_min: int=.1,
        q_init: float=1e-3,
        eps: float=1e-6,
    ):
        assert groups == 1 or groups == in_channels, " ".join([
            "'groups' needs to be 1 or 'in_channels' ",
            f"({in_channels})."])
        assert out_channels % groups == 0, " ".join([
            "The number of",
            "output channels needs to be a multiple of the number",
            "of groups.\nHere there are",
            f"{out_channels} output channels and {groups}",
            "groups."])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.shared_weights = shared_weights

        if self.groups == 1:
            self.shared_weights = False

        super(SharpCosSim2d, self).__init__(
            self.in_channels,
            self.out_channels,
            kernel_size,
            bias=False,
            padding=padding,
            stride=stride,
            groups=self.groups)

        # Overwrite self.kernel_size created in the 'super' above.
        # We want an int, assuming a square kernel, rather than a tuple.
        self.kernel_size = kernel_size

        # Scaling weights in this way generates kernels that have
        # an l2-norm of about 1. Since they get normalized to 1 during
        # the forward pass anyway, this prevents any numerical
        # or gradient weirdness that might result from large amounts of
        # rescaling.
        self.channels_per_kernel = self.in_channels // self.groups
        if self.shared_weights:
            self.n_kernels = self.out_channels // self.groups
        else:
            self.n_kernels = self.out_channels
        self.w_max = w_max
        scaled_weight = np.random.uniform(
            low=-self.w_max,
            high=self.w_max,
            size=(
                self.n_kernels,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size)
        )
        self.weight = torch.nn.Parameter(torch.Tensor(scaled_weight))

        # Initialize p values on a uniform interval from 1 to 3.
        # The final values of p are pretty insensitive to this,
        # so I don't think it's worth it to expose this is a parameter.
        self.p_min = p_min
        p_values = np.random.uniform(
            low=1,
            high=3,
            size=(1, self.n_kernels, 1, 1)
        )
        self.p = torch.nn.Parameter(torch.Tensor(p_values))

        self.log_q = torch.nn.Parameter(torch.full(
            (1, 1, 1, 1), float(np.log(q_init))))
        self.eps = eps

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Enforce limits on parameters
        self.weight.data = torch.clamp(
            self.weight.data,
            min=-self.w_max,
            max=self.w_max)
        self.p.data = torch.clamp(
            self.p.data,
            min=self.p_min)

        # Scale and transform the p and q parameters
        # to ensure that their magnitudes are appropriate
        # and their gradients are smooth
        # so that they will be learned well.
        p = self.p
        q = torch.exp(self.log_q)

        # If necessary, expand out the weight and p parameters.
        if self.shared_weights:
            weight = torch.tile(self.weight, (self.groups, 1, 1, 1))
            p = torch.tile(p, (1, self.groups, 1, 1))
        else:
            weight = self.weight

        return self.scs(inp, weight, p, q)

    def scs(self, inp, weight, p, q):
        # Normalize the kernel weights.
        weight = weight / self.weight_norm(weight)

        # Normalize the inputs and
        # Calculate the dot product of the normalized kernels and the
        # normalized inputs.
        cos_sim = F.conv2d(
            inp,
            weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        ) / self.input_norm(inp, q)

        # Raise the result to the power p, keeping the sign of the original.
        return cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

    def weight_norm(self, weight):
        # Find the l2-norm of the weights in each kernel.
        return weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()

    def input_norm(self, inp, q):
        # Find the l2-norm of the inputs at each position of the kernels.
        # Sum the squared inputs over each set of kernel positions
        # by convolving them with the mock all-ones kernel weights.
        xnorm = F.conv2d(
            inp.square(),
            torch.ones((
                self.groups,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size)),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)

        # Add in the q parameter. 
        xnorm = (xnorm + self.eps).sqrt() + q
        outputs_per_group = self.out_channels // self.groups
        return torch.repeat_interleave(xnorm, outputs_per_group, axis=1)


# Aliases for the class name
SharpCosSim2D = SharpCosSim2d
SharpCosSim = SharpCosSim2d
SCS = SharpCosSim2d
SharpenedCosineSimilarity = SharpCosSim2d
sharp_cos_sim = SharpCosSim2d
