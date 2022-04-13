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
        shared_weights: bool = True,
        log_p_init: float=.7,
        log_q_init: float=1.,
        log_p_scale: float=5.,
        log_q_scale: float=.3,
        alpha: Optional[float] = None,
        autoinit: bool = True,
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
        weights_per_kernel = self.channels_per_kernel * self.kernel_size ** 2
        if self.shared_weights:
            self.n_kernels = self.out_channels // self.groups
        else:
            self.n_kernels = self.out_channels
        initialization_scale = (3 / weights_per_kernel) ** .5
        scaled_weight = np.random.uniform(
            low=-initialization_scale,
            high=initialization_scale,
            size=(
                self.n_kernels,
                self.channels_per_kernel,
                self.kernel_size,
                self.kernel_size)
        )
        self.weight = torch.nn.Parameter(torch.Tensor(scaled_weight))

        self.log_p_scale = log_p_scale
        self.log_q_scale = log_q_scale
        self.p = torch.nn.Parameter(torch.full(
            (1, self.n_kernels, 1, 1),
            float(log_p_init * self.log_p_scale)))
        self.q = torch.nn.Parameter(torch.full(
            (1, 1, 1, 1), float(log_q_init * self.log_q_scale)))
        self.eps = eps

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.full((self.out_channels,),
                                        float(alpha)))
        else:
            self.alpha = None
        if autoinit and (alpha is not None):
            self.LSUV_like_init()

    def LSUV_like_init(self):
        BS, CH = 32, int(self.weight.shape[1]*self.groups)
        H, W = self.weight.shape[2], self.weight.shape[3]
        device = self.weight.device
        inp = torch.rand(BS, CH, H, W, device=device)
        with torch.no_grad():
            out = self.forward(inp)
            coef = out.std(dim=(0, 2, 3)) + self.eps
            self.alpha.data *= 1.0 / coef.view_as(self.alpha)
        return

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Scale and transform the p and q parameters
        # to ensure that their magnitudes are appropriate
        # and their gradients are smooth
        # so that they will be learned well.
        p = torch.exp(self.p / self.log_p_scale)
        q = torch.exp(-self.q / self.log_q_scale)

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
        out = cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

        # Apply learned scale parameter
        if self.alpha is not None:
            out = self.alpha.view(1, -1, 1, 1) * out
        return out

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
