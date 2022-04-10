from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class SharpCosSim2d(nn.Conv2d):
    def __init__(
        self,
        n_channels_in: int,
        n_kernels: int,
        kernel_size: int=3,
        padding: int=0,
        stride: int=1,
        depthwise: bool=False,
        p_init: float=.7,
        q_init: float=1.,
        p_scale: float=5.,
        q_scale: float=.3,
        eps: float=1e-6,
        alpha: Optional[float] = None
    ):
        kernel_size = (kernel_size, kernel_size)

        if depthwise:
            self.groups = n_channels_in
        else:
            self.groups = 1

        super(SharpCosSim2d, self).__init__(
            n_channels_in,
            n_kernels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=self.groups)

        # Create a mock set of kernel weights that are all ones.
        if self.groups == 1:
            self.kernel_size_ones = torch.ones_like(self.weight[:1, :1, :, :])
        else:
            self.kernel_size_ones = torch.ones_like(self.weight)

        # Initialize the weights so that each kernel starts with
        # an l2-norm of 1.
        normalized_weights = self.weight / (
            self.weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt())
        self.weight = torch.nn.Parameter(normalized_weights)

        self.p_scale = p_scale
        self.q_scale = q_scale
        self.p = torch.nn.Parameter(
            torch.full((n_kernels,), float(p_init * self.p_scale)))
        self.q = torch.nn.Parameter(
            torch.full((1,), float(q_init * self.q_scale)))
        self.eps = eps
        if alpha is not None:
            self.a = torch.nn.Parameter(torch.full((n_kernels,),
                                        float(alpha)))
        else:
            self.a = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # 1. Find the l2-norm of the inputs at each position of the kernels.
        # 1a. Square each input element.
        out = inp.square()

        # 1b. Sum each element over all channels
        # if this isn't a depthwise layer.
        if self.groups == 1:
            out = out.sum(1, keepdim=True)

        # 1c. Sum the squared inputs over each set of kernel positions
        # by convolving them with the mock all-ones kernel weights.
        norm = F.conv2d(
            out,
            self.kernel_size_ones,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)

        # 2. Add in the q parameter. The scaling and transforming ensure
        # that its magnitude is appropriate and its gradient is smooth
        # so that it will be learned well.
        q = torch.exp(-self.q / self.q_scale)
        norm = (norm + self.eps).sqrt() + q

        # 3. Find the l2-norm of the weights in each kernel and
        # 4. Normalize the kernel weights.
        weight = self.weight / (
            self.weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt())

        # 5. Normalize the inputs and
        # 6. Calculate the dot product of the normalized kernels and the
        # normalized inputs.
        out = F.conv2d(
            inp,
            weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups) / norm

        # 7. Raise the result to the power p, keeping the sign of the original.
        magnitude = out.abs() + self.eps
        sign = out.sign()

        # 7a. Like with q, the scaling and transforming of the p parameter
        # ensure that its magnitude is appropriate and its gradient is smooth
        # so that it will be learned well.
        p = torch.exp(self.p / self.p_scale)

        # 7b. Broadcast the p's so that each gets applied to its own kernel.
        out = magnitude.pow(p.view(1, -1, 1, 1))

        out = out * sign

        # 8. learned scale parameter
        if self.a is not None:
            out = self.a.view(1, -1, 1, 1) * out
        return out


# Aliases for the class name
SharpCosSim2D = SharpCosSim2d
SharpCosSim = SharpCosSim2d
SCS = SharpCosSim2d
SharpenedCosineSimilarity = SharpCosSim2d
sharp_cos_sim = SharpCosSim2d
