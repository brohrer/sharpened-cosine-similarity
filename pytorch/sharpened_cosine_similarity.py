"""
Slightly modified version of code
https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80
from @clashluke
https://gist.github.com/ClashLuke

Also based on the TensorFlow implementation
https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB
and blog post
https://www.rpisoni.dev/posts/cossim-convolution/
from Raphael Pisoni
https://twitter.com/ml_4rtemi5

Check here to get the full story.
https://e2eml.school/scs.html
"""

import torch
from torch import nn
from torch.nn import functional as F


class SharpenedCosineSimilarity(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups: int = 1,
        q_init: float = 1,
        p_init: float = 1,
        q_scale: float = .3,
        p_scale: float = 5,
        eps: float = 1e-6,
    ):
        if padding is None:
            if int(torch.__version__.split('.')[1]) >= 10:
                padding = "same"
            else:
                # This doesn't support even kernels
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert dilation == 1, \
            "Dilation has to be 1 to use AvgPool2d as L2-Norm backend."
        assert groups == in_channels or groups == 1, \
            "Either depthwise or full convolution. Grouped not supported"

        super(SharpenedCosineSimilarity, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups)

        self.p_scale = p_scale
        self.q_scale = q_scale
        self.p = torch.nn.Parameter(
            torch.full((out_channels,), float(p_init * self.p_scale)))
        self.q = torch.nn.Parameter(
            torch.full((1,), float(q_init * self.q_scale)))
        self.eps = eps

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # 1. Find the l2-norm of the inputs at each position of the kernels.
        # 1a. Square each input element.
        out = inp.square()
        # 1b. Sum each element over all channels
        # if this isn't a depthwise scs layer.
        if self.groups == 1:
            out = out.sum(1, keepdim=True)

        # 1c. Create a mock set of kernel weights that are all ones.
        # These will have a different shape, depending on whether this
        # is a depthwise scs layer.
        if self.groups == 1:
            kernel_size_ones = torch.ones_like(self.weight[:1, :1])
        else:
            kernel_size_ones = torch.ones_like(self.weight)

        # 1d. Sum the squared inputs over each set of kernel positions
        # by convolving them with the mock all-ones kernel weights.
        norm = F.conv2d(
            out,
            kernel_size_ones,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)

        # 2. Add in the q parameter. The scaling and transforming ensure
        # that its magnitude is appropriate and its gradient is smooth
        # so that it will be learned well.
        q = torch.exp(-self.q / self.q_scale)
        norm = (norm + self.eps).sqrt() + q
        # norm = (norm + (q + self.eps)).sqrt()

        # 3. Find the l2-norm of the weights in each kernel.
        # 4. Normalize the kernel weights.
        weight = self.weight / (
            self.weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt())

        # 5. Normalize the inputs.
        # 6. Calculate the dot product of the normalized kernels and the
        # normalized inputs.
        out = F.conv2d(
            inp,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups) / norm

        # 7. Raise the result to the power p, keeping the sign of the original.
        magnitude = out.abs() + self.eps
        sign = out.sign()

        # 7a. Like with q, the scaling and transforming of the p parameter
        # ensure that its magnitude is appropriate and its gradient is smooth
        # so that it will be learned well.
        p = torch.exp(self.p / self.p_scale)
        # 7b. Broadcast the p's so that each one gets applied to its own unit.
        out = magnitude.pow(p.view(1, -1, 1, 1))
        out = out * sign
        return out
