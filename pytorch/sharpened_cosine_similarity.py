import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class SharpCosSim2d(nn.Conv2d):
    def __init__(
        self,
        channels_in: int,
        features: int,
        kernel_size: int=3,
        padding: int=0,
        padding_mode: str="zeros",
        stride: int=1,
        groups: int=1,
        # depthwise: bool=False,
        shared_weights: bool = True,
        p_init: float=.7,
        q_init: float=1.,
        p_scale: float=5.,
        q_scale: float=.3,
        eps: float=1e-6,
    ):
        assert groups == 1 or groups == channels_in, " ".join([
            "'groups' needs to be 1 or 'channels_in'",
            f"({channels_in})."])
        self.channels_in = channels_in
        self.features = features
        self.padding_mode = padding_mode
        self.stride = stride
        self.groups = groups
        self.shared_weights = shared_weights

        if self.groups == channels_in:
            if self.shared_weights:
                channels_out = self.features * channels_in
            else:
                channels_out = self.features
        else:
            self.shared_weights = False
            channels_out = self.features

        super(SharpCosSim2d, self).__init__(
            self.channels_in,
            channels_out,
            kernel_size,
            bias=False,
            padding=padding,
            stride=stride,
            groups=self.groups)


        # Scaling weights in this way generates kernels that have
        # an l2-norm of about 1. Since they get normalized to 1 during
        # the forward pass anyway, this prevents any numerical
        # or gradient weirdness that might result from large amounts of
        # rescaling.
        weights_per_kernel = (
            (self.channels_in // self.groups) * kernel_size ** 2)
        initialization_scale = (3 / weights_per_kernel) ** .5
        scaled_weights = np.random.uniform(
            low=-initialization_scale,
            high=initialization_scale,
            size=(
                self.features,
                self.channels_in // self.groups,
                kernel_size,
                kernel_size)
        )

        self.weight = torch.nn.Parameter(torch.Tensor(scaled_weights))
        # (features, channels_in, kernel_size, kernel_size)
        self.p_scale = p_scale
        self.q_scale = q_scale
        self.p = torch.nn.Parameter(
            torch.full((1, self.features, 1, 1), float(p_init * self.p_scale)))
        self.q = torch.nn.Parameter(
            torch.full((1, 1, 1, 1), float(q_init * self.q_scale)))
        self.eps = eps

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Scale and transform the p and q parameters
        # to ensure that their magnitudes are appropriate
        # and their gradients are smooth
        # so that they will be learned well.
        p = torch.exp(self.p / self.p_scale)
        q = torch.exp(-self.q / self.q_scale)

        # If necessary, expand out the weights and p parameters.
        if self.shared_weights:
            weights = torch.tile(self.weight, (self.groups, 1, 1, 1))
            p = torch.tile(self.p, (1, self.groups, 1, 1))
        else:
            weights = self.weight

        # 1. Find the l2-norm of the inputs at each position of the kernels.
        # 1a. Square each input element.
        out = inp.square()

        # 1b. Sum the squared inputs over each set of kernel positions
        # by convolving them with the mock all-ones kernel weights.
        o, i, kh, kw = weights.shape
        if self.shared_weights:
            norm = F.conv2d(
                out,
                torch.ones(
                    (self.groups, self.channels_in // self.groups, kw, kh)),
                stride=self.stride,
                padding=self.padding,
                groups=self.groups)
        else:
            norm = F.conv2d(
                out,
                torch.ones((1, self.channels_in, kw, kw)),
                stride=self.stride,
                padding=self.padding)

        # 2. Add in the q parameter. 
        norm = (norm + self.eps).sqrt() + q
        norm = torch.repeat_interleave(norm, self.features, axis=1)

        # 3. Find the l2-norm of the weights in each kernel and
        # 4. Normalize the kernel weights.
        weights = weights / (
            weights.square().sum(dim=(0, 2, 3), keepdim=True).sqrt())

        # 5. Normalize the inputs and
        # 6. Calculate the dot product of the normalized kernels and the
        # normalized inputs.
        out = F.conv2d(
            inp,
            weights,
            stride=self.stride,
            padding=self.padding,
            # padding_mode=self.padding_mode,
            groups=self.groups) / norm

        # 7. Raise the result to the power p, keeping the sign of the original.
        magnitude = out.abs() + self.eps
        out = out.sign() * magnitude ** p
        return out


# Aliases for the class name
SharpCosSim2D = SharpCosSim2d
SharpCosSim = SharpCosSim2d
SCS = SharpCosSim2d
SharpenedCosineSimilarity = SharpCosSim2d
sharp_cos_sim = SharpCosSim2d
