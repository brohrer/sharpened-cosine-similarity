import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

n_classes = 10
n_input_channels = 3
n_units_1 = 16
n_units_2 = 16
n_units_3 = 16



class DemoNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.scs1 = SharpenedCosineSimilarity(
            in_channels=n_input_channels,
            out_channels=n_units_1,
            kernel_size=5,
            padding=0)
        self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs2 = SharpenedCosineSimilarity(
            in_channels=n_units_1,
            out_channels=n_units_2,
            kernel_size=5,
            padding=1)
        self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs3 = SharpenedCosineSimilarity(
            in_channels=n_units_2,
            out_channels=n_units_3,
            kernel_size=5,
            padding=1)
        self.pool3 = MaxAbsPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.out = nn.Linear(in_features=n_units_3, out_features=n_classes)

    def forward(self, t):
        t = self.scs1(t)
        t = self.pool1(t)

        t = self.scs2(t)
        t = self.pool2(t)

        t = self.scs3(t)
        t = self.pool3(t)

        t = t.reshape(-1, n_units_3)
        t = self.out(t)

        return t