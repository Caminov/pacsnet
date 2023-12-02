from typing import Callable, Tuple

import torch
from torch import nn

from pacsnet.blocks import CatConv, Downsampler, Upsampler


class UNet(nn.Module):

    def __init__(self, img_shape: Tuple[int], merger: Callable = CatConv):
        super().__init__()

        B, _, D, _, _ = img_shape

        self.downsampler = Downsampler(img_shape, padding='valid')
        self.upsampler = Upsampler(input_shape=(B, D // 2, 2, 2, 2),
                                   merger=merger)

    def forward(self, x):
        features = self.downsampler(x)
        x = self.upsampler(features)

        return x
