import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

import pacsnet.layers as L


class SingleConv(nn.Sequential):
    '''
    Basic convolutional module consisting of conv3d, non-linearity, and
    optional normmalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``
        norm_layer (nn.Module): Normalization layer. Default: ``None``
        act_layer (nn.Module): Activation layer. Default: ``nn.GELU``
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True,
                 norm_layer: Optional[Callable] = L.SwitchNorm3d,
                 act_layer: Callable = nn.GELU,
                 **kwargs):
        super().__init__()

        self.add_module(
            'conv',
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias))

        self.add_module('activation', act_layer())

        if norm_layer is not None:
            self.add_module('norm', norm_layer(out_channels, **kwargs))


class CatConv(nn.Module):
    '''
    Concatenation module consisting of concatenation, then SingleConv module.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the convolving kernel.
        **kwargs: Additional keyword arguments to pass to SingleConv.
    '''

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.conv = SingleConv(
            in_channels * 2,
            in_channels,
            kernel_size,
            **kwargs,
        )

    def forward(self, x_b, x_l):
        x_b = torch.cat([x_b, x_l], dim=1)
        x_b = self.conv(x_b)
        return x_b


class Downsampler(nn.Module):
    '''
    Downsampling module consisting of multiple SingleConv modules.

    Args:
        input_shape (tuple): Shape of the input tensor.
        kernel_size (int): Size of the convolving kernel. Default: 2
        padding (str): Zero-padding added to both sides of the input. Default: valid
    '''

    def __init__(
        self,
        input_shape: tuple,
        kernel_size: int = 3,
        padding: 'str' = 'valid',
    ):
        super().__init__()
        self.kernel_size = kernel_size

        B, C, D, H, W = input_shape

        assert D == H == W, "Input shape must be cubic"

        n_convs = int(math.log2(D)) - 1
        ins = [2**i for i in range(n_convs)]
        outs = [2**i for i in range(1, n_convs + 2)]
        cube_sizes = [D // 2**i for i in range(n_convs + 1)]
        self.convs = nn.ModuleList([
            SingleConv(
                in_channels=ins[i],
                out_channels=outs[i],
                kernel_size=min(cube_sizes[i], kernel_size),
                stride=2,
                padding=1,
            ) for i in range(n_convs)
        ])

    def forward(self, x):
        features = [x]

        for conv in self.convs:
            x = conv(x)
            features.append(x)

        return features


class Upsampler(nn.Module):
    '''
    Upsampling module consisting of transposed convolutions, concatenation,
    then SingleConv modules.
    '''

    def __init__(
        self,
        input_shape: tuple,
        kernel_size: int = 2,
        merger: Callable = CatConv,
    ):
        super().__init__()

        B, C, D, H, W = input_shape

        assert D == H == W, "Input shape must be cubic"

        n_convs = int(math.log2(C))
        ins = [2**i for i in range(n_convs, 0, -1)]
        outs = [2**i for i in range(n_convs - 1, -1, -1)]
        cube_sizes = [D**(i + 1) for i in range(n_convs)]
        self.convs = nn.ModuleList([
            nn.ConvTranspose3d(
                in_channels=ins[i],
                out_channels=outs[i],
                kernel_size=min(cube_sizes[i], kernel_size),
                stride=2,
                padding=0,
            ) for i in range(n_convs)
        ])

        self.mergers = nn.ModuleList([
            merger(
                in_channels=outs[i],
                kernel_size=3,
                padding='same',
            ) for i in range(n_convs)
        ])

    def forward(self, features):
        x_b = features.pop()

        for x_l, conv, merger in zip(
                reversed(features),
                self.convs,
                self.mergers,
        ):
            x_b = conv(x_b)
            x_b = merger(x_b, x_l)

        return x_b


class Mlp(nn.Module):
    ''' MLP as used in Vision Transformer, MLP-Mixer and related networks
    '''

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: int = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
        drop: float = 0.,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [bias] * 2
        drop_probs = [drop] * 2
        linear_layer = partial(nn.Conv2d,
                               kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LatentSpaceBuilder(nn.Module):
    ''' Builds a latent space from the output of the downsampler
    This is where the fun part begins.
    '''

    def __init__(
        self,
        input_shape: tuple,
        hidden_features: Optional[int] = None,
        out_features: int = 256,
        **kwargs,
    ):
        super().__init__()

        B, C, D, H, W = input_shape
        in_channels = C * D * H * W
        hidden_features = hidden_features or in_channels

        self.mlp = Mlp(
            in_channels,
            hidden_features=hidden_features,
            out_features=out_features,
            **kwargs,
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.mlp(x)
        return x


x = torch.randn(1, 1, 256, 256, 256)
downsampler = Downsampler(x.shape, padding='valid')
upsampler = Upsampler(input_shape=(1, 128, 2, 2, 2))
out = downsampler(x)
seg = upsampler(out)
# lsb = LatentSpaceBuilder((1, x.shape[2] // 2, 2, 2, 2))
# latent = lsb(out[-1])

# B, F = latent.shape

# #params in lsb
# print(sum(p.numel() for p in lsb.parameters()))

# l1 = latent.clone().view(B, 1, F, 1, 1)
# l2 = latent.clone().view(B, 1, 1, F, 1)
# l3 = latent.clone().view(B, 1, 1, 1, F)

# Obtain a volume of size (B, 1, F, F, F)
# by multiplying the three latent vectors
# volume = l1 * l2 * l3
