import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn


class RandomRotationNd(nn.Module):
    ''' This augmentation first permutes the dimensions as an initial rotation
        to select the rotation axis, then rotates around the (fixed) z axis. 
        The result is zoomed in to remove empty space and finally permuted 
        once more to move randomize the rotation axis.
    '''

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        angle = torch.rand(1).item() * 360
        keep = torch.arange(x.dim() - self.dims)
        perm = -torch.randperm(self.dims) - 1
        x = x.clone().permute(*[k.item() for k in keep],
                              *[p.item() for p in perm])
        rad = math.pi * angle / 180
        scale = abs(math.sin(rad)) + abs(math.cos(rad))
        for i in range(0, x.shape[-3], 8):  # presumptuous
            v = x[..., i:i + 8, :, :]
            w = v.view(-1, *v.shape[-3:])
            w = TF.rotate(w, angle)
            v = w.view(*v.shape)
            x[..., i:i + 8, :, :] = v
        s = x.shape
        x = F.interpolate(x, scale_factor=scale, mode="bilinear")
        x = TF.center_crop(x, s[-2:])
        perm = -torch.randperm(self.dims) - 1
        x = x.permute(*[k.item() for k in keep], *[p.item() for p in perm])
        return x


class RandomRot90Nd(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        dims = -torch.randperm(self.dims)[:2] - 1
        dims = [d.item() for d in dims]
        rot = torch.randint(4, (1,)).item()
        return x.rot90(rot, dims)


class RandomPermuteNd(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        perm = -torch.randperm(self.dims) - 1
        keep = torch.arange(x.dim() - self.dims)
        return x.permute(*[k.item() for k in keep], *[p.item() for p in perm])


class RandomFlipNd(nn.Module):

    def __init__(self, dims, p=0.5):
        super().__init__()
        self.dims = dims
        self.p = p

    def forward(self, x):
        for i in range(self.dims):
            if torch.rand(1) < self.p:
                x = x.flip(-i - 1)
        return x


class ToDevice(nn.Module):
    ''' Sometimes it helps to move the tensor to the gpu before augmentations like
        rotation. Note however that you need to set num_workers to 0 in the dataloader
    '''

    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return x.to(self.device)


class RandomCrop3d(nn.Module):
    ''' Randomly crop a subvolume between min_shape and the shape of the input.
        If the input tensor is smaller than shape, it is padded with zeros.
        The crop is then interpolated to the original size.
    '''

    def __init__(self, min_shape):
        super().__init__()
        self.min_shape = min_shape

    def forward(self, x):
        # Randomly select the crop size
        h, w, d = x.shape[-3:]

        height = torch.randint(self.min_shape[-3], h, (1,)).item()
        width = torch.randint(self.min_shape[-2], w, (1,)).item()
        depth = torch.randint(self.min_shape[-1], d, (1,)).item()

        # Randomly select the crop location
        top = torch.randint(0, h - height, (1,)).item()
        left = torch.randint(0, w - width, (1,)).item()
        front = torch.randint(0, d - depth, (1,)).item()

        # Crop and interpolate
        x = x[..., top:top + height, left:left + width, front:front + depth]
        x = F.interpolate(x, size=(h, w, d), mode="trilinear")

        return x


class RandomGaussianNoise(nn.Module):
    ''' Add gaussian noise to the input tensor. The noise is scaled by the
        standard deviation of the input.
    '''

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x + torch.randn_like(x) * x.std()
        return x


class RandomGamma(nn.Module):
    ''' Randomly change contrast of an image by raising its values to the power gamma
    '''

    def __init__(self, p=0.5, log_gamma_range=(-0.3, 0.3)):
        super().__init__()
        self.p = p
        self.log_gamma_range = log_gamma_range

    def forward(self, x):
        if torch.rand(1) < self.p:
            log_gamma = torch.rand(1).item() * (self.log_gamma_range[1] -
                                                self.log_gamma_range[0]) + \
                self.log_gamma_range[0]
            gamma = math.exp(log_gamma)
            return x**gamma
        return x


class RandomBlur:
    pass
