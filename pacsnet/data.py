import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from torch.utils.data import Dataset

from pacsnet.augmentations import (RandomCrop3d, RandomFlipNd, RandomGamma,
                                   RandomGaussianNoise, RandomRotationNd)


def load_volume(dataset, labeled=True, slice_range=None):
    ''' Load slices into a volume. Keeps the memory requirement
        as low as possible by using uint8 and uint16 in CPU memory.
    '''
    if labeled:
        path = os.path.join(dataset, "labels", "*.tif")
    else:
        path = os.path.join(dataset, "images", "*.tif")

    dataset = sorted(glob.glob(path))
    volume = None
    target = None
    keys = []
    offset = 0 if slice_range is None else slice_range[0]
    depth = len(
        dataset) if slice_range is None else slice_range[1] - slice_range[0]

    for z, path in enumerate(tqdm.tqdm(dataset)):
        if slice_range is not None:
            if z < slice_range[0]:
                continue
            if z >= slice_range[1]:
                continue

        parts = path.split(os.path.sep)
        key = parts[-3] + "_" + parts[-1].split(".")[0]
        keys.append(key)

        if labeled:
            label = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            label = np.array(label, dtype=np.uint8)
            if target is None:
                target = np.zeros((1, depth, *label.shape[-2:]), dtype=np.uint8)
            target[:, z - offset] = label

        path = path.replace("/labels/", "/images/")
        path = path.replace("/kidney_3_dense/", "/kidney_3_sparse/")
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        image = np.array(image, dtype=np.uint16)

        if volume is None:
            volume = np.zeros((1, depth, *image.shape[-2:]), dtype=np.uint16)
        volume[:, z - offset] = image

    return volume, target, keys


class RandomVolumetricDataset(Dataset):
    ''' Dataset for segmentation of a sparse class. Keeps
        track of positive samples and favors samples that
        contain a positive sample.
        WARNING: do not use in a distributed setting.
    '''

    def __init__(self,
                 datasets,
                 shape=(256, 256, 256),
                 length=1000,
                 transform=None):
        self.volumes = []
        self.targets = []
        self.length = length
        self.shape = shape
        self.transform = transform
        self.nonzero = []

        for dataset in datasets:
            print("loading volume", dataset)
            volume, target, _ = load_volume(dataset)
            self.volumes.append(volume)
            self.targets.append(target)
            self.nonzero.append(np.argwhere(target > 0))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vidx = torch.randint(len(self.volumes), (1,)).item()
        volume = self.volumes[vidx]
        target = self.targets[vidx]
        nonzero = self.nonzero[vidx]
        random = torch.rand(1)

        if random > 0.9:
            # Load a random subvolume
            z,y,x = torch.randint(volume.shape[-3]-self.shape[-3], (1,)).item(), \
                    torch.randint(volume.shape[-2]-self.shape[-2], (1,)).item(), \
                    torch.randint(volume.shape[-1]-self.shape[-1], (1,)).item()
        else:
            # Load a subvolume containing a random sample
            idx = torch.randint(len(nonzero), (1,)).item()
            c, z, y, x = nonzero[idx]

            z += torch.randint(self.shape[-3],
                               (1,)).sub(self.shape[-3] // 2).item()
            y += torch.randint(self.shape[-2],
                               (1,)).sub(self.shape[-2] // 2).item()
            x += torch.randint(self.shape[-1],
                               (1,)).sub(self.shape[-1] // 2).item()

            z = min(max(0, z + self.shape[-3] // 2),
                    volume.shape[-3] - self.shape[-3])
            y = min(max(0, y + self.shape[-2] // 2),
                    volume.shape[-2] - self.shape[-2])
            x = min(max(0, x + self.shape[-3] // 2),
                    volume.shape[-1] - self.shape[-1])

        volume = volume[:, z:z + self.shape[-3], y:y + self.shape[-2],
                        x:x + self.shape[-1]]
        target = target[:, z:z + self.shape[-3], y:y + self.shape[-2],
                        x:x + self.shape[-1]]

        volume = torch.from_numpy((volume / 65536).astype(np.float32))
        target = torch.from_numpy(target > 0).float()
        if self.transform is not None:
            rng = torch.get_rng_state()
            volume = self.transform(volume)
            torch.set_rng_state(rng)
            target = self.transform(target)

        return volume, target
