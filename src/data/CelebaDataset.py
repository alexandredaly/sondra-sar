# coding: utf-8

# Standard imports
import pathlib

# External imports
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

_NUM_SAMPLES_DRY_RUN = 20

_IMG_HEIGHT = 218
_IMG_WIDTH = 178


class CelebaDataset(Dataset):
    def __init__(self, rootdir: pathlib.Path, dry_run=False):
        self.dataset = torchvision.datasets.CelebA(
            str(rootdir), transform=transforms.Compose([transforms.Grayscale()])
        )

        if dry_run:
            indices = list(range(len(self.dataset)))[:_NUM_SAMPLES_DRY_RUN]
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        self.totensor = transforms.PILToTensor()
        self.highres_to_lowres = transforms.Compose(
            [transforms.Resize(size=(_IMG_HEIGHT // 2, _IMG_WIDTH // 2))]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        highres, meta = self.dataset[idx]

        lowres = self.highres_to_lowres(highres)

        return self.totensor(lowres).float(), self.totensor(highres).float()
