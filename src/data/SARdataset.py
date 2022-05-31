import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from torch.utils.data import Dataset
from skimage import exposure
from skimage import data, img_as_float

from data.utils import to_db

_NUM_SAMPLES_DRY_RUN = 20


class SARdataset(Dataset):
    """Store the SAR data into a torch dataset like object.

    Args:
        Dataset (class): pytorch dataset object
    """

    def __init__(
        self, root, use_fake_high=False, test=False, augment_valid=False, dry_run=False
    ):
        """
        Args:
            root (str): absolute path of the data files
            image_type (str): type of bandwith used for SAR images. Defaults to 'HH'.
            preprocessing (str): type of preprocessing to perform. Can be either 'padding', 'hanning' or None. Defaults to 'None'.
        """
        self.test = test
        self.root = root
        self.augment_valid = augment_valid

        self.low_resolution_dir = "low_resolution"
        self.high_resolution_dir = (
            "fake_high_resolution" if use_fake_high else "high_resolution"
        )

        if self.test:
            self.files_names = [
                os.path.join(self.root, f)
                for f in os.listdir(self.root)
                if os.path.isfile(os.path.join(self.root, f))
            ]

        else:
            self.files_names = [
                f
                for f in os.listdir(os.path.join(self.root, self.high_resolution_dir))
                if os.path.isfile(os.path.join(self.root, self.high_resolution_dir, f))
            ]
        # For a dry run experiment, just take few samples
        if dry_run:
            self.files_names = self.files_names[:_NUM_SAMPLES_DRY_RUN]

    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): idx-th item to retrieve

        Returns:
            image_input, image_target: the low resolution image and the high resolution image
        """

        if self.test:
            return (
                to_db(np.load(self.files_names[idx])),
                self.files_names[idx].split("/")[-1],
            )

        else:
            image_input = to_db(
                np.load(
                    os.path.join(
                        self.root, self.low_resolution_dir, self.files_names[idx]
                    )
                )
            )
            image_target = to_db(
                np.load(
                    os.path.join(
                        self.root, self.high_resolution_dir, self.files_names[idx]
                    )
                )
            )

            return image_input, image_target

    def __len__(self):
        """Operator len that returns the size of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.files_names)
