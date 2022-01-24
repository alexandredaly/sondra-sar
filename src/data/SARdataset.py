import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import random

from torch.utils.data import Dataset
from PIL import Image
from skimage import exposure
from skimage import data, img_as_float


class SARdataset(Dataset):
    """Store the SAR data into a torch dataset like object.

    Args:
        Dataset (class): pytorch dataset object
    """

    def __init__(self, root, test=False):
        """
        Args:
            root (str): absolute path of the data files
            image_type (str): type of bandwith used for SAR images. Defaults to 'HH'.
            preprocessing (str): type of preprocessing to perform. Can be either 'padding', 'hanning' or None. Defaults to 'None'.
        """
        self.test = test
        self.root = root

        if self.test:
            self.files_names = [
                os.path.join(self.root, f)
                for f in os.listdir(self.root)
                if os.path.isfile(os.path.join(self.root, f))
            ]

        else:
            self.files_names = [
                f
                for f in os.listdir(os.path.join(self.root, "high_resolution"))
                if os.path.isfile(os.path.join(self.root, "high_resolution", f))
            ]

    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): idx-th item to retrieve

        Returns:
            image_input, image_target: the low resolution image and the high resolution image
        """

        if self.test:
            return (
                Image.fromarray(
                    np.uint8(apply_processing(np.load(self.files_names[idx])))
                ),
                self.files_names[idx].split("/")[-1],
            )

        else:
            image_input = apply_processing(
                np.load(
                    os.path.join(self.root, "low_resolution", self.files_names[idx])
                )
            )
            image_target = apply_processing(
                np.load(
                    os.path.join(self.root, "high_resolution", self.files_names[idx])
                )
            )

            # Perform augmentation on images
            mode = random.randint(0, 7)

            return (
                Image.fromarray(np.uint8(augment_img(image_input, mode=mode))),
                Image.fromarray(np.uint8(augment_img(image_target, mode=mode))),
            )

    def __len__(self):
        """Operator len that returns the size of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.files_names)


def apply_processing(data):
    """A function to have the images in log mode

    Args:
        data (np.array): the images as a numpy array

    Return:
        img (np.array): the processed image
    """

    img = np.log10(np.abs(data) + 1e-8)
    return img


def plot_sample(item, method="stretch"):
    """Function to plot a sample (low_resolution,high_resolution)

    Args:
        item (tuple): an item from the SARdataset
    """

    # Process image to better visualize when plotting
    if method == "stretch":
        p2, p98 = np.percentile(item[0], (2, 98))
        img_low = exposure.rescale_intensity(item[0], in_range=(p2, p98))
        p2, p98 = np.percentile(item[1], (2, 98))
        img_low = exposure.rescale_intensity(item[1], in_range=(p2, p98))

    elif method == "equal":
        img_low = exposure.equalize_hist(item[0])
        img_high = exposure.equalize_hist(item[1])

    else:
        raise NameError("wrong 'method' or not defined")

    # Plot low resolution image
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution")
    plt.imshow(img_as_float(item[0]), cmap=plt.cm.gray)

    # Plot high resolution image
    plt.subplot(1, 2, 2)
    plt.title("High Resolution")
    plt.imshow(img_as_float(item[1]), cmap=plt.cm.gray)
    plt.show()


def augment_img(img, mode=0):
    """Perform augmentation either flip and/or rotation
       From Kai Zhang (github: https://github.com/cszn)

    Args:
        img (np.array): image
        mode (int): transformation mode. Defaults to 0.

    Returns:
        np.array: trasnformed image
    """

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img)).copy()
    elif mode == 2:
        return np.flipud(img).copy()
    elif mode == 3:
        return np.rot90(img, k=3).copy()
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2)).copy()
    elif mode == 5:
        return np.rot90(img).copy()
    elif mode == 6:
        return np.rot90(img, k=2).copy()
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3)).copy()
