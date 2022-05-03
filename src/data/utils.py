import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import pathlib

from skimage import exposure
from skimage import data, img_as_float


def to_db(data, maxi=None):
    """A function to have the images in log mode

    Args:
        data (np.array): the images as a numpy array

    Return:
        img (np.array): the processed image
    """

    img = 20 * np.log10(np.abs(data) + 1e-15)
    return img


def plot_hist(img):

    _ = plt.hist(img, bins="auto")  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def plot_sample(item, method="stretch"):
    """Function to plot a sample (low_resolution,high_resolution)

    Args:
        item (tuple): an item from the SARdataset
    """

    item = list(map(to_db, item))

    # Process image to better visualize when plotting
    if method == "stretch":
        p2, p98 = np.percentile(item[0], (2, 98))
        img_low = exposure.rescale_intensity(item[0], in_range=(p2, p98))
        p2, p98 = np.percentile(item[1], (2, 98))
        img_high = exposure.rescale_intensity(item[1], in_range=(p2, p98))

    elif method == "equal":
        img_low = exposure.equalize_hist(item[0])
        img_high = exposure.equalize_hist(item[1])

    else:
        raise NameError("wrong 'method' or not defined")

    # Plot low resolution image
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution")
    plt.imshow(img_as_float(img_low), cmap=plt.cm.gray)

    # Plot high resolution image
    plt.subplot(1, 2, 2)
    plt.title("High Resolution")
    plt.imshow(img_as_float(img_high), cmap=plt.cm.gray)
    plt.show()


def equalize(image, p2=None, p98=None):
    if not p2:
        p2, p98 = np.percentile(image, (1, 99))
    img = exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(0, 1))
    return img, p2, p98


def augment_img(img):
    """Perform augmentation either flip and/or rotation
           From Kai Zhang (github: https://github.com/cszn)

        Args:
            img (np.array): image
            mode (int): transformation mode. Defaults to 0.
    +60)/60
        Returns:
            np.array: trasnformed image
    """

    mode = random.randint(0, 7)

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


if __name__ == "__main__":
    # Example for plotting a high/low res sample
    cfgpath = "./config.yaml"
    with open(cfgpath, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    fname = "SSurge_12600_14170_008_141120_L090HH_01_BC_s1_1x1_2469.npy"
    highres_datapath = pathlib.Path(cfg["TRAIN_DATA_DIR"]) / "high_resolution"
    high_res = np.load(highres_datapath / fname)
    lowres_datapath = pathlib.Path(cfg["TRAIN_DATA_DIR"]) / "low_resolution"
    low_res = np.load(lowres_datapath / fname)
    item = (low_res, high_res)

    plot_sample(item)
