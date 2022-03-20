import numpy as np
import matplotlib.pyplot as plt

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


def equalize(image, p2=None, p98=None):
    if not p2:
        p2, p98 = np.percentile(image, (1, 99))
    img = exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(0, 1))
    return img, p2, p98
