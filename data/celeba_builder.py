# coding: utf-8

import pathlib
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


# def process_image(srcpath, dstpath):

IMG_HEIGHT = 218
IMG_WIDTH = 178


def main(args):
    dataset = torchvision.datasets.CelebA(
        str(args.rootdir), transform=transforms.Compose([transforms.Grayscale()])
    )

    highres_to_lowres = transforms.Compose(
        [transforms.Resize(size=(IMG_HEIGHT // 2, IMG_WIDTH // 2))]
    )

    print(f"I loaded {len(dataset)} samples")
    for highres, metadata in dataset:
        lowres = highres_to_lowres(highres)
        print(highres.size, lowres.size)


if __name__ == "__main__":

    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--rootdir",
        type=pathlib.Path,
        required=True,
        help="path to the root dir of celeba",
    )
    parser.add_argument(
        "--datadir",
        type=pathlib.Path,
        required=True,
        help="path to store the preprocessed data",
    )
    args = parser.parse_args()

    main(args)
