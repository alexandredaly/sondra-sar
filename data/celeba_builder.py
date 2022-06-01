# coding: utf-8

import pathlib
import argparse
import torchvision


def main(args):
    dataset = torchvision.datasets.CelebA(str(args.rootdir))
    print(f"I loaded {len(dataset)} samples")


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
