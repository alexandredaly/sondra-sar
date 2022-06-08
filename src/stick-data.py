# coding: utf-8

import pathlib
import argparse


def main(args):
    # List all the npy in the directory
    # and keep the x , y coordinate values
    azimuth_values = set()
    range_values = set()
    basename = None
    for f in args.datadir.glob("*.npy"):
        filename = f.name
        fields = filename[:-4].split("_")
        azimuth, range = fields[-2], fields[-1]
        azimuth_values.add(int(azimuth))
        range_values.add(int(range))
        if basename is None:
            basename = "_".join(fields[:-2])
    # organize them
    azimuth_values = sorted(list(azimuth_values))
    range_values = sorted(list(range_values))
    for azi in azimuth_values:
        for rng in range_values:
            filename = args.datadir / pathlib.Path(f"{basename}_{azi}_{rng}.npy")
            if not filename.exists():
                print(f"{filename} does not exist")


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--datadir",
        type=pathlib.Path,
        required=True,
        help="path to datadir (low, high, fake high, outputs)",
    )
    args = parser.parse_args()

    main(args)
