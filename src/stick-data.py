# coding: utf-8

import pathlib
import argparse
import math
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from data.utils import equalize, to_db


def main(args):
    # List all the npy in the directory
    # and keep the x , y coordinate values
    azimuth_values = set()
    range_values = set()
    basename = None
    datashape = None
    for f in args.datadir.glob("*.npy"):
        filename = f.name
        fields = filename[:-4].split("_")
        azimuth, rng = int(fields[-2]), int(fields[-1])
        if args.azi_start <= azimuth <= args.azi_end:
            azimuth_values.add(azimuth)
        if args.rng_start <= rng <= args.rng_end:
            range_values.add(rng)
        if basename is None:
            basename = "_".join(fields[:-2])
        if datashape is None:
            datashape = np.load(f).shape
    # organize them
    azimuth_values = sorted(list(azimuth_values))
    range_values = sorted(list(range_values))
    n_azimuth = len(azimuth_values)
    n_rows = n_azimuth * datashape[0]
    n_range = len(range_values)
    n_cols = n_range * datashape[1]
    print(f"I will generate an image of size {n_rows} x {n_cols}")
    print(
        f"With azi range in ({min(azimuth_values)}, {max(azimuth_values)}) x ({min(range_values)},{max(range_values)})"
    )
    fulldata = np.zeros((n_rows, n_cols))
    for iazi, azi in tqdm.tqdm(enumerate(azimuth_values)):
        for irng, rng in enumerate(range_values):
            filename = args.datadir / pathlib.Path(f"{basename}_{azi}_{rng}.npy")
            if not filename.exists():
                print(f"{filename} does not exist")
            data = np.load(filename)
            fulldata[
                iazi * datashape[0] : ((iazi + 1) * datashape[0]),
                irng * datashape[1] : ((irng + 1) * datashape[1]),
            ] = to_db(data[...])
    plt.figure()
    plt.imshow(equalize(fulldata.T)[0], cmap=plt.cm.gray)
    plt.axis("off")
    plt.savefig("stick.png", bbox_inches="tight", dpi=1200)


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

    parser.add_argument("--azi_start", type=int, required=False, default=-math.inf)
    parser.add_argument("--azi_end", type=int, required=False, default=math.inf)
    parser.add_argument("--rng_start", type=int, required=False, default=-math.inf)
    parser.add_argument("--rng_end", type=int, required=False, default=math.inf)
    args = parser.parse_args()

    main(args)
