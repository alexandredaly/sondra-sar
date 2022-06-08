# coding: utf-8

import pathlib
import argparse


def main(args):
    # List all the npy in the directory
    # and keep the x , y coordinate values
    x_values = set()
    y_values = set()
    basename = None
    for f in args.datadir.glob("*.npy"):
        filename = f.name
        fields = filename[:-4].split("_")
        x, y = fields[-2], fields[-1]
        if not int(x) in x_values:
            x_values.add(int(x))
            print(f"Add {x}")
            print(filename)
        y_values.add(int(y))
        if basename is None:
            basename = "_".join(fields[:-2])
    # organize them
    print(list(x_values))
    x_values = sorted(list(x_values))
    y_values = sorted(list(y_values))
    print(x_values)
    print(basename)
    for x in x_values:
        for y in y_values:
            filename = args.datadir / pathlib.Path(f"{basename}_{x}_{y}.npy")
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
