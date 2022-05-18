import os
import yaml
import argparse
import pathlib

from data_reader import Uavsar_slc_stack_1x1


def build_dataset(cfg):
    # Create directory for saving training images
    datadir = pathlib.Path(cfg["TRAIN_DATA_DIR"])
    if not datadir.exists():
        datadir.mkdir()
    lowres_datadir = datadir / "low_resolution"
    if not lowres_datadir.exists():
        lowres_datadir.mkdir()
    highres_datadir = datadir / "high_resolution"
    if not highres_datadir.exists():
        highres_datadir.mkdir()
    fake_highres_datadir = datadir / "fake_high_resolution"
    if not fake_highres_datadir.exists():
        fake_highres_datadir.mkdir()

    # Init data reader
    sardata = Uavsar_slc_stack_1x1(cfg)
    sardata.read_meta_data(polarisation=cfg["DATASET"]["POLARISATION"])

    # Read all sar data
    print(5)
    datapath = pathlib.Path(cfg["RAW_DATA_DIR"])
    for identifier in sardata.meta_data:
        # Build the training set
        print(f"Globing for {identifier}*_1x1.slc in {datapath}")
        print(list(datapath.glob(f"{identifier}*.slc")))
        for filepath in datapath.glob(f"{identifier}*.slc"):
            # Get the segment number
            sstr = filepath.name.split("_")[-2]  # should s1 or s2, or ...
            seg = int(sstr[1:])
            sardata.read_data(identifier, seg, crop=cfg["DATASET"]["IMAGE_SIZE"])
            sardata.subband_process(
                f"{identifier}_s{seg}_1x1.slc",
                downscale_factor=cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"],
                decimation=cfg["DATASET"]["PREPROCESSING"]["DECIMATION"],
                wd=cfg["DATASET"]["PREPROCESSING"]["WINDOW"],
            )


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="../config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    # Build the dataset
    build_dataset(cfg)
