import os
import yaml
import argparse

from data_reader import Uavsar_slc_stack_1x1


def build_dataset(cfg):
    # Create directory for saving training images
    if not os.path.isdir(cfg["TRAIN_DATA_DIR"]):
        os.mkdir(cfg["TRAIN_DATA_DIR"])
        os.mkdir(os.path.join(cfg["TRAIN_DATA_DIR"], "low_resolution"))
        os.mkdir(os.path.join(cfg["TRAIN_DATA_DIR"], "high_resolution"))

    # Init data reader
    sardata = Uavsar_slc_stack_1x1(cfg)
    sardata.read_meta_data(polarisation=cfg["DATASET"]["POLARISATION"])

    # Read all sar data
    print(5)
    datapath = pathlib.Path(cfg["TRAIN_DATA_DIR"])
    for identifier in sardata.meta_data:
        # Build the training set
        for filepath in datapath.glob(f"{identifier}*_1x1.slc"):
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
