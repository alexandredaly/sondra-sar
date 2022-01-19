import os
import yaml
import argparse

from data_reader import Uavsar_slc_stack_1x1


def build_dataset(cfg):
    # Create directory for saving training images
    if not os.path.isdir(cfg["TRAIN_DATA_DIR"]):
        os.mkdir(cfg["TRAIN_DATA_DIR"])
        os.mkdir(os.path.join(cfg["TRAIN_DATA_DIR"], 'low_resolution'))
        os.mkdir(os.path.join(cfg["TRAIN_DATA_DIR"], 'high_resolution'))
    
    # Init data reader
    sardata = Uavsar_slc_stack_1x1(cfg["RAW_DATA_DIR"])
    sardata.read_meta_data(polarisation=cfg["DATASET"]["POLARISATION"])

    # Read all sar data
    for identifier in sardata.meta_data:

        # Build the training set 
        sardata.read_data(identifier, crop = cfg["DATASET"]["IMAGE_SIZE"])
        for i in range(1,8):
            try:
                sardata.subband_process(identifier+"_s{}_1x1.slc".format(i), decimation = cfg["DATASET"]["PREPROCESSING"]["DECIMATION"],
                                            wd=cfg["DATASET"]["PREPROCESSING"]["WINDOW"])
            except:
                break


if __name__=="__main__":
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