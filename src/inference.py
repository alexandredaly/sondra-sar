import os
import yaml
import argparse
import shutil
import pathlib
import torch
from PIL import Image
import numpy as np
import data.loader as loader

from tools.train_utils import get_model, load_network


def main(cfg):
    """Main pipeline to run inference on images

    Args:
        cfg (dict): config with all the necessary parameters
    """

    # Load data
    test_data = loader.load_test(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    model = get_model(cfg)
    model = model.to(device)

    # Load model parameters
    if cfg["MODEL"]["NAME"] not in ["Nearest", "Bilinear", "Bicubic"]:
        load_network(cfg["INFERENCE"]["PATH_TO_MODEL"], model, strict=True)
        print("\n Model has been load !")

    # Init directory to save images if not created
    path_to_save = pathlib.Path(cfg["INFERENCE"]["PATH_TO_SAVE"])
    if path_to_save.exists():
        print(f"Removing {path_to_save}")
        shutil.rmtree(path_to_save)
    path_to_save.mkdir()

    # Start testing loop
    for images, filepaths in test_data:
        images = images.to(device)
        # Pass images through the model
        outputs = model(images)

        # Save images
        for output, filepath in zip(outputs, filepaths):
            print(filepath)
            filepath = pathlib.Path(filepath)
            output = output.cpu().detach().numpy()
            np.save(path_to_save / filepath.name, output)


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    main(cfg)
