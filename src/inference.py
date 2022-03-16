import os
import yaml
import argparse
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
    load_network(cfg["INFERENCE"]["PATH_TO_MODEL"], model, strict=True)
    print("\n Model has been load !")

    # Init directory to save images if not created
    if not os.path.exists(cfg["INFERENCE"]["PATH_TO_SAVE"]):
        os.mkdir(cfg["INFERENCE"]["PATH_TO_SAVE"])

    # Start testing loop
    for images, names in test_data:
        images = images.to(device)
        print(images.shape)
        # Pass images through the model
        outputs = model(images)

        # Save images
        for i, image in enumerate(outputs):
            image = image.cpu().detach().numpy()

            # Save output numpy array
            if cfg["INFERENCE"]["SAVE_ARRAY"]:
                np.save(os.path.join(cfg["INFERENCE"]["PATH_TO_SAVE"], names[i]),image)

            # Save output png image
            if cfg["INFERENCE"]["SAVE_PNG"]:
                img = Image.fromarray(np.uint8(image.squeeze().squeeze()))
                img.save(
                    os.path.join(
                        cfg["INFERENCE"]["PATH_TO_SAVE"], names[i][:-3] + "png"
                    )
                )


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
