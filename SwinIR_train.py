import os
import yaml
import argparse
import torch                                                  
import torchvision.transforms as transforms
from data.SARdataset import SARdataset
from data.dataset_transformer import DatasetTransformer

# Create datasets
def create_dataset(cfg):
    # Get the validation ratio from the config.yaml file
    valid_ratio = cfg['DATASET']['VALID_RATIO']

    # Get the dataset for the training/validation sets
    train_valid_dataset = SARdataset(os.path.join('data/', cfg['TRAIN_DATA_DIR'][2:]))

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset)) + 1
    nb_valid =  int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])
    
    # Define train and validation sets
    train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())
    return train_dataset, valid_dataset


if __name__=="__main__":
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

    # Build the datasets
    train_dataset, valid_dataset = create_dataset(cfg)

