import os
import yaml
import argparse
import torch                                                  
import torchvision.transforms as transforms

from SARdataset import SARdataset

############################################################################################ Datasets

def load_dataset(cfg):
    valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

    # Get the dataset for the training/validation sets
    train_valid_dataset = SARdataset("./data_files/train")

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid =  int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                    transform= None, #transforms.ToTensor(),
                                                    train=False)

    train_dataset = SARdataset(train_dataset, transforms.ToTensor())
    valid_dataset = SARdataset(valid_dataset, transforms.ToTensor())
    test_dataset  = SARdataset(test_dataset , transforms.ToTensor())


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
    load_dataset(cfg)