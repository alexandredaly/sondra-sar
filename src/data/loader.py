import datetime
import torch
import torchvision.transforms as transforms
import numpy as np
import joblib

from data.SARdataset import SARdataset
from data.utils import augment_img


class DatasetTransformer(torch.utils.data.Dataset):
    """Apply transformation to a torch Dataset"""

    def __init__(self, base_dataset, transform, augment=False, test=False):
        """Initialize DatasetTransformer class

        Args:
            base_dataset (torchvision.datasets.folder.ImageFolder): Image dataset
            transform (torchvision.transforms.Compose): List of transformation to apply
        """
        self.augment = augment
        self.test = test
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        if self.test:
            return self.transform(np.expand_dims(img, 0)).float(), target
        else:
            if self.augment:
                img = augment_img(img)
                target = augment_img(target)

            return (
                self.transform(np.expand_dims(img, 0)).float(),
                self.transform(np.expand_dims(target, 0)).float(),
            )

    def __len__(self):
        return len(self.base_dataset)


def get_max(loader):
    """
    Get the max pixel value of the whole dataset
    """
    # Init max
    # maxi = -np.inf
    # Loop over the dataset
    print("####################################################")
    print("####### COMPUTE MAX OVER THE WHOLE DATASET #########")
    # for imgs, _ in tqdm.tqdm(loader):
    #     candidate = joblib.delay(np.max)(imgs) for (imgs, _) in loader)
    #     if candidate > maxi:
    #         maxi = candidate
    print(str(datetime.datetime.now()))
    maxi = max(
        joblib.Parallel(n_jobs=-2)(joblib.delayed(np.max)(imgs) for (imgs, _) in loader)
    )
    print(str(datetime.datetime.now()))

    print(f"######### Max = {maxi} db has been saved #########")
    print("#############################################################")
    return maxi


def create_dataset(cfg):
    """Function to create datasets torchget_max
    torch.dataset: validation and train sets
    """

    # Get the validation ratio from the config.yaml file
    valid_ratio = cfg["DATASET"]["VALID_RATIO"]

    # Get the dataset for the training/validation sets
    train_valid_dataset = SARdataset(cfg["TRAIN_DATA_DIR"])

    # Get dataset maximum
    maxi = get_max(train_valid_dataset)

    # Store max value of preprocessed dataset
    np.save("./data/dataset_maximum.npy", maxi)

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset)) + 1
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset,
        [nb_train, nb_valid],
        generator=torch.Generator().manual_seed(55),
    )

    # Apply transforms (to Tensor - retrieve max - clip )
    train_dataset = DatasetTransformer(
        train_dataset,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.permute(1, 0, 2) - maxi),
                transforms.Lambda(
                    lambda x: x.clamp_(
                        min=cfg["DATASET"]["CLIP"]["MIN"],
                        max=cfg["DATASET"]["CLIP"]["MAX"],
                    )
                ),
                transforms.Lambda(
                    lambda x: x
                    if not cfg["TRAIN"]["LOSS"]["NAME"] == "SSIM"
                    else x
                    / (cfg["DATASET"]["CLIP"]["MAX"] - cfg["DATASET"]["CLIP"]["MIN"])
                    + 1
                ),
                transforms.Lambda(
                    lambda x: x.expand(3, -1, -1)
                    if cfg["DATASET"]["IN_CHANNELS"] == 3
                    else x
                ),
            ]
        ),
        augment=True,
    )

    valid_dataset = DatasetTransformer(
        valid_dataset,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.permute(1, 0, 2) - maxi),
                transforms.Lambda(
                    lambda x: x.clamp_(
                        min=cfg["DATASET"]["CLIP"]["MIN"],
                        max=cfg["DATASET"]["CLIP"]["MAX"],
                    )
                ),
                transforms.Lambda(
                    lambda x: x
                    if not cfg["TRAIN"]["LOSS"]["NAME"] == "SSIM"
                    else x
                    / (cfg["DATASET"]["CLIP"]["MAX"] - cfg["DATASET"]["CLIP"]["MIN"])
                    + 1
                ),
                transforms.Lambda(
                    lambda x: x.expand(3, -1, -1)
                    if cfg["DATASET"]["IN_CHANNELS"] == 3
                    else x
                ),
            ]
        ),
    )

    return train_dataset, valid_dataset


def create_dataloader(cfg, train_dataset, valid_dataset):
    """Generate torch loader from torch datasets

    Args:
        cfg (dic): config
        train_dataset (torch.dataset): training data
        valid_dataset (torch.dataset): validation data

    Returns:
        torch loaders: train and val loaders
    """

    # Define the dataloaders for the train and validation sets
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=True,  # <-- this reshuffles the data at every epoch
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    return train_loader, valid_loader


def load_train(cfg):
    """Main fonction of the loader. Last step concerning the data processing

    Args:
        cfg (dict): config

    Returns:
        torch loaders: train and val loaders
    """

    # Build the datasets
    train_dataset, valid_dataset = create_dataset(cfg)

    # Build the dataloaders
    train_loader, valid_loader = create_dataloader(cfg, train_dataset, valid_dataset)

    # Print informations about the dataset size and number of batches
    print(
        "The train set contains {} images, in {} batches".format(
            len(train_loader.dataset), len(train_loader)
        )
    )
    print(
        "The validation set contains {} images, in {} batches".format(
            len(valid_loader.dataset), len(valid_loader)
        )
    )

    return train_loader, valid_loader


def load_test(cfg):
    """Loads testing images

    Args:
        cfg (dict): config file

    Returns :
        Tensor of images
    """

    # Init test dataset
    test_data = SARdataset(cfg["INFERENCE"]["PATH_TO_IMAGES"], test=True)

    # Get the maximum of the training data
    maxi = np.load("./data/dataset_maximum.npy")

    # Apply transforms
    test_dataset = DatasetTransformer(
        test_data,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.permute(1, 0, 2) - maxi),
                transforms.Lambda(
                    lambda x: x.clamp_(
                        min=cfg["DATASET"]["CLIP"]["MIN"],
                        max=cfg["DATASET"]["CLIP"]["MAX"],
                    )
                ),
                transforms.Lambda(
                    lambda x: x.expand(3, -1, -1)
                    if cfg["DATASET"]["IN_CHANNELS"] == 3
                    else x
                ),
            ]
        ),
        test=True,
    )

    # Build Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg["INFERENCE"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    return test_loader
