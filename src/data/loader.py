import torch
import torchvision.transforms as transforms

from data.SARdataset import SARdataset


class DatasetTransformer(torch.utils.data.Dataset):
    """Apply transformation to a torch Dataset
    """

    def __init__(self, base_dataset, transform, test=False):
        """Initialize DatasetTransformer class

        Args:
            base_dataset (torchvision.datasets.folder.ImageFolder): Image dataset
            transform (torchvision.transforms.Compose): List of transformation to apply
        """
        self.test = test
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        if self.test:
            return self.transform(img), target
        else:
            return self.transform(img), self.transform(target)

    def __len__(self):
        return len(self.base_dataset)


def create_dataset(cfg):
    """Function to create datasets torch object

    Args:
        cfg (dict): config

    Returns:
        torch.dataset: validation and train sets
    """

    # Get the validation ratio from the config.yaml file
    valid_ratio = cfg["DATASET"]["VALID_RATIO"]

    # Get the dataset for the training/validation sets
    train_valid_dataset = SARdataset(cfg["TRAIN_DATA_DIR"])
    # Store max value of preprocessed dataset in max attribute  of SARDataset
    train_valid_dataset.compute_max_dataset()
    print(len(train_valid_dataset))
    print((1.0 - valid_ratio) * len(train_valid_dataset), "nb_train")
    print(valid_ratio * len(train_valid_dataset), "nb_valid")
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset)) + 1
    print(nb_train)
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    print(nb_valid)
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )
    print(nb_valid)

    # Apply transforms (unsqueez dim 1 and convert to tensor)
    train_dataset = DatasetTransformer(
        train_dataset,
        transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        ),
    )
    valid_dataset = DatasetTransformer(
        valid_dataset,
        transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        ),
    )

    return train_dataset, valid_dataset


def create_dataloader(cfg, train_dataset, valid_dataset):
    """ Generate torch loader from torch datasets

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

    # Apply transforms
    test_dataset = DatasetTransformer(
        test_data,
        transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
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
