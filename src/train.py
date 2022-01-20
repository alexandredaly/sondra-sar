import os
import yaml
import argparse
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

import data.loader as loader

from tools.train_utils import (
    ModelCheckpoint,
    get_loss,
    get_model,
    get_optimizer,
    get_scheduler,
    generate_unique_logpath,
    load_network,
)
from tools.trainer import train_one_epoch
from tools.valid import valid_one_epoch


def main(cfg, path_to_config):
    """Main pipeline to train a model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """

    # Load data
    train_loader, valid_loader = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    model = get_model(cfg)
    model = model.to(device)

    # Load pre trained model parameters
    if cfg["TRAIN"]["PRETRAINED"]["BOOL"]:
        load_network(
            cfg["TRAIN"]["PRETRAINED"]["PATH"],
            model,
            strict=cfg["TRAIN"]["PRETRAINED"]["STRICT"],
            param_key="params",
        )
        print("\n Model has been load !")

    # Define the loss
    f_loss = get_loss(cfg).to(device)

    # Define the optimizer
    optimizer = get_optimizer(cfg, model.parameters())

    # Define Scheduler
    scheduler = get_scheduler(cfg, optimizer)

    # Tracking with tensorboard
    tensorboard_writer = SummaryWriter(log_dir=cfg["TRAIN"]["TENSORBOARD_DIR"])

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_MODEL_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODEL"]["NAME"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save config for each training
    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    # Init Checkpoint class
    checkpoint = ModelCheckpoint(
        save_dir, model, cfg["TRAIN"]["EPOCH"], cfg["TRAIN"]["CHECKPOINT_STEP"]
    )

    # Start training loop
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH : {}".format(epoch))

        training_loss = train_one_epoch(
            model,
            train_loader,
            f_loss,
            optimizer,
            device,
            cfg["TRAIN"]["LOSS"]["WEIGHT"],
            clipgrad = cfg["TRAIN"]["OPTIMIZER"]["CLIPGRAD"]
        )
        valid_loss, psnr, restored_images, target_restored_images = valid_one_epoch(
            model, valid_loader, f_loss, device, cfg["TRAIN"]["LOSS"]["WEIGHT"]
        )

        # Update scheduler
        scheduler.step()

        # Save best model
        checkpoint.update(val_loss, epoch)

        # Get current learning rate
        learning_rate = scheduler.optimizer.param_groups[0]["lr"]

        # Track performances with tensorboard
        tensorboard_writer.add_scalar("training_loss", training_loss, epoch)
        tensorboard_writer.add_scalar("valid_loss", valid_loss, epoch)
        tensorboard_writer.add_scalar("psnr", psnr, epoch)
        tensorboard_writer.add_scalar("lr", learning_rate, epoch)

        # Save images in tensorboard
        restored_img_grid = make_grid(restored_images)
        target_restored_img_grid = make_grid(target_restored_images)

        tensorboard_writer.add_image("Restored images", restored_img_grid)
        tensorboard_writer.add_image("Target Restored images", target_restored_img_grid)


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

    main(cfg, args.path_to_config)
