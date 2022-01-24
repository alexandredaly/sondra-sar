import os
import torch
import torch.nn as nn
import numpy as np

from models.SwinTransformer import SwinIR
from tools.SSIMLoss import SSIMLoss


class ModelCheckpoint:

    """Define the model checkpoint class
    """

    def __init__(self, dir_path, model, epochs, checkpoint_step):
        self.min_loss = None
        self.dir_path = dir_path
        self.best_model_filepath = os.path.join(self.dir_path, "best_model.pth")
        self.model = model
        self.epochs = epochs
        self.checkpoint_step = checkpoint_step

    def update(self, loss, epoch):
        """Update the model if the we get a smaller lost

        Args:
            loss (float): Loss over one epoch
        """

        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.best_model_filepath)
            self.min_loss = loss

        if epoch in np.arange(
            self.checkpoint_step - 1, self.epochs, self.checkpoint_step
        ):

            print(f"Saving model at Epoch {epoch}")

            filename = "epoch_" + str(epoch) + "_model.pth"

            filepath = os.path.join(self.dir_path, filename)
            torch.save(self.model.state_dict(), filepath)


def get_model(cfg):
    """This function loads the right model

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """

    if cfg["MODEL"]["NAME"] == "SwinTransformer":
        # Create an instance of SwinIR model
        return SwinIR(
            upscale=cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"],
            in_chans=cfg["MODEL"]["SWINTRANSFORMER"]["IN_CHANNELS"],
            img_size=cfg["MODEL"]["SWINTRANSFORMER"]["IMG_SIZE"],
            window_size=cfg["MODEL"]["SWINTRANSFORMER"]["WINDOW_SIZE"],
            img_range=cfg["MODEL"]["SWINTRANSFORMER"]["IMG_RANGE"],
            depths=cfg["MODEL"]["SWINTRANSFORMER"]["DEPTHS"],
            embed_dim=cfg["MODEL"]["SWINTRANSFORMER"]["EMBED_DIM"],
            num_heads=cfg["MODEL"]["SWINTRANSFORMER"]["NUM_HEADS"],
            mlp_ratio=cfg["MODEL"]["SWINTRANSFORMER"]["MLP_RATIO"],
            upsampler=cfg["MODEL"]["SWINTRANSFORMER"]["UPSAMPLER"],
            resi_connection=cfg["MODEL"]["SWINTRANSFORMER"]["RESI_CONNECTION"],
        )

    elif cfg["MODEL"]["NAME"] == "SomethingElse":
        return None


def get_loss(cfg):
    """This function returns the loss from the config

    Args:
        cfg (dic): config file

    Returns:
        loss: loss
    """
    if cfg["TRAIN"]["LOSS"]["NAME"] == "SSIM":
        return SSIMLoss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l1":
        return nn.L1Loss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l2":
        return nn.MSELoss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l2sum":
        return nn.MSELoss(reduction="sum")
    else:
        raise NotImplementedError(
            "Loss type [{:s}] is not found.".format(cfg["TRAIN"]["LOSS"])
        )


def get_optimizer(cfg, params):
    """This function returns the correct optimizer

    Args:
        cfg (dic): config

    Returns:
        torch.optimizer: train optimizer
    """
    if cfg["TRAIN"]["OPTIMIZER"]["NAME"] == "Adam":
        return torch.optim.Adam(
            params,
            lr=cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["LR_INITIAL"],
            weight_decay=cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["WEIGHT_DECAY"],
        )
    else:
        raise NotImplementedError(
            "Optimizer type [{:s}] is not found.".format(
                cfg["TRAIN"]["OPTIMIZER"]["NAME"]
            )
        )


def get_scheduler(cfg, optimizer):
    """This function returns the correct learning rate scheduler

    Args:
        cfg (dic): config

    Returns:
        torch.optim.lr_scheduler: learning rate scheduler
    """
    if cfg["TRAIN"]["SCHEDULER"]["NAME"] == "MultiStep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["TRAIN"]["SCHEDULER"]["MULTISTEP"]["STEPS"],
            gamma=cfg["TRAIN"]["SCHEDULER"]["MULTISTEP"]["GAMMA"],
        )
    else:
        raise NotImplementedError(
            "Scheduler type [{:s}] is not found.".format(
                cfg["TRAIN"]["SCHEDULER"]["NAME"]
            )
        )


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def load_network(load_path, model, strict=True, param_key="params"):
    """Function to load pretrained model or checkpoint

    Args:
        load_path (string): the path of the checkpoint to load
        model (torch.nn.module): the network
        strict (bool, optional): If the model is strictly the same as the one we load. Defaults to True.
        param_key (str, optional): the key inside state dict. Defaults to 'params'.
    """

    if strict:
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        model.load_state_dict(state_dict, strict=strict)
    else:
        state_dict_old = torch.load(load_path)
        if param_key in state_dict_old.keys():
            state_dict_old = state_dict_old[param_key]
        state_dict = model.state_dict()
        for ((key_old, param_old), (key, param)) in zip(
            state_dict_old.items(), state_dict.items()
        ):
            state_dict[key] = param_old
        model.load_state_dict(state_dict, strict=True)
        del state_dict_old, state_dict
