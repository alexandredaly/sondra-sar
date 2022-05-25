import os
import yaml
import argparse
import torch
import numpy as np
from shutil import copyfile
import torchinfo

import data.loader as loader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tools.regularizers import regularizer_orth2, regularizer_clip
from data.utils import equalize

import neptune.new as neptune
from neptune.new.types import File
from skimage import img_as_float


def eval_upsample(mode, valid_loader, f_loss, device, cfg):
    model = torch.nn.Upsample(scale_factor=2, mode=mode)
    (_, psnr, _, _, _, l1_loss, l2_loss, ssim_loss) = valid_one_epoch(
        model,
        valid_loader,
        f_loss,
        device,
        cfg["DATASET"]["CLIP"]["MAX"] - cfg["DATASET"]["CLIP"]["MIN"],
    )
    print(
        f"Baseline for upsampling {mode} : \n L1 = {l1_loss:.2f}, L2 = {l2_loss:.2f}, psnr={psnr:.2f}, ssim: {ssim_loss:.2f}"
    )
    return psnr, l1_loss, l2_loss


def main(cfg, path_to_config, runid):
    """Main pipeline to train a model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """

    # TODO: Check what is merge_bn and tidy_sequential and where it is used
    # links : https://github.com/cszn/KAIR/blob/72e93351bca41d1b1f6a4c3e1957f5bffccc7101/models/model_base.py#L205

    # TODO: Check what is update_E and if used or not :
    # links : https://github.com/cszn/KAIR/blob/72e93351bca41d1b1f6a4c3e1957f5bffccc7101/models/model_base.py#L188

    # TODO: Check whats is img_size in swintransformer
    # (int: it is not the input image size because we can feed images of different size than im_size)

    # TODO: Check what is depths and HEAD in SwinTransformers

    run = neptune.init(
        project="Sondra-SAR", api_token=cfg["TRAIN"]["NEPTUNE_API_TOKEN"]
    )
    print(f"Neptune id is : {run._id}")

    # Log Neptune config & pararameters
    params = {
        "im_size": cfg["DATASET"]["IMAGE_SIZE"],
        "downscale_factor": cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"],
        "batch_size": cfg["DATASET"]["BATCH_SIZE"],
        "loss": cfg["TRAIN"]["LOSS"]["NAME"],
        "optimizer": cfg["TRAIN"]["OPTIMIZER"]["NAME"],
        "lr": cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["LR_INITIAL"],
        "weight_decay": cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["WEIGHT_DECAY"],
        "pretrained": cfg["TRAIN"]["PRETRAINED"]["BOOL"],
        "epochs": cfg["TRAIN"]["EPOCH"],
    }

    run["algorithm"] = cfg["MODEL"]["NAME"].lower()
    run["config/yaml"] = "".join(open(path_to_config, "r").readlines())
    run["config/dataset/path"] = cfg["TRAIN_DATA_DIR"]
    run["config/params"] = params

    # Load data
    train_loader, valid_loader = loader.load_train(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    model = get_model(cfg, pretrained=cfg["TRAIN"]["PRETRAINED"]["BOOL"])
    model = model.to(device)

    summary = torchinfo.summary(
        model,
        verbose=0,
        input_size=(
            cfg["DATASET"]["BATCH_SIZE"],
            cfg["DATASET"]["IN_CHANNELS"],
            cfg["DATASET"]["IMAGE_SIZE"] // 2,
            cfg["DATASET"]["IMAGE_SIZE"] // 2,
        ),
    )
    print(summary)

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

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_MODEL_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODEL"]["NAME"].lower(), runid)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save config for each training
    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    # Init Checkpoint class
    checkpoint = ModelCheckpoint(
        save_dir, model, cfg["TRAIN"]["EPOCH"], cfg["TRAIN"]["CHECKPOINT_STEP"]
    )

    image_first_epoch = None

    # Evaluate baselines with naive upsampling
    eval_upsample("nearest", valid_loader, f_loss, device, cfg)
    eval_upsample("bilinear", valid_loader, f_loss, device, cfg)
    eval_upsample("bicubic", valid_loader, f_loss, device, cfg)

    # Start training loop
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH : {}".format(epoch))

        # if epoch == 0:
        #     (
        #         valid_loss,
        #         psnr,
        #         input_image,
        #         restored_images,
        #         target_images,
        #         l1_loss,
        #         l2_loss,
        #     ) = valid_one_epoch(
        #         model,
        #         valid_loader,
        #         f_loss,
        #         device,
        #         cfg["DATASET"]["CLIP"]["MAX"] - cfg["DATASET"]["CLIP"]["MIN"],
        #     )
        #     # Scatter all images with the same transformation
        #     target_scattered, p2, p98 = equalize(target_images)
        #     # Log Neptune losses, psnr and lr
        #     run["logs/training/batch/valid_loss"].log(valid_loss)
        #     run["logs/training/batch/psnr"].log(psnr)
        #     run["logs/training/batch/L2_loss"].log(l2_loss)
        #     run["logs/training/batch/L1_loss"].log(l1_loss)
        #     run["logs/valid/batch/Input_image"].log(
        #         File.as_image(equalize(input_image, p2, p98)[0])
        #     )
        #     run["logs/valid/batch/Restored_image"].log(
        #         File.as_image(equalize(restored_images, p2, p98)[0])
        #     )
        #     run["logs/valid/batch/Target_image"].log(File.as_image(target_scattered))
        #     run["logs/valid/batch/Diff_Target_Restored"].log(
        #         File.as_image(np.abs(restored_images - target_images))
        #     )

        # Train
        training_loss = train_one_epoch(
            model,
            train_loader,
            f_loss,
            optimizer,
            device,
            clipgrad=cfg["TRAIN"]["OPTIMIZER"]["CLIPGRAD"],
        )

        # Apply Regularization
        if cfg["TRAIN"]["REGULARIZER"]["ORTHSTEP"] > 0:
            model.apply(regularizer_orth2)
        if cfg["TRAIN"]["REGULARIZER"]["CLIPSTEP"] > 0:
            model.apply(regularizer_clip)

        # Validation
        (
            valid_loss,
            psnr,
            input_image,
            restored_images,
            target_images,
            l1_loss,
            l2_loss,
            ssim_loss,
        ) = valid_one_epoch(
            model,
            valid_loader,
            f_loss,
            device,
            cfg["DATASET"]["CLIP"]["MAX"] - cfg["DATASET"]["CLIP"]["MIN"],
        )

        if epoch == 0:
            image_first_epoch = restored_images

        # Update scheduler
        scheduler.step(valid_loss)

        # Save best model
        checkpoint.update(valid_loss, epoch)

        # Get current learning rate
        learning_rate = scheduler.optimizer.param_groups[0]["lr"]

        # Scatter all images with the same transformation
        target_scattered, p2, p98 = equalize(target_images)

        # Log Neptune losses, psnr and lr
        run["logs/training/loss"].log(training_loss)
        run["logs/valid/loss"].log(valid_loss)
        run["logs/valid/psnr"].log(psnr)
        run["logs/learning_rate"].log(learning_rate)
        run["logs/valid/L2_loss"].log(l2_loss)
        run["logs/valid/L1_loss"].log(l1_loss)
        run["logs/valid/SSIM_loss"].log(ssim_loss)

        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(equalize(input_image, p2, p98)[0], cmap=plt.cm.gray)
        plt.title("Input")

        plt.subplot(1, 3, 2)
        plt.imshow(target_scattered, cmap=plt.cm.gray)
        plt.title("Target")

        plt.subplot(1, 3, 3)
        plt.imshow(equalize(restored_images, p2, p98)[0], cmap=plt.cm.gray)
        plt.title("Restored")

        run["logs/valid/batch/Input_target_restored"].log(fig)
        plt.close(fig)

        diff_restored_target = np.abs(restored_images - target_images)
        max_diff = diff_restored_target.max()
        min_diff = diff_restored_target.min()
        print(f"Diff restored-target : min = {min_diff:.2f}; max = {max_diff:.2f}")
        run["logs/valid/batch/Diff_Target_Restored_normalized"].log(
            File.as_image(diff_restored_target / (max_diff if max_diff != 0 else 1.0))
        )

        run["logs/valid/batch/diff_restored_over_epochs"].log(
            File.as_image(np.abs(restored_images - image_first_epoch))
        )

        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.hist(target_images, bins=30)
        plt.xlim(-150, -25)
        plt.title("Target")

        plt.subplot(1, 3, 2)
        plt.hist(restored_images, bins=30)
        plt.xlim(-150, -25)
        plt.title("Restored")

        plt.subplot(1, 3, 3)
        plt.xlim(-50, 50)
        plt.hist(target_images - restored_images, bins=30)
        plt.title("Diff (target - restored)")

        run["logs/valid/batch/histograms"].log(fig)
        plt.close(fig)

    # Stop Neptune logging
    run.stop()


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

    parser.add_argument(
        "--runid",
        type=int,
        required=False,
        default=None,
        help="Optional : a run id to suffix the logdirs",
    )
    args = parser.parse_args()

    # Load config
    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    main(cfg, args.path_to_config, args.runid)
