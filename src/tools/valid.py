"""This module define the function to visualise the images during training."""
import torch
import tqdm
import numpy as np
from . import SSIMLoss
import torch.nn as nn


def calculate_psnr(img1, img2, scale, border=0):
    """Function to computer peak to signal ratio

    Args:
        img1 ([type]): [description]
        img2 ([type]): [description]
        border (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns
        [type]: [description]
    """
    # img1 and img2 have range [-60, 0]

    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[2:]
    img1 = img1[:, :, border : (h - border), border : (w - border)]
    img2 = img2[:, :, border : (h - border), border : (w - border)]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2, axis=(2, 3))
    mse[mse == 0] = float("inf")

    return np.mean(10 * np.log10(scale**2 / mse))  # <=> 1/B * sum_i


def valid_one_epoch(model, loader, f_loss, device, scale):
    """Train the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        device (torch.device): cuda

    Return:
        tot_loss : computed loss over one epoch
    """
    ssim = SSIMLoss.SSIMLoss(size_average=True)
    with torch.no_grad():

        model.eval()

        n_samples = 0
        tot_loss = 0.0
        tot_l1loss = 0.0
        tot_l2loss = 0.0
        tot_ssim = 0.0
        avg_psnr = 0

        for low, high in tqdm.tqdm(loader):
            low, high = low.to(device), high.to(device)

            # with torch.cuda.amp.autocast():
            # Compute the forward pass through the network up to the loss
            outputs = model(low)

            batch_size = low.shape[0]

            # WARN: if using reduction = "mean", the avg
            # is computed over batch_size * Height * Width
            l1_loss = torch.nn.functional.l1_loss(outputs, high, reduction="mean")
            tot_l1loss += batch_size * l1_loss.item()
            l2_loss = torch.nn.functional.mse_loss(outputs, high, reduction="mean")
            tot_l2loss += batch_size * l2_loss.item()
            ssim_loss = ssim(outputs, high)
            tot_ssim += batch_size * ssim_loss.item()

            n_samples += batch_size
            tot_loss += batch_size * f_loss(outputs, high).item()

            # We need to denormalize the PSNR to correctly average
            psnr = batch_size * calculate_psnr(
                outputs.cpu().numpy(), high.cpu().numpy(), scale
            )
            avg_psnr += psnr

        return (
            tot_loss / n_samples,
            avg_psnr / n_samples,
            low.cpu().numpy(),
            outputs.cpu().numpy(),
            high.cpu().numpy(),
            tot_l1loss / n_samples,
            tot_l2loss / n_samples,
            -tot_ssim / n_samples,
        )
