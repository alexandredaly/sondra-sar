"""This module define the function to visualise the images during training."""
import torch
import tqdm
import numpy as np


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

    return np.mean(10 * np.log10(scale**2 / mse))


def valid_one_epoch(model, loader, f_loss, device, loss_weight, scale):
    """Train the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        device (torch.device): cuda

    Return:
        tot_loss : computed loss over one epoch
    """
    with torch.no_grad():

        model.eval()

        n_samples = 0
        tot_loss = 0.0
        avg_psnr = 0

        for low, high in tqdm.tqdm(loader):
            low, high = low.to(device), high.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(low)
            l1_loss = torch.nn.functional.l1_loss(outputs, high)
            l2_loss = torch.nn.functional.mse_loss(outputs, high)
            n_samples += low.shape[0]
            tot_loss += low.shape[0] * f_loss(outputs, high).item()

            psnr = calculate_psnr(outputs.cpu().numpy(), high.cpu().numpy(), scale)
            avg_psnr += psnr

        return (
            tot_loss / n_samples,
            avg_psnr / n_samples,
            np.mean(low[0].cpu().numpy(), axis=0),
            np.mean(outputs[0].cpu().numpy(), axis=0),
            np.mean(high[0].cpu().numpy(), axis=0),
            l1_loss,
            l2_loss,
        )
