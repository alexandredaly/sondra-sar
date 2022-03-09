"""This module define the function to visualise the images during training."""
import torch
import tqdm
import random
import numpy as np


def tensor2uint(img):
    """Converts tensor image to numpy uint8 array

    Args:
        img (torch.tensor): tensor image on GPU

    Returns:
        np.uint8array: numpy image on CPU
    """
    img = img.data.squeeze(axis=1).float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (0, 2, 1))
    return np.uint8((img * 255.0).round())


def calculate_psnr(img1, img2, border=0):
    """Function to computer peak to signal ratio

    Args:
        img1 ([type]): [description]
        img2 ([type]): [description]
        border (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # img1 and img2 have range [-60, 0]

    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[1:]
    img1 = img1[:, border : h - border, border : w - border]
    img2 = img2[:, border : h - border, border : w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2, axis=(1, 2))
    mse[mse == 0] = float("inf")
    #TODO factor the min clip value from the data set
    return np.mean(10 * np.log10(60**2 / mse))


def valid_one_epoch(model, loader, f_loss, device, loss_weight):
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
        restored_images = []
        target_images = []
        avg_psnr = 0

        for low, high in tqdm.tqdm(loader):
            low, high = low.to(device), high.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(low)
            loss = loss_weight * f_loss(outputs, high)

            n_samples += low.shape[0]
            tot_loss += low.shape[0] * f_loss(outputs, high).item()

            pred = tensor2uint(outputs)
            target = tensor2uint(high)

            psnr = calculate_psnr(pred, target)
            avg_psnr += psnr

            # Return reconstructed images
            restored_images.append(outputs[0])
            target_images.append(high[0])

        return (
            tot_loss / n_samples,
            psnr / n_samples,
            restored_images,
            target_images,
        )
