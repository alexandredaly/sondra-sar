"""This module define the function to visualise the images during training."""
import torch
import tqdm
import random


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
        target_restored_images = []

        for low, high in tqdm.tqdm(loader):
            low, high = low.to(device), high.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(low)
            loss = loss_weight * f_loss(outputs, high)

            n_samples += low.shape[0]
            tot_loss += low.shape[0] * f_loss(outputs, high).item()

            # Return 5 random reconstructed images
            count = 0
            previous_idx = []
            while count < 5:
                idx = random.randint(0, outputs.shape[0])
                if idx not in previous_idx:
                    restored_images.append(outputs[idx])
                    target_restored_images(high[idx])
                    count += 1
                    previous_idx.append(idx)

        return tot_loss / n_samples, restored_images, target_restored_images
