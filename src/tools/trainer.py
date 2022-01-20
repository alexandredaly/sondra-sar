"""This module define the function to train the model on one epoch."""
import torch
import tqdm


def train_one_epoch(model, loader, f_loss, optimizer, device, loss_weight):
    """Train the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        optimizer (torch.optim.Optimzer object): adam optimizer
        device (torch.device): cuda

    Return:
        tot_loss : computed loss over one epoch
    """

    model.train()

    n_samples = 0
    tot_loss = 0.0

    for low, high in tqdm.tqdm(loader):
        low, high = low.to(device), high.to(device)
        # Compute the forward pass through the network up to the loss
        outputs = model(low)
        loss = loss_weight * f_loss(outputs, high)

        n_samples += low.shape[0]
        tot_loss += low.shape[0] * f_loss(outputs, high).item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tot_loss / n_samples
