"""
training/train_simclr.py
------------------------
SimCLR self-supervised pre-training loop.

For each mini-batch the loader yields a positive pair (x1, x2) — two
independently augmented views of the same set of images. The model encodes
both views and projects them to the contrastive embedding space. The NT-Xent
loss then maximises agreement between paired projections while pushing apart
all other pairs in the batch.
"""

import torch
from tqdm import tqdm
from methods.losses import nt_xent_loss


def train_simclr(model, loader, optimizer, device):
    """
    Run a single epoch of SimCLR contrastive pre-training.

    Each batch step:
      1. Receive positive pair (x1, x2) from the SimCLRDataset loader.
      2. Encode both views through the shared encoder + projector.
      3. Compute NT-Xent loss on the projected embeddings z1 and z2.
      4. Backpropagate and update model parameters.

    Note: only the projected embeddings (z1, z2) are used for the loss;
    the backbone features (h1, h2) are returned by the model but discarded
    here — they are used later during linear-probe evaluation.

    Parameters
    ----------
    model     : SimCLR. Model with `.forward(x) → (h, z)`.
    loader    : DataLoader. Yields (x1, x2) positive-pair batches.
    optimizer : Optimizer. E.g. Adam.
    device    : torch.device. Target device for tensor operations.

    Returns
    -------
    float. Average NT-Xent loss per mini-batch over the full epoch.
    """

    model.train()
    total_loss = 0

    for x1, x2 in tqdm(loader):

        # Move both views to the target device.
        x1, x2 = x1.to(device), x2.to(device)

        # Forward pass: obtain (backbone_features, projected_embeddings).
        # Underscore discards the backbone features (h); only projections (z)
        # are needed to compute the contrastive loss.
        _, z1 = model(x1)
        _, z2 = model(x2)

        # NT-Xent loss: treats (z1[i], z2[i]) as positives and all other
        # pairs in the batch as negatives.
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
