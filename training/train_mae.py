"""
training/train_mae.py
---------------------
MAE self-supervised pre-training loop.

In the full MAE paper the training objective is pixel-level reconstruction of
masked patches. Here a variance-maximisation proxy loss is used instead:

    L = -log( Var(decoded_output) + ε )

Maximising the variance of the decoder's output encourages the model to
produce diverse, non-collapsed representations across the batch, which serves
as a lightweight proxy for meaningful patch reconstruction. This avoids the
need to implement a pixel-level decoder target while still preventing the
encoder from learning trivial constant representations.

Note: the encoder is a pretrained ViT-Base/16 in the evaluation scripts, so
this pre-training loop is primarily for experimental / from-scratch runs.
"""

import torch
from tqdm import tqdm


def train_mae(model, loader, optimizer, device):
    """
    Run a single epoch of MAE proxy pre-training.

    Each batch step:
      1. Receive a batch of images from the MAE loader (labels discarded).
      2. Forward pass through the MAE model: embed → mask → encode → decode.
      3. Compute the proxy reconstruction loss (negative log variance).
      4. Backpropagate and update both encoder and decoder parameters.

    Parameters
    ----------
    model     : MAE. Model with `.forward(x) → decoded_tokens`.
    loader    : DataLoader. Yields (images, labels) batches from the MAE
                dataset; labels are intentionally ignored during pre-training.
    optimizer : Optimizer. E.g. Adam with a small learning rate (1e-4).
    device    : torch.device. Target device for tensor operations.

    Returns
    -------
    float. Average proxy loss per mini-batch over the full epoch.
    """

    model.train()
    total_loss = 0

    for x, _ in tqdm(loader):
        # Labels are not used during self-supervised pre-training.
        x = x.to(device)

        # Full forward pass: image → patch embeddings → random masking →
        # ViT encoder blocks → lightweight MLP decoder.
        output = model(x)

        # Proxy reconstruction loss: maximise output variance across the
        # feature dimension to prevent representational collapse.
        # Var is computed per feature dimension (dim=0 over the batch), then
        # averaged. The log-variance loss penalises near-zero variance.
        loss = -torch.log(output.var(dim=0).mean() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
