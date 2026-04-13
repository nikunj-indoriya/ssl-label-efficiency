"""
training/train_byol.py
----------------------
BYOL self-supervised pre-training loop with cosine EMA scheduling.

For each mini-batch the loader yields a positive pair (x1, x2). The BYOL
model computes predictions from the online branch and stop-gradient projections
from the target branch. The symmetric BYOL regression loss drives the online
predictor to match the target projections.

EMA schedule
------------
The target network is updated via Exponential Moving Average (EMA) after
every optimiser step. The EMA momentum τ is annealed with a cosine schedule:

    τ(t) = 1 - (1 - τ_base) · (cos(π · t / T) + 1) / 2

where t is the current global step and T is the total number of steps. This
gradually increases τ toward 1.0, making the target network update more
slowly as training progresses and providing increasingly stable targets.
"""

import torch
from tqdm import tqdm
from methods.byol_loss import byol_loss
import math


def train_byol(model, loader, optimizer, device, epoch, total_epochs):
    """
    Run a single epoch of BYOL contrastive pre-training.

    Each batch step:
      1. Receive positive pair (x1, x2) from the two-view loader.
      2. Forward pass through the BYOL model to get online predictions
         (p1, p2) and target projections (z1, z2).
      3. Compute the symmetric BYOL loss: loss(p1, z2) + loss(p2, z1).
      4. Backpropagate through the online branch only (target is frozen).
      5. Update model parameters.
      6. Compute the cosine-annealed EMA momentum τ.
      7. Apply the EMA update to the target network.

    Parameters
    ----------
    model        : BYOL. Model with `.forward(x1, x2)` and `.update_target(τ)`.
    loader       : DataLoader. Yields (x1, x2) positive-pair batches from
                   SimCLRDataset (two independently augmented views per image).
    optimizer    : Optimizer. E.g. Adam applied to online branch parameters.
    device       : torch.device. Target device for tensor operations.
    epoch        : int. Current epoch index (0-based), used for τ scheduling.
    total_epochs : int. Total number of pre-training epochs, used to compute
                   the maximum number of global steps.

    Returns
    -------
    float. Average BYOL loss per mini-batch over the full epoch.
    """

    model.train()
    total_loss = 0

    for step, (x1, x2) in enumerate(tqdm(loader)):

        x1, x2 = x1.to(device), x2.to(device)

        # Forward pass through both branches.
        # p1, p2 — online predictor outputs (gradients flow through these).
        # z1, z2 — target projections (stop-gradient / detached).
        p1, p2, z1, z2 = model(x1, x2)

        # Symmetric BYOL loss: the online branch predicts each view's
        # target representation from the other view's online output.
        loss = byol_loss(p1, z2) + byol_loss(p2, z1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Cosine EMA schedule for τ ----
        # Compute the current global training step (across all epochs).
        global_step = epoch * len(loader) + step
        max_steps = total_epochs * len(loader)

        # Cosine schedule: τ starts near τ_base (0.99) and increases to ~1.0.
        # The (cos + 1) / 2 term maps the cosine from [-1, 1] to [0, 1],
        # so τ rises monotonically from 0.99 toward 1.0 over training.
        tau = 1 - (1 - 0.99) * (math.cos(math.pi * global_step / max_steps) + 1) / 2

        # Apply EMA update to target encoder and projector.
        model.update_target(tau)

        total_loss += loss.item()

    return total_loss / len(loader)
