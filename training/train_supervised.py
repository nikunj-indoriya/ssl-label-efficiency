"""
training/train_supervised.py
-----------------------------
Training and evaluation routines for standard supervised learning.

Provides two functions:
  `train`    — run one epoch of supervised mini-batch gradient descent.
  `evaluate` — compute top-1 accuracy on a given DataLoader (no gradients).

These functions are used by `main.py` for the supervised baseline and by
`scripts/evaluate_noise.py` for the label-noise robustness experiment.
"""

import torch
from tqdm import tqdm


def train(model, loader, optimizer, criterion, device):
    """
    Run a single training epoch with mini-batch stochastic gradient descent.

    Each batch:
      1. Forward pass through the model to obtain class logits.
      2. Compute the scalar loss via `criterion` (typically CrossEntropyLoss).
      3. Backward pass to compute gradients.
      4. Optimizer step to update model parameters.

    Parameters
    ----------
    model     : nn.Module. The model to train (must be in training mode).
    loader    : DataLoader. Yields (images, labels) mini-batches.
    optimizer : Optimizer. E.g. Adam or SGD.
    criterion : Loss function. E.g. nn.CrossEntropyLoss().
    device    : torch.device. Target device for tensor operations.

    Returns
    -------
    float. Average loss per mini-batch over the full epoch.
    """
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        # Move data to the target device (GPU if available, else CPU).
        images, labels = images.to(device), labels.to(device)

        # Zero out accumulated gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward pass: compute raw class logits.
        outputs = model(images)

        # Compute cross-entropy loss between logits and ground-truth labels.
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients w.r.t. model parameters.
        loss.backward()

        # Update model parameters using the computed gradients.
        optimizer.step()

        total_loss += loss.item()

    # Return the mean loss per batch for logging purposes.
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """
    Compute top-1 classification accuracy on a DataLoader.

    Runs in inference mode (`torch.no_grad()`) to avoid storing activations
    and gradients, reducing memory usage during evaluation.

    Parameters
    ----------
    model  : nn.Module. Trained model to evaluate.
    loader : DataLoader. Yields (images, labels) batches.
    device : torch.device. Target device for tensor operations.

    Returns
    -------
    float. Top-1 accuracy in [0, 1] over all samples in `loader`.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass: obtain class logits.
            outputs = model(images)

            # Predicted class = index of the maximum logit.
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
