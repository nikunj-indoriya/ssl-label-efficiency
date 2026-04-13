"""
scripts/run_byol.py
-------------------
Self-supervised pre-training script for BYOL on CIFAR-10.

Trains a BYOL model (online ResNet-50 encoder + projector + predictor, and
an EMA-updated target network) for 40 epochs using the symmetric BYOL
regression loss. The EMA momentum τ is annealed via a cosine schedule inside
the training loop.

At the end of training, only the online encoder backbone is saved — the
projector, predictor, and the entire target network are discarded. Downstream
linear probing is performed on raw 2048-d backbone features.

Saved artefact
--------------
byol_encoder.pth : state dict of the online ResNet-50 backbone. Loaded by
                   `evaluate_byol.py` and `evaluate_noise.py` for downstream
                   evaluation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from methods.byol import BYOL
from datasets.simclr_dataset import SimCLRDataset
from training.train_byol import train_byol


def main():

    # Use GPU for training if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse SimCLRDataset since BYOL also requires two-view positive pairs.
    # The two-augmentation format is identical to SimCLR's data requirements.
    dataset = SimCLRDataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    # Initialise the full BYOL model (online + target networks).
    model = BYOL().to(device)

    # Adam applied only to the online branch parameters (target parameters
    # have requires_grad=False and are not in the optimiser graph).
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # BYOL benefits from longer training than SimCLR because the EMA target
    # network converges gradually; 40 epochs gives a reasonable schedule.
    total_epochs = 40

    for epoch in range(total_epochs):
        # Pass epoch and total_epochs so the training loop can compute the
        # correct global step index for the cosine EMA momentum schedule.
        loss = train_byol(model, loader, optimizer, device, epoch, total_epochs)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    # Save only the online encoder backbone; the target network, projector,
    # and predictor are not needed after pre-training.
    torch.save(model.online_encoder.state_dict(), "byol_encoder.pth")


if __name__ == "__main__":
    main()
