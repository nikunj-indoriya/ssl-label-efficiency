"""
scripts/run_mae.py
------------------
Self-supervised pre-training script for MAE on CIFAR-10.

Trains a Masked Autoencoder (ViT-Base/16 encoder + lightweight MLP decoder)
for 20 epochs using a variance-maximisation proxy reconstruction loss. Images
are resized to 224×224 to match the ViT patch-embedding resolution.

Note: the evaluation script (`evaluate_mae.py`) loads a *pretrained* ViT from
`timm` rather than this from-scratch checkpoint, because training a full ViT-
Base from scratch on CIFAR-10 requires significantly more compute and data
than is available in this experimental setup. This run_mae.py script is
retained for completeness and experimental exploration.

Saved artefact
--------------
mae_encoder.pth : state dict of the ViT encoder backbone. Can be loaded for
                  linear probing, though the pretrained ViT used in
                  evaluate_mae.py typically yields better results.
"""

import torch
import torch.optim as optim

from methods.mae import MAE
from datasets.mae_dataset import get_mae_loader
from training.train_mae import train_mae


def main():

    # Use GPU for training; ViT-Base is computationally expensive.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MAE data loader: 224×224 CIFAR-10 images with spatial augmentation.
    loader = get_mae_loader()

    # Initialise the MAE model (ViT-Base/16 encoder + MLP decoder proxy).
    model = MAE().to(device)

    # Adam with a smaller learning rate (1e-4) suits the ViT encoder's
    # attention-based architecture better than the 1e-3 used for ResNets.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        loss = train_mae(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    # Save only the encoder (ViT backbone), discarding the decoder.
    torch.save(model.encoder.state_dict(), "mae_encoder.pth")


if __name__ == "__main__":
    main()
