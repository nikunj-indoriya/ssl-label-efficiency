"""
scripts/run_simclr.py
---------------------
Self-supervised pre-training script for SimCLR on CIFAR-10.

Trains a SimCLR model (ResNet-50 encoder + 2-layer MLP projector) for 20
epochs using the NT-Xent contrastive loss. At the end of training, only the
encoder backbone is saved — the projector is discarded because downstream
linear probing is performed on the raw backbone features, not on the
projection space.

Saved artefact
--------------
simclr_encoder.pth : state dict of the ResNet-50 backbone (without the
                     projector head). Loaded by `evaluate_simclr.py` and
                     `evaluate_noise.py` for downstream evaluation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from methods.simclr import SimCLR
from datasets.simclr_dataset import SimCLRDataset
from training.train_simclr import train_simclr


def main():

    # Use GPU for training if available; SimCLR with large batches benefits
    # greatly from GPU parallelism.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SimCLRDataset yields (x1, x2) positive pairs — two randomly augmented
    # views of each CIFAR-10 training image.
    dataset = SimCLRDataset()

    # Large batch sizes are important for SimCLR: more negatives per step
    # generally improves representation quality.
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    # Initialise the SimCLR model (ResNet-50 encoder + projection MLP).
    model = SimCLR().to(device)

    # Adam with lr=1e-3 is a reasonable default for SimCLR on CIFAR-10.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Pre-train for 20 epochs (a reduced schedule for quick experimentation).
    for epoch in range(20):
        loss = train_simclr(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    # Save only the encoder backbone state dict; the projector head is
    # not needed for downstream linear probing.
    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")


if __name__ == "__main__":
    main()
