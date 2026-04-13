"""
datasets/mae_dataset.py
-----------------------
DataLoader factory for Masked Autoencoder (MAE) pre-training on CIFAR-10.

MAE operates on patch sequences derived from full images and reconstructs
randomly masked patches. The augmentation pipeline therefore applies spatial
transforms (resize + random crop + flip) before converting to a tensor, but
does NOT apply colour jitter or grayscale — patch-level pixel reconstruction
requires the original colour information to remain meaningful.

Images are resized to 224×224 to match the patch-embedding input resolution
expected by the ViT-Base/16 backbone (16×16 patches → 14×14 = 196 tokens).
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mae_loader(batch_size=128):
    """
    Build a shuffled DataLoader for MAE self-supervised pre-training.

    The augmentation pipeline:
      - Resize(224)           : upscale 32×32 CIFAR images to 224×224.
      - RandomResizedCrop(224): random scale & crop for position diversity.
      - RandomHorizontalFlip  : mirror augmentation with p=0.5.
      - ToTensor              : convert PIL image to float tensor in [0, 1].

    Parameters
    ----------
    batch_size : int. Number of samples per mini-batch (default 128).

    Returns
    -------
    loader : DataLoader that yields (images, labels) batches.
             Labels are unused during MAE pre-training but are kept so the
             loader can be passed to generic utility functions if needed.
    """

    # Augmentation tailored for MAE: spatial transforms only, no colour drops.
    # Large crops encourage the model to reconstruct diverse image regions.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Shuffle to break temporal correlation between consecutive batches.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
