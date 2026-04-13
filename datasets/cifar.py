"""
datasets/cifar.py
-----------------
CIFAR-10 data-loader factory functions used throughout the project.

Four loaders are provided to cover the different experimental settings:

  get_cifar10           — standard training loader with augmentation, supports
                          label fraction sub-sampling and label noise injection.
  get_cifar10_eval      — clean evaluation loader (no augmentation, no subset)
                          for ResNet-based models (32×32 images).
  get_cifar10_eval_vit  — clean evaluation loader resized to 224×224 for
                          ViT / MAE models that require ImageNet-size inputs.
  get_cifar10_vit_subset— 224×224 training loader with optional label-fraction
                          sub-sampling, used during MAE linear-probe evaluation.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils import create_label_subset
from .utils import add_label_noise


def get_cifar10(batch_size=128, num_workers=4, label_fraction=1.0, noise_ratio=0.0):
    """
    Build CIFAR-10 train/test DataLoaders for supervised or SSL evaluation.

    Training augmentation pipeline:
      - RandomResizedCrop(32): random scale and crop for positional invariance.
      - RandomHorizontalFlip: horizontal mirror with p=0.5.
      - ToTensor: converts PIL image to float tensor in [0, 1].

    Optional dataset modifications applied in order:
      1. Stratified sub-sampling via `create_label_subset` when
         label_fraction < 1.0.
      2. Symmetric label noise injection via `add_label_noise` when
         noise_ratio > 0.0.

    Parameters
    ----------
    batch_size     : int. Number of samples per mini-batch.
    num_workers    : int. Parallel data-loading workers.
    label_fraction : float. Fraction of training labels to keep (0, 1].
    noise_ratio    : float. Fraction of training labels to corrupt [0, 1).

    Returns
    -------
    (train_loader, test_loader) — tuple of DataLoader objects.
    """

    # Training augmentation: random crop + flip encourage invariance learning.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Test set uses no augmentation — deterministic evaluation.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    # Apply stratified sub-sampling if we are in a low-label regime.
    if label_fraction < 1.0:
        train_dataset = create_label_subset(train_dataset, label_fraction)

    # Optionally corrupt a fraction of the training labels to test robustness.
    if noise_ratio > 0.0:
        train_dataset = add_label_noise(train_dataset, noise_ratio)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Shuffle training data each epoch; keep test order fixed for consistency.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar10_eval(batch_size=128, num_workers=4):
    """
    Build clean CIFAR-10 train/test DataLoaders for feature extraction with
    ResNet-based encoders (32×32 resolution, no augmentation).

    Used when evaluating SimCLR, BYOL, and supervised models via linear probe.
    No augmentation is applied so that extracted features are deterministic.

    Parameters
    ----------
    batch_size  : int. Number of samples per mini-batch.
    num_workers : int. Parallel data-loading workers.

    Returns
    -------
    (train_loader, test_loader) — tuple of DataLoader objects.
    """

    # Plain tensor conversion — preserves raw pixel statistics.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # No shuffle: consistent ordering is useful for feature-label alignment.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar10_eval_vit(batch_size=128, num_workers=4):
    """
    Build clean CIFAR-10 train/test DataLoaders resized to 224×224 for
    ViT / MAE encoders that expect ImageNet-resolution inputs.

    Parameters
    ----------
    batch_size  : int. Number of samples per mini-batch.
    num_workers : int. Parallel data-loading workers.

    Returns
    -------
    (train_loader, test_loader) — tuple of DataLoader objects.
    """

    import torchvision.transforms as transforms
    import torchvision
    from torch.utils.data import DataLoader

    # Resize 32×32 CIFAR images to 224×224 to match ViT patch-embedding input.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar10_vit_subset(label_fraction=1.0, batch_size=128, num_workers=4):
    """
    Build a 224×224 CIFAR-10 training DataLoader with optional label-fraction
    sub-sampling, designed for MAE linear-probe evaluation at various label
    budgets.

    Only a training loader is returned (no test loader) because the test set
    is loaded separately via `get_cifar10_eval_vit`.

    Parameters
    ----------
    label_fraction : float. Fraction of training labels to keep (0, 1].
    batch_size     : int. Number of samples per mini-batch.
    num_workers    : int. Parallel data-loading workers.

    Returns
    -------
    train_loader — DataLoader for the (possibly sub-sampled) training split.
    """

    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from .utils import create_label_subset

    # Resize to 224×224 to match the ViT encoder's expected input size.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Apply stratified sub-sampling to create the desired low-label regime.
    if label_fraction < 1.0:
        train_dataset = create_label_subset(train_dataset, label_fraction)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader
