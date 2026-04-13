"""
datasets/utils.py
-----------------
Dataset utility functions shared across all training and evaluation scripts.

Two core operations are provided:
  1. `create_label_subset`  — stratified random sampling of a fraction of
     the training set while keeping class balance.
  2. `add_label_noise`      — symmetric label corruption at a given noise
     ratio, applicable to both full datasets and Subset wrappers.
"""

import numpy as np
from torch.utils.data import Subset


def create_label_subset(dataset, fraction, num_classes=10, seed=42):
    """
    Build a class-balanced subset of `dataset` containing `fraction` of the
    samples from each class.

    Stratified sampling ensures that every class contributes equally —
    i.e. if fraction=0.1 and each class has 5,000 samples, the returned
    subset contains 500 samples per class (5,000 total).

    Parameters
    ----------
    dataset    : torchvision Dataset with a `.targets` attribute.
    fraction   : float in (0, 1]. Proportion of each class to keep.
    num_classes: int. Number of distinct classes (default 10 for CIFAR-10).
    seed       : int. NumPy random seed for reproducibility.

    Returns
    -------
    torch.utils.data.Subset wrapping `dataset` with the selected indices.
    """
    np.random.seed(seed)

    targets = np.array(dataset.targets)
    indices = []

    for c in range(num_classes):
        # Gather all sample indices that belong to class `c`.
        class_indices = np.where(targets == c)[0]

        # Compute how many samples to draw from this class.
        n_samples = int(len(class_indices) * fraction)

        # Sample without replacement to avoid duplicate examples.
        selected = np.random.choice(class_indices, n_samples, replace=False)
        indices.extend(selected)

    return Subset(dataset, indices)


def add_label_noise(dataset, noise_ratio, num_classes=10, seed=42):
    """
    Corrupt a random subset of training labels by replacing them with a
    uniformly sampled *different* class label (symmetric noise).

    Handles two cases transparently:
      - A full torchvision dataset (has `.targets` directly).
      - A `torch.utils.data.Subset` wrapper (modifies the underlying dataset
        but only touches indices belonging to the subset).

    Parameters
    ----------
    dataset    : torchvision Dataset or Subset with accessible `.targets`.
    noise_ratio: float in [0, 1). Fraction of samples whose labels are flipped.
    num_classes: int. Total number of classes (default 10 for CIFAR-10).
    seed       : int. NumPy random seed for reproducibility.

    Returns
    -------
    The same dataset object with `.targets` modified in-place.

    Raises
    ------
    ValueError if `dataset` is neither a plain dataset nor a Subset.
    """
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Case 1: Full dataset — the targets list is directly on the object.
    # ------------------------------------------------------------------
    if hasattr(dataset, "targets"):

        targets = np.array(dataset.targets)
        n_samples = len(targets)

        # Determine how many labels to corrupt.
        n_noisy = int(noise_ratio * n_samples)

        # Choose `n_noisy` indices at random (without replacement).
        indices = np.random.choice(n_samples, n_noisy, replace=False)

        for idx in indices:
            original = targets[idx]

            # Draw a replacement label that is *different* from the original
            # to guarantee symmetric (not self-referential) noise.
            new_label = np.random.randint(num_classes)
            while new_label == original:
                new_label = np.random.randint(num_classes)

            targets[idx] = new_label

        # Write the modified labels back to the dataset.
        dataset.targets = targets.tolist()
        return dataset

    # ------------------------------------------------------------------
    # Case 2: Subset dataset — only corrupt labels within the subset's
    # selected indices; leave the rest of the parent dataset untouched.
    # ------------------------------------------------------------------
    elif isinstance(dataset, Subset):

        base_dataset = dataset.dataset
        indices = dataset.indices  # indices into the parent dataset

        targets = np.array(base_dataset.targets)

        n_samples = len(indices)
        n_noisy = int(noise_ratio * n_samples)

        # Pick which subset indices will have their labels flipped.
        noisy_indices = np.random.choice(indices, n_noisy, replace=False)

        for idx in noisy_indices:
            original = targets[idx]

            # Resample until we get a label different from the original.
            new_label = np.random.randint(num_classes)
            while new_label == original:
                new_label = np.random.randint(num_classes)

            targets[idx] = new_label

        # Propagate the changes back to the parent dataset's target list.
        base_dataset.targets = targets.tolist()
        return dataset

    else:
        raise ValueError("Unsupported dataset type for noise injection")
