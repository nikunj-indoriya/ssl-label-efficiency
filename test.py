"""
test.py
-------
Sanity-check script for the label-subset sampling utility.

Loads the full CIFAR-10 training split, then calls `create_label_subset`
at several label fractions (100 %, 10 %, 5 %, 1 %) and verifies that:
  - The returned subset has the expected number of samples.
  - Each of the 10 classes is represented proportionally (balanced sampling).

This script is intended to be run manually during development; it is NOT
part of the main training or evaluation pipeline.
"""

import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

from datasets.utils import create_label_subset


def main():

    # Minimal transform: convert PIL images to tensors without any augmentation.
    # Augmentation is irrelevant here since we only inspect labels, not pixels.
    transform = transforms.ToTensor()

    # Load the full CIFAR-10 training split (50,000 samples, 10 classes).
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print("Total dataset size:", len(dataset))

    # Label fractions to test: 100 %, 10 %, 5 %, and 1 % of training data.
    fractions = [1.0, 0.1, 0.05, 0.01]

    for frac in fractions:

        # Create a balanced subset: `frac` fraction of samples per class,
        # selected uniformly at random with a fixed seed for reproducibility.
        subset = create_label_subset(dataset, frac)

        print(f"\n--- Fraction: {frac} ---")
        print("Subset size:", len(subset))

        # Recover the ground-truth labels for all selected indices so we can
        # verify the per-class distribution directly from the parent dataset.
        labels = [dataset.targets[i] for i in subset.indices]

        counter = Counter(labels)

        # Print the sample count for each class (0–9) in sorted order.
        print("Class distribution:")
        for k in sorted(counter.keys()):
            print(f"Class {k}: {counter[k]}")


if __name__ == "__main__":
    main()
