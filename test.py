import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

from datasets.utils import create_label_subset


def main():

    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print("Total dataset size:", len(dataset))

    fractions = [1.0, 0.1, 0.05, 0.01]

    for frac in fractions:

        subset = create_label_subset(dataset, frac)

        print(f"\n--- Fraction: {frac} ---")
        print("Subset size:", len(subset))

        # Extract labels
        labels = [dataset.targets[i] for i in subset.indices]

        counter = Counter(labels)

        print("Class distribution:")
        for k in sorted(counter.keys()):
            print(f"Class {k}: {counter[k]}")


if __name__ == "__main__":
    main()