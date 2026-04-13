"""
datasets/simclr_dataset.py
--------------------------
Custom PyTorch Dataset that produces the *two-view* batches required by the
SimCLR and BYOL contrastive pre-training objectives.

For each image, two independent random augmentations are applied, yielding a
positive pair (x1, x2). The contrastive loss then maximises agreement between
these two views of the same image while pushing apart views from different
images.
"""

import torchvision
from torch.utils.data import Dataset
from methods.augmentations import get_simclr_augmentation


class SimCLRDataset(Dataset):
    """
    Wraps CIFAR-10 (training split) and returns two independently augmented
    views of each image instead of the original (image, label) pair.

    Labels are discarded at this stage because self-supervised pre-training
    is entirely label-free. Labels are only used during downstream linear-
    probe evaluation.

    Attributes
    ----------
    dataset   : the underlying torchvision CIFAR-10 dataset.
    transform : the SimCLR augmentation pipeline (applied twice independently).
    """

    def __init__(self, root="./data"):
        # Load raw CIFAR-10 images without any transform; augmentation is
        # applied on-the-fly in __getitem__ to ensure statistical independence
        # between the two views.
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True
        )
        # Shared augmentation pipeline; called twice per sample to produce
        # two different random views of the same image.
        self.transform = get_simclr_augmentation()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the raw PIL image; the class label is intentionally ignored.
        x, _ = self.dataset[idx]

        # Apply the stochastic augmentation pipeline twice with different
        # random seeds, producing two correlated-but-distinct views.
        x1 = self.transform(x)
        x2 = self.transform(x)

        # Return both views as a tuple; the training loop unpacks them.
        return x1, x2
