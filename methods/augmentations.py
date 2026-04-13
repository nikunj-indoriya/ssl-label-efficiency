"""
methods/augmentations.py
------------------------
Augmentation pipelines for self-supervised contrastive pre-training.

The SimCLR augmentation strategy follows the paper
  "A Simple Framework for Contrastive Learning of Visual Representations"
  (Chen et al., 2020). The key insight is that the network must learn
  representations that are *invariant* to the applied transformations, so
  the augmentations should cover transformations that do not change the
  semantic meaning of an image (crop, flip, colour distortion, blur).
"""

import torchvision.transforms as transforms


def get_simclr_augmentation(size=32):
    """
    Compose the standard SimCLR stochastic augmentation pipeline.

    Each call to this transform produces a *different* random augmentation,
    so applying it twice to the same image yields two distinct views — the
    positive pair used by the NT-Xent contrastive loss.

    Pipeline stages
    ---------------
    1. RandomResizedCrop(size)
       Crops a random region of the image and resizes it back to `size`.
       Encourages spatial invariance and scale invariance.

    2. RandomHorizontalFlip()
       Mirror the image horizontally with probability 0.5.
       Provides left-right invariance.

    3. RandomApply([ColorJitter(...)], p=0.8)
       With probability 0.8, randomly jitter brightness (0.4), contrast (0.4),
       saturation (0.4), and hue (0.1). Teaches the model to rely on shape and
       texture rather than absolute colour statistics.

    4. RandomGrayscale(p=0.2)
       Convert to grayscale with probability 0.2, encouraging the model to
       build colour-agnostic representations for 20 % of views.

    5. GaussianBlur(kernel_size=3)
       Apply Gaussian blur with a fixed small kernel; softens high-frequency
       texture detail and emphasises mid-level features.

    6. ToTensor()
       Convert PIL image to a float32 tensor with values in [0, 1].

    Parameters
    ----------
    size : int. Target spatial resolution after cropping (default 32 for CIFAR-10).

    Returns
    -------
    transforms.Compose — a callable augmentation pipeline.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])
