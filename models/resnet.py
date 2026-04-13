"""
models/resnet.py
----------------
Defines the ResNet-50 backbone used for supervised training and as a
feature extractor for SimCLR and BYOL linear-probe evaluation.

The model is split into two parts:
  - `backbone` : a ResNet-50 without the final classification head,
                 outputting 2048-dimensional feature vectors.
  - `classifier`: a single linear layer mapping features to class logits.

This separation allows the backbone weights to be loaded from an SSL
pre-trained checkpoint (e.g. `simclr_encoder.pth`) and then frozen, while
only the classifier head is trained during linear-probe evaluation.
"""

import timm
import torch.nn as nn


class ResNet50(nn.Module):
    """
    ResNet-50 with a detachable linear classifier head.

    Architecture
    ------------
    backbone   : ResNet-50 (from `timm`) with `num_classes=0`, meaning the
                 global-average-pool layer produces raw 2048-d feature vectors
                 with no projection head attached.
    classifier : nn.Linear(2048, num_classes) — the downstream head trained
                 via cross-entropy loss.

    Parameters
    ----------
    num_classes : int. Number of output classes (default 10 for CIFAR-10).
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Create ResNet-50 without a pre-trained initialisation (`pretrained=False`)
        # so that SSL checkpoints or random initialisation can be loaded explicitly.
        # `num_classes=0` removes the default FC head and exposes raw pool features.
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)

        # Lightweight linear head mapping 2048-d backbone output to class logits.
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        """
        Full forward pass: backbone → classifier.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W). Input image batch.

        Returns
        -------
        Tensor of shape (B, num_classes). Raw (pre-softmax) class logits.
        """
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """
        Extract backbone features without applying the classifier head.

        Used during linear-probe evaluation: features are extracted once in
        a frozen forward pass and then a separate linear classifier is trained
        on top of them.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W). Input image batch.

        Returns
        -------
        Tensor of shape (B, 2048). L2-unnormalised backbone features.
        """
        return self.backbone(x)
