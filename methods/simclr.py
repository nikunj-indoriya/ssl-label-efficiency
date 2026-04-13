"""
methods/simclr.py
-----------------
SimCLR model definition.

SimCLR (Chen et al., 2020) consists of two components:
  1. Encoder  (f): a ResNet-50 backbone that maps augmented images to
                   2048-dimensional representation vectors h.
  2. Projector (g): a 2-layer MLP that maps h → z (128-dimensional), where
                    the NT-Xent contrastive loss is applied.

The projection head is used *only* during pre-training. At evaluation time,
the encoder features h are used directly for linear-probe classification,
discarding the projector g — this is a key design choice from the paper.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations", ICML 2020.
"""

import torch
import torch.nn as nn
import timm


class SimCLR(nn.Module):
    """
    SimCLR encoder + projection head.

    Attributes
    ----------
    encoder  : ResNet-50 backbone (without classification head) that outputs
               h ∈ R^2048 for each image.
    projector: Two-layer MLP: Linear(2048→512) → ReLU → Linear(512→128).
               Maps backbone features to the contrastive embedding space z.

    Parameters
    ----------
    projection_dim : int. Dimensionality of the projected embedding z
                     (default 128, following the SimCLR paper).
    """

    def __init__(self, projection_dim=128):
        super().__init__()

        # ResNet-50 backbone without the final FC head (`num_classes=0`),
        # outputting 2048-d global-average-pool features.
        self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)

        # Non-linear projection head: maps backbone output to the lower-
        # dimensional contrastive space where the NT-Xent loss is computed.
        # The paper shows that learning in a lower-dimensional space improves
        # the quality of the upstream representation.
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        """
        Compute backbone features and projected embeddings for a batch of images.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W). Augmented image batch.

        Returns
        -------
        h : Tensor of shape (B, 2048). Backbone representation (used for
            downstream linear probing).
        z : Tensor of shape (B, projection_dim). Projected embedding (used
            for NT-Xent contrastive loss during pre-training).
        """
        h = self.encoder(x)      # backbone representation
        z = self.projector(h)    # contrastive projection
        return h, z
