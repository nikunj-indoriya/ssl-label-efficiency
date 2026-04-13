"""
methods/byol_loss.py
--------------------
BYOL (Bootstrap Your Own Latent) regression loss.

Unlike contrastive methods that require negative pairs, BYOL trains a
predictor on the online branch to match the target-branch projections.
The loss is the negative cosine similarity between the normalised prediction
`p` and the stop-gradient target projection `z`.

    L(p, z) = 2 - 2 · (p̂ · ẑ)     where  p̂ = p / ‖p‖,  ẑ = z / ‖z‖

The value 2 is a constant offset so that L ≥ 0 (perfect alignment → L = 0).
The full BYOL objective is the *symmetric* sum:
    L_total = byol_loss(p1, z2) + byol_loss(p2, z1)

Reference: Grill et al., "Bootstrap Your Own Latent — A New Approach to
Self-Supervised Learning", NeurIPS 2020.
"""

import torch.nn.functional as F


def byol_loss(p, z):
    """
    Compute the BYOL regression loss between a predictor output `p` and a
    target projection `z`.

    Both tensors are L2-normalised before computing the dot product, making
    the result equivalent to negative cosine similarity shifted to [0, 2].

    Parameters
    ----------
    p : Tensor of shape (N, D). Online branch predictor output.
        Gradients flow through this term.
    z : Tensor of shape (N, D). Target branch projection (stop-gradient).
        Should be passed as `.detach()` by the caller to prevent target
        network collapse.

    Returns
    -------
    Scalar loss tensor. Value of 0 means perfect alignment; value of 2 means
    completely anti-aligned representations.
    """

    # L2-normalise both vectors before computing the dot product.
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    # Mean negative cosine similarity, shifted so the minimum is 0.
    # Equivalent to: 1 - cosine_similarity(p, z).mean()  ×  2
    return 2 - 2 * (p * z).sum(dim=1).mean()
