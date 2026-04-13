"""
methods/losses.py
-----------------
Contrastive loss functions for self-supervised learning.

NT-Xent Loss (Normalised Temperature-scaled Cross-Entropy)
----------------------------------------------------------
Introduced in SimCLR (Chen et al., 2020). Given a batch of N images, each
represented by two augmented views, the loss treats the two views of the
*same* image as positives and all 2(N-1) other views as negatives.

For a projected embedding pair (z_i, z_j):

    L_i = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )

where sim(·,·) is cosine similarity and τ is a temperature hyper-parameter.
The full loss averages over both directions (i→j and j→i).
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Compute the NT-Xent (InfoNCE) contrastive loss for a batch of paired
    projections produced by SimCLR.

    Implementation details
    ----------------------
    1. L2-normalise z1 and z2 so that dot products equal cosine similarities.
    2. Concatenate into a single (2N, D) matrix `representations`.
    3. Compute the full (2N, 2N) cosine similarity matrix.
    4. Remove self-similarity entries (diagonal) to exclude i vs i comparisons.
    5. Treat (z_i, z_j) and (z_j, z_i) as the positive pair; the remaining
       2N-2 entries per row are the negatives.
    6. Formulate as cross-entropy where the positive logit is always placed
       first (index 0) and the label is 0 for every sample.

    Parameters
    ----------
    z1          : Tensor of shape (N, D). Projections from the first views.
    z2          : Tensor of shape (N, D). Projections from the second views.
    temperature : float. Scaling factor τ; lower values produce sharper
                  distributions. Default 0.5 follows the SimCLR paper.

    Returns
    -------
    Scalar loss tensor (average NT-Xent over all 2N samples in the batch).
    """

    # Step 1: L2-normalise so that matmul gives cosine similarities.
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    # Step 2: Stack both views into one (2N, D) tensor.
    representations = torch.cat([z1, z2], dim=0)

    # Step 3: Full (2N, 2N) pairwise cosine similarity matrix.
    similarity_matrix = torch.matmul(representations, representations.T)

    # Step 4: Mask out self-similarities (diagonal) to avoid trivial i==i pairs.
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    # Step 5: Gather the positive similarities — z1[i] vs z2[i] and vice versa.
    positives = torch.cat([
        torch.sum(z1 * z2, dim=1),   # similarities for the z1 → z2 direction
        torch.sum(z2 * z1, dim=1)    # similarities for the z2 → z1 direction
    ], dim=0)

    # Step 6: Apply temperature scaling to all terms.
    logits = similarity_matrix / temperature
    positives = positives / temperature

    # Label 0 means "the positive is always at index 0 in the logit vector".
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    # Prepend the positive logit to the negative logits along dim=1.
    logits = torch.cat([positives.unsqueeze(1), logits], dim=1)

    # Cross-entropy with label=0 forces the model to rank the positive first.
    return F.cross_entropy(logits, labels)
