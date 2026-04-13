"""
analysis/representation_geometry.py
-------------------------------------
Geometric metrics for characterising learned feature spaces.

These functions quantify the quality of a representation in terms of how
well it separates different classes and how compactly it clusters same-class
examples. They are used in `run_representation_geometry.py` to compare the
geometric properties of Supervised, SimCLR, BYOL, and MAE feature spaces.

Metrics
-------
  compute_effective_rank       — dimensionality utilisation (entropy of spectrum).
  compute_intra_class_variance — within-class compactness.
  compute_inter_class_distance — between-class separation (centroid distances).
  compute_separation           — composite ratio: inter / intra.

All functions expect features that have already been L2-normalised to the
unit hypersphere (as done in `run_representation_geometry.py`).
"""

import torch
import numpy as np


def compute_effective_rank(features):
    """
    Compute the effective rank of a feature matrix using the eigenvalue
    entropy of its covariance matrix.

    Parameters
    ----------
    features : Tensor of shape (N, D). L2-normalised feature vectors.

    Returns
    -------
    float. Effective rank in [1, D]; higher = more dimensions used.
    """

    # Mean-centre the features to compute a proper covariance matrix.
    features = features - features.mean(dim=0)

    # Compute the biased covariance matrix (normalised by N, not N-1).
    cov = torch.matmul(features.T, features) / features.shape[0]

    # Compute eigenvalues of the symmetric covariance matrix.
    eigvals = torch.linalg.eigvalsh(cov)

    # Clamp tiny negative eigenvalues (numerical artefacts) to a small positive
    # value before normalising to a probability distribution.
    eigvals = torch.clamp(eigvals, min=1e-8)

    # Normalise to a probability distribution over eigenvalues.
    eigvals = eigvals / eigvals.sum()

    # Shannon entropy of the normalised eigenvalue distribution.
    entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum()

    # Effective rank = exp(H) — the number of dimensions with significant variance.
    return torch.exp(entropy).item()


def compute_intra_class_variance(features, labels):
    """
    Compute the mean intra-class variance across all classes.

    For each class c, the intra-class variance is defined as:
        V_c = (1/|S_c|) Σ_{i ∈ S_c} ‖x_i - μ_c‖²

    where S_c is the set of indices for class c and μ_c is the class centroid.
    The overall metric is the mean V_c across all classes.

    Lower intra-class variance means same-class examples cluster more tightly,
    which corresponds to higher class purity in the feature space.

    Parameters
    ----------
    features : Tensor of shape (N, D). L2-normalised feature vectors.
    labels   : Tensor of shape (N,).   Integer class labels.

    Returns
    -------
    float. Mean intra-class variance across all classes.
    """

    classes = torch.unique(labels)
    intra_var = 0.0

    for c in classes:
        # Extract feature vectors belonging to class c.
        class_feats = features[labels == c]

        # Compute the centroid for class c.
        mean = class_feats.mean(dim=0)

        # Sum of squared distances from each sample to its class centroid,
        # normalised by the number of samples in the class.
        intra_var += ((class_feats - mean) ** 2).sum() / len(class_feats)

    # Average across all classes for a class-balanced metric.
    return (intra_var / len(classes)).item()


def compute_inter_class_distance(features, labels):
    """
    Compute the mean pairwise L2 distance between class centroids.

    A higher inter-class distance means the class centroids are well-separated
    in feature space, which makes linear classification easier.

    Parameters
    ----------
    features : Tensor of shape (N, D). L2-normalised feature vectors.
    labels   : Tensor of shape (N,).   Integer class labels.

    Returns
    -------
    float. Mean pairwise L2 distance between all pairs of class centroids.
    """

    classes = torch.unique(labels)
    means = []

    # Compute the centroid for each class.
    for c in classes:
        class_feats = features[labels == c]
        means.append(class_feats.mean(dim=0))

    means = torch.stack(means)   # shape: (num_classes, D)

    dist = 0.0
    count = 0

    # Enumerate all unique pairs of class centroids (upper triangle only).
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            dist += torch.norm(means[i] - means[j])
            count += 1

    # Normalise by the number of pairs to get the mean pairwise distance.
    return (dist / count).item()


def compute_separation(features, labels):
    """
    Compute the class separation ratio: inter-class distance / intra-class variance.

    A high separation ratio indicates that class centroids are far apart
    relative to within-class spread — i.e. the representation is both
    compact and discriminative.

    Parameters
    ----------
    features : Tensor of shape (N, D). L2-normalised feature vectors.
    labels   : Tensor of shape (N,).   Integer class labels.

    Returns
    -------
    float. Separation ratio (inter / (intra + ε)).
           Higher is better for downstream classification.
    """

    intra = compute_intra_class_variance(features, labels)
    inter = compute_inter_class_distance(features, labels)

    # Small epsilon prevents division by zero when intra-class variance ≈ 0.
    return inter / (intra + 1e-8)
