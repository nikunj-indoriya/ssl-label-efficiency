"""
analysis/representation_analysis.py
-------------------------------------
Spectral analysis utilities for evaluating learned representations.

These functions operate on the empirical covariance matrix of a feature
matrix to characterise the dimensionality and spread of the representation:

  compute_covariance  — centred empirical covariance matrix.
  compute_spectrum    — sorted eigenvalue decomposition.
  effective_rank      — entropy-based effective dimensionality.
  plot_spectrum       — log-scale eigenvalue decay plot.

Effective rank
--------------
Defined as exp(H(p)) where H is the Shannon entropy of the normalised
eigenvalue distribution p = λ / Σλ. A representation with a flat spectrum
(all eigenvalues equal) has maximum effective rank = D. A representation
that collapses to a single direction has effective rank = 1.
"""

import torch
import numpy as np


def compute_covariance(features):
    """
    Compute the empirical covariance matrix of a feature matrix.

    The features are mean-centred before computing the outer product to
    remove the effect of the global feature mean.

    Parameters
    ----------
    features : Tensor of shape (N, D). Feature vectors for N samples.

    Returns
    -------
    cov : Tensor of shape (D, D). Empirical covariance matrix.
          Normalised by (N - 1) for the unbiased estimator.
    """
    # Centre features by subtracting the per-dimension mean.
    features = features - features.mean(dim=0)

    # Unbiased covariance estimator: X^T X / (N - 1).
    cov = torch.matmul(features.T, features) / (features.shape[0] - 1)
    return cov


def compute_spectrum(cov):
    """
    Compute the eigenvalue spectrum of a symmetric positive semi-definite matrix.

    Uses `torch.linalg.eigvalsh` which is numerically stable for symmetric
    matrices and more efficient than the general eigendecomposition.

    Parameters
    ----------
    cov : Tensor of shape (D, D). Symmetric covariance matrix.

    Returns
    -------
    eigvals : numpy.ndarray of shape (D,). Eigenvalues sorted in descending
              order (largest first), on CPU.
    """
    # eigvalsh returns eigenvalues in ascending order; reverse for descending.
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.sort(eigvals, descending=True).values
    return eigvals.cpu().numpy()


def effective_rank(eigvals):
    """
    Compute the effective rank as the exponential of the Shannon entropy of
    the normalised eigenvalue distribution.

    Formula:
        p_i   = λ_i / Σ λ_j          (normalised eigenvalue distribution)
        H(p)  = -Σ p_i log(p_i + ε)  (Shannon entropy with numerical guard)
        rank  = exp(H(p))

    Parameters
    ----------
    eigvals : numpy.ndarray of shape (D,). Eigenvalue spectrum (any order).

    Returns
    -------
    float. Effective rank in [1, D].
    """
    # Normalise to a probability distribution.
    eigvals = eigvals / np.sum(eigvals)

    # Shannon entropy of the normalised eigenvalue distribution.
    # Small epsilon prevents log(0) for near-zero eigenvalues.
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))

    return np.exp(entropy)


import matplotlib.pyplot as plt


def plot_spectrum(eigvals, title):
    """
    Plot and save the eigenvalue decay curve on a log-scale y-axis.

    A steep decay indicates most variance is concentrated in a few dimensions
    (low effective rank). A flat decay suggests the representation uses many
    dimensions (high effective rank).

    Parameters
    ----------
    eigvals : numpy.ndarray. Eigenvalue spectrum (descending order).
    title   : str. Plot title and base name of the saved PNG file.
    """
    plt.figure()
    plt.plot(eigvals)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue (log)')
    plt.grid(True)
    plt.show()
    # Save with the same name as the title for easy identification.
    plt.savefig(f"{title}.png")
