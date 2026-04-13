"""
scripts/run_representation_analysis.py
---------------------------------------
Spectral analysis of learned representations for Supervised and SimCLR models.

This script performs covariance-spectrum analysis on the feature spaces of
pre-trained encoders to understand how different training objectives shape
the geometry of the learned representations.

Analysis pipeline (per model)
------------------------------
1. Extract feature vectors for the entire clean CIFAR-10 training set.
2. Compute the empirical covariance matrix of the centred features.
3. Compute the eigenvalue spectrum of the covariance matrix.
4. Derive the *effective rank* — an entropy-based measure of how many
   dimensions are meaningfully used by the representation.
5. Plot and save the eigenvalue spectrum (log scale) for visual inspection.

Effective rank interpretation
------------------------------
- High effective rank → the representation is spread across many dimensions
  (good for downstream tasks; more linear separability).
- Low effective rank → the representation collapses to a low-dimensional
  subspace (feature collapse, common in poorly trained SSL models).
"""

import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10_eval
from evaluation.extract_features import extract_features

from analysis.representation_analysis import (
    compute_covariance,
    compute_spectrum,
    effective_rank,
    plot_spectrum
)


def analyze_model(model, name, device):
    """
    Run the full spectral analysis pipeline for a single model.

    Parameters
    ----------
    model  : nn.Module. Pre-trained encoder to analyse.
    name   : str. Display name used for logging and plot titles.
    device : torch.device. Device on which the model resides.
    """

    print(f"\n===== {name} Analysis =====")

    # Load the clean CIFAR-10 training split (no augmentation) for
    # deterministic feature extraction.
    train_loader, _ = get_cifar10_eval()

    # Extract 2048-d backbone feature vectors for all 50,000 training samples.
    features, _ = extract_features(model, train_loader, device)

    # Compute the empirical covariance matrix of mean-centred features.
    # Shape: (2048, 2048) — symmetric positive semi-definite.
    cov = compute_covariance(features)

    # Compute the sorted eigenvalue spectrum of the covariance matrix.
    # Eigenvalues represent the variance explained by each principal component.
    eigvals = compute_spectrum(cov)

    # Effective rank: exp(H(p)) where p = normalised eigenvalue distribution.
    # Provides a single scalar summary of how spread out the spectrum is.
    rank = effective_rank(eigvals)

    print(f"{name} Effective Rank: {rank:.4f}")

    # Save the eigenvalue decay plot for visual inspection of dimensionality usage.
    plot_spectrum(eigvals, f"{name} Spectrum")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Supervised Model
    # -----------------------------
    supervised_model = ResNet50(num_classes=10).to(device)

    # Load the checkpoint saved by main.py after full-data supervised training.
    supervised_model.load_state_dict(torch.load("supervised_model.pth"))
    supervised_model.eval()

    analyze_model(supervised_model, "Supervised", device)

    # -----------------------------
    # SimCLR Model
    # -----------------------------
    simclr_model = ResNet50(num_classes=10).to(device)

    # Load only the backbone weights from the SimCLR pre-training checkpoint.
    simclr_model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))
    simclr_model.eval()

    analyze_model(simclr_model, "SimCLR", device)


if __name__ == "__main__":
    main()
