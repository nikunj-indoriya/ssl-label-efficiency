"""
scripts/run_representation_geometry.py
---------------------------------------
Comprehensive representation geometry analysis across all four methods.

For each pre-trained encoder (Supervised, SimCLR, BYOL, MAE), this script
computes four complementary geometric metrics on the learned feature space:

  1. Effective Rank
     Entropy-based measure of how many dimensions the representation actively
     uses. Higher rank → less dimensional collapse → more expressive features.

  2. Intra-class Variance
     Average within-class feature spread. Lower intra-class variance indicates
     that the same-class images cluster tightly in feature space.

  3. Inter-class Distance
     Average pairwise distance between class centroids. Higher values indicate
     that different classes are well-separated in feature space.

  4. Separation Ratio (= inter-class distance / intra-class variance)
     A single composite metric summarising class discriminability. Higher
     separation → better linear separability → higher linear-probe accuracy.

All features are L2-normalised before computing geometry metrics so that
the analysis is scale-invariant and operates on the unit hypersphere.

Models
------
- Supervised  : ResNet-50 (full-data supervised training, `supervised_model.pth`).
- SimCLR      : ResNet-50 backbone (SimCLR checkpoint, `simclr_encoder.pth`).
- BYOL        : ResNet-50 backbone (BYOL checkpoint, `byol_encoder.pth`).
- MAE (ViT)   : Pretrained ViT-Base/16 (224×224 inputs, timm pretrained weights).
"""

import torch
import timm

from datasets.cifar import get_cifar10_eval, get_cifar10_eval_vit
from models.resnet import ResNet50
from evaluation.extract_features import extract_features

from analysis.representation_geometry import (
    compute_effective_rank,
    compute_intra_class_variance,
    compute_inter_class_distance,
    compute_separation
)

import torch.nn.functional as F


def analyze(name, model, loader, device):
    """
    Extract features, apply L2 normalisation, and compute geometry metrics.

    Parameters
    ----------
    name   : str. Display name used for logging output.
    model  : nn.Module. Pre-trained frozen encoder.
    loader : DataLoader. Clean evaluation loader (no augmentation).
    device : torch.device. Device on which the model resides.

    Prints
    ------
    Effective Rank, Intra-class Variance, Inter-class Distance, and
    Separation Ratio for the given model.
    """

    print(f"\n===== {name} =====")

    # Extract all feature vectors and labels for the evaluation split.
    features, labels = extract_features(model, loader, device)

    # Move to the target device for geometry computations.
    features = features.to(device)
    labels = labels.to(device)

    # L2-normalise features to the unit hypersphere before computing distances.
    # This makes all metrics scale-invariant and comparable across models with
    # different feature magnitudes (e.g. ResNet vs ViT).
    features = F.normalize(features, dim=1)

    # Compute and log all four geometry metrics.
    rank = compute_effective_rank(features)
    intra = compute_intra_class_variance(features, labels)
    inter = compute_inter_class_distance(features, labels)
    separation = compute_separation(features, labels)

    print(f"Effective Rank: {rank:.4f}")
    print(f"Intra-class Variance: {intra:.4f}")
    print(f"Inter-class Distance: {inter:.4f}")
    print(f"Separation Ratio: {separation:.4f}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Supervised
    # -----------------------
    # Load the full-data supervised ResNet-50 checkpoint.
    sup_model = ResNet50(num_classes=10).to(device)
    sup_model.load_state_dict(torch.load("supervised_model.pth"))

    # Use 32×32 clean evaluation loader for ResNet-based models.
    _, test_loader = get_cifar10_eval()

    analyze("Supervised", sup_model, test_loader, device)

    # -----------------------
    # SimCLR
    # -----------------------
    # Load SimCLR backbone checkpoint into the ResNet50 wrapper.
    simclr_model = ResNet50(num_classes=10).to(device)
    simclr_model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))

    analyze("SimCLR", simclr_model, test_loader, device)

    # -----------------------
    # BYOL
    # -----------------------
    # Load BYOL online-encoder checkpoint into the ResNet50 wrapper.
    byol_model = ResNet50(num_classes=10).to(device)
    byol_model.backbone.load_state_dict(torch.load("byol_encoder.pth"))

    analyze("BYOL", byol_model, test_loader, device)

    # -----------------------
    # MAE (ViT)
    # -----------------------
    # Use a pretrained ViT-Base/16 as the MAE encoder proxy.
    # ViT requires 224×224 inputs — use the dedicated ViT evaluation loader.
    mae_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=0
    ).to(device)

    _, vit_test_loader = get_cifar10_eval_vit()

    analyze("MAE", mae_model, vit_test_loader, device)


if __name__ == "__main__":
    main()
