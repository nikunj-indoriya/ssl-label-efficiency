"""
scripts/evaluate_mae.py
-----------------------
Linear-probe evaluation of the MAE (pretrained ViT) encoder across label fractions.

Unlike SimCLR and BYOL which load from-scratch SSL checkpoints, this script
uses a *pretrained* ViT-Base/16 from `timm` as the MAE encoder proxy. This
decision reflects the practical reality that training a ViT-Base from scratch
on 50K CIFAR-10 images is data-limited and compute-intensive — the pretrained
ImageNet ViT serves as a strong upper-bound reference for the MAE architecture.

Images are resized to 224×224 to match the ViT patch-embedding input
resolution (16×16 patches → 14×14 = 196 patch tokens).

Protocol
--------
1. Load pretrained ViT-Base/16 from `timm` with `num_classes=0` (feature
   extractor mode — returns 768-d CLS token embeddings).
2. Extract test features once (224×224 evaluation split).
3. For each label fraction in {100%, 50%, 20%, 10%, 5%, 1%}:
     a. Build a 224×224 training subset at the target fraction.
     b. Extract training features with the frozen ViT encoder.
     c. Train a linear classifier on those features.
     d. Report top-1 accuracy on the pre-computed test features.
"""

import torch
import timm

from datasets.cifar import get_cifar10_eval_vit, get_cifar10_vit_subset
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load PRETRAINED ViT (MAE-style encoder proxy) ----
    # `pretrained=True` loads ImageNet-21k weights — this is the key
    # difference from SimCLR/BYOL which use randomly initialised ResNet-50s.
    # `num_classes=0` removes the classification head so `forward()` returns
    # the pooled CLS token embedding (768-d for ViT-Base).
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,   # IMPORTANT: load ImageNet pretrained weights
        num_classes=0
    ).to(device)

    model.eval()   # freeze BN / dropout — ViT has no BN, but good practice

    # ---- Evaluation data (224×224) ----
    # Both train and test loaders use 224×224 images with no augmentation
    # to ensure deterministic feature extraction.
    eval_train_loader, eval_test_loader = get_cifar10_eval_vit()

    # Pre-compute test features once — reused across all fraction experiments.
    test_features, test_labels = extract_features(model, eval_test_loader, device)

    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    print("\n=== MAE (Pretrained ViT) Evaluation ===\n")

    for frac in fractions:

        # Get a 224×224 training subset at the target label fraction.
        train_loader = get_cifar10_vit_subset(label_fraction=frac)

        # Extract features from the labelled training subset using the frozen
        # pretrained ViT encoder.
        train_features, train_labels = extract_features(model, train_loader, device)

        # Train a linear classifier on top of the frozen ViT features.
        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        # Evaluate on the pre-computed 224×224 test features.
        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        print(f"{int(frac*100)}%: {acc:.4f}")


if __name__ == "__main__":
    main()
