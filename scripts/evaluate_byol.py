"""
scripts/evaluate_byol.py
------------------------
Linear-probe evaluation of the BYOL pre-trained encoder across label fractions.

Protocol
--------
1. Load the BYOL online-encoder checkpoint (`byol_encoder.pth`) into a
   ResNet50 wrapper (backbone weights only).
2. Extract test set features once with the frozen encoder.
3. For each label fraction in {100%, 50%, 20%, 10%, 5%, 1%}:
     a. Build a labelled training subset at the target fraction.
     b. Extract training features using the frozen BYOL encoder.
     c. Train a linear classifier on those features.
     d. Report top-1 accuracy on the pre-computed test features.

This is identical in structure to `evaluate_simclr.py`; only the checkpoint
that is loaded differs (byol_encoder.pth vs simclr_encoder.pth).
"""

import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10_eval, get_cifar10
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the BYOL online encoder. Only backbone weights are restored;
    # the projector, predictor, and target network are not needed here.
    model = ResNet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load("byol_encoder.pth"))
    model.eval()   # freeze BN statistics; disable dropout

    # Build clean evaluation loaders (no augmentation) for deterministic
    # feature extraction.
    eval_train_loader, eval_test_loader = get_cifar10_eval()

    # Pre-compute test features once to avoid redundant forward passes.
    test_features, test_labels = extract_features(model, eval_test_loader, device)

    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    print("\n=== BYOL Evaluation ===\n")

    for frac in fractions:

        # Build training loader for the target label fraction.
        train_loader, _ = get_cifar10(label_fraction=frac)

        # Extract training features from the labelled subset using the frozen
        # BYOL encoder backbone.
        train_features, train_labels = extract_features(model, train_loader, device)

        # Train the linear probe on top of frozen BYOL features.
        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        # Evaluate on pre-computed test features.
        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        print(f"{int(frac*100)}%: {acc:.4f}")


if __name__ == "__main__":
    main()
