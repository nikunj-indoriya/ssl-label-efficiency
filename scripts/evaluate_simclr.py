"""
scripts/evaluate_simclr.py
--------------------------
Linear-probe evaluation of the SimCLR pre-trained encoder across label fractions.

Protocol
--------
1. Load the SimCLR encoder checkpoint (`simclr_encoder.pth`) into a ResNet50
   wrapper (backbone only — the classifier head is unused here).
2. Extract test set features once with the frozen encoder (reused across all
   fraction experiments to avoid redundant computation).
3. For each label fraction in {100%, 50%, 20%, 10%, 5%, 1%}:
     a. Build a labelled training subset at the target fraction.
     b. Extract training features from that subset using the frozen encoder.
     c. Train a linear classifier on the extracted training features.
     d. Evaluate the classifier on the pre-computed test features.
     e. Report linear probe accuracy.

Key design note
---------------
Feature extraction uses the evaluation data loaders (no augmentation) rather
than the training loaders. This ensures that the extracted features are
deterministic and not distorted by random augmentation, which would add
variance to the linear classifier's training signal.
"""

import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10, get_cifar10_eval
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load SimCLR Encoder --------
    # Instantiate the ResNet50 wrapper and load only the backbone weights
    # from the SimCLR pre-training checkpoint. The classifier head is ignored.
    model = ResNet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))
    model.eval()   # freeze batch-norm statistics and disable dropout

    # -------- Clean data for feature extraction --------
    # Use the evaluation loader (no augmentation) so features are
    # deterministic. The training split is needed for train features;
    # the test split provides the fixed evaluation set.
    eval_train_loader, eval_test_loader = get_cifar10_eval()

    # Extract test features once — these are reused for every fraction.
    # Computing them once saves significant time when sweeping many fractions.
    test_features, test_labels = extract_features(model, eval_test_loader, device)

    # -------- Evaluate across label fractions --------
    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    print("\n=== SimCLR Evaluation ===\n")

    for frac in fractions:

        print(f"--- Label Fraction: {int(frac*100)}% ---")

        # Build a training loader whose indices correspond to the desired
        # fraction. The same stratified sampling as in main.py is applied.
        train_loader, _ = get_cifar10(label_fraction=frac)

        # Extract features from the labelled training subset using the frozen
        # SimCLR encoder. Only the subset indices are forwarded — the rest
        # of the training data is not used for the linear classifier.
        train_features, train_labels = extract_features(model, train_loader, device)

        # Train a linear classifier on top of the frozen features.
        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        # Evaluate the trained linear classifier on the held-out test features.
        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        print(f"Linear Probe Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()
