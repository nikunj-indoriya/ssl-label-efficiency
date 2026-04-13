"""
main.py
-------
Supervised baseline training and label-efficiency evaluation on CIFAR-10.

This script is the entry point for the supervised learning component of the
SSL label-efficiency study. It runs the following experiment for each label
fraction in {1.0, 0.5, 0.2, 0.1, 0.05, 0.01}:

  1. Build a labelled training subset at the target fraction.
  2. Train a ResNet-50 from scratch with cross-entropy loss (supervised).
  3. Record the fine-tune accuracy (end-to-end supervised performance).
  4. Evaluate representation quality via two linear-probe variants:
       a. Full-data probe  : features from full-dataset training split →
          linear classifier → test accuracy. Measures representation quality
          independent of the label budget.
       b. Same-fraction probe: features from the same subset used for training →
          linear classifier → test accuracy. Measures what a linear head can
          extract from the fraction-specific representation.

Results are printed after each fraction and summarised at the end. The model
trained on 100 % of labels is saved to `supervised_model.pth` for downstream
use in representation geometry and noise robustness experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar import get_cifar10
from models.resnet import ResNet50
from training.train_supervised import train, evaluate

from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label fractions to sweep: from full supervision down to 1 % labels.
    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    # Stores (label_fraction, fine_tune_accuracy) for the final summary.
    results = []

    for label_fraction in fractions:

        print(f"\n========== Fraction: {label_fraction} ==========")

        # Build the stratified training subset and the fixed test loader.
        # The test loader always uses the full test set (no sub-sampling).
        train_loader, test_loader = get_cifar10(label_fraction=label_fraction)

        # Fresh ResNet-50 initialised from scratch for each fraction.
        # No pre-trained weights — this is the purely supervised baseline.
        model = ResNet50(num_classes=10).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # ----- Supervised Training -----
        # Train for 10 epochs — a short run to approximate relative label
        # efficiency trends rather than reach peak accuracy.
        for epoch in range(10):
            loss = train(model, train_loader, optimizer, criterion, device)
            finetune_acc = evaluate(model, test_loader, device)

            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={finetune_acc:.4f}")

        print(f"Final Fine-tune Accuracy: {finetune_acc:.4f}")

        results.append((label_fraction, finetune_acc))

        # Save the full-data supervised model checkpoint. Only done at
        # label_fraction=1.0 to avoid overwriting with lower-quality models.
        if label_fraction == 1.0:
            torch.save(model.state_dict(), "supervised_model.pth")
            print("Supervised model saved as supervised_model.pth")

        # ----- Linear Probe -----

        # --- Probe variant 1: full-data features ---
        # Extract features using the entire training set (all 50,000 images).
        # This isolates representation quality from the label budget — the
        # linear classifier sees the same amount of data regardless of fraction.
        full_train_loader, _ = get_cifar10(label_fraction=1.0)

        train_features_full, train_labels_full = extract_features(model, full_train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)

        classifier_full = train_linear_probe(
            train_features_full,
            train_labels_full,
            num_classes=10
        )

        linear_acc_full = evaluate_linear(
            classifier_full,
            test_features.to(device),
            test_labels.to(device)
        )

        # --- Probe variant 2: same-fraction features ---
        # Extract features from the same labelled subset used for supervised
        # training. Measures what the linear head can learn from the limited
        # feature set that the model actually trained on.
        train_features_small, train_labels_small = extract_features(model, train_loader, device)

        classifier_small = train_linear_probe(
            train_features_small,
            train_labels_small,
            num_classes=10
        )

        linear_acc_small = evaluate_linear(
            classifier_small,
            test_features.to(device),
            test_labels.to(device)
        )

        print("Linear Probe (full data):", linear_acc_full)
        print("Linear Probe (same fraction):", linear_acc_small)

    # ----- Final Summary -----
    print("\n========== FINAL RESULTS ==========")
    for frac, acc in results:
        print(f"{frac}: {acc:.4f}")


if __name__ == "__main__":
    main()
