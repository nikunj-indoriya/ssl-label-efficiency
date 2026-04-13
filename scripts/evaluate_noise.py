"""
scripts/evaluate_noise.py
-------------------------
Label-noise robustness experiment across all four methods.

For each method (Supervised, SimCLR, BYOL, MAE), this script measures how
performance degrades as the training labels become increasingly corrupted
with symmetric noise at levels {0%, 20%, 40%}.

A *robustness score* is computed for each method as:
    robustness = acc_noisy / acc_clean

Values close to 1.0 indicate that the method retains most of its performance
under noisy labels. SSL methods are expected to be more robust than supervised
training because their representations are learned from the data distribution
rather than from the (potentially corrupted) labels.

Experiment details
------------------
- All experiments use label_fraction=0.1 (10% labelled data) to simulate
  a realistic semi-supervised setting.
- Supervised model: retrained from scratch for each noise level (10 epochs).
- SSL models (SimCLR, BYOL): frozen encoder + linear probe trained on the
  noisy-labelled subset. Only the linear probe sees noisy labels; the
  encoder weights are never modified.
- MAE: uses a pretrained ViT-Base encoder (frozen); the linear probe is
  trained on the full clean eval training split (MAE is label-agnostic in
  its pre-training, so noise only affects the linear-probe training labels).
"""

import torch
import timm

from datasets.cifar import get_cifar10, get_cifar10_eval_vit
from models.resnet import ResNet50
from training.train_supervised import train, evaluate

from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def compute_robustness(acc_clean, acc_noisy):
    """
    Compute the robustness ratio: how much accuracy is retained under noise.

    Parameters
    ----------
    acc_clean : float. Accuracy on clean labels (noise=0).
    acc_noisy : float. Accuracy under noisy labels.

    Returns
    -------
    float. acc_noisy / acc_clean, or 0.0 if acc_clean is zero.
    """
    return acc_noisy / acc_clean if acc_clean > 0 else 0.0


def evaluate_supervised(device, noise_levels, label_fraction):
    """
    Train and evaluate a supervised ResNet-50 at each noise level.

    A fresh model is trained from scratch for each noise level so that the
    reported accuracy reflects end-to-end performance under the specified
    degree of label corruption.

    Parameters
    ----------
    device         : torch.device. Target computation device.
    noise_levels   : list of float. Noise ratios to sweep (e.g. [0.0, 0.2, 0.4]).
    label_fraction : float. Fraction of training labels to use.

    Returns
    -------
    dict mapping noise_level → (accuracy, robustness_score).
    """

    results = {}

    print("\n===== Supervised =====")

    # Track the clean (noise=0) accuracy to compute robustness for subsequent levels.
    clean_acc = None

    for noise in noise_levels:

        print(f"\n--- Noise: {int(noise*100)}% ---")

        # Build a training loader with the specified label fraction AND noise ratio.
        train_loader, test_loader = get_cifar10(
            label_fraction=label_fraction,
            noise_ratio=noise
        )

        # Fresh model for each noise level — no weight sharing across runs.
        model = ResNet50(num_classes=10).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # Train for 10 epochs (abbreviated for experimental efficiency).
        for epoch in range(10):
            train(model, train_loader, optimizer, criterion, device)

        # Evaluate on the clean (unaugmented) test set.
        acc = evaluate(model, test_loader, device)

        # Store the clean accuracy from the first (noise=0) iteration.
        if noise == 0.0:
            clean_acc = acc

        robustness = compute_robustness(clean_acc, acc)

        print(f"Accuracy: {acc:.4f} | Robustness: {robustness:.4f}")

        results[noise] = (acc, robustness)

    return results


def evaluate_ssl(method_name, encoder, device, noise_levels, label_fraction):
    """
    Evaluate an SSL encoder under label noise using linear probing.

    The encoder is frozen throughout — only the linear probe sees noisy labels.
    This reflects the SSL paradigm: representations are learned from the
    unlabelled data distribution and are therefore inherently label-noise robust.

    Parameters
    ----------
    method_name    : str. Display name for logging ("SimCLR", "BYOL", "MAE").
    encoder        : nn.Module. Frozen SSL encoder in eval mode.
    device         : torch.device. Target computation device.
    noise_levels   : list of float. Noise ratios to sweep.
    label_fraction : float. Fraction of training labels to use.

    Returns
    -------
    dict mapping noise_level → (linear_probe_accuracy, robustness_score).
    """

    results = {}

    print(f"\n===== {method_name} =====")

    encoder.eval()

    # Pre-compute test features once; the test set is always clean.
    # MAE uses 224×224 inputs (ViT); others use 32×32 (ResNet).
    if method_name == "MAE":
        _, test_loader = get_cifar10_eval_vit()
    else:
        _, test_loader = get_cifar10(label_fraction=1.0)

    test_features, test_labels = extract_features(encoder, test_loader, device)

    # Track the clean (noise=0) accuracy for the robustness ratio.
    clean_acc = None

    for noise in noise_levels:

        print(f"\n--- Noise: {int(noise*100)}% ---")

        # For MAE, the linear probe is trained on the clean eval train split
        # because the ViT features are from a pretrained ImageNet model and
        # injecting CIFAR noise into the linear probe training labels is the
        # relevant perturbation (not the MAE pre-training data).
        if method_name == "MAE":
            train_loader, _ = get_cifar10_eval_vit()
        else:
            # For ResNet-based SSL methods, build a noisy labelled subset.
            train_loader, _ = get_cifar10(
                label_fraction=label_fraction,
                noise_ratio=noise
            )

        # Extract training features using the frozen SSL encoder.
        train_features, train_labels = extract_features(encoder, train_loader, device)

        # Train the linear probe on noisy training features / labels.
        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        # Evaluate on the clean test features.
        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        # Record clean accuracy from the first (noise=0) run.
        if noise == 0.0:
            clean_acc = acc

        robustness = compute_robustness(clean_acc, acc)

        print(f"Linear Probe Acc: {acc:.4f} | Robustness: {robustness:.4f}")

        results[noise] = (acc, robustness)

    return results


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Noise levels to sweep: 0%, 20%, 40% symmetric label corruption.
    noise_levels = [0.0, 0.2, 0.4]

    # Use 10% label fraction to simulate a semi-supervised setting.
    label_fraction = 0.1

    print("\n========== Label Noise Experiment ==========\n")

    # -------------------------------
    # Supervised baseline
    # -------------------------------
    supervised_results = evaluate_supervised(
        device, noise_levels, label_fraction
    )

    # -------------------------------
    # SimCLR
    # -------------------------------
    simclr_model = ResNet50(num_classes=10).to(device)
    simclr_model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))

    simclr_results = evaluate_ssl(
        "SimCLR",
        simclr_model,
        device,
        noise_levels,
        label_fraction
    )

    # -------------------------------
    # BYOL
    # -------------------------------
    byol_model = ResNet50(num_classes=10).to(device)
    byol_model.backbone.load_state_dict(torch.load("byol_encoder.pth"))

    byol_results = evaluate_ssl(
        "BYOL",
        byol_model,
        device,
        noise_levels,
        label_fraction
    )

    # -------------------------------
    # MAE (Pretrained ViT)
    # -------------------------------
    # Use a pretrained ViT-Base/16 as the MAE encoder proxy.
    # `num_classes=0` returns pooled CLS token features (768-d).
    mae_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=0
    ).to(device)

    mae_results = evaluate_ssl(
        "MAE",
        mae_model,
        device,
        noise_levels,
        label_fraction
    )


if __name__ == "__main__":
    main()
