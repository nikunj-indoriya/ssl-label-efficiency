import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10, get_cifar10_eval
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load SimCLR Encoder --------
    model = ResNet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))
    model.eval()

    # -------- Clean data for feature extraction --------
    eval_train_loader, eval_test_loader = get_cifar10_eval()

    # Extract test features once (reuse)
    test_features, test_labels = extract_features(model, eval_test_loader, device)

    # -------- Evaluate across label fractions --------
    fractions = [1.0, 0.1, 0.05, 0.01]

    print("\n=== SimCLR Evaluation ===\n")

    for frac in fractions:

        print(f"--- Label Fraction: {int(frac*100)}% ---")

        # Get subset loader (for labels)
        train_loader, _ = get_cifar10(label_fraction=frac)

        # IMPORTANT: extract features from SAME subset indices but CLEAN images
        train_features, train_labels = extract_features(model, train_loader, device)

        # Train linear classifier
        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        # Evaluate
        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        print(f"Linear Probe Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()