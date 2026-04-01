import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10_eval, get_cifar10
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load("byol_encoder.pth"))
    model.eval()

    eval_train_loader, eval_test_loader = get_cifar10_eval()

    test_features, test_labels = extract_features(model, eval_test_loader, device)

    fractions = [1.0, 0.1, 0.01]

    print("\n=== BYOL Evaluation ===\n")

    for frac in fractions:

        train_loader, _ = get_cifar10(label_fraction=frac)

        train_features, train_labels = extract_features(model, train_loader, device)

        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        print(f"{int(frac*100)}%: {acc:.4f}")


if __name__ == "__main__":
    main()