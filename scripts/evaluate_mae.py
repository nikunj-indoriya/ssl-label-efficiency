import torch
import timm

from datasets.cifar import get_cifar10_eval_vit, get_cifar10_vit_subset
from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load PRETRAINED ViT (MAE-style encoder proxy) ----
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,   # IMPORTANT
        num_classes=0
    ).to(device)

    model.eval()

    # ---- Evaluation data (224x224) ----
    eval_train_loader, eval_test_loader = get_cifar10_eval_vit()

    test_features, test_labels = extract_features(model, eval_test_loader, device)

    fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    print("\n=== MAE (Pretrained ViT) Evaluation ===\n")

    for frac in fractions:

        train_loader = get_cifar10_vit_subset(label_fraction=frac)

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