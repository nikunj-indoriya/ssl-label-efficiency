import torch
import timm

from datasets.cifar import get_cifar10, get_cifar10_eval_vit
from models.resnet import ResNet50
from training.train_supervised import train, evaluate

from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear


def compute_robustness(acc_clean, acc_noisy):
    return acc_noisy / acc_clean if acc_clean > 0 else 0.0


def evaluate_supervised(device, noise_levels, label_fraction):

    results = {}

    print("\n===== Supervised =====")

    clean_acc = None

    for noise in noise_levels:

        print(f"\n--- Noise: {int(noise*100)}% ---")

        train_loader, test_loader = get_cifar10(
            label_fraction=label_fraction,
            noise_ratio=noise
        )

        model = ResNet50(num_classes=10).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            train(model, train_loader, optimizer, criterion, device)

        acc = evaluate(model, test_loader, device)

        if noise == 0.0:
            clean_acc = acc

        robustness = compute_robustness(clean_acc, acc)

        print(f"Accuracy: {acc:.4f} | Robustness: {robustness:.4f}")

        results[noise] = (acc, robustness)

    return results


def evaluate_ssl(method_name, encoder, device, noise_levels, label_fraction):

    results = {}

    print(f"\n===== {method_name} =====")

    encoder.eval()

    # Precompute test features (clean test set)
    if method_name == "MAE":
        _, test_loader = get_cifar10_eval_vit()
    else:
        _, test_loader = get_cifar10(label_fraction=1.0)

    test_features, test_labels = extract_features(encoder, test_loader, device)

    clean_acc = None

    for noise in noise_levels:

        print(f"\n--- Noise: {int(noise*100)}% ---")

        if method_name == "MAE":
            train_loader, _ = get_cifar10_eval_vit()
        else:
            train_loader, _ = get_cifar10(
                label_fraction=label_fraction,
                noise_ratio=noise
            )

        train_features, train_labels = extract_features(encoder, train_loader, device)

        classifier = train_linear_probe(train_features, train_labels, num_classes=10)

        acc = evaluate_linear(
            classifier,
            test_features.to(device),
            test_labels.to(device)
        )

        if noise == 0.0:
            clean_acc = acc

        robustness = compute_robustness(clean_acc, acc)

        print(f"Linear Probe Acc: {acc:.4f} | Robustness: {robustness:.4f}")

        results[noise] = (acc, robustness)

    return results


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_levels = [0.0, 0.2, 0.4]
    label_fraction = 0.1

    print("\n========== Label Noise Experiment ==========\n")

    # -------------------------------
    # Supervised
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