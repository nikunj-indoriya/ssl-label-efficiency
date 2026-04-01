import torch

from models.resnet import ResNet50
from datasets.cifar import get_cifar10_eval
from evaluation.extract_features import extract_features

from analysis.representation_analysis import (
    compute_covariance,
    compute_spectrum,
    effective_rank,
    plot_spectrum
)


def analyze_model(model, name, device):

    print(f"\n===== {name} Analysis =====")

    # Load clean dataset
    train_loader, _ = get_cifar10_eval()

    # Extract features
    features, _ = extract_features(model, train_loader, device)

    # Compute covariance
    cov = compute_covariance(features)

    # Spectrum
    eigvals = compute_spectrum(cov)

    # Effective rank
    rank = effective_rank(eigvals)

    print(f"{name} Effective Rank: {rank:.4f}")

    # Plot
    plot_spectrum(eigvals, f"{name} Spectrum")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Supervised Model
    # -----------------------------
    supervised_model = ResNet50(num_classes=10).to(device)

    # IMPORTANT: load your trained supervised weights
    supervised_model.load_state_dict(torch.load("supervised_model.pth"))
    supervised_model.eval()

    analyze_model(supervised_model, "Supervised", device)

    # -----------------------------
    # SimCLR Model
    # -----------------------------
    simclr_model = ResNet50(num_classes=10).to(device)

    simclr_model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))
    simclr_model.eval()

    analyze_model(simclr_model, "SimCLR", device)


if __name__ == "__main__":
    main()