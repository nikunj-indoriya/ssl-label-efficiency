import torch
import timm

from datasets.cifar import get_cifar10_eval, get_cifar10_eval_vit
from models.resnet import ResNet50
from evaluation.extract_features import extract_features

from analysis.representation_geometry import (
    compute_effective_rank,
    compute_intra_class_variance,
    compute_inter_class_distance,
    compute_separation
)


import torch.nn.functional as F

def analyze(name, model, loader, device):

    print(f"\n===== {name} =====")

    features, labels = extract_features(model, loader, device)

    features = features.to(device)
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    rank = compute_effective_rank(features)
    intra = compute_intra_class_variance(features, labels)
    inter = compute_inter_class_distance(features, labels)
    separation = compute_separation(features, labels)

    print(f"Effective Rank: {rank:.4f}")
    print(f"Intra-class Variance: {intra:.4f}")
    print(f"Inter-class Distance: {inter:.4f}")
    print(f"Separation Ratio: {separation:.4f}")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Supervised
    # -----------------------
    sup_model = ResNet50(num_classes=10).to(device)
    sup_model.load_state_dict(torch.load("supervised_model.pth"))

    _, test_loader = get_cifar10_eval()

    analyze("Supervised", sup_model, test_loader, device)

    # -----------------------
    # SimCLR
    # -----------------------
    simclr_model = ResNet50(num_classes=10).to(device)
    simclr_model.backbone.load_state_dict(torch.load("simclr_encoder.pth"))

    analyze("SimCLR", simclr_model, test_loader, device)

    # -----------------------
    # BYOL
    # -----------------------
    byol_model = ResNet50(num_classes=10).to(device)
    byol_model.backbone.load_state_dict(torch.load("byol_encoder.pth"))

    analyze("BYOL", byol_model, test_loader, device)

    # -----------------------
    # MAE (ViT)
    # -----------------------
    mae_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=0
    ).to(device)

    _, vit_test_loader = get_cifar10_eval_vit()

    analyze("MAE", mae_model, vit_test_loader, device)


if __name__ == "__main__":
    main()