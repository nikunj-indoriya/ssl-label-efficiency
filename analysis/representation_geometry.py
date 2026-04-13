import torch
import numpy as np


def compute_effective_rank(features):
    features = features - features.mean(dim=0)

    cov = torch.matmul(features.T, features) / features.shape[0]

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=1e-8)
    eigvals = eigvals / eigvals.sum()

    entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum()

    return torch.exp(entropy).item()


def compute_intra_class_variance(features, labels):

    classes = torch.unique(labels)
    intra_var = 0.0

    for c in classes:
        class_feats = features[labels == c]
        mean = class_feats.mean(dim=0)

        intra_var += ((class_feats - mean) ** 2).sum() / len(class_feats)

    return (intra_var / len(classes)).item()


def compute_inter_class_distance(features, labels):

    classes = torch.unique(labels)
    means = []

    for c in classes:
        class_feats = features[labels == c]
        means.append(class_feats.mean(dim=0))

    means = torch.stack(means)

    dist = 0.0
    count = 0

    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            dist += torch.norm(means[i] - means[j])
            count += 1

    return (dist / count).item()


def compute_separation(features, labels):

    intra = compute_intra_class_variance(features, labels)
    inter = compute_inter_class_distance(features, labels)

    return inter / (intra + 1e-8)