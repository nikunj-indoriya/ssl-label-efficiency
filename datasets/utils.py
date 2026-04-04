import numpy as np
from torch.utils.data import Subset

def create_label_subset(dataset, fraction, num_classes=10, seed=42):
    np.random.seed(seed)

    targets = np.array(dataset.targets)
    indices = []

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        n_samples = int(len(class_indices) * fraction)

        selected = np.random.choice(class_indices, n_samples, replace=False)
        indices.extend(selected)

    return Subset(dataset, indices)

def add_label_noise(dataset, noise_ratio, num_classes=10, seed=42):

    np.random.seed(seed)

    # Case 1: Full dataset (normal CIFAR)
    if hasattr(dataset, "targets"):

        targets = np.array(dataset.targets)
        n_samples = len(targets)

        n_noisy = int(noise_ratio * n_samples)
        indices = np.random.choice(n_samples, n_noisy, replace=False)

        for idx in indices:
            original = targets[idx]
            new_label = np.random.randint(num_classes)

            while new_label == original:
                new_label = np.random.randint(num_classes)

            targets[idx] = new_label

        dataset.targets = targets.tolist()
        return dataset

    # Case 2: Subset dataset
    elif isinstance(dataset, Subset):

        base_dataset = dataset.dataset
        indices = dataset.indices

        targets = np.array(base_dataset.targets)

        n_samples = len(indices)
        n_noisy = int(noise_ratio * n_samples)

        noisy_indices = np.random.choice(indices, n_noisy, replace=False)

        for idx in noisy_indices:
            original = targets[idx]
            new_label = np.random.randint(num_classes)

            while new_label == original:
                new_label = np.random.randint(num_classes)

            targets[idx] = new_label

        base_dataset.targets = targets.tolist()
        return dataset

    else:
        raise ValueError("Unsupported dataset type for noise injection")