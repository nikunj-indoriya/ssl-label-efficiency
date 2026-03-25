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