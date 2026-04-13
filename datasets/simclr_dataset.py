import torchvision
from torch.utils.data import Dataset
from methods.augmentations import get_simclr_augmentation

class SimCLRDataset(Dataset):
    def __init__(self, root="./data"):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True
        )
        self.transform = get_simclr_augmentation()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]

        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2