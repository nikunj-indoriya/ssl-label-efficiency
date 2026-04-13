import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mae_loader(batch_size=128):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader