import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils import create_label_subset
from .utils import add_label_noise

def get_cifar10(batch_size=128, num_workers=4, label_fraction=1.0, noise_ratio=0.0):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    if label_fraction < 1.0:
        train_dataset = create_label_subset(train_dataset, label_fraction)

    if noise_ratio > 0.0:
        train_dataset = add_label_noise(train_dataset, noise_ratio)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_cifar10_eval(batch_size=128, num_workers=4):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_cifar10_eval_vit(batch_size=128, num_workers=4):

    import torchvision.transforms as transforms
    import torchvision
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_cifar10_vit_subset(label_fraction=1.0, batch_size=128, num_workers=4):

    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from .utils import create_label_subset

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    if label_fraction < 1.0:
        train_dataset = create_label_subset(train_dataset, label_fraction)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader