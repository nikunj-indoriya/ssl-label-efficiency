import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar import get_cifar10
from models.resnet import ResNet50
from training.train_supervised import train, evaluate

from evaluation.extract_features import extract_features
from evaluation.linear_probe import train_linear_probe, evaluate_linear

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_fraction = 1.0  #change this to 0.01, 0.05, etc.

    train_loader, test_loader = get_cifar10(label_fraction=label_fraction)

    model = ResNet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    results = []

    for epoch in range(10):
        loss = train(model, train_loader, optimizer, criterion, device)
        finetune_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={finetune_acc:.4f}")

    results.append((label_fraction, finetune_acc))

    # -----Linear Probe-----
    # IMPORTANT: use full dataset for probing

    full_train_loader, _ = get_cifar10(label_fraction=1.0)

    train_features, train_labels = extract_features(model, full_train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    classifier = train_linear_probe(train_features, train_labels, num_classes=10)

    linear_acc = evaluate_linear(classifier, test_features.to(device), test_labels.to(device))

    print("Final Fine-tune Accuracy:", finetune_acc)
    print("Linear Probe Accuracy:", linear_acc)

if __name__ == "__main__":
    main()