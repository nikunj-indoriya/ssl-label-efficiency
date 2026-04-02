import torch
import torch.nn as nn
import torch.optim as optim

def train_linear_probe(features, labels, num_classes, epochs=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize features (VERY IMPORTANT)
    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

    features = features.to(device)
    labels = labels.to(device)

    classifier = nn.Linear(features.shape[1], num_classes).to(device)

    # Use SGD instead of Adam (more stable for linear probe)
    optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return classifier

def evaluate_linear(classifier, features, labels):

    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

    classifier.eval()

    with torch.no_grad():
        outputs = classifier(features)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    return acc