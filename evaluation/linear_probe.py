import torch
import torch.nn as nn
import torch.optim as optim

def train_linear_probe(features, labels, num_classes, epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = features.to(device)
    labels = labels.to(device)

    classifier = nn.Linear(features.shape[1], num_classes).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return classifier

def evaluate_linear(classifier, features, labels):
    classifier.eval()

    with torch.no_grad():
        outputs = classifier(features)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    return acc