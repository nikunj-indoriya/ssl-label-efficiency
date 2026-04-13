"""
evaluation/linear_probe.py
--------------------------
Linear probing utility for evaluating the quality of SSL representations.

Linear probing is the standard downstream evaluation protocol for self-
supervised methods:
  1. Freeze the pre-trained encoder — weights are never updated.
  2. Extract fixed feature vectors for the entire training and test sets.
  3. Train a single linear layer (logistic regression) on the training features.
  4. Report top-1 accuracy on the test features.

A high linear-probe accuracy indicates that the SSL encoder has learned
linearly separable class representations, even without seeing any labels
during pre-training.

Design choices
--------------
- Features are L2-normalised before training and evaluation, which is
  essential for stable convergence of the linear classifier.
- SGD with momentum is used instead of Adam, which is empirically more
  stable for linear probe training on normalised features.
- The entire feature set is trained in a single full-batch step per epoch
  (no mini-batch DataLoader), which is computationally efficient because the
  features are pre-extracted and fixed.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_linear_probe(features, labels, num_classes, epochs=50):
    """
    Train a linear classifier on top of pre-extracted frozen features.

    Parameters
    ----------
    features    : Tensor of shape (N, D). Pre-extracted feature vectors.
    labels      : Tensor of shape (N,).   Corresponding ground-truth labels.
    num_classes : int. Number of output classes (10 for CIFAR-10).
    epochs      : int. Number of full-batch gradient descent steps (default 50).

    Returns
    -------
    classifier : nn.Linear. Trained linear classifier (on the same device
                 as `features` after normalisation).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # L2-normalise features so that the linear classifier operates on the
    # unit hypersphere. This removes scale differences between dimensions and
    # makes cosine similarity the implicit metric — critical for good accuracy.
    # A small epsilon (1e-6) prevents division by zero for zero-norm vectors.
    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

    features = features.to(device)
    labels = labels.to(device)

    # Single linear layer: D → num_classes. No bias normalisation needed
    # because features are already unit-normalised.
    classifier = nn.Linear(features.shape[1], num_classes).to(device)

    # SGD with momentum is empirically more stable than Adam for linear probe
    # training. Weight decay provides mild L2 regularisation.
    optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Full-batch gradient step: all N samples are used at once.
        # This is efficient because features are pre-computed and fixed.
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return classifier


def evaluate_linear(classifier, features, labels):
    """
    Evaluate the linear classifier's top-1 accuracy on held-out features.

    Parameters
    ----------
    classifier : nn.Linear. Trained linear classifier.
    features   : Tensor of shape (M, D). Test feature vectors (on device).
    labels     : Tensor of shape (M,).   Ground-truth test labels (on device).

    Returns
    -------
    float. Top-1 accuracy in [0, 1] over all M test samples.
    """

    # Apply the same L2 normalisation used during training to ensure
    # consistent scale between train and test feature spaces.
    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

    classifier.eval()

    with torch.no_grad():
        outputs = classifier(features)

        # Predicted class = argmax of the linear layer's output logits.
        preds = outputs.argmax(dim=1)

        # Fraction of correctly classified test samples.
        acc = (preds == labels).float().mean().item()

    return acc
