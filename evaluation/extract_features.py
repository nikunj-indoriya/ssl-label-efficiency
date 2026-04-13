"""
evaluation/extract_features.py
-------------------------------
Feature extraction utility used across all linear-probe evaluations.

Runs a frozen model in inference mode over a DataLoader and collects the
backbone feature vectors and corresponding labels for the entire dataset.
The returned tensors are then passed to `train_linear_probe` for downstream
evaluation.

Two model families are supported transparently:
  - ResNet / SimCLR / BYOL models that expose a `get_features` method.
  - ViT / MAE models loaded via `timm` whose `forward` returns feature tensors
    directly (since they are created with `num_classes=0`).
"""

import torch


def extract_features(model, loader, device):
    """
    Extract backbone feature representations for all samples in `loader`.

    The model is run in `eval` mode with `torch.no_grad()` so that no
    gradients are accumulated and batch-normalisation statistics are frozen.

    Model dispatch logic
    --------------------
    - If `model` has a `get_features` attribute (ResNet50, SimCLR, BYOL):
      call `model.get_features(images)` to bypass the classifier head and
      return raw backbone features.
    - Otherwise (timm ViT / MAE): call `model(images)` directly. Models
      created with `num_classes=0` return pooled feature vectors, not logits.
      A safety check handles the rare case where `timm` returns a tuple.

    Parameters
    ----------
    model  : nn.Module. Encoder whose features are to be extracted.
    loader : DataLoader. Yields (images, labels) batches.
    device : torch.device. Device on which the model lives.

    Returns
    -------
    features : Tensor of shape (N, D). All feature vectors concatenated on CPU.
    labels   : Tensor of shape (N,).   Corresponding ground-truth labels (CPU).
    """
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # ---- Handle different model types ----
            if hasattr(model, "get_features"):
                # ResNet / SimCLR / BYOL (your existing pipeline):
                # `get_features` routes through the backbone only, skipping
                # the classifier head so we get raw 2048-d representations.
                features = model.get_features(images)

            else:
                # ViT / MAE (timm models):
                # Models created with `num_classes=0` return pooled features
                # from their forward pass rather than classification logits.
                features = model(images)

                # Some timm ViTs return a tuple (features, aux_output);
                # take the first element to ensure we always get a tensor.
                if isinstance(features, tuple):
                    features = features[0]

            # Accumulate features on CPU to avoid GPU memory exhaustion
            # when processing the full 50,000-sample training set.
            all_features.append(features.cpu())
            all_labels.append(labels)

    # Concatenate all batches into single tensors.
    return torch.cat(all_features), torch.cat(all_labels)
