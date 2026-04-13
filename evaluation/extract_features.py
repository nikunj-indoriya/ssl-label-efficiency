import torch

def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # ---- Handle different model types ----
            if hasattr(model, "get_features"):
                # ResNet / SimCLR / BYOL (your existing pipeline)
                features = model.get_features(images)

            else:
                # ViT / MAE (timm models)
                features = model(images)

                # Some timm ViTs return tuple (rare case safety)
                if isinstance(features, tuple):
                    features = features[0]

            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)