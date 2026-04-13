import torch
import torch.nn as nn
import timm

class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z