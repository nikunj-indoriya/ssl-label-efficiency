import torch
import torch.nn as nn
import timm
import copy


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()

        # Online network
        self.online_encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)
        self.online_projector = MLP(self.online_encoder.num_features)
        self.online_predictor = MLP(256, hidden_dim=512, out_dim=256)

        # Target network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):

        # Online branch
        o1 = self.online_encoder(x1)
        z1 = self.online_projector(o1)
        p1 = self.online_predictor(z1)

        o2 = self.online_encoder(x2)
        z2 = self.online_projector(o2)
        p2 = self.online_predictor(z2)

        # Target branch (no gradients)
        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t_z1 = self.target_projector(t1)

            t2 = self.target_encoder(x2)
            t_z2 = self.target_projector(t2)

        return p1, p2, t_z1.detach(), t_z2.detach()

    def update_target(self, tau):
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data