"""
methods/byol.py
---------------
BYOL (Bootstrap Your Own Latent) model definition.

BYOL (Grill et al., 2020) avoids the need for negative pairs by maintaining
two networks:

  Online network  : encoder → projector → predictor  (parameters trained via
                    gradient descent).
  Target network  : encoder → projector               (parameters updated via
                    exponential moving average, EMA, of the online network;
                    gradients do NOT flow through here).

The online network predicts the target network's projections for two augmented
views, and the objective is to minimise the (symmetric) BYOL loss between the
prediction and the stop-gradient target projection.

EMA update rule (target ← τ · target + (1-τ) · online):
  τ is annealed using a cosine schedule from an initial value toward 1.0,
  making the target network update more slowly as training progresses.

Reference: Grill et al., "Bootstrap Your Own Latent — A New Approach to
Self-Supervised Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import timm
import copy


class MLP(nn.Module):
    """
    2-layer MLP with BatchNorm used as the BYOL projector and predictor.

    Architecture: Linear(in_dim → hidden_dim) → BN → ReLU → Linear(hidden_dim → out_dim)

    BatchNorm is applied after the first linear layer (before ReLU) to
    stabilise training and match the architecture described in the paper.

    Parameters
    ----------
    in_dim     : int. Input feature dimensionality.
    hidden_dim : int. Width of the hidden layer (default 512).
    out_dim    : int. Output dimensionality (default 256).
    """

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
    """
    Full BYOL model with online and target networks.

    Online branch  (trained): encoder → projector → predictor
    Target branch  (EMA):     encoder → projector

    The target network is initialised as a deep copy of the online network
    and then frozen (requires_grad=False). Its parameters are updated
    exclusively via the EMA rule in `update_target`.

    Attributes
    ----------
    online_encoder   : ResNet-50 backbone for the online branch.
    online_projector : MLP projector for the online branch (ResNet → 256-d).
    online_predictor : MLP predictor applied on top of online projections.
    target_encoder   : Deep copy of online_encoder; EMA-updated, no grad.
    target_projector : Deep copy of online_projector; EMA-updated, no grad.
    """

    def __init__(self):
        super().__init__()

        # ---- Online network (full gradient flow) ----
        self.online_encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)
        # Projects 2048-d backbone features to 256-d embedding space.
        self.online_projector = MLP(self.online_encoder.num_features)
        # Predictor head: maps 256-d projections to 256-d predictions.
        # This asymmetry between online (with predictor) and target (without)
        # is what prevents representational collapse.
        self.online_predictor = MLP(256, hidden_dim=512, out_dim=256)

        # ---- Target network (no gradient, EMA-updated) ----
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network parameters so that gradients never flow
        # through the target branch — updates come only from the EMA rule.
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        """
        Compute online predictions and target projections for a positive pair.

        Parameters
        ----------
        x1 : Tensor (B, C, H, W). First augmented view of each image.
        x2 : Tensor (B, C, H, W). Second augmented view of each image.

        Returns
        -------
        p1     : Tensor (B, 256). Online predictor output for view 1.
        p2     : Tensor (B, 256). Online predictor output for view 2.
        t_z1   : Tensor (B, 256). Target projection for view 1 (detached).
        t_z2   : Tensor (B, 256). Target projection for view 2 (detached).

        Training objective (symmetric BYOL loss):
            L = byol_loss(p1, t_z2) + byol_loss(p2, t_z1)
        """

        # ---- Online branch: encoder → projector → predictor ----
        o1 = self.online_encoder(x1)
        z1 = self.online_projector(o1)
        p1 = self.online_predictor(z1)   # prediction for view 1

        o2 = self.online_encoder(x2)
        z2 = self.online_projector(o2)
        p2 = self.online_predictor(z2)   # prediction for view 2

        # ---- Target branch: encoder → projector (no gradients) ----
        # torch.no_grad() ensures no gradients accumulate in the target path;
        # target weights are updated solely via the EMA in `update_target`.
        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t_z1 = self.target_projector(t1)

            t2 = self.target_encoder(x2)
            t_z2 = self.target_projector(t2)

        # Detach target projections — they serve as fixed regression targets.
        return p1, p2, t_z1.detach(), t_z2.detach()

    def update_target(self, tau):
        """
        Apply the EMA update to the target network parameters.

        Update rule:  θ_target ← τ · θ_target + (1 - τ) · θ_online

        A higher τ (close to 1) makes the target network change more slowly,
        providing stable regression targets. τ is typically annealed from
        ~0.996 towards 1.0 over training using a cosine schedule.

        Parameters
        ----------
        tau : float. EMA momentum coefficient in [0, 1).
        """

        # Update target encoder parameters.
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

        # Update target projector parameters.
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
