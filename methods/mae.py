"""
methods/mae.py
--------------
Masked Autoencoder (MAE) model definition.

MAE (He et al., 2022) is a self-supervised method for Vision Transformers.
The training objective is to reconstruct randomly masked image patches from
the unmasked context patches alone.

Architecture used here
----------------------
Encoder : ViT-Base/16 (vit_base_patch16_224) — processes only the *visible*
          (unmasked) patch tokens. Removing masked tokens reduces the encoder's
          computational cost significantly.
Decoder : A lightweight 2-layer MLP that maps each encoder output token back
          to the original feature space. A full pixel-level decoder (as in the
          original paper) is replaced by this proxy decoder for simplicity.

Masking strategy
----------------
A random subset (mask_ratio × N) of the N patch tokens is discarded before
the encoder. The encoder therefore never sees the masked tokens, forcing it to
build rich contextual representations from partial observations.

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners",
CVPR 2022.
"""

import torch
import torch.nn as nn
import timm


class MAE(nn.Module):
    """
    Lightweight MAE with a ViT-Base/16 encoder and an MLP decoder proxy.

    Parameters
    ----------
    mask_ratio : float. Fraction of patch tokens to mask during pre-training
                 (default 0.75, following the original MAE paper).

    Attributes
    ----------
    mask_ratio : float. Stored for use in `random_mask`.
    encoder    : ViT-Base/16 backbone (no classification head).
    decoder    : 2-layer MLP that reconstructs features for each token.
    """

    def __init__(self, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio

        # ViT encoder: processes 224×224 images divided into 16×16 patches.
        # `num_classes=0` removes the classification head, returning token
        # sequences instead of a single class logit.
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.encoder.num_features  # 768 for ViT-Base

        # Decoder: a lightweight MLP used as a reconstruction proxy.
        # Maps each encoded token back to the embedding dimension.
        # A production MAE would use a separate smaller Transformer decoder
        # and target raw pixel values; this MLP is a simplified stand-in.
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def random_mask(self, x):
        """
        Randomly mask a fraction of patch tokens by discarding them.

        Token removal (as opposed to masking with a [MASK] token) is the key
        efficiency gain in MAE: the encoder operates on a much smaller sequence.

        Parameters
        ----------
        x : Tensor of shape (B, N, D). Patch token sequence (no CLS token).

        Returns
        -------
        x_masked   : Tensor of shape (B, N - num_mask, D). Visible token subset.
        ids_shuffle: Tensor of shape (B, N). Permutation indices (kept for
                     potential downstream use, e.g. reconstruction ordering).
        """
        B, N, D = x.shape

        # Number of tokens to discard.
        num_mask = int(self.mask_ratio * N)

        # Generate random scores per token and sort to get a random permutation.
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # The first (N - num_mask) indices after shuffling are the kept tokens.
        ids_keep = ids_shuffle[:, :-num_mask]

        # Gather only the visible tokens using the kept indices.
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        return x_masked, ids_shuffle

    def forward(self, x):
        """
        MAE forward pass: embed patches → mask → encode → decode.

        Parameters
        ----------
        x : Tensor of shape (B, 3, 224, 224). Input image batch.

        Returns
        -------
        decoded : Tensor of shape (B, 1 + N_visible, embed_dim).
                  Decoder outputs for the CLS token and all visible patch
                  tokens. Used to compute the reconstruction proxy loss.
        """

        # Step 1: Convert image pixels to patch embeddings.
        # Shape: (B, N, embed_dim) where N = (224/16)^2 = 196 patches.
        x = self.encoder.patch_embed(x)

        # Step 2: Prepend the CLS token and add learnable positional embeddings.
        # CLS token carries global image context throughout the encoder.
        cls_token = self.encoder.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)    # (B, N+1, D)
        x = x + self.encoder.pos_embed           # add positional information

        # Step 3: Separate the CLS token before masking (masking is applied
        # only to patch tokens, not to the CLS token).
        x_patch = x[:, 1:, :]                   # (B, N, D) — patches only

        # Step 4: Randomly discard mask_ratio fraction of patch tokens.
        x_masked, ids_shuffle = self.random_mask(x_patch)

        # Step 5: Restore the CLS token at position 0 before the encoder.
        x_masked = torch.cat((x[:, :1, :], x_masked), dim=1)

        # Step 6: Pass the visible token sequence through the ViT encoder blocks.
        for blk in self.encoder.blocks:
            x_masked = blk(x_masked)

        # Apply final layer normalisation.
        x_masked = self.encoder.norm(x_masked)

        # Step 7: Decode each token back to the embedding space.
        # The decoder serves as a reconstruction proxy; in a full MAE it would
        # predict raw pixel values of the masked patches.
        decoded = self.decoder(x_masked)

        return decoded
