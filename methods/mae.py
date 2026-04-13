import torch
import torch.nn as nn
import timm


class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio

        # ViT encoder
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.encoder.num_features

        # Decoder (lightweight)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def random_mask(self, x):
        B, N, D = x.shape

        num_mask = int(self.mask_ratio * N)

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        ids_keep = ids_shuffle[:, :-num_mask]

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        return x_masked, ids_shuffle

    def forward(self, x):

        # Patch embeddings
        x = self.encoder.patch_embed(x)

        # Add positional encoding
        cls_token = self.encoder.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.encoder.pos_embed

        # Remove cls token for masking
        x_patch = x[:, 1:, :]

        x_masked, ids_shuffle = self.random_mask(x_patch)

        # Add cls token back
        x_masked = torch.cat((x[:, :1, :], x_masked), dim=1)

        # Encoder forward
        for blk in self.encoder.blocks:
            x_masked = blk(x_masked)

        x_masked = self.encoder.norm(x_masked)

        # Decoder
        decoded = self.decoder(x_masked)

        return decoded