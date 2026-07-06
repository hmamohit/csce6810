import os
import sys
import torch
import torch.nn as nn
from embedding import PatchEmbedding
from regression_head import RegressionHead
from vit import VisionTransformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class VisionModel(nn.Module):
    def __init__(self, ftr_size: int = 200, patch_size: int = 8,
                 embed_dim: int = 256, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            ftr_size=ftr_size, patch_size=patch_size,
            in_channels=2, embed_dim=embed_dim,
        )
        self.vit = VisionTransformer(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout,
        )
        self.head = RegressionHead(
            embed_dim=embed_dim, hidden_dim=embed_dim // 2, dropout=dropout,
        )

    @staticmethod
    def preprocess(feature: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        feature = feature.float()
        attention = attention.float().clamp(min=1.0)

        feature_log = torch.log1p(feature)
        attn_log = torch.log(attention)

        weighted_feature = feature_log * attention
        x = torch.cat([weighted_feature, attn_log], dim=1)
        return x

    def forward(self, feature: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(feature, attention)
        patch_tokens = self.patch_embed(x)
        encoded = self.vit(patch_tokens)
        cls_out = encoded[:, 0]
        pred = self.head(cls_out)
        return pred
