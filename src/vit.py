import os
import sys
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder import TransformerEncoderBlock
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class VisionTransformer(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, depth: int,
                 num_heads: int, mlp_ratio: float,
                 dropout: float, attn_dropout: float):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(
            num_patches + 1, embed_dim, dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls_tokens, patch_tokens], dim=1)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
