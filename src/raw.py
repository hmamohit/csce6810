"""
Vision Transformer for continuous regression, driven by a feature map + an
attention map, both 256x256, processed as 8x8 patches.

Shapes (as specified):
    feature:    (B, 1, 256, 256)  int,   value >= 0
    attention:  (B, 1, 256, 256)  float, value >= 1
    target:     (B, 1)            float, value >  0

Design notes
------------
- 256x256 split into 8x8 patches -> 32x32 = 1024 patches per image.
- `attention` isn't fed as a separate branch/mask trick; instead it is used
  in two simple, effective ways:
    1. It rescales the (log-compressed) feature map pixel-wise before
       patchifying, so patches in "important" regions get amplified.
    2. Its own log value is kept as a second channel, so the model also
       sees *how much* attention was applied at every pixel, not just the
       attention-weighted feature.
- `feature` is int and unbounded, so we log1p-compress it before use.
- The regression head outputs `softplus(x) + eps`, which is strictly
  positive, matching the target's `> 0` constraint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Positional Encoding
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Learnable positional embedding added to (CLS + patch) tokens.

    A learnable table is simple, effective, and lets the model learn
    whatever spatial prior it needs directly from data (standard choice
    in ViT-style models).
    """

    def __init__(self, num_tokens: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        # num_tokens = num_patches + 1 (for the CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_tokens, D)
        x = x + self.pos_embed
        return self.dropout(x)


# --------------------------------------------------------------------------
# 2. Attention mechanism (Multi-Head Self-Attention)
# --------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                              # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block: MHSA + MLP, each with a residual connection."""

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# 3. Patch Embedding (8x8 patches, feature + attention combined)
# --------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    Splits a (C, 256, 256) tensor into non-overlapping 8x8 patches and
    linearly projects each patch to `embed_dim`. Implemented as a single
    strided convolution, which is mathematically identical to
    "flatten each patch -> linear layer" but faster.
    """

    def __init__(self, img_size: int = 256, patch_size: int = 8,
                 in_channels: int = 2, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.patch_size = patch_size
        self.grid_size = img_size // patch_size            # 32
        self.num_patches = self.grid_size * self.grid_size  # 1024

        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 256, 256)
        x = self.proj(x)                 # (B, D, 32, 32)
        x = x.flatten(2).transpose(1, 2)  # (B, 1024, D)
        return x


# --------------------------------------------------------------------------
# 4. Vision Transformer backbone
# --------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """
    Prepends a CLS token to the patch tokens, adds positional encoding,
    and runs everything through a stack of Transformer encoder blocks.
    """

    def __init__(self, num_patches: int, embed_dim: int = 256, depth: int = 6,
                 num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(num_patches + 1, embed_dim, dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, N, D)
        B = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)      # (B, 1, D)

        x = torch.cat([cls_tokens, patch_tokens], dim=1)   # (B, N+1, D)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x  # (B, N+1, D)


# --------------------------------------------------------------------------
# 5. Regression head (final projection -> positive continuous value)
# --------------------------------------------------------------------------
class RegressionHead(nn.Module):
    """
    Maps the CLS token embedding to a single strictly-positive continuous
    value, matching target constraint (value > 0).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        # cls_token: (B, D)
        out = self.net(cls_token)          # (B, 1), unbounded
        out = F.softplus(out) + 1e-6       # (B, 1), strictly > 0
        return out


# --------------------------------------------------------------------------
# 6. Full Vision Model (feature + attention -> patches -> ViT -> regression)
# --------------------------------------------------------------------------
class VisionModel(nn.Module):
    """
    End-to-end model.

    Pipeline:
        feature (B,1,256,256) int>=0 ---\
                                          >-- preprocess --> (B,2,256,256)
        attention (B,1,256,256) float>=1-/                        |
                                                                    v
                                                      PatchEmbedding (8x8)
                                                                    |
                                                                    v
                                                          VisionTransformer
                                                                    |
                                                             CLS token (B,D)
                                                                    |
                                                                    v
                                                            RegressionHead
                                                                    |
                                                                    v
                                                              pred (B,1) > 0
    """

    def __init__(self, img_size: int = 256, patch_size: int = 8,
                 embed_dim: int = 256, depth: int = 6, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
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
        """
        feature:   (B, 1, 256, 256) int,   >= 0
        attention: (B, 1, 256, 256) float, >= 1
        returns:   (B, 2, 256, 256) float, ready for patch embedding
        """
        feature = feature.float()
        attention = attention.float().clamp(min=1.0)

        feature_log = torch.log1p(feature)     # compress unbounded int range
        attn_log = torch.log(attention)        # >= 0 (log of values >= 1)

        # channel 1: feature amplified where attention is high
        weighted_feature = feature_log * attention
        # channel 2: raw (log) attention strength itself
        x = torch.cat([weighted_feature, attn_log], dim=1)  # (B, 2, 256, 256)
        return x

    def forward(self, feature: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(feature, attention)     # (B, 2, 256, 256)
        patch_tokens = self.patch_embed(x)          # (B, 1024, D)
        encoded = self.vit(patch_tokens)            # (B, 1025, D)
        cls_out = encoded[:, 0]                     # (B, D)
        pred = self.head(cls_out)                   # (B, 1)
        return pred


# --------------------------------------------------------------------------
# Quick sanity check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 4
    feature = torch.randint(0, 256, (B, 1, 256, 256)).float()   # int-valued, >= 0
    attention = (torch.rand(B, 1, 256, 256) * 3 + 1)             # >= 1
    target = torch.rand(B, 1) * 10 + 0.1                         # > 0

    model = VisionModel(
        img_size=256, patch_size=8, embed_dim=256,
        depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1,
    )

    pred = model(feature, attention)
    print("pred shape:", pred.shape)     # (4, 1)
    print("pred values:", pred.squeeze(-1))
    assert (pred > 0).all(), "regression head must output strictly positive values"

    loss = F.mse_loss(pred, target)
    loss.backward()
    print("loss:", loss.item())
    print("num params:", sum(p.numel() for p in model.parameters()))