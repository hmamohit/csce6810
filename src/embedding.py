import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, ftr_size: int = 256, patch_size: int = 8,
                 in_channels: int = 2, embed_dim: int = 256):
        super().__init__()
        assert ftr_size % patch_size == 0, "ftr_size must be divisible by patch_size"

        self.patch_size = patch_size
        self.grid_size = ftr_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
