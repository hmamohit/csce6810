import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return self.dropout(x)
