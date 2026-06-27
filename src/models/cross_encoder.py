"""Single cross-attention block."""

import torch
import torch.nn as nn

from attention import Attention


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.attn = Attention(d_model=d_model, num_heads=num_heads, dropout=dropout, bias=bias)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.attn(ftr=query, key_val_states=context, attn_mask=attn_mask)
        out = self.dropout(out)
        return self.norm(query + out)
