import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from feed_fwn import FFN


class Encoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.attn = Attention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, bias=bias)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)

        self.dropout = nn.Dropout(p=dropout)

        self.attn_lnorm = nn.LayerNorm(d_model)
        self.ffn_lnorm = nn.LayerNorm(d_model)

    def forward(self, proj_ftr: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:

        attn_sublayer = self.attn(
            ftr=proj_ftr, key_val_states=None, attn_mask=padding_mask)
        attn_sublayer = self.dropout(attn_sublayer)
        attn_norm = self.attn_lnorm(proj_ftr + attn_sublayer)

        ffn_sublayer = self.ffn(ftr=attn_norm)
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_norm = self.ffn_lnorm(attn_norm + ffn_sublayer)

        return ffn_norm


class TransformerEncoder(nn.Module):
    def __init__(self, num_encoders: int, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()

        self.encoders = nn.ModuleList([Encoder(
            d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout, bias=bias) for _ in range(num_encoders)])

    def forward(self, proj_ftr: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        output = proj_ftr
        for encoder in self.encoders:
            output = encoder(proj_ftr=output, padding_mask=padding_mask)

        return output
