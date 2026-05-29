import torch
import torch.nn as nn
from projection import HiCProjection
from encoder import TransformerEncoder


class GeneExpression(nn.Module):
    def __init__(
        self,
        num_encoders: int,
        d_model: int,
        d_ff: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        tda_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.tda_dim = int(tda_dim)

        self.input_proj = HiCProjection(d_model, dropout)
        self.encoder = TransformerEncoder(
            num_encoders=num_encoders,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        if self.tda_dim > 0:
            self.tda_proj = nn.Sequential(
                nn.Linear(self.tda_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
        else:
            self.tda_proj = None

        head_in = d_model * d_model + (d_model if self.tda_dim > 0 else 0)
        self.output = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        ftr: torch.Tensor,
        attn_mask: torch.Tensor = None,
        tda: torch.Tensor = None,
    ) -> torch.Tensor:
        proj_ftr = self.input_proj(ftr=ftr)
        self_attn_output = self.encoder(
            proj_ftr=proj_ftr, padding_mask=attn_mask
        )
        self_attn_output = self_attn_output.view(self_attn_output.size(0), -1)

        if self.tda_proj is not None:
            if tda is None:
                raise ValueError("Model expects TDA features but tda=None.")
            tda_emb = self.tda_proj(tda)
            self_attn_output = torch.cat([self_attn_output, tda_emb], dim=-1)

        output = self.output(self_attn_output)
        return output
