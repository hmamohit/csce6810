import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        out = self.net(cls_token)
        out = F.softplus(out) + 1e-10
        return out
