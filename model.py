import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import warnings
warnings.filterwarnings("ignore")


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                    beta=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = self.dropout(F.gelu(x))

        expr_pred = self.reg_head(x).squeeze(-1)
        class_logits = self.cls_head(x)

        return expr_pred, class_logits
