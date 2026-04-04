import torch
import torch.nn as nn


class ExpTransformer(nn.Module):
    def __init__(
        self,
        num_neighbors: int,
        hidden_dim:    int = 64,
        num_heads:     int = 4,
        num_layers:    int = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim

        self.weight_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.pos_embedding = nn.Embedding(num_neighbors, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B = weights.size(0)
        tokens = self.weight_proj(weights.unsqueeze(-1))
        positions = torch.arange(self.num_neighbors, device=weights.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)
        coord_ctx = self.coord_proj(coords)
        tokens = tokens + coord_ctx.unsqueeze(1)
        out = self.transformer(tokens)
        agg = out.mean(dim=1)
        combined = torch.cat([agg, coord_ctx], dim=-1)
        return self.head(combined).squeeze(-1)
