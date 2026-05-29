import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiCProjection(nn.Module):
    # def __init__(self, d_ftr_size: int, d_model: int, dropout: float = 0.1):
    #     super().__init__()
    #     self.d_model = d_model
    #     self.avg_pool = nn.AdaptiveAvgPool1d(output_size=d_ftr_size)
    #     self.projection = nn.Linear(1, self.d_model)
    #     self.scaling = float(math.sqrt(self.d_model))
    #     self.layernorm = nn.LayerNorm(self.d_model)
    #     self.dropout = nn.Dropout(p=dropout)

    # @staticmethod
    # def create_positional_encoding(ftr_length: int, d_model: int, batch_size: int = 32):
    #     position = torch.arange(ftr_length).unsqueeze(1).float()
    #     div_term = torch.exp(torch.arange(
    #         0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    #     pe = torch.zeros(ftr_length, d_model)
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)

    #     return pe

    # def forward(self, ftr: torch.Tensor) -> torch.Tensor:
    #     assert ftr.dtype == torch.float32, f"Input tensor must have dtype torch.float32 and got {ftr.dtype}"

    #     avg_pooled_ftr = self.avg_pool(ftr)

    #     batch_size, ftr_length = avg_pooled_ftr.size()
    #     ftr_unsqz = avg_pooled_ftr.unsqueeze(-1)

    #     ftr_embedding = self.projection(ftr_unsqz) * self.scaling
    #     positional_encoding = self.create_positional_encoding(
    #         ftr_length, self.d_model, batch_size).to(ftr.device)
    #     normalized_embedding = self.layernorm(
    #         ftr_embedding + positional_encoding)
    #     output = self.dropout(normalized_embedding)

    #     return output

    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=d_model,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(d_model)
        )

        self.scaling = float(math.sqrt(d_model))
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.output_projection = nn.Linear(d_model, d_model)

    @staticmethod
    def create_positional_encoding(ftr_length: int, d_model: int):
        position = torch.arange(ftr_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(ftr_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, ftr: torch.Tensor) -> torch.Tensor:
        assert ftr.dtype == torch.float32, f"Input must be float32, got {ftr.dtype}"

        x = ftr.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = x * self.scaling
        pe = self.create_positional_encoding(
            x.size(1), self.d_model).to(ftr.device)
        x = self.layernorm(x + pe)
        x = self.dropout(x)
        output = self.output_projection(x)

        return output
