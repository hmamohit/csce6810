import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, ftr: torch.Tensor) -> torch.Tensor:
        batch_size, ftr_length, d_input = ftr.size()
        assert self.d_model == d_input, f"Input feature dimension {d_input} does not match model dimension {self.d_model}"

        f1 = F.relu(self.fc1(ftr))
        f2 = self.fc2(f1)
        return f2
