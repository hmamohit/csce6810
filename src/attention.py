import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self.dropout_rate = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scaler = float(1.0/math.sqrt(self.d_head))

    def forward(self, ftr: torch.Tensor, key_val_states: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, ftr_length, model_dim = ftr.size()

        assert model_dim == self.d_model, f"Input feature dimension {model_dim} does not match model dimension {self.d_model}"

        if key_val_states is not None:
            assert key_val_states.size(
                -1) == self.d_model, f"Cross attention key/value dimension {key_val_states.size(-1)} does not match model dimension {self.d_model}"

        is_cross_attention = key_val_states is not None

        Q_state = self.q_proj(ftr)
        if is_cross_attention:
            kv_ftr_len = key_val_states.size(1)
            K_state = self.k_proj(key_val_states)
            V_state = self.v_proj(key_val_states)
        else:
            kv_ftr_len = ftr_length
            K_state = self.k_proj(ftr)
            V_state = self.v_proj(ftr)

        Q_state = Q_state.view(batch_size, ftr_length,
                               self.num_heads, self.d_head).transpose(1, 2)
        K_state = K_state.view(batch_size, kv_ftr_len,
                               self.num_heads, self.d_head).transpose(1, 2)
        V_state = V_state.view(batch_size, kv_ftr_len,
                               self.num_heads, self.d_head).transpose(1, 2)

        Q_state = Q_state * self.scaler
        self.attn_weights = torch.matmul(Q_state, K_state.transpose(-1, -2))
        if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
            raise TypeError(
                f"Attention mask must be a tensor, but got {type(attn_mask)}")

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            self.attn_weights = self.attn_weights + attn_mask
            # attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            # self.attn_weights = self.attn_weights.masked_fill(
            #     attn_mask,
            #     float('-inf')
            # )


        attn_score = F.softmax(self.attn_weights, dim=-1)
        attn_score = self.dropout(attn_score)
        attn_output = torch.matmul(attn_score, V_state)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(
            batch_size, ftr_length, self.num_heads*self.d_head)

        attn_output = self.out_proj(attn_output)
        assert attn_output.size() == (batch_size, ftr_length,
                                      self.d_model), f"Attention output shape {attn_output.size()} does not match expected shape {(batch_size, ftr_length, self.d_model)}"

        return attn_output
