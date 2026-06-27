"""Multimodal Hi-C transformer for TPM prediction."""

import torch
import torch.nn as nn

from encoder import TransformerEncoder
from models.cross_encoder import CrossAttentionBlock
from projection import HiCProjection


def uses_shared_ep_encoder(state_dict: dict) -> bool:
    return any(k.startswith("enc_ep.") for k in state_dict)


class MultimodalGeneExpression(nn.Module):
    def __init__(
        self,
        num_encoders: int = 4,
        d_model: int = 256,
        d_ff: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        shared_ep_encoder: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_ep_encoder = shared_ep_encoder
        self.proj = HiCProjection(d_model=d_model, dropout=dropout)
        self.modality_emb = nn.Embedding(4, d_model)

        self.enc_gene = TransformerEncoder(num_encoders, d_model, d_ff, num_heads, dropout, bias)
        if shared_ep_encoder:
            self.enc_ep = TransformerEncoder(num_encoders, d_model, d_ff, num_heads, dropout, bias)
        else:
            self.enc_pels = TransformerEncoder(num_encoders, d_model, d_ff, num_heads, dropout, bias)
            self.enc_dels = TransformerEncoder(num_encoders, d_model, d_ff, num_heads, dropout, bias)
        self.enc_pls = TransformerEncoder(num_encoders, d_model, d_ff, num_heads, dropout, bias)

        self.cross_ep_to_gene = CrossAttentionBlock(d_model, num_heads, dropout, bias)
        self.cross_pls_to_gene = CrossAttentionBlock(d_model, num_heads, dropout, bias)

        # +1 for has_annotated_pls flag (metadata only; PLS Hi-C always used)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _encode_modality(
        self,
        vectors: torch.Tensor,
        modality_id: int,
        encoder: TransformerEncoder,
    ) -> torch.Tensor:
        proj = self.proj(vectors)
        proj = proj + self.modality_emb.weight[modality_id].view(1, 1, -1)
        # HiCProjection fixes seq len to d_model; raw pad masks do not apply post-proj.
        return encoder(proj_ftr=proj, padding_mask=None)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def forward(
        self,
        gene_body: torch.Tensor,
        pls: torch.Tensor,
        pels: torch.Tensor,
        dels: torch.Tensor,
        mask_gene: torch.Tensor,
        mask_pls: torch.Tensor,
        mask_pels: torch.Tensor,
        mask_dels: torch.Tensor,
        has_pls: torch.Tensor,
        binary_gene: torch.Tensor | None = None,
        binary_pls: torch.Tensor | None = None,
        binary_pels: torch.Tensor | None = None,
        binary_dels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_gene = self._encode_modality(gene_body, 0, self.enc_gene)
        h_pls = self._encode_modality(pls, 1, self.enc_pls)
        if self.shared_ep_encoder:
            h_pels = self._encode_modality(pels, 2, self.enc_ep)
            h_dels = self._encode_modality(dels, 3, self.enc_ep)
        else:
            h_pels = self._encode_modality(pels, 2, self.enc_pels)
            h_dels = self._encode_modality(dels, 3, self.enc_dels)

        h_ep = torch.cat([h_pels, h_dels], dim=1)

        h_gene = self.cross_ep_to_gene(h_gene, h_ep, attn_mask=None)
        # has_pls = annotated PLS in cCRE BED only; Hi-C at TSS±200bp always flows through PLS branch
        h_gene = self.cross_pls_to_gene(h_gene, h_pls, attn_mask=None)

        pool_gene = self._pool(h_gene)
        pool_ep = self._pool(h_ep)
        pool_pls = self._pool(h_pls)

        pls_meta = has_pls.float().unsqueeze(-1)
        fused = torch.cat([pool_gene, pool_ep, pool_pls, pls_meta], dim=-1)
        return self.head(fused)
