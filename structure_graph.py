

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


EdgeWeight = Literal["rbf", "inv"]


@dataclass(frozen=True)
class StructureGraphConfig:
    chrom: str
    bin_size: int = 1_000_000
    include_xyz_as_node_features: bool = False
    include_bin_index_as_node_features: bool = True
    neighbor_mode: Literal["knn", "radius"] = "knn"
    k: int = 8
    radius: float = 15.0
    edge_weight: EdgeWeight = "rbf"
    sigma: float = 10.0  # used for RBF
    eps: float = 1e-6    # used for inverse distance
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test by coord


def _read_tsv(path: Path) -> Tuple[list[str], list[list[str]]]:
    with path.open("r") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    return header, rows


def load_bin_expression_tsv(
    bin_expression_tsv: str | Path,
    chrom: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    path = Path(bin_expression_tsv)
    header, rows = _read_tsv(path)
    col = {name: i for i, name in enumerate(header)}

    required = ["chr", "bin_index", "gene_count", "bin_expr_log2_mean"]
    missing = [c for c in required if c not in col]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    recs = []
    for r in rows:
        if r[col["chr"]] != chrom:
            continue
        recs.append(
            (
                int(r[col["bin_index"]]),
                float(r[col["gene_count"]]),
                float(r[col["bin_expr_log2_mean"]]),
            )
        )

    if not recs:
        raise ValueError(f"No rows found for {chrom} in {path}")

    recs.sort(key=lambda x: x[0])
    bin_index = np.array([x[0] for x in recs], dtype=np.int64)
    gene_count = np.array([x[1] for x in recs], dtype=np.float32)
    expr = np.array([x[2] for x in recs], dtype=np.float32)
    return bin_index, gene_count, expr


def parse_pdb_ca_coordinates(pdb_file: str | Path) -> np.ndarray:
    
    coords = []
    with Path(pdb_file).open("r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            # Example: ATOM 1 CA MET B1 40.030 -9.254 5.711 0.20 10.00
            # indices:            5      6      7
            if len(parts) < 8:
                continue
            if parts[2] != "CA":
                continue
            
            xy = parts[6].split('-') # sometimes the y and z coords would come out like "11.091-100.794" with no space in between 
            if len(xy) == 2 and xy[0] != '':
                parts[6] = xy[0]
                parts[7] = xy[1]
            
            x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
            print(x,y,z)
            coords.append((x, y, z))

    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_file}")
    return np.asarray(coords, dtype=np.float32)


def _pairwise_edges_knn(coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    # (n, n)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = nn_idx.reshape(-1)
    dist = np.sqrt(d2[np.arange(n)[:, None], nn_idx]).reshape(-1)
    return np.stack([src, dst], axis=0), dist


def _pairwise_edges_radius(coords: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    mask = d2 <= (radius * radius)
    src, dst = np.where(mask)
    dist = np.sqrt(d2[src, dst])
    return np.stack([src, dst], axis=0), dist


def _edge_weights(dist: np.ndarray, mode: EdgeWeight, sigma: float, eps: float) -> np.ndarray:
    if mode == "rbf":
        return np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    if mode == "inv":
        return (1.0 / (dist + eps)).astype(np.float32)
    raise ValueError(f"Unknown edge_weight mode: {mode}")


def _make_split_masks(n: int, split: Tuple[float, float, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tr, va, te = split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError(f"split must sum to 1.0; got {split}")
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    n_test = n - n_train - n_val

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == n
    return train_mask, val_mask, test_mask


def build_structure_graph(
    pdb_file: str | Path,
    bin_expression_tsv: str | Path,
    cfg: Optional[StructureGraphConfig] = None,
    ) -> Data:
    """Build a PyG `Data` graph for chr19 based on 3D neighbors only."""

    bin_index, gene_count, y = load_bin_expression_tsv(bin_expression_tsv, chrom=cfg.chrom)
    coords = parse_pdb_ca_coordinates(pdb_file)

    # Assumption: one CA atom per 1Mb bin and ordered by bin index.
    n = len(bin_index)
    if coords.shape[0] != n:
        print(f"Node count mismatch: {cfg.chrom} has {n} bins in expression TSV, but PDB has {coords.shape[0]} CA atoms.")
        return None
    
    print(f'Bins and atoms match for {cfg.chrom}')
    # Node features
    feats = [gene_count.reshape(-1, 1)]
    if cfg.include_bin_index_as_node_features:
        feats.append((bin_index.astype(np.float32).reshape(-1, 1)))
    if cfg.include_xyz_as_node_features:
        feats.append(coords.astype(np.float32))
    x = np.concatenate(feats, axis=1).astype(np.float32)

    # Edges
    if cfg.neighbor_mode == "knn":
        edge_index_np, dist = _pairwise_edges_knn(coords, k=cfg.k)
    elif cfg.neighbor_mode == "radius":
        edge_index_np, dist = _pairwise_edges_radius(coords, radius=cfg.radius)
    else:
        raise ValueError(f"Unknown neighbor_mode: {cfg.neighbor_mode}")

    # Make undirected by adding reverse edges
    src, dst = edge_index_np
    edge_index_np = np.concatenate([edge_index_np, np.stack([dst, src], axis=0)], axis=1)
    dist = np.concatenate([dist, dist], axis=0)

    w = _edge_weights(dist, cfg.edge_weight, sigma=cfg.sigma, eps=cfg.eps)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(w.reshape(-1, 1), dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
        num_nodes=n,
    )

    train_mask, val_mask, test_mask = _make_split_masks(n, cfg.split)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
