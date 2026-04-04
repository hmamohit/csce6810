"""
Node Transformer Regressor v4  —  Real Data, Stacked Datasets
==============================================================

Key changes from v3 based on actual data analysis:

  1. REAL DATA LOADER         — reads your 4 actual files (weights, coords, ids, targets)
  2. NO weight rescaling      — weights already in [0,1], skip RobustScaler
  3. NON-SEQUENTIAL node IDs  — IDs like 1,3,4,10,11... with gaps are mapped to a
                                compact embedding index via a lookup table
  4. DATASET-AWARE IDs        — same node ID can appear in multiple stacked datasets,
                                so we track (dataset_index, node_id) for uniqueness
  5. SPARSE target handling   — focal-style weighted loss that extra-penalises
                                the rare non-zero targets
  6. COORDS scaled only       — StandardScaler on raw xyz (handles negatives)
  7. MEMORY-EFFICIENT loading — streams the weight file line by line (no full matrix load)

File format expected:
    weights.txt  : N rows, M space-separated floats per row (M varies, already [0,1])
    coords.txt   : N rows, 3 space-separated floats (raw, can be negative)
    ids.txt      : N rows, 1 integer per row (non-sequential, can repeat across datasets)
    targets.txt  : N rows, 1 float per row (raw, many zeros)

Usage:
    pip install torch numpy matplotlib scikit-learn
    python node_transformer_v4.py --weights weights.txt --coords coords.txt
                                  --ids ids.txt --targets targets.txt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import sys
from datetime import datetime
from tqdm.auto import tqdm

# ─────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # ── Neighbor selection ───────────────────────────────────────────────
    "strategy":          "topk_spatial",   # "topk_spatial" | "topk"
    "topk_k":             128,             # keep K neighbours per node
                                           # (tune based on avg non-zero count)

    # ── Model ────────────────────────────────────────────────────────────
    "hidden_dim":         128,
    "num_heads":            4,
    "num_layers":           4,
    "dropout":            0.1,

    # ── Training ─────────────────────────────────────────────────────────
    "batch_size":          512,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "num_epochs":         200,
    "patience":            25,
    "train_ratio":        0.70,
    "val_ratio":          0.15,
    "train_nonzero_oversample": 6.0,

    # ── Loss ─────────────────────────────────────────────────────────────
    # Focal-style weighting: non-zero targets get higher loss weight
    # alpha > 1 → stronger emphasis on non-zero samples
    "loss_alpha":         2.0,
    "calib_min_nz_recall": 0.80,
    "calib_nz_weight":    3.0,
    "use_tqdm":           True,
    "log_dir":            "logs",

    "checkpoint_path":  "best_model_v4.pt",
}


class TeeStream:
    """Mirror stdout/stderr to both terminal and file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def setup_run_logging(log_dir: str, prefix: str = "node_transformer_v4"):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{stamp}.log")

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_stdout, log_file)
    sys.stderr = TeeStream(old_stderr, log_file)
    print(f"[logging] Console output is being written to: {log_path}")
    return log_path, log_file, old_stdout, old_stderr


def close_run_logging(log_file, old_stdout, old_stderr):
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()


def print_section(title: str):
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def split_indices(indices, n_train, n_val, rng):
    """Shuffle indices and split into train/val/test chunks."""
    idx = np.array(indices, dtype=np.int64)
    rng.shuffle(idx)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return tr, va, te


# ─────────────────────────────────────────────
# 1.  Real Data Loader
# ─────────────────────────────────────────────
def detect_datasets(node_ids_raw: np.ndarray):
    """
    Detect dataset boundaries in the stacked ID file.

    Strategy: a new dataset starts whenever the ID sequence resets
    (i.e. current ID <= previous ID, indicating a new dataset began).

    Returns:
        dataset_indices : np.ndarray [N]  int  — which dataset each row belongs to
        boundaries      : list of (start, end) row pairs per dataset
    """
    dataset_indices = np.zeros(len(node_ids_raw), dtype=np.int64)
    boundaries = []
    ds_idx = 0
    start  = 0

    for i in range(1, len(node_ids_raw)):
        if node_ids_raw[i] <= node_ids_raw[i - 1]:
            boundaries.append((start, i))
            ds_idx += 1
            start   = i
        dataset_indices[i] = ds_idx

    boundaries.append((start, len(node_ids_raw)))
    print(f"  Detected {ds_idx + 1} dataset(s) in stacked file")
    for d, (s, e) in enumerate(boundaries):
        print(f"    Dataset {d}: rows {s}–{e-1}  ({e-s} nodes)")
    return dataset_indices, boundaries


def build_id_lookup(node_ids_raw: np.ndarray, dataset_indices: np.ndarray):
    """
    Map (dataset_idx, node_id) → compact embedding index [1, 2, 3, ...].
    Index 0 is reserved for padding.

    Also builds the reverse: for each row, what is its embedding index?
    """
    lookup  = {}   # (ds_idx, node_id) → embed_idx
    counter = 1    # start from 1 (0 = padding)

    row_embed_ids = np.zeros(len(node_ids_raw), dtype=np.int64)

    for row, (ds, nid) in enumerate(zip(dataset_indices, node_ids_raw)):
        key = (int(ds), int(nid))
        if key not in lookup:
            lookup[key] = counter
            counter    += 1
        row_embed_ids[row] = lookup[key]

    print(f"  Unique (dataset, node_id) pairs: {counter - 1}")
    return lookup, row_embed_ids, counter  # counter = vocab size


def load_real_data(weights_path, coords_path, ids_path, targets_path, k, use_tqdm=True):
    """
    Load the four real data files.

    weights : N rows, variable M floats  (already [0,1])
    coords  : N rows, 3 floats           (raw, negative ok)
    ids     : N rows, 1 int              (non-sequential, repeats across datasets)
    targets : N rows, 1 float            (raw, many zeros)

    For each node i, its neighbours are all other nodes j in the SAME dataset
    where weight[i][j] > 0. Because within one dataset n=m (square matrix),
    column index j directly gives the neighbour's position in that dataset,
    and we can look up its coordinates and embed ID from there.
    """
    print("  Reading coords  ...", end=" ", flush=True)
    coords = np.loadtxt(coords_path, dtype=np.float32)          # [N, 3]
    print(f"shape={coords.shape}")

    print("  Reading ids     ...", end=" ", flush=True)
    node_ids_raw = np.loadtxt(ids_path, dtype=np.int64).ravel() # [N]
    print(f"shape={node_ids_raw.shape}  range=[{node_ids_raw.min()},{node_ids_raw.max()}]")

    print("  Reading targets ...", end=" ", flush=True)
    targets = np.loadtxt(targets_path, dtype=np.float32).ravel()# [N]
    print(f"shape={targets.shape}  non-zero={(targets>0).sum()}  "
          f"zero={(targets==0).sum()}  "
          f"max={targets.max():.4f}")

    assert len(coords) == len(node_ids_raw) == len(targets), \
        "coords, ids, targets must all have same number of rows"

    N = len(targets)

    # Detect dataset boundaries
    dataset_indices, boundaries = detect_datasets(node_ids_raw)

    # Build compact node ID embedding table
    id_lookup, row_embed_ids, vocab_size = build_id_lookup(node_ids_raw, dataset_indices)

    # Build dataset-local position maps:
    # For dataset d spanning rows [start, end),
    # local_pos[row] = local position of that row within its dataset (0-based)
    local_positions = np.zeros(N, dtype=np.int64)
    for ds_idx, (start, end) in enumerate(boundaries):
        for local_pos, row in enumerate(range(start, end)):
            local_positions[row] = local_pos

    # Read weights line by line (memory-efficient for large files)
    print("  Reading weights ...", flush=True)
    raw_weights         = []   # [N] arrays of variable length (non-zero weights)
    raw_neighbor_coords = []   # [N] arrays [L, 3]
    raw_neighbor_ids    = []   # [N] arrays [L] embed_idx of neighbours

    with open(weights_path, "r") as f:
        iterator = tqdm(
            enumerate(f),
            total=N,
            desc="  Weight rows",
            unit="row",
            disable=not use_tqdm,
            dynamic_ncols=True,
        )
        for row_idx, line in iterator:

            vals = np.array(line.strip().split(), dtype=np.float32)

            # Find which dataset this row belongs to
            ds_idx       = int(dataset_indices[row_idx])
            ds_start     = boundaries[ds_idx][0]
            ds_end       = boundaries[ds_idx][1]
            ds_size      = ds_end - ds_start

            # Sanity: weight row length should match dataset size
            # Truncate or pad if needed (shouldn't happen with well-formed data)
            if len(vals) > ds_size:
                vals = vals[:ds_size]
            elif len(vals) < ds_size:
                vals = np.pad(vals, (0, ds_size - len(vals)))

            # Column j in this row = local position j within the dataset
            # → global row index = ds_start + j
            nonzero_local = np.where(vals > 1e-9)[0]          # local positions with weight>0
            nonzero_vals  = vals[nonzero_local]

            # Map local positions → global row indices → coords & embed IDs
            global_indices     = ds_start + nonzero_local
            neighbour_coords_i = coords[global_indices]        # [L, 3]
            neighbour_eids_i   = row_embed_ids[global_indices] # [L]

            raw_weights.append(nonzero_vals)
            raw_neighbor_coords.append(neighbour_coords_i)
            raw_neighbor_ids.append(neighbour_eids_i)

    print(f"\n  Weight loading complete.")

    nonzero_counts = np.array([len(w) for w in raw_weights])
    print(f"  Non-zero neighbours per node: "
          f"min={nonzero_counts.min()}  max={nonzero_counts.max()}  "
          f"mean={nonzero_counts.mean():.1f}  "
          f"p90={np.percentile(nonzero_counts, 90):.0f}")
    print(f"  Recommend topk_k <= {int(np.percentile(nonzero_counts, 90))}")

    return (raw_weights, raw_neighbor_coords, raw_neighbor_ids,
            coords, targets, row_embed_ids, vocab_size, dataset_indices)


# ─────────────────────────────────────────────
# 2.  Preprocessing
# ─────────────────────────────────────────────
def transform_targets(targets):
    """log1p compresses large values, handles zeros: log1p(0)=0."""
    return np.log1p(targets).astype(np.float32)

def inverse_transform_targets(t):
    if isinstance(t, torch.Tensor):
        return torch.expm1(t)
    return np.expm1(t)


# ── Top-K with spatial scoring ────────────────────────────────────────────
def apply_topk_spatial(raw_weights, raw_neighbor_coords, raw_neighbor_ids,
                       node_coords, k, use_tqdm=True, verbose=True):
    """
    Select up to K neighbours ranked by:  weight × (1 / euclidean_distance)

    Since weights are already [0,1], no additional scaling needed.
    Zero-weight positions are masked so Transformer ignores them.

    Returns:
        weights_out  : [N, k]  float32
        nids_out     : [N, k]  int64   embed IDs of selected neighbours
        mask_out     : [N, k]  bool    True = ignore (zero / padding)
    """
    N           = len(raw_weights)
    weights_out = np.zeros((N, k), dtype=np.float32)
    nids_out    = np.zeros((N, k), dtype=np.int64)
    mask_out    = np.ones((N, k),  dtype=bool)   # True = masked (ignored)

    row_iter = tqdm(
        range(N),
        desc=f"  Top-K spatial (k={k})",
        unit="row",
        disable=not use_tqdm,
        dynamic_ncols=True,
    )
    for i in row_iter:

        w    = raw_weights[i]           # [L]  already [0,1]
        nc   = raw_neighbor_coords[i]  # [L, 3]
        nids = raw_neighbor_ids[i]     # [L]
        own  = node_coords[i]          # [3]

        if len(w) == 0:
            continue   # isolated node — all masked

        dists  = np.linalg.norm(nc - own[np.newaxis, :], axis=1) + 1e-8
        scores = w / dists             # weight × proximity

        L = len(w)
        if L >= k:
            idx   = np.argpartition(scores, -k)[-k:]
            top_w = w[idx];   top_nid = nids[idx]
            order = np.argsort(top_w)[::-1]
            weights_out[i] = top_w[order]
            nids_out[i]    = top_nid[order]
        else:
            order = np.argsort(w)[::-1]
            weights_out[i, :L] = w[order]
            nids_out[i,    :L] = nids[order]

        mask_out[i] = (weights_out[i] < 1e-9)   # mask zero slots

    real_pct = (~mask_out).mean() * 100
    if verbose:
        print(f"  [topk_spatial k={k}]  real={real_pct:.1f}%  masked={100-real_pct:.1f}%")
    return weights_out, nids_out, mask_out


def apply_topk(raw_weights, raw_neighbor_ids, k, use_tqdm=True, verbose=True):
    """Top-K by weight magnitude only (fallback)."""
    N           = len(raw_weights)
    weights_out = np.zeros((N, k), dtype=np.float32)
    nids_out    = np.zeros((N, k), dtype=np.int64)
    mask_out    = np.ones((N, k),  dtype=bool)

    row_iter = tqdm(
        range(N),
        desc=f"  Top-K weight-only (k={k})",
        unit="row",
        disable=not use_tqdm,
        dynamic_ncols=True,
    )
    for i in row_iter:
        w    = raw_weights[i]
        nids = raw_neighbor_ids[i]
        L    = len(w)
        if L == 0:
            continue
        if L >= k:
            idx   = np.argpartition(w, -k)[-k:]
            top_w = w[idx];   top_nid = nids[idx]
            order = np.argsort(top_w)[::-1]
            weights_out[i] = top_w[order]
            nids_out[i]    = top_nid[order]
        else:
            order = np.argsort(w)[::-1]
            weights_out[i, :L] = w[order]
            nids_out[i,    :L] = nids[order]
        mask_out[i] = (weights_out[i] < 1e-9)

    real_pct = (~mask_out).mean() * 100
    if verbose:
        print(f"  [topk k={k}]  real={real_pct:.1f}%  masked={100-real_pct:.1f}%")
    return weights_out, nids_out, mask_out


# ─────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────
class NodeDataset(Dataset):
    """
    Each item:
        weights      : [k]     float32  — top-K neighbour weights (already [0,1])
        neighbor_ids : [k]     int64    — embed IDs of selected neighbours
        own_id       : scalar  int64    — this node's embed ID
        coords       : [3]     float32  — scaled xyz
        target_log   : scalar  float32  — log1p(true value)
        mask         : [k]     bool     — True = ignore
    """
    def __init__(self, weights, neighbor_ids, own_ids, coords, targets_log, mask):
        self.weights      = torch.from_numpy(weights)
        self.neighbor_ids = torch.from_numpy(neighbor_ids)
        self.own_ids      = torch.from_numpy(own_ids)
        self.coords       = torch.from_numpy(coords)
        self.targets_log  = torch.from_numpy(targets_log)
        self.mask         = torch.from_numpy(mask)

    def __len__(self):
        return len(self.targets_log)

    def __getitem__(self, idx):
        return (self.weights[idx], self.neighbor_ids[idx], self.own_ids[idx],
                self.coords[idx], self.targets_log[idx], self.mask[idx])


def build_dataloaders(weights, neighbor_ids, own_ids, coords,
                      targets_log, mask, cfg):
    dataset = NodeDataset(weights, neighbor_ids, own_ids, coords, targets_log, mask)
    n       = len(dataset)
    nonzero_idx = np.where(targets_log > 0)[0]
    zero_idx    = np.where(targets_log == 0)[0]
    print(f"  Non-zero targets: {len(nonzero_idx)}  Zero targets: {len(zero_idx)}")

    rng = np.random.default_rng(SEED)
    tr_r, va_r = cfg["train_ratio"], cfg["val_ratio"]

    nz_tr = int(len(nonzero_idx) * tr_r)
    nz_va = int(len(nonzero_idx) * va_r)
    z_tr  = int(len(zero_idx) * tr_r)
    z_va  = int(len(zero_idx) * va_r)

    nz_train, nz_val, nz_test = split_indices(nonzero_idx, nz_tr, nz_va, rng)
    z_train,  z_val,  z_test  = split_indices(zero_idx, z_tr, z_va, rng)

    train_idx = np.concatenate([nz_train, z_train])
    val_idx   = np.concatenate([nz_val, z_val])
    test_idx  = np.concatenate([nz_test, z_test])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())

    print(f"  Splits  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"  Train non-zero fraction: {(targets_log[train_idx] > 0).mean()*100:.2f}%")

    train_targets = targets_log[train_idx]
    sample_w = np.where(train_targets > 0, cfg["train_nonzero_oversample"], 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(train_ds),
        replacement=True,
    )

    kw = dict(batch_size=cfg["batch_size"], num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 4.  Model
# ─────────────────────────────────────────────
class NodeTransformerV4(nn.Module):
    """
    Input streams:
      • weight_proj(w_i)              — connection strength token  [B, k, H]
      • neighbor_id_embed(nid_j)      — where is neighbour j in sequence?
      • coord_proj(x,y,z)             — own 3D physical position   [B, H]
      • node_id_embed(own_id)         — own sequence position       [B, H]

    Weights are already [0,1] so no internal rescaling applied.
    Zero-weight positions are fully masked in attention.

    Regression head uses: neighbour aggregate + own coord + own id
    """

    def __init__(self, seq_len, vocab_size, hidden_dim=128,
                 num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim

        # Weight encoder (already [0,1] — just project to hidden)
        self.weight_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3D coordinate encoder (StandardScaler applied upstream)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Node ID embedding: vocab_size = unique (dataset, node_id) pairs + 1 padding
        self.node_id_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head: aggregate + coord context + id context → scalar
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, weights, neighbor_ids, own_ids, coords, mask):
        """
        Args:
            weights      : [B, k]   float32  already [0,1]
            neighbor_ids : [B, k]   int64
            own_ids      : [B]      int64
            coords       : [B, 3]   float32  StandardScaled
            mask         : [B, k]   bool     True = ignore
        Returns:
            pred : [B]  log-space
        """
        # Own context
        coord_ctx = self.coord_proj(coords)            # [B, H]
        id_ctx    = self.node_id_embed(own_ids)        # [B, H]

        # Neighbour tokens
        tokens  = self.weight_proj(weights.unsqueeze(-1))          # [B, k, H]
        tokens  = tokens + self.node_id_embed(neighbor_ids)        # + neighbour ID
        tokens  = tokens + (coord_ctx + id_ctx).unsqueeze(1)       # + own context

        # Guard fully-masked rows (isolated nodes)
        all_masked = mask.all(dim=1, keepdim=True)
        mask       = mask & ~all_masked

        # Transformer
        out = self.transformer(tokens, src_key_padding_mask=mask)  # [B, k, H]

        # Masked mean-pool
        real = (~mask).float().unsqueeze(-1)                       # [B, k, 1]
        agg  = (out * real).sum(1) / real.sum(1).clamp(min=1)     # [B, H]

        combined = torch.cat([agg, coord_ctx, id_ctx], dim=-1)    # [B, H*3]
        return self.head(combined).squeeze(-1)                     # [B]


# ─────────────────────────────────────────────
# 5.  Sparse-Aware Loss
# ─────────────────────────────────────────────
class SparseFocalMSELoss(nn.Module):
    """
    Weighted MSE where non-zero targets get weight=alpha, zero targets get weight=1.
    This prevents the model from ignoring rare non-zero nodes.

    alpha=5  →  errors on non-zero samples count 5× more than zero samples.
    """
    def __init__(self, alpha=5.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target_log):
        is_nonzero = (target_log > 0).float()
        weights    = 1.0 + (self.alpha - 1.0) * is_nonzero  # 1 for zero, alpha for nonzero
        return ((pred - target_log) ** 2 * weights).mean()


# ─────────────────────────────────────────────
# 6.  Training
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, loss_fn, device, train):
    model.train() if train else model.eval()
    total_loss = total_mae_orig = n_total = 0.0

    # Track separately: zero-target MAE vs nonzero-target MAE
    mae_zero = mae_nonzero = n_zero = n_nonzero = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for weights, neighbor_ids, own_ids, coords, targets_log, mask in loader:
            weights      = weights.to(device)
            neighbor_ids = neighbor_ids.to(device)
            own_ids      = own_ids.to(device)
            coords       = coords.to(device)
            targets_log  = targets_log.to(device)
            mask         = mask.to(device)

            preds_log = model(weights, neighbor_ids, own_ids, coords, mask)
            loss      = loss_fn(preds_log, targets_log)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = len(targets_log)
            total_loss   += loss.item() * n
            n_total      += n

            p_orig = inverse_transform_targets(preds_log.cpu())
            t_orig = inverse_transform_targets(targets_log.cpu())
            total_mae_orig += (p_orig - t_orig).abs().sum().item()

            # Split MAE
            nz_mask = (targets_log.cpu() > 0)
            if nz_mask.any():
                mae_nonzero += (p_orig[nz_mask]  - t_orig[nz_mask]).abs().sum().item()
                n_nonzero   += nz_mask.sum().item()
            if (~nz_mask).any():
                mae_zero    += (p_orig[~nz_mask] - t_orig[~nz_mask]).abs().sum().item()
                n_zero      += (~nz_mask).sum().item()

    return (total_loss / n_total,
            total_mae_orig / n_total,
            mae_nonzero / max(n_nonzero, 1),
            mae_zero    / max(n_zero, 1))


def train_model(model, train_loader, val_loader, device, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"], eta_min=1e-6
    )
    loss_fn  = SparseFocalMSELoss(alpha=cfg["loss_alpha"])
    history  = defaultdict(list)
    best_val = float("inf")
    stagnant = 0

    print("\nEpoch metrics: TrLoss, VaLoss, TrMAE, VaMAE, VaNZ_MAE, LR")

    epoch_iter = range(1, cfg["num_epochs"] + 1)
    if cfg.get("use_tqdm", True):
        epoch_iter = tqdm(
            epoch_iter,
            desc="Training epochs",
            unit="epoch",
            dynamic_ncols=True,
        )

    for epoch in epoch_iter:
        tr_loss, tr_mae, tr_nz, tr_z = run_epoch(
            model, train_loader, optimizer, loss_fn, device, True)
        va_loss, va_mae, va_nz, va_z = run_epoch(
            model, val_loader,   optimizer, loss_fn, device, False)
        scheduler.step()

        for k, v in [("train_loss",tr_loss),("val_loss",va_loss),
                     ("train_mae",tr_mae),("val_mae",va_mae),
                     ("val_nz_mae",va_nz),("val_z_mae",va_z)]:
            history[k].append(v)

        lr_now = optimizer.param_groups[0]["lr"]
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                tr=f"{tr_loss:.4f}",
                va=f"{va_loss:.4f}",
                va_nz=f"{va_nz:.4f}",
                lr=f"{lr_now:.2e}",
            )
        else:
            print(f"{epoch:>6}  {tr_loss:>9.5f}  {va_loss:>9.5f}  "
                  f"{tr_mae:>9.5f}  {va_mae:>9.5f}  {va_nz:>10.5f}  {lr_now:>9.2e}")

        if va_loss < best_val:
            best_val = va_loss
            stagnant = 0
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            msg = f"Best val loss={best_val:.5f} saved (epoch {epoch})"
            if hasattr(epoch_iter, "write"):
                epoch_iter.write(msg)
            else:
                print(msg)
        else:
            stagnant += 1
            if stagnant >= cfg["patience"]:
                msg = f"Early stopping at epoch {epoch}."
                if hasattr(epoch_iter, "write"):
                    epoch_iter.write(msg)
                else:
                    print(f"\n{msg}")
                break

    model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location=device))
    print(f"\nBest checkpoint loaded  (val loss={best_val:.5f})")
    return dict(history)


def collect_predictions(model, loader, device):
    """Collect model outputs and targets on original scale."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for weights, neighbor_ids, own_ids, coords, targets_log, mask in loader:
            pred = model(
                weights.to(device),
                neighbor_ids.to(device),
                own_ids.to(device),
                coords.to(device),
                mask.to(device),
            )
            all_preds.append(inverse_transform_targets(pred.cpu()))
            all_targets.append(inverse_transform_targets(targets_log))

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return preds, targets


def calibrate_zero_threshold(model, val_loader, device, grid_size=200,
                             min_nz_recall=0.80, nz_weight=3.0):
    """
    Find threshold tau such that preds < tau are set to zero.
    Chooses tau minimizing validation MAE on original scale.
    """
    preds_raw, targets = collect_predictions(model, val_loader, device)
    if len(preds_raw) == 0:
        return {"tau": 0.0, "scale": 1.0, "recall": 1.0}

    nz = targets > 0
    z = ~nz
    if not nz.any():
        return {"tau": 0.0, "scale": 1.0, "recall": 1.0}

    zero_preds = preds_raw[z]
    upper = float(np.percentile(zero_preds, 99.9)) if z.any() else float(np.percentile(preds_raw, 95.0))
    if upper <= 0:
        print("  Threshold calibration skipped (predictions are non-positive).")
        return {"tau": 0.0, "scale": 1.0, "recall": 1.0}

    taus = np.linspace(0.0, upper, grid_size)
    best = None

    for tau in taus:
        clipped = preds_raw.copy()
        clipped[clipped < tau] = 0.0
        nz_recall = float((clipped[nz] > 0).mean())
        if nz_recall < min_nz_recall:
            continue

        active_nz = nz & (clipped > 0)
        scale = 1.0
        if active_nz.any():
            ratios = targets[active_nz] / np.maximum(clipped[active_nz], 1e-8)
            scale = float(np.clip(np.median(ratios), 0.5, 20.0))

        adjusted = clipped.copy()
        adjusted[adjusted > 0] *= scale

        mae_z = float(np.abs(adjusted[z] - targets[z]).mean()) if z.any() else 0.0
        mae_nz = float(np.abs(adjusted[nz] - targets[nz]).mean())
        score = mae_z + nz_weight * mae_nz

        if best is None or score < best["score"]:
            best = {
                "tau": float(tau),
                "scale": scale,
                "mae_z": mae_z,
                "mae_nz": mae_nz,
                "score": score,
                "recall": nz_recall,
            }

    base_mae = float(np.abs(preds_raw - targets).mean())
    if best is None:
        print("  Calibration fallback: no threshold satisfied recall constraint; using tau=0, scale=1")
        return {"tau": 0.0, "scale": 1.0, "recall": 1.0}

    adjusted = preds_raw.copy()
    adjusted[adjusted < best["tau"]] = 0.0
    adjusted[adjusted > 0] *= best["scale"]
    calibrated_mae = float(np.abs(adjusted - targets).mean())
    print(
        "  Calibration: "
        f"tau={best['tau']:.6f}, scale={best['scale']:.4f}, nz_recall={best['recall']:.3f}  "
        f"val_MAE(base={base_mae:.6f} -> calibrated={calibrated_mae:.6f})"
    )
    return {"tau": best["tau"], "scale": best["scale"], "recall": best["recall"]}


# ─────────────────────────────────────────────
# 7.  Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, device, postproc=None):
    preds_raw, targets = collect_predictions(model, test_loader, device)

    preds = preds_raw.copy()
    postproc = postproc or {"tau": 0.0, "scale": 1.0}
    tau = float(postproc.get("tau", 0.0))
    scale = float(postproc.get("scale", 1.0))
    if tau > 0:
        preds[preds < tau] = 0.0
    if scale != 1.0:
        preds[preds > 0] *= scale

    mse    = float(((preds - targets)**2).mean())
    mae    = float(np.abs(preds - targets).mean())
    rmse   = float(mse**0.5)
    ss_res = ((targets - preds)**2).sum()
    ss_tot = ((targets - targets.mean())**2).sum()
    r2     = float(1 - ss_res / (ss_tot + 1e-10))

    nz   = targets > 0
    mae_nz = float(np.abs(preds[nz]  - targets[nz]).mean())  if nz.any()  else 0.
    mae_z  = float(np.abs(preds[~nz] - targets[~nz]).mean()) if (~nz).any() else 0.

    print(f"\n{'='*50}")
    print(f"  TEST RESULTS  (original scale)")
    print(f"{'='*50}")
    print(f"  Samples          : {len(targets)}")
    print(f"  Non-zero targets : {nz.sum()}  ({nz.mean()*100:.1f}%)")
    print(f"  MSE              : {mse:.6f}")
    print(f"  RMSE             : {rmse:.6f}")
    mae_raw = float(np.abs(preds_raw - targets).mean())
    print(f"  MAE (all)        : {mae:.6f}")
    if tau > 0 or scale != 1.0:
        print(f"  MAE (raw preds)  : {mae_raw:.6f}")
        print(f"  Zero threshold   : {tau:.6f}")
        print(f"  Positive scale   : {scale:.4f}")
    print(f"  MAE (non-zero)   : {mae_nz:.6f}  ← most important")
    print(f"  MAE (zero)       : {mae_z:.6f}")
    print(f"  R²               : {r2:.4f}")
    print(f"{'='*50}\n")
    return preds, targets


# ─────────────────────────────────────────────
# 8.  Plots
# ─────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Focal MSE Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_mae"], label="Train")
    axes[1].plot(history["val_mae"],   label="Val")
    axes[1].set_title("MAE (original scale, all)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_nz_mae"],  label="Non-zero targets")
    axes[2].plot(history["val_z_mae"],   label="Zero targets")
    axes[2].set_title("Val MAE by target type"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_v4.png", dpi=150)
    print("Saved → training_v4.png")


def plot_predictions(preds, targets):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Only non-zero for scatter (zeros cluster at origin and hide pattern)
    nz = targets > 0
    axes[0].scatter(targets[nz], preds[nz], alpha=0.6, s=20, c="steelblue", label="non-zero")
    axes[0].scatter(targets[~nz], preds[~nz], alpha=0.1, s=5, c="gray", label="zero")
    mn, mx = targets.min(), targets.max()
    axes[0].plot([mn, mx], [mn, mx], "r--")
    axes[0].set_xlabel("True"); axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs True"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log scale view
    axes[1].scatter(np.log1p(targets[nz]),
                    np.log1p(np.maximum(preds[nz], 0)),
                    alpha=0.6, s=20, c="darkorange")
    lmn = np.log1p(mn); lmx = np.log1p(mx)
    axes[1].plot([lmn, lmx], [lmn, lmx], "r--")
    axes[1].set_xlabel("log1p(True)"); axes[1].set_ylabel("log1p(Pred)")
    axes[1].set_title("Log-scale (non-zero only)")
    axes[1].grid(True, alpha=0.3)

    # Residuals
    residuals = preds - targets
    axes[2].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_xlabel("Residual (Pred − True)")
    axes[2].set_title("Residuals"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions_v4.png", dpi=150)
    print("Saved → predictions_v4.png")


# ─────────────────────────────────────────────
# 9.  Single-Node Inference
# ─────────────────────────────────────────────
def predict_single_node(model, weight_row, neighbor_coords, neighbor_embed_ids,
                        own_embed_id, own_coords_scaled, coord_scaler,
                        strategy, k, device, postproc=None):
    """
    Predict for one node using raw inputs.

    Args:
        weight_row           : 1-D np.array  [M]  full weight row for this node
                               (only non-zero entries will be used)
        neighbor_coords      : np.array [L, 3]  coords of non-zero neighbours (raw)
        neighbor_embed_ids   : np.array [L]  int64  embed IDs of neighbours
        own_embed_id         : int  — this node's embed ID from training lookup
        own_coords_scaled    : np.array [3]  — already StandardScaled
        coord_scaler         : fitted StandardScaler (to scale neighbour coords)
        strategy / k         : matching training config
    Returns:
        float — prediction in original scale
    """
    # Scale neighbour coords
    nc_scaled = coord_scaler.transform(neighbor_coords).astype(np.float32)

    if strategy == "topk_spatial":
        w_f, nid_f, zmask = apply_topk_spatial(
            [weight_row], [nc_scaled], [neighbor_embed_ids],
            own_coords_scaled.reshape(1, -1), k, use_tqdm=False, verbose=False
        )
    else:
        w_f, nid_f, zmask = apply_topk([weight_row], [neighbor_embed_ids], k, use_tqdm=False, verbose=False)

    model.eval()
    w_t   = torch.from_numpy(w_f).to(device)
    nid_t = torch.from_numpy(nid_f).to(device)
    oid_t = torch.tensor([own_embed_id], dtype=torch.int64).to(device)
    c_t   = torch.from_numpy(own_coords_scaled.reshape(1, -1)).to(device)
    m_t   = torch.from_numpy(zmask).to(device)

    with torch.no_grad():
        pred_log = model(w_t, nid_t, oid_t, c_t, m_t)

    pred = float(inverse_transform_targets(pred_log.cpu()).item())
    postproc = postproc or {"tau": 0.0, "scale": 1.0}
    tau = float(postproc.get("tau", 0.0))
    scale = float(postproc.get("scale", 1.0))
    if pred < tau:
        pred = 0.0
    elif scale != 1.0:
        pred *= scale
    return pred

ROOT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/csce6810/data/preprocessing"
WEIGHTS= "contact_matrix.txt"
COORDS = "coordinates.txt"
IDS    = "bins.txt"
TARGETS= "gene_exp.txt" 
# ─────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────
def main():
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=f"{ROOT_DIR}/{WEIGHTS}")
    parser.add_argument("--coords",  default=f"{ROOT_DIR}/{COORDS}")
    parser.add_argument("--ids",     default=f"{ROOT_DIR}/{IDS}")
    parser.add_argument("--targets", default=f"{ROOT_DIR}/{TARGETS}")
    parser.add_argument("--log-dir", default=CONFIG["log_dir"])
    parser.add_argument("--single-sample-count", type=int, default=10)
    parser.add_argument("--single-sample-seed", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--disable-threshold-calibration", action="store_true")
    args = parser.parse_args()

    CONFIG["use_tqdm"] = not args.disable_tqdm

    _, log_file, old_stdout, old_stderr = setup_run_logging(args.log_dir)
    try:
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategy = CONFIG["strategy"]
        k        = CONFIG["topk_k"]
        print_section("Run Configuration")
        print(f"Device   : {device}")
        print(f"Strategy : {strategy}  k={k}")
        print(f"Loss alpha: {CONFIG['loss_alpha']}")
        print(f"tqdm enabled: {CONFIG['use_tqdm']}")

        # -- 1. Load
        print_section("[1/7] Loading Real Data")
        (raw_weights, raw_neighbor_coords, raw_neighbor_ids,
         coords, targets, row_embed_ids, vocab_size,
         dataset_indices) = load_real_data(
            args.weights, args.coords, args.ids, args.targets, k,
            use_tqdm=CONFIG["use_tqdm"]
        )
        N = len(targets)

        # -- 2. Scale
        print_section("[2/7] Scaling")
        print("  Weights: already [0,1] - skipping rescaler")

        coord_scaler  = StandardScaler()
        coords_scaled = coord_scaler.fit_transform(coords).astype(np.float32)
        print(f"  Coord scaler mean={coord_scaler.mean_} std={coord_scaler.scale_}")

        targets_log = transform_targets(targets)
        nz = (targets > 0)
        print(f"  Targets log1p: non-zero={nz.sum()} ({nz.mean()*100:.2f}%), max_log={targets_log.max():.4f}")

        joblib.dump(coord_scaler, "coord_scaler_v4.pkl")
        print("  Saved -> coord_scaler_v4.pkl")

        # -- 3. Top-K
        print_section("[3/7] Selecting Neighbors")
        if strategy == "topk_spatial":
            weights_fixed, nids_fixed, zero_mask = apply_topk_spatial(
                raw_weights, raw_neighbor_coords, raw_neighbor_ids,
                coords_scaled, k, use_tqdm=CONFIG["use_tqdm"]
            )
        else:
            weights_fixed, nids_fixed, zero_mask = apply_topk(
                raw_weights, raw_neighbor_ids, k, use_tqdm=CONFIG["use_tqdm"]
            )

        # -- 4. Dataloaders
        print_section("[4/7] Building Dataloaders")
        own_ids = row_embed_ids

        train_loader, val_loader, test_loader = build_dataloaders(
            weights_fixed, nids_fixed, own_ids,
            coords_scaled, targets_log, zero_mask, CONFIG
        )

        # -- 5. Model
        print_section("[5/7] Building Model")
        model = NodeTransformerV4(
            seq_len=k,
            vocab_size=vocab_size,
            hidden_dim=CONFIG["hidden_dim"],
            num_heads=CONFIG["num_heads"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        print(f"  Sequence length (k): {k}")
        print(f"  Node ID vocab size : {vocab_size}")

        history = train_model(model, train_loader, val_loader, device, CONFIG)
        plot_training(history)

        # -- 6. Calibrate + Evaluate
        print_section("[6/7] Calibration and Evaluation")
        postproc = {"tau": 0.0, "scale": 1.0}
        if not args.disable_threshold_calibration:
            postproc = calibrate_zero_threshold(
                model,
                val_loader,
                device,
                min_nz_recall=CONFIG["calib_min_nz_recall"],
                nz_weight=CONFIG["calib_nz_weight"],
            )
        else:
            print("  Threshold calibration disabled.")

        preds, true_vals = evaluate_model(
            model, test_loader, device, postproc=postproc
        )
        plot_predictions(preds, true_vals)

        # -- 7. Single-node demo
        print_section("[7/7] Single-Node Inference Demo")
        print(f"  {'#':>3}  {'True':>12}  {'Pred':>12}  {'AbsErr':>12}  {'NZ?':>5}")
        print("  " + "-" * 48)

        sample_count = max(1, min(args.single_sample_count, N))
        rng = np.random.default_rng(args.single_sample_seed)
        idxs = rng.choice(N, size=sample_count, replace=False)

        for num, idx in enumerate(idxs, 1):
            w_row = raw_weights[idx]
            nc    = raw_neighbor_coords[idx]
            nids  = raw_neighbor_ids[idx]

            pred_val = predict_single_node(
                model,
                weight_row          = w_row,
                neighbor_coords     = nc if len(nc) > 0 else np.zeros((1, 3), np.float32),
                neighbor_embed_ids  = nids if len(nids) > 0 else np.zeros(1, np.int64),
                own_embed_id        = int(own_ids[idx]),
                own_coords_scaled   = coords_scaled[idx],
                coord_scaler        = coord_scaler,
                strategy            = strategy,
                k                   = k,
                device              = device,
                postproc            = postproc,
            )
            true_val = float(targets[idx])
            is_nz    = "Y" if true_val > 0 else "N"
            print(f"  {num:>3}  {true_val:>12.6f}  {pred_val:>12.6f}  "
                  f"{abs(pred_val-true_val):>12.6f}  {is_nz:>5}")

        print("\nRun complete.")
    finally:
        close_run_logging(log_file, old_stdout, old_stderr)


if __name__ == "__main__":
    main()