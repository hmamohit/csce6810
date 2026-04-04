import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm


SEED = 42


def set_global_seed(seed: int):
    global SEED
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


set_global_seed(SEED)

TQDM_DISABLE = not sys.stdout.isatty()
LOG_FILE_PATH: Optional[Path] = None

CLASS_NAMES = ["Low", "Medium", "High"]

CONFIG = {
    "data_k": 512,
    "graph_k": 12,
    "data_subdir": "data/preprocessing",
    "log_subdir": "logs",
    "float_precision": 6,
    "sci_precision": 6,
    "hidden_dim": 128,
    "heads": 4,
    "dropout": 0.2,
    "batch_size": 256,
    "lr": 7e-4,
    "weight_decay": 1e-4,
    "num_epochs": 120,
    "patience": 20,
    "scheduler": "plateau",  # "plateau" | "cosine"
    "plateau_factor": 0.5,
    "plateau_patience": 6,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "label_mode": "fixed_thresholds",
    "low_max": 0.01,
    "med_max": 1.0,
    "class_weight_power": 0.35,
    "use_balanced_sampler": True,
    "sampler_power": 0.5,
    "balance_loaded_data": True,
    "balance_mode": "undersample",
    "weights_log1p": True,
    "weights_clip_quantile": 0.995,
    "normalize_weights": True,
    "normalize_coords": True,
    "use_amp": True,
}


def console_log(message: str = ""):
    if TQDM_DISABLE:
        print(message)
    else:
        tqdm.write(message)

    if LOG_FILE_PATH is not None:
        with LOG_FILE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def init_log_file(base_dir: Path, prefix: str) -> Path:
    global LOG_FILE_PATH
    log_dir = base_dir / CONFIG["log_subdir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{prefix}_{ts}.log"
    log_path.write_text("", encoding="utf-8")
    LOG_FILE_PATH = log_path
    return log_path


def fmt_float(value: float) -> str:
    return f"{float(value):.{CONFIG['float_precision']}f}"


def fmt_sci(value: float) -> str:
    return f"{float(value):.{CONFIG['sci_precision']}e}"


def console_rule(char: str = "-", width: int = 86):
    console_log(char * width)


def console_section(title: str, width: int = 86):
    console_rule("=", width)
    console_log(title)
    console_rule("=", width)


def console_kv(key: str, value):
    console_log(f"{key:<24}: {value}")


def load_preprocessed_data(data_dir: Path, k: int):
    weights_path = data_dir / "contact_matrix.txt"
    coords_path = data_dir / "coordinates.txt"
    targets_path = data_dir / "gene_exp.txt"

    missing = [
        str(path)
        for path in (weights_path, coords_path, targets_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing preprocessed input files:\n" + "\n".join(missing))

    coords = np.loadtxt(coords_path, dtype=np.float32)
    targets = np.loadtxt(targets_path, dtype=np.float32)

    if coords.ndim == 1:
        if coords.shape[0] == 3:
            coords = np.expand_dims(coords, axis=0)
        else:
            raise ValueError(f"Expected coordinates with shape (N, 3), got {coords.shape}.")

    if targets.ndim == 0:
        targets = np.array([float(targets)], dtype=np.float32)
    elif targets.ndim == 2:
        if targets.shape[1] == 1:
            targets = targets[:, 0]
        else:
            raise ValueError(f"Expected targets with shape (N,), got {targets.shape}.")

    n_coords = len(coords)
    n_targets = len(targets)
    n_meta = min(n_coords, n_targets)
    if n_meta <= 0:
        raise ValueError(f"Empty coords/targets after loading: coords={n_coords}, targets={n_targets}")

    if n_coords != n_targets:
        console_log(
            "[ALIGN] coords/targets row mismatch detected. "
            f"coords={n_coords}, targets={n_targets}. Using first {n_meta} rows."
        )

    weights_rows = []
    neighbor_rows = []
    iterator = tqdm(
        range(n_meta),
        desc=f"Top-{k} from contact_matrix rows",
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
    )

    with weights_path.open("r", encoding="utf-8") as handle:
        for _ in iterator:
            line = handle.readline()
            if line == "":
                break

            vals = np.fromstring(line, sep=" ", dtype=np.float32)
            row_out = np.zeros(k, dtype=np.float32)
            idx_out = np.full(k, -1, dtype=np.int64)

            if vals.size > 0:
                if vals.size >= k:
                    idx = np.argpartition(vals, -k)[-k:]
                    top = vals[idx]
                    order = np.argsort(top)[::-1]
                    row_out[:] = top[order]
                    idx_out[:] = idx[order]
                else:
                    order = np.argsort(vals)[::-1]
                    row_out[:vals.size] = vals[order]
                    idx_out[:vals.size] = order

            weights_rows.append(row_out)
            neighbor_rows.append(idx_out)

    n_weights = len(weights_rows)
    n_shared = min(n_meta, n_weights)
    if n_shared <= 0:
        raise ValueError(
            f"No usable rows loaded from contact matrix: weights_rows={n_weights}, n_meta={n_meta}"
        )

    if n_weights != n_meta:
        console_log(
            "[ALIGN] weights row count differs from coords/targets. "
            f"weights={n_weights}, coords/targets_min={n_meta}. Using first {n_shared} rows."
        )

    weights = np.vstack(weights_rows[:n_shared]).astype(np.float32)
    neighbor_idx = np.vstack(neighbor_rows[:n_shared]).astype(np.int64)
    coords = coords[:n_shared]
    targets = targets[:n_shared]

    paths = {
        "weights": weights_path,
        "coords": coords_path,
        "targets": targets_path,
    }
    return weights, coords, neighbor_idx, targets, paths


def preprocess_features(
    weights: np.ndarray,
    coords: np.ndarray,
    weights_log1p: bool,
    weights_clip_quantile: float,
    normalize_weights: bool,
    normalize_coords: bool,
):
    out_w = weights.astype(np.float32, copy=True)
    out_c = coords.astype(np.float32, copy=True)

    if weights_log1p:
        out_w = np.log1p(np.maximum(out_w, 0.0))

    clip_q = float(weights_clip_quantile)
    if 0.0 < clip_q < 1.0:
        hi = float(np.quantile(out_w, clip_q))
        out_w = np.minimum(out_w, hi)
    else:
        hi = float(np.max(out_w))

    w_mean = float(out_w.mean())
    w_std = float(out_w.std())
    if normalize_weights:
        out_w = (out_w - w_mean) / (w_std + 1e-6)

    c_mean = out_c.mean(axis=0, keepdims=True).astype(np.float32)
    c_std = out_c.std(axis=0, keepdims=True).astype(np.float32)
    if normalize_coords:
        out_c = (out_c - c_mean) / (c_std + 1e-6)

    stats = {
        "weights_clip_hi": hi,
        "weights_mean_pre_norm": w_mean,
        "weights_std_pre_norm": w_std,
        "coords_mean_pre_norm": c_mean.reshape(-1),
        "coords_std_pre_norm": c_std.reshape(-1),
    }
    return out_w.astype(np.float32), out_c.astype(np.float32), stats


def make_labels(
    targets: np.ndarray,
    mode: str = "fixed_thresholds",
    low_max: float = 0.01,
    med_max: float = 1.0,
):
    n = len(targets)
    labels = np.zeros(n, dtype=np.int64)

    if mode == "fixed_thresholds":
        if low_max >= med_max:
            raise ValueError(f"Expected low_max < med_max. Got low_max={low_max}, med_max={med_max}")
        boundaries = (float(low_max), float(med_max))
        labels = np.digitize(targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    elif mode == "balanced_rank":
        order = np.argsort(targets, kind="mergesort")
        cut1 = n // 3
        cut2 = (2 * n) // 3
        labels[order[:cut1]] = 0
        labels[order[cut1:cut2]] = 1
        labels[order[cut2:]] = 2
        t1 = float(targets[order[cut1 - 1]]) if cut1 > 0 else float(targets.min())
        t2 = float(targets[order[cut2 - 1]]) if cut2 > 0 else float(targets.max())
        boundaries = (t1, t2)
    elif mode == "quantile":
        q1, q2 = np.quantile(targets, [1.0 / 3.0, 2.0 / 3.0])
        if q2 <= q1:
            q2 = q1 + 1e-12
        boundaries = (float(q1), float(q2))
        labels = np.digitize(targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    else:
        raise ValueError(f"Unknown label mode: {mode}")

    return labels, boundaries


def balance_loaded_data(
    weights: np.ndarray,
    coords: np.ndarray,
    neighbor_idx: np.ndarray,
    edge_strength: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    mode: str = "undersample",
):
    counts = np.bincount(labels, minlength=3)
    if counts.min() <= 0:
        raise ValueError(f"Cannot balance because at least one class is empty: {counts.tolist()}")

    if mode != "undersample":
        raise ValueError(f"Unsupported balance mode: {mode}")

    rng = np.random.default_rng(SEED)
    per_class = int(counts.min())
    kept = []
    for cls in range(3):
        cls_idx = np.where(labels == cls)[0]
        pick = rng.choice(cls_idx, size=per_class, replace=False)
        kept.append(pick)

    kept_idx = np.concatenate(kept)
    rng.shuffle(kept_idx)

    return (
        weights[kept_idx],
        coords[kept_idx],
        neighbor_idx[kept_idx],
        edge_strength[kept_idx],
        targets[kept_idx],
        labels[kept_idx],
        original_indices[kept_idx],
        counts,
        np.bincount(labels[kept_idx], minlength=3),
    )


class ExpClassDataset(Dataset):
    def __init__(
        self,
        weights: np.ndarray,
        coords: np.ndarray,
        neighbor_idx: np.ndarray,
        edge_strength: np.ndarray,
        labels: np.ndarray,
        global_indices: np.ndarray,
    ):
        self.weights = torch.from_numpy(weights)
        self.coords = torch.from_numpy(coords)
        self.neighbor_idx = torch.from_numpy(neighbor_idx.astype(np.int64))
        self.edge_strength = torch.from_numpy(edge_strength.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.global_indices = torch.from_numpy(global_indices.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.weights[idx],
            self.coords[idx],
            self.neighbor_idx[idx],
            self.edge_strength[idx],
            self.global_indices[idx],
            self.labels[idx],
        )


def build_dataloaders(
    weights: np.ndarray,
    coords: np.ndarray,
    neighbor_idx: np.ndarray,
    edge_strength: np.ndarray,
    labels: np.ndarray,
    global_indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    use_balanced_sampler: bool,
    sampler_power: float,
):
    n = len(labels)
    indices = np.arange(n)

    class_counts = np.bincount(labels, minlength=3)
    can_stratify = class_counts.min() >= 2

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=SEED,
        stratify=labels if can_stratify else None,
    )

    val_fraction = val_ratio / (1.0 - train_ratio)
    temp_labels = labels[temp_idx]
    temp_counts = np.bincount(temp_labels, minlength=3)
    temp_can_stratify = temp_counts.min() >= 2
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction,
        random_state=SEED,
        stratify=temp_labels if (can_stratify and temp_can_stratify) else None,
    )

    dataset = ExpClassDataset(weights, coords, neighbor_idx, edge_strength, labels, global_indices)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    train_labels = labels[train_idx]
    train_counts = np.bincount(train_labels, minlength=3).astype(np.float64)

    train_min = float(np.maximum(train_counts.min(), 1.0))
    balance_ratio = float(train_counts.max() / train_min)
    if use_balanced_sampler and balance_ratio <= 1.05:
        console_log(
            f"[INFO] Train split already balanced (max/min={balance_ratio:.3f}); "
            "using shuffle instead of weighted sampler."
        )
        use_balanced_sampler = False

    if use_balanced_sampler:
        train_sample_weights = 1.0 / np.maximum(train_counts[train_labels], 1.0) ** sampler_power
        train_sample_weights = torch.as_tensor(train_sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=False, **kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    split_stats = {
        "train_counts": np.bincount(labels[train_idx], minlength=3),
        "val_counts": np.bincount(labels[val_idx], minlength=3),
        "test_counts": np.bincount(labels[test_idx], minlength=3),
    }

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, split_stats


def compute_class_weights(labels: np.ndarray, num_classes: int = 3, power: float = 0.35):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** float(power)
    weights = weights / weights.mean()
    return weights, counts


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.edge_scale = nn.Parameter(torch.zeros(heads))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def _build_knn_edges(self, coords: torch.Tensor, k: int):
        bsz = coords.shape[0]
        if bsz <= 1:
            idx = torch.zeros(1, device=coords.device, dtype=torch.long)
            return idx, idx

        k_eff = int(min(max(k, 1), bsz - 1))
        dist = torch.cdist(coords, coords, p=2)
        knn_idx = dist.topk(k=k_eff + 1, largest=False).indices[:, 1:]

        src = torch.arange(bsz, device=coords.device).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        dst = knn_idx.reshape(-1)
        return src, dst

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        graph_k: int,
        edge_index=None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        bsz = x.shape[0]
        if edge_index is None:
            src, dst = self._build_knn_edges(coords, graph_k)
        else:
            src, dst = edge_index

        h = self.linear(x).view(bsz, self.heads, self.out_dim)
        h_src = h[src]
        h_dst = h[dst]

        e = (h_src * self.attn_src).sum(dim=-1) + (h_dst * self.attn_dst).sum(dim=-1)
        e = e.float()
        if edge_weight is not None:
            ew = edge_weight.float()
            if ew.numel() > 1:
                ew = (ew - ew.mean()) / (ew.std(unbiased=False) + 1e-6)
            else:
                ew = torch.zeros_like(ew)
            e = e + ew.unsqueeze(1) * self.edge_scale.unsqueeze(0)
        e = self.leaky_relu(e)
        alpha = torch.zeros_like(e)
        for hd in range(self.heads):
            e_h = e[:, hd]
            e_h = e_h - e_h.max()
            exp_e = torch.exp(e_h)
            denom = torch.zeros(bsz, device=x.device, dtype=exp_e.dtype)
            denom.index_add_(0, src, exp_e)
            alpha[:, hd] = exp_e / denom[src].clamp_min(1e-12)

        alpha = self.dropout(alpha)

        out = torch.zeros(bsz, self.heads, self.out_dim, device=x.device, dtype=h.dtype)
        messages = alpha.unsqueeze(-1).to(dtype=h_dst.dtype) * h_dst
        out.index_add_(0, src, messages)

        if self.concat:
            return out.reshape(bsz, self.heads * self.out_dim)
        return out.mean(dim=1)


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        num_classes: int,
        graph_k: int,
        dropout: float,
    ):
        super().__init__()
        self.graph_k = graph_k
        self.fallback_k = max(2, min(graph_k // 2, 6))
        self.dropout = nn.Dropout(dropout)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        per_head = max(hidden_dim // heads, 8)
        self.block_dim = heads * per_head
        self.gat1 = BatchGATLayer(hidden_dim, per_head, heads=heads, dropout=dropout, concat=True)
        self.gat2 = BatchGATLayer(self.block_dim, per_head, heads=heads, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(self.block_dim)
        self.norm2 = nn.LayerNorm(self.block_dim)

        self.head = nn.Sequential(
            nn.Linear(self.block_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_contact_edges(
        self,
        global_idx: torch.Tensor,
        neighbor_idx: torch.Tensor,
        edge_strength: torch.Tensor,
        coords: torch.Tensor,
    ):
        bsz = global_idx.shape[0]
        if bsz <= 1:
            return None

        valid_neighbors = neighbor_idx[neighbor_idx >= 0]
        if valid_neighbors.numel() == 0:
            return None

        max_gid = int(torch.maximum(global_idx.max(), valid_neighbors.max()).item())
        lookup = torch.full((max_gid + 1,), -1, dtype=torch.long, device=coords.device)
        lookup[global_idx] = torch.arange(bsz, device=coords.device)

        nbr = neighbor_idx.clamp(min=0, max=max_gid)
        dst_pos = lookup[nbr]
        src_pos = torch.arange(bsz, device=coords.device).unsqueeze(1).expand_as(dst_pos)

        mask = (neighbor_idx >= 0) & (dst_pos >= 0) & (dst_pos != src_pos)
        if not torch.any(mask):
            return None

        src = src_pos[mask]
        dst = dst_pos[mask]
        ew = edge_strength[mask]
        return src, dst, ew

    def _augment_edges_with_knn(self, coords: torch.Tensor, edge_index):
        src_knn, dst_knn = self.gat1._build_knn_edges(coords, self.fallback_k)
        ew_knn = torch.zeros(src_knn.shape[0], device=coords.device, dtype=torch.float32)
        if edge_index is None:
            return src_knn, dst_knn, ew_knn

        src, dst, ew = edge_index
        src = torch.cat([src, src_knn], dim=0)
        dst = torch.cat([dst, dst_knn], dim=0)
        ew = torch.cat([ew.float(), ew_knn], dim=0)
        return src, dst, ew

    def forward(
        self,
        weights: torch.Tensor,
        coords: torch.Tensor,
        neighbor_idx: torch.Tensor,
        edge_strength: torch.Tensor,
        global_idx: torch.Tensor,
    ):
        x = torch.cat([weights, coords], dim=1)
        x = self.input_proj(x)

        edge_index = self._build_contact_edges(global_idx, neighbor_idx, edge_strength, coords)
        src, dst, ew = self._augment_edges_with_knn(coords, edge_index)

        h = self.gat1(x, coords, self.graph_k, edge_index=(src, dst), edge_weight=ew)
        h = F.elu(h)
        x = self.norm1(x + self.dropout(h))

        h = self.gat2(x, coords, self.graph_k, edge_index=(src, dst), edge_weight=ew)
        h = F.elu(h)
        x = self.norm2(x + self.dropout(h))

        combined = torch.cat([x, coords], dim=1)
        return self.head(combined)


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    train: bool,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_total = 0
    y_true_all = []
    y_pred_all = []

    phase = "Train" if train else "Val"
    desc = f"{phase} {epoch}/{total_epochs}" if (epoch is not None and total_epochs is not None) else f"{phase} Batches"

    iterator = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
        position=1,
    )

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for weights, coords, neighbor_idx, edge_strength, global_idx, labels in iterator:
            weights = weights.to(device)
            coords = coords.to(device)
            neighbor_idx = neighbor_idx.to(device)
            edge_strength = edge_strength.to(device)
            global_idx = global_idx.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(weights, coords, neighbor_idx, edge_strength, global_idx)
                loss = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            preds = torch.argmax(logits, dim=1)

            n = len(labels)
            total_loss += loss.item() * n
            n_total += n

            y_true_all.append(labels.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

            running_acc = (np.concatenate(y_true_all) == np.concatenate(y_pred_all)).mean()
            iterator.set_postfix(loss=fmt_float(total_loss / n_total), acc=fmt_float(running_acc))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    avg_loss = total_loss / max(n_total, 1)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return avg_loss, acc, macro_f1


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_weights,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    patience: int,
    scheduler_type: str,
    plateau_factor: float,
    plateau_patience: int,
    checkpoint_path: str,
    use_amp: bool,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    best_val_f1 = -1.0
    best_epoch = 0
    stagnant = 0
    final_epoch = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    console_log("")
    console_section("TRAINING", width=100)
    console_log(
        f"{'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>12}  {'TrainAcc':>10}  {'ValAcc':>10}  {'TrainF1':>10}  {'ValF1':>10}  {'LR':>12}"
    )
    console_rule("-", width=100)

    epoch_iterator = tqdm(range(1, num_epochs + 1), desc="Epochs", dynamic_ncols=True, disable=TQDM_DISABLE)

    for epoch in epoch_iterator:
        final_epoch = epoch

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            train=True,
            scaler=scaler,
            use_amp=amp_enabled,
            epoch=epoch,
            total_epochs=num_epochs,
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model,
            val_loader,
            optimizer,
            loss_fn,
            device,
            train=False,
            scaler=None,
            use_amp=amp_enabled,
            epoch=epoch,
            total_epochs=num_epochs,
        )

        if scheduler_type == "plateau":
            scheduler.step(va_f1)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_iterator.set_postfix(val_f1=fmt_float(va_f1), val_acc=fmt_float(va_acc), lr=fmt_sci(current_lr))

        console_log(
            f"{epoch:>6}  {tr_loss:>12.{CONFIG['float_precision']}f}  {va_loss:>12.{CONFIG['float_precision']}f}  "
            f"{tr_acc:>10.{CONFIG['float_precision']}f}  {va_acc:>10.{CONFIG['float_precision']}f}  "
            f"{tr_f1:>10.{CONFIG['float_precision']}f}  {va_f1:>10.{CONFIG['float_precision']}f}  "
            f"{current_lr:>12.{CONFIG['sci_precision']}e}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            stagnant = 0
            torch.save(model.state_dict(), checkpoint_path)
            console_log(f"[BEST] epoch={epoch:03d} val_macro_f1={fmt_float(best_val_f1)} checkpoint_saved={checkpoint_path}")
        else:
            stagnant += 1
            if stagnant >= patience:
                console_log(f"[EARLY-STOP] epoch={epoch:03d} no_improvement_for={patience} epochs")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    console_log(f"[CHECKPOINT] Loaded best checkpoint from epoch {best_epoch:03d} (val_macro_f1={fmt_float(best_val_f1)})")
    console_log(f"[SUMMARY] epochs_ran={final_epoch} best_epoch={best_epoch} best_val_macro_f1={fmt_float(best_val_f1)}")


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        iterator = tqdm(test_loader, desc="Test Batches", leave=False, dynamic_ncols=True, disable=TQDM_DISABLE)
        for weights, coords, neighbor_idx, edge_strength, global_idx, labels in iterator:
            logits = model(
                weights.to(device),
                coords.to(device),
                neighbor_idx.to(device),
                edge_strength.to(device),
                global_idx.to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(labels.numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    console_log("")
    console_section("TEST RESULTS (GAT CLASSIFICATION)", width=86)
    console_kv("samples", len(test_loader.dataset))
    console_kv("accuracy", fmt_float(acc))
    console_kv("macro_f1", fmt_float(macro_f1))
    console_rule("-", width=86)

    console_log(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    console_rule("-", width=86)
    for cls_name in CLASS_NAMES:
        row = report[cls_name]
        console_log(
            f"{cls_name:<12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1-score']:>10.4f} {int(row['support']):>10}"
        )
    console_rule("-", width=86)

    console_log("Confusion Matrix (rows=true, cols=pred)")
    console_log("            Pred-Low  Pred-Med  Pred-High")
    for i, row_name in enumerate(CLASS_NAMES):
        console_log(f"True-{row_name:<5} {cm[i, 0]:>9d} {cm[i, 1]:>9d} {cm[i, 2]:>10d}")
    console_rule("-", width=86)

    return y_true, y_pred


def predict_single_node_class(
    model,
    weights_array: np.ndarray,
    coords_array: np.ndarray,
    neighbor_idx_array: np.ndarray,
    edge_strength_array: np.ndarray,
    global_idx_value: int,
    device,
):
    model.eval()
    w = torch.from_numpy(weights_array.astype(np.float32)).unsqueeze(0).to(device)
    c = torch.from_numpy(coords_array.astype(np.float32)).unsqueeze(0).to(device)
    nidx = torch.from_numpy(neighbor_idx_array.astype(np.int64)).unsqueeze(0).to(device)
    ew = torch.from_numpy(edge_strength_array.astype(np.float32)).unsqueeze(0).to(device)
    gidx = torch.tensor([int(global_idx_value)], dtype=torch.int64, device=device)

    with torch.no_grad():
        logits = model(w, c, nidx, ew, gidx)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_class, confidence, probs


def main():
    global TQDM_DISABLE

    parser = argparse.ArgumentParser(description="Gene expression 3-class GAT trainer (Low/Medium/High)")
    parser.add_argument("--label-mode", choices=["fixed_thresholds", "balanced_rank", "quantile"], default=CONFIG["label_mode"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--low-max", type=float, default=CONFIG["low_max"])
    parser.add_argument("--med-max", type=float, default=CONFIG["med_max"])
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--num-sample-infer", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--graph-k", type=int, default=CONFIG["graph_k"])
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default=CONFIG["scheduler"])
    parser.add_argument("--plateau-factor", type=float, default=CONFIG["plateau_factor"])
    parser.add_argument("--plateau-patience", type=int, default=CONFIG["plateau_patience"])
    parser.add_argument("--class-weight-power", type=float, default=CONFIG["class_weight_power"])
    parser.add_argument("--sampler-power", type=float, default=CONFIG["sampler_power"])
    parser.add_argument("--disable-balanced-sampler", action="store_true")
    parser.add_argument("--disable-balance-loaded-data", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    if args.no_tqdm:
        TQDM_DISABLE = True

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / CONFIG["data_subdir"]
    log_path = init_log_file(base_dir, prefix="train_gat_cls")

    console_section("RUN START", width=86)
    console_kv("device", device)
    console_kv("seed", args.seed)
    console_kv("label_mode", args.label_mode)
    console_kv("low_max", args.low_max)
    console_kv("med_max", args.med_max)
    console_kv("log_file", log_path)
    console_kv("data_topk_k", CONFIG["data_k"])
    console_kv("graph_k", args.graph_k)
    console_kv("batch_size", args.batch_size)
    console_kv("num_epochs", args.num_epochs)
    console_kv("patience", args.patience)
    console_kv("learning_rate", fmt_sci(args.lr))
    console_kv("scheduler", args.scheduler)
    if args.scheduler == "plateau":
        console_kv("plateau_factor", args.plateau_factor)
        console_kv("plateau_patience", args.plateau_patience)
    console_kv("weight_decay", fmt_sci(CONFIG["weight_decay"]))
    console_kv("class_weight_power", args.class_weight_power)

    effective_balanced_sampler = CONFIG["use_balanced_sampler"] and (not args.disable_balanced_sampler)
    effective_balance_loaded = CONFIG["balance_loaded_data"] and (not args.disable_balance_loaded_data)
    effective_amp = CONFIG["use_amp"] and (not args.disable_amp)

    console_kv("use_balanced_sampler", effective_balanced_sampler)
    console_kv("sampler_power", args.sampler_power)
    console_kv("balance_loaded_data", effective_balance_loaded)
    console_kv("balance_mode", CONFIG["balance_mode"])
    console_kv("weights_log1p", CONFIG["weights_log1p"])
    console_kv("weights_clip_q", CONFIG["weights_clip_quantile"])
    console_kv("normalize_weights", CONFIG["normalize_weights"])
    console_kv("normalize_coords", CONFIG["normalize_coords"])
    console_kv("use_amp", effective_amp)

    console_log("\n[1/4] Loading preprocessed data ...")
    weights, coords, neighbor_idx, targets, data_paths = load_preprocessed_data(data_dir=data_dir, k=CONFIG["data_k"])

    labels, boundaries = make_labels(targets, mode=args.label_mode, low_max=args.low_max, med_max=args.med_max)
    targets_full = targets.copy()
    original_indices = np.arange(len(targets), dtype=np.int64)
    edge_strength = weights.copy()

    if effective_balance_loaded:
        (
            weights,
            coords,
            neighbor_idx,
            edge_strength,
            targets,
            labels,
            original_indices,
            before_counts,
            after_counts,
        ) = balance_loaded_data(
            weights,
            coords,
            neighbor_idx,
            edge_strength,
            targets,
            labels,
            original_indices,
            mode=CONFIG["balance_mode"],
        )
        console_kv("balance counts before", {CLASS_NAMES[i]: int(before_counts[i]) for i in range(3)})
        console_kv("balance counts after", {CLASS_NAMES[i]: int(after_counts[i]) for i in range(3)})

    weights, coords, feat_stats = preprocess_features(
        weights=weights,
        coords=coords,
        weights_log1p=CONFIG["weights_log1p"],
        weights_clip_quantile=CONFIG["weights_clip_quantile"],
        normalize_weights=CONFIG["normalize_weights"],
        normalize_coords=CONFIG["normalize_coords"],
    )

    class_weights, class_counts = compute_class_weights(labels, num_classes=3, power=args.class_weight_power)
    class_frac = class_counts / max(class_counts.sum(), 1)

    console_kv("weights file", data_paths["weights"])
    console_kv("coords file", data_paths["coords"])
    console_kv("targets file", data_paths["targets"])
    console_kv("weights shape", weights.shape)
    console_kv("coords shape", coords.shape)
    console_kv("targets shape", targets.shape)
    console_kv("target range", f"[{fmt_float(targets.min())}, {fmt_float(targets.max())}]")
    console_kv("label boundaries", f"low<= {fmt_float(boundaries[0])}, med<= {fmt_float(boundaries[1])}, else high")
    console_kv("class counts", {CLASS_NAMES[i]: int(class_counts[i]) for i in range(3)})
    console_kv("class fractions", {CLASS_NAMES[i]: f"{class_frac[i]*100:.2f}%" for i in range(3)})
    console_kv("class weights", {CLASS_NAMES[i]: fmt_float(class_weights[i]) for i in range(3)})
    console_kv("weights clip hi", fmt_float(feat_stats["weights_clip_hi"]))
    console_kv("weights mean/std", f"{fmt_float(feat_stats['weights_mean_pre_norm'])} / {fmt_float(feat_stats['weights_std_pre_norm'])}")

    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, split_stats = build_dataloaders(
        weights=weights,
        coords=coords,
        neighbor_idx=neighbor_idx,
        edge_strength=edge_strength,
        labels=labels,
        global_indices=original_indices,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        batch_size=args.batch_size,
        use_balanced_sampler=effective_balanced_sampler,
        sampler_power=args.sampler_power,
    )

    console_kv("dataset splits", f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    console_kv("split train counts", {CLASS_NAMES[i]: int(split_stats["train_counts"][i]) for i in range(3)})
    console_kv("split val counts", {CLASS_NAMES[i]: int(split_stats["val_counts"][i]) for i in range(3)})
    console_kv("split test counts", {CLASS_NAMES[i]: int(split_stats["test_counts"][i]) for i in range(3)})

    console_log("\n[2/4] Building GAT model ...")
    model = GATClassifier(
        in_dim=weights.shape[1] + coords.shape[1],
        hidden_dim=CONFIG["hidden_dim"],
        heads=CONFIG["heads"],
        num_classes=3,
        graph_k=args.graph_k,
        dropout=CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console_kv("trainable_parameters", f"{total_params:,}")

    console_log("\n[3/4] Training ...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=CONFIG["weight_decay"],
        num_epochs=args.num_epochs,
        patience=args.patience,
        scheduler_type=args.scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        checkpoint_path="best_model_gat_cls.pt",
        use_amp=effective_amp,
    )

    console_log("\n[4/4] Evaluating on test set ...")
    y_true, y_pred = evaluate_model(model, test_loader, device)

    console_log("")
    console_section("SINGLE-NODE INFERENCE (RANDOM TEST NODES)", width=100)

    test_size = len(test_loader.dataset)
    n_demo = min(max(args.num_sample_infer, 1), test_size)
    rng = np.random.default_rng(SEED)
    selected_local = rng.choice(test_size, size=n_demo, replace=False)
    selected_global = original_indices[test_idx[selected_local]]

    console_kv("sample_source", "random from test split")
    console_kv("num_nodes", n_demo)
    console_kv("global_indices", selected_global.tolist())
    console_rule("-", width=100)
    console_log(
        f"{'Node#':>6}  {'GlobalIdx':>9}  {'TrueVal':>10}  {'TrueCls':>8}  {'PredCls':>8}  "
        f"{'Conf':>8}  {'P(L/M/H)':>25}"
    )
    console_rule("-", width=100)

    for i, local_idx in enumerate(selected_local, start=1):
        global_idx = int(original_indices[test_idx[int(local_idx)]])
        sample_weights_t, sample_coords_t, sample_neighbor_idx_t, sample_edge_strength_t, sample_global_idx_t, sample_label_t = test_loader.dataset[int(local_idx)]

        sample_weights = sample_weights_t.cpu().numpy()
        sample_coords = sample_coords_t.cpu().numpy()
        sample_neighbor_idx = sample_neighbor_idx_t.cpu().numpy()
        sample_edge_strength = sample_edge_strength_t.cpu().numpy()
        sample_global_idx = int(sample_global_idx_t.item())
        true_label = int(sample_label_t.item())

        pred_label, confidence, probs = predict_single_node_class(
            model,
            sample_weights,
            sample_coords,
            sample_neighbor_idx,
            sample_edge_strength,
            sample_global_idx,
            device,
        )

        probs_str = f"[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]"
        console_log(
            f"{i:>6}  {global_idx:>9}  {targets_full[global_idx]:>10.{CONFIG['float_precision']}f}  "
            f"{CLASS_NAMES[true_label]:>8}  {CLASS_NAMES[pred_label]:>8}  "
            f"{confidence:>8.3f}  {probs_str:>25}"
        )

    console_rule("-", width=100)


if __name__ == "__main__":
    main()
