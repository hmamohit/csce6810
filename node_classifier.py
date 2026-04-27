import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from tqdm.auto import tqdm


SEED = 42
TQDM_DISABLE = not sys.stdout.isatty()
LOG_FILE_PATH: Optional[Path] = None

CLASS_NAMES = ["Low", "Medium", "High"]

CONFIG = {
    "data_subdir": "data/preprocessing",
    "log_subdir": "logs",
    "data_k": 0,
    "graph_k": 0,
    "float_precision": 6,
    "sci_precision": 6,
    "hidden_dim": 64,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 200,
    "patience": 30,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "label_mode": "fixed_thresholds",
    "low_max": 0.01,
    "med_max": 1.0,
    "class_weight_power": 0.35,
    "label_smoothing": 0.02,
    "balance_loaded_data": True,
    "balance_mode": "undersample",
    "max_nodes": 0,
    "node_batch_size": 0,
    "num_neighbors": "20,10",
}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def console_rule(char: str = "-", width: int = 100):
    console_log(char * width)


def console_section(title: str, width: int = 100):
    console_rule("=", width)
    console_log(title)
    console_rule("=", width)


def console_kv(key: str, value):
    console_log(f"{key:<24}: {value}")


def load_preprocessed_data_for_graph(data_dir: Path, k: int, max_nodes: Optional[int] = None):
    use_topk = k is not None and int(k) > 0
    k = int(k)

    weights_path = data_dir / "contact_matrix.txt"
    coords_path = data_dir / "coordinates.txt"
    targets_path = data_dir / "gene_exp.txt"

    missing = [
        str(path)
        for path in (weights_path, coords_path, targets_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessed input files:\n" + "\n".join(missing))

    coords = np.loadtxt(coords_path, dtype=np.float32)
    targets = np.loadtxt(targets_path, dtype=np.float32)

    if coords.ndim == 1:
        if coords.shape[0] == 3:
            coords = np.expand_dims(coords, axis=0)
        else:
            raise ValueError(
                f"Expected coordinates with shape (N, 3), got {coords.shape}.")

    if targets.ndim == 0:
        targets = np.array([float(targets)], dtype=np.float32)
    elif targets.ndim == 2:
        if targets.shape[1] == 1:
            targets = targets[:, 0]
        else:
            raise ValueError(
                f"Expected targets with shape (N,), got {targets.shape}.")

    n_coords = len(coords)
    n_targets = len(targets)
    n_meta = min(n_coords, n_targets)
    n_meta_before_cap = int(n_meta)
    if n_meta <= 0:
        raise ValueError(
            f"Empty coords/targets after loading: coords={n_coords}, targets={n_targets}")

    if n_coords != n_targets:
        console_log(
            "[ALIGN] coords/targets row mismatch detected. "
            f"coords={n_coords}, targets={n_targets}. Using first {n_meta} rows."
        )

    if max_nodes is not None and max_nodes > 0:
        n_meta = min(n_meta, int(max_nodes))

    edge_strength_rows = []
    neighbor_rows = []
    iterator = tqdm(
        range(n_meta),
        desc=(
            f"Reading rows; keep top-{k} interactions/row"
            if use_topk
            else "Reading rows; keep all positive interactions/row"
        ),
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
    )

    with weights_path.open("r", encoding="utf-8") as handle:
        for _ in iterator:
            line = handle.readline()
            if line == "":
                break

            vals = np.fromstring(line, sep=" ", dtype=np.float32)
            if use_topk:
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

                edge_strength_rows.append(row_out)
                neighbor_rows.append(idx_out)
            else:
                # Keep all positive contacts and store ragged rows to avoid huge dense allocation.
                pos_idx = np.flatnonzero(vals > 0.0).astype(np.int64)
                pos_w = vals[pos_idx].astype(np.float32)
                if pos_w.size > 1:
                    order = np.argsort(pos_w)[::-1]
                    pos_idx = pos_idx[order]
                    pos_w = pos_w[order]
                edge_strength_rows.append(pos_w)
                neighbor_rows.append(pos_idx)

    n_weights = len(edge_strength_rows)
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

    if use_topk:
        edge_strength = np.vstack(edge_strength_rows[:n_shared]).astype(np.float32)
        neighbor_idx = np.vstack(neighbor_rows[:n_shared]).astype(np.int64)
    else:
        edge_strength = edge_strength_rows[:n_shared]
        neighbor_idx = neighbor_rows[:n_shared]

    coords = coords[:n_shared]
    targets = targets[:n_shared]

    paths = {
        "weights": weights_path,
        "coords": coords_path,
        "targets": targets_path,
    }
    stats = {
        "rows_coords": int(n_coords),
        "rows_targets": int(n_targets),
        "rows_before_cap": int(n_meta_before_cap),
        "rows_after_cap": int(n_shared),
        "max_nodes": None if max_nodes is None else int(max_nodes),
    }
    return edge_strength, neighbor_idx, coords, targets, paths, stats


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
            raise ValueError(
                f"Expected low_max < med_max. Got low_max={low_max}, med_max={med_max}")
        boundaries = (float(low_max), float(med_max))
        labels = np.digitize(
            targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    elif mode == "balanced_rank":
        order = np.argsort(targets, kind="mergesort")
        cut1 = n // 3
        cut2 = (2 * n) // 3
        labels[order[:cut1]] = 0
        labels[order[cut1:cut2]] = 1
        labels[order[cut2:]] = 2

        t1 = float(targets[order[cut1 - 1]]
                   ) if cut1 > 0 else float(targets.min())
        t2 = float(targets[order[cut2 - 1]]
                   ) if cut2 > 0 else float(targets.max())
        boundaries = (t1, t2)
    elif mode == "quantile":
        q1, q2 = np.quantile(targets, [1.0 / 3.0, 2.0 / 3.0])
        if q2 <= q1:
            q2 = q1 + 1e-12
        boundaries = (float(q1), float(q2))
        labels = np.digitize(
            targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    else:
        raise ValueError(f"Unknown label mode: {mode}")

    return labels, boundaries


def balance_loaded_data(
    edge_strength,
    neighbor_idx,
    coords: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    mode: str = "undersample",
):
    counts = np.bincount(labels, minlength=3)
    if counts.min() <= 0:
        raise ValueError(
            f"Cannot balance because at least one class is empty: {counts.tolist()}")

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

    if isinstance(edge_strength, list):
        edge_strength_bal = [edge_strength[int(i)] for i in kept_idx]
        neighbor_idx_bal = [neighbor_idx[int(i)] for i in kept_idx]
    else:
        edge_strength_bal = edge_strength[kept_idx]
        neighbor_idx_bal = neighbor_idx[kept_idx]

    return (
        edge_strength_bal,
        neighbor_idx_bal,
        coords[kept_idx],
        targets[kept_idx],
        labels[kept_idx],
        original_indices[kept_idx],
        counts,
        np.bincount(labels[kept_idx], minlength=3),
    )


def compute_class_weights(labels: np.ndarray, num_classes: int = 3, power: float = 0.35):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** float(power)
    weights = weights / weights.mean()
    return weights, counts


def build_graph_node_classification(
    interaction_matrix,
    positions,
    node_labels,
    threshold=0.1,
    top_k=None,
    undirected=True,
):
    positions = np.asarray(positions, dtype=np.float32)
    node_labels = np.asarray(node_labels)
    n_nodes = positions.shape[0]

    if positions.ndim != 2:
        raise ValueError(
            f"Expected positions shape (N, D), got {positions.shape}.")
    if node_labels.shape[0] != n_nodes:
        raise ValueError(
            f"Expected {n_nodes} node labels, got {node_labels.shape[0]}.")

    edge_src = []
    edge_dst = []
    edge_w = []

    if isinstance(interaction_matrix, tuple):
        neighbor_idx, edge_strength = interaction_matrix
        if len(neighbor_idx) != n_nodes or len(edge_strength) != n_nodes:
            raise ValueError(
                f"Expected neighbor/edge rows={n_nodes}, got {len(neighbor_idx)} and {len(edge_strength)}."
            )

        dense_mode = isinstance(neighbor_idx, np.ndarray)

        if dense_mode:
            if neighbor_idx.shape != edge_strength.shape:
                raise ValueError(
                    f"neighbor_idx and edge_strength must have same shape. Got {neighbor_idx.shape} and {edge_strength.shape}."
                )

            use_k = neighbor_idx.shape[1] if top_k is None else min(int(top_k), neighbor_idx.shape[1])
            for i in range(n_nodes):
                for j_pos in range(use_k):
                    j = int(neighbor_idx[i, j_pos])
                    w = float(edge_strength[i, j_pos])
                    if j < 0 or j >= n_nodes or j == i:
                        continue
                    if w <= threshold:
                        continue
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_w.append(w)
        else:
            for i in range(n_nodes):
                row_idx = np.asarray(neighbor_idx[i], dtype=np.int64)
                row_w = np.asarray(edge_strength[i], dtype=np.float32)
                if row_idx.shape[0] != row_w.shape[0]:
                    raise ValueError(
                        f"Ragged row mismatch at i={i}: idx={row_idx.shape[0]} vs w={row_w.shape[0]}"
                    )

                if top_k is not None and int(top_k) > 0 and row_idx.shape[0] > int(top_k):
                    row_idx = row_idx[:int(top_k)]
                    row_w = row_w[:int(top_k)]

                for j, w in zip(row_idx, row_w):
                    j = int(j)
                    w = float(w)
                    if j < 0 or j >= n_nodes or j == i:
                        continue
                    if w <= threshold:
                        continue
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_w.append(w)
    else:
        matrix = np.asarray(interaction_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != n_nodes or matrix.shape[1] != n_nodes:
            raise ValueError(
                "Dense interaction matrix must be shape (N, N). "
                f"Got {matrix.shape} for N={n_nodes}."
            )

        for i in range(n_nodes):
            row = matrix[i].copy()
            row[i] = -np.inf

            if top_k is not None and int(top_k) > 0:
                k_eff = min(int(top_k), n_nodes)
                candidates = np.argpartition(row, -k_eff)[-k_eff:]
            else:
                candidates = np.arange(n_nodes)

            for j in candidates:
                w = float(matrix[i, j])
                if i == j:
                    continue
                if w <= threshold:
                    continue
                edge_src.append(i)
                edge_dst.append(int(j))
                edge_w.append(w)

    if undirected:
        rev_src = edge_dst.copy()
        rev_dst = edge_src.copy()
        rev_w = edge_w.copy()
        edge_src.extend(rev_src)
        edge_dst.extend(rev_dst)
        edge_w.extend(rev_w)

    if len(edge_src) == 0:
        raise ValueError(
            "No graph edges were created. Lower threshold or increase graph-k.")

    x = torch.tensor(positions, dtype=torch.float32)
    y = torch.tensor(node_labels, dtype=torch.long)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_w, dtype=torch.float32).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_index_dataloaders(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    node_batch_size: int,
):
    n = len(labels)
    indices = np.arange(n)
    class_counts = np.bincount(labels, minlength=3)
    can_stratify = class_counts.min() >= 2

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    val_fraction = val_ratio / (1.0 - train_ratio)
    temp_labels = labels[temp_idx]
    temp_counts = np.bincount(temp_labels, minlength=3)
    temp_can_stratify = temp_counts.min() >= 2

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction,
        random_state=seed,
        stratify=temp_labels if (can_stratify and temp_can_stratify) else None,
    )

    if node_batch_size is None or int(node_batch_size) <= 0:
        train_bs = max(1, len(train_idx))
        val_bs = max(1, len(val_idx))
        test_bs = max(1, len(test_idx))
    else:
        train_bs = min(int(node_batch_size), max(1, len(train_idx)))
        val_bs = min(int(node_batch_size), max(1, len(val_idx)))
        test_bs = min(int(node_batch_size), max(1, len(test_idx)))

    split_stats = {
        "train_counts": np.bincount(labels[train_idx], minlength=3),
        "val_counts": np.bincount(labels[val_idx], minlength=3),
        "test_counts": np.bincount(labels[test_idx], minlength=3),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_batch_size": int(train_bs),
        "val_batch_size": int(val_bs),
        "test_batch_size": int(test_bs),
    }
    return train_idx, val_idx, test_idx, split_stats


def parse_num_neighbors(text: str):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 0:
        return [20, 10]
    out = [int(p) for p in parts]
    return out


def build_neighbor_loaders(
    data: Data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    node_batch_size: int,
    num_neighbors,
):
    if node_batch_size is None or int(node_batch_size) <= 0:
        bs = 1024
    else:
        bs = int(node_batch_size)

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=torch.as_tensor(train_idx, dtype=torch.long),
        batch_size=min(bs, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=torch.as_tensor(val_idx, dtype=torch.long),
        batch_size=min(bs, max(1, len(val_idx))),
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=torch.as_tensor(test_idx, dtype=torch.long),
        batch_size=min(bs, max(1, len(test_idx))),
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


class GATNodeClassifier(nn.Module):
    def __init__(self, in_channels=3, hidden=64, num_classes=3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=4, dropout=0.3)
        self.gat2 = GATConv(hidden * 4, hidden, heads=1,
                            concat=False, dropout=0.3)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.lin(x)
        return x


def run_epoch(model, data, loader, optimizer, loss_fn, train: bool):
    model.train() if train else model.eval()

    if len(loader) == 0:
        return 0.0, 0.0, 0.0

    total_loss = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            logits = model(batch.x, batch.edge_index)
            n_seed = int(batch.batch_size)
            seed_logits = logits[:n_seed]
            seed_y = batch.y[:n_seed]
            loss = loss_fn(seed_logits, seed_y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            pred = torch.argmax(seed_logits, dim=1)
            true = seed_y
            n = n_seed

            total_loss += float(loss.item()) * n
            total_n += n
            y_true_all.append(true.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

    if total_n <= 0:
        return 0.0, 0.0, 0.0

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    return float(total_loss / total_n), acc, f1


@torch.no_grad()
def evaluate_test(model, data, loader, loss_fn):
    model.eval()

    if len(loader) == 0:
        return 0.0, 0.0, 0.0

    total_loss = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []

    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        logits = model(batch.x, batch.edge_index)
        n_seed = int(batch.batch_size)
        seed_logits = logits[:n_seed]
        seed_y = batch.y[:n_seed]
        loss = loss_fn(seed_logits, seed_y)
        pred = torch.argmax(seed_logits, dim=1)
        true = seed_y
        n = n_seed

        total_loss += float(loss.item()) * n
        total_n += n
        y_true_all.append(true.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    if total_n <= 0:
        return 0.0, 0.0, 0.0

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    return float(total_loss / total_n), acc, f1


def save_metrics_csv(rows, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss",
                        "train_acc", "val_acc", "train_f1", "val_f1", "lr"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Node classification with Hi-C graph + coordinates + class labels.")
    parser.add_argument("--data-dir", type=str, default=CONFIG["data_subdir"])
    parser.add_argument(
        "--data-k",
        type=int,
        default=CONFIG["data_k"],
        help="Top-K interactions retained per row while loading. Use 0 to keep all positive interactions per row.",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=CONFIG["graph_k"],
        help="Top-K neighbors per node during graph build. Use 0 to keep all available neighbors.",
    )
    parser.add_argument("--max-nodes", type=int, default=CONFIG["max_nodes"])
    parser.add_argument("--node-batch-size", type=int, default=CONFIG["node_batch_size"])
    parser.add_argument("--num-neighbors", type=str, default=CONFIG["num_neighbors"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--label-mode", choices=[
                        "fixed_thresholds", "balanced_rank", "quantile"], default=CONFIG["label_mode"])
    parser.add_argument("--low-max", type=float, default=CONFIG["low_max"])
    parser.add_argument("--med-max", type=float, default=CONFIG["med_max"])
    parser.add_argument("--no-balance-loaded-data", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--hidden-dim", type=int, default=CONFIG["hidden_dim"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float,
                        default=CONFIG["weight_decay"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float,
                        default=CONFIG["train_ratio"])
    parser.add_argument("--val-ratio", type=float, default=CONFIG["val_ratio"])
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main():
    global TQDM_DISABLE

    args = parse_args()
    if args.no_tqdm:
        TQDM_DISABLE = True
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    log_path = init_log_file(base_dir, prefix="node_cls")

    console_section("RUN START")
    console_kv("device", device)
    console_kv("seed", args.seed)
    console_kv("log_file", log_path)
    console_kv("data_dir", data_dir)
    console_kv("data_k", args.data_k)
    console_kv("graph_k", args.graph_k)
    console_kv("max_nodes", args.max_nodes)
    console_kv("node_batch_size", args.node_batch_size)
    console_kv("num_neighbors", args.num_neighbors)
    console_kv("threshold", args.threshold)
    console_kv("label_mode", args.label_mode)
    if args.label_mode == "fixed_thresholds":
        console_kv("low_max", args.low_max)
        console_kv("med_max", args.med_max)
    console_kv("balance_loaded_data", not args.no_balance_loaded_data)
    console_kv("num_epochs", args.num_epochs)
    console_kv("patience", args.patience)
    console_kv("hidden_dim", args.hidden_dim)
    console_kv("learning_rate", fmt_sci(args.lr))
    console_kv("weight_decay", fmt_sci(args.weight_decay))

    console_log("\n[1/4] Loading preprocessed data ...")
    edge_strength, neighbor_idx, coords, targets, data_paths, load_stats = load_preprocessed_data_for_graph(
        data_dir=data_dir,
        k=args.data_k,
        max_nodes=args.max_nodes,
    )

    labels, boundaries = make_labels(
        targets, mode=args.label_mode, low_max=args.low_max, med_max=args.med_max)
    original_indices = np.arange(len(labels), dtype=np.int64)

    if not args.no_balance_loaded_data:
        (
            edge_strength,
            neighbor_idx,
            coords,
            targets,
            labels,
            original_indices,
            before_counts,
            after_counts,
        ) = balance_loaded_data(
            edge_strength,
            neighbor_idx,
            coords,
            targets,
            labels,
            original_indices,
            mode=CONFIG["balance_mode"],
        )
        console_kv("balance counts before", {
                   CLASS_NAMES[i]: int(before_counts[i]) for i in range(3)})
        console_kv("balance counts after", {
                   CLASS_NAMES[i]: int(after_counts[i]) for i in range(3)})

    class_weights, class_counts = compute_class_weights(
        labels, num_classes=3, power=CONFIG["class_weight_power"])
    class_frac = class_counts / max(class_counts.sum(), 1)

    console_kv("weights file", data_paths["weights"])
    console_kv("coords file", data_paths["coords"])
    console_kv("targets file", data_paths["targets"])
    console_kv("rows available", load_stats["rows_before_cap"])
    console_kv("rows used after max_nodes", load_stats["rows_after_cap"])
    if load_stats["max_nodes"] is not None and load_stats["max_nodes"] > 0:
        console_kv("max_nodes cap", load_stats["max_nodes"])
    if isinstance(edge_strength, list):
        total_pos = int(sum(len(r) for r in edge_strength))
        avg_pos = float(total_pos / max(len(edge_strength), 1))
        console_kv("interaction storage", "ragged (all positive interactions)")
        console_kv("positive interactions", total_pos)
        console_kv("avg interactions/node", fmt_float(avg_pos))
    else:
        console_kv("interaction storage", "dense top-k matrix")
        console_kv("edge_strength shape", edge_strength.shape)
        console_kv("neighbor_idx shape", neighbor_idx.shape)
    console_kv("coords shape", coords.shape)
    console_kv("targets shape", targets.shape)
    console_kv("target range",
               f"[{fmt_float(targets.min())}, {fmt_float(targets.max())}]")
    console_kv("label boundaries",
               f"low<= {fmt_float(boundaries[0])}, med<= {fmt_float(boundaries[1])}, else high")
    console_kv("class counts", {CLASS_NAMES[i]: int(
        class_counts[i]) for i in range(3)})
    console_kv("class fractions", {
               CLASS_NAMES[i]: f"{class_frac[i]*100:.2f}%" for i in range(3)})
    console_kv("class weights", {CLASS_NAMES[i]: fmt_float(
        class_weights[i]) for i in range(3)})
    console_kv("rows after balance/label", len(labels))

    console_log("\n[2/4] Building graph data ...")
    data = build_graph_node_classification(
        interaction_matrix=(neighbor_idx, edge_strength),
        positions=coords,
        node_labels=labels,
        threshold=args.threshold,
        top_k=args.graph_k,
        undirected=True,
    )

    train_idx, val_idx, test_idx, split_stats = build_index_dataloaders(
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        node_batch_size=args.node_batch_size,
    )
    neighbor_sizes = parse_num_neighbors(args.num_neighbors)
    train_loader, val_loader, test_loader = build_neighbor_loaders(
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        node_batch_size=args.node_batch_size,
        num_neighbors=neighbor_sizes,
    )

    console_kv("graph nodes", data.num_nodes)
    console_kv("graph edges", data.edge_index.size(1))
    console_kv("node feature dim", data.x.size(1))
    console_kv("split train counts", {CLASS_NAMES[i]: int(
        split_stats["train_counts"][i]) for i in range(3)})
    console_kv("split val counts", {CLASS_NAMES[i]: int(
        split_stats["val_counts"][i]) for i in range(3)})
    console_kv("split test counts", {CLASS_NAMES[i]: int(
        split_stats["test_counts"][i]) for i in range(3)})
    console_kv("split sizes", f"train={split_stats['train_size']} val={split_stats['val_size']} test={split_stats['test_size']}")
    console_kv("loader batch sizes", f"train={split_stats['train_batch_size']} val={split_stats['val_batch_size']} test={split_stats['test_batch_size']}")
    console_kv("neighbor sampling", neighbor_sizes)

    console_log("\n[3/4] Building GAT model ...")
    model = GATNodeClassifier(
        in_channels=data.x.size(1),
        hidden=args.hidden_dim,
        num_classes=int(np.max(labels) + 1),
    ).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    console_kv("trainable_parameters", f"{total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=CONFIG["label_smoothing"],
    )

    metrics_rows = []
    best_val_f1 = -1.0
    best_epoch = 0
    stagnant = 0
    ckpt_path = base_dir / "best_model_node_cls.pt"

    console_log("\n[4/4] Training ...")
    console_section("TRAINING")
    console_log(
        f"{'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>12}  {'TrainAcc':>10}  {'ValAcc':>10}  {'TrainF1':>10}  {'ValF1':>10}  {'LR':>12}"
    )
    console_rule("-")

    epoch_iterator = tqdm(
        range(1, args.num_epochs + 1),
        desc="Epochs",
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
    )

    for epoch in epoch_iterator:
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, data, train_loader, optimizer, loss_fn, train=True)
        va_loss, va_acc, va_f1 = run_epoch(
            model, data, val_loader, optimizer, loss_fn, train=False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        metrics_rows.append(
            [epoch, tr_loss, va_loss, tr_acc, va_acc, tr_f1, va_f1, lr_now])

        epoch_iterator.set_postfix(
            val_f1=fmt_float(va_f1),
            val_acc=fmt_float(va_acc),
            lr=fmt_sci(lr_now),
        )

        console_log(
            f"{epoch:>6}  {tr_loss:>12.{CONFIG['float_precision']}f}  {va_loss:>12.{CONFIG['float_precision']}f}  "
            f"{tr_acc:>10.{CONFIG['float_precision']}f}  {va_acc:>10.{CONFIG['float_precision']}f}  "
            f"{tr_f1:>10.{CONFIG['float_precision']}f}  {va_f1:>10.{CONFIG['float_precision']}f}  "
            f"{lr_now:>12.{CONFIG['sci_precision']}e}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            stagnant = 0
            torch.save(model.state_dict(), ckpt_path)
            console_log(
                f"[BEST] epoch={epoch:03d} val_macro_f1={fmt_float(best_val_f1)} checkpoint_saved={ckpt_path}")
        else:
            stagnant += 1
            if stagnant >= args.patience:
                console_log(
                    f"[EARLY-STOP] epoch={epoch:03d} no_improvement_for={args.patience} epochs")
                break

    model.load_state_dict(torch.load(
        ckpt_path, map_location=device, weights_only=True))

    test_loss, test_acc, test_f1 = evaluate_test(model, data, test_loader, loss_fn)
    metrics_csv = log_path.with_name(f"{log_path.stem}_metrics.csv")
    save_metrics_csv(metrics_rows, metrics_csv)

    console_log("")
    console_section("TEST RESULTS")
    console_kv("best_epoch", best_epoch)
    console_kv("best_val_macro_f1", fmt_float(best_val_f1))
    console_kv("test_loss", fmt_float(test_loss))
    console_kv("test_accuracy", fmt_float(test_acc))
    console_kv("test_macro_f1", fmt_float(test_f1))
    console_kv("metrics_csv", metrics_csv)
    console_kv("checkpoint", ckpt_path)


if __name__ == "__main__":
    main()
