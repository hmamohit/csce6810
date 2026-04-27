import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm


SEED = 42
TQDM_DISABLE = not sys.stdout.isatty()
LOG_FILE_PATH: Optional[Path] = None

CLASS_NAMES = ["Low", "Medium", "High"]

CONFIG = {
    "data_k": 512,
    "data_subdir": "data/preprocessing",
    "log_subdir": "logs",
    "float_precision": 6,
    "sci_precision": 6,
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 600,
    "patience": 30,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "label_smoothing": 0.02,
    "class_weight_power": 0.35,
    "use_balanced_sampler": True,
    "sampler_power": 0.5,
    "balance_loaded_data": True,
    "balance_mode": "undersample",
    "label_mode": "fixed_thresholds",
    "low_max": 0.01,
    "med_max": 1.0,
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
    if n_meta <= 0:
        raise ValueError(
            f"Empty coords/targets after loading: coords={n_coords}, targets={n_targets}")

    if n_coords != n_targets:
        console_log(
            "[ALIGN] coords/targets row mismatch detected. "
            f"coords={n_coords}, targets={n_targets}. Using first {n_meta} rows."
        )

    weights_rows = []
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

            if vals.size > 0:
                if vals.size >= k:
                    idx = np.argpartition(vals, -k)[-k:]
                    top = vals[idx]
                    row_out[:] = top[np.argsort(top)[::-1]]
                else:
                    order = np.argsort(vals)[::-1]
                    row_out[:vals.size] = vals[order]

            weights_rows.append(row_out)

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
    coords = coords[:n_shared]
    targets = targets[:n_shared]

    paths = {
        "weights": weights_path,
        "coords": coords_path,
        "targets": targets_path,
    }
    return weights, coords, targets, paths


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
    weights: np.ndarray,
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

    return (
        weights[kept_idx],
        coords[kept_idx],
        targets[kept_idx],
        labels[kept_idx],
        original_indices[kept_idx],
        counts,
        np.bincount(labels[kept_idx], minlength=3),
    )


class ExpClassDataset(Dataset):
    def __init__(self, weights: np.ndarray, coords: np.ndarray, labels: np.ndarray):
        self.weights = torch.from_numpy(weights)
        self.coords = torch.from_numpy(coords)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.weights[idx], self.coords[idx], self.labels[idx]


class ExpTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_neighbors: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_neighbors = num_neighbors

        self.weight_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        tokens = self.weight_proj(weights.unsqueeze(-1))
        positions = torch.arange(self.num_neighbors, device=weights.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)

        coord_ctx = self.coord_proj(coords)
        tokens = tokens + coord_ctx.unsqueeze(1)

        out = self.transformer(tokens)
        agg = out.mean(dim=1)
        combined = torch.cat([agg, coord_ctx], dim=-1)
        return self.head(combined)


def build_dataloaders(
    weights: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    use_balanced_sampler: bool,
    sampler_power: float,
    seed: int,
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

    dataset = ExpClassDataset(weights, coords, labels)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    train_labels = labels[train_idx]
    train_counts = np.bincount(train_labels, minlength=3).astype(np.float64)

    if use_balanced_sampler:
        train_sample_weights = 1.0 / \
            np.maximum(train_counts[train_labels], 1.0) ** sampler_power
        train_sample_weights = torch.as_tensor(
            train_sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, sampler=train_sampler, shuffle=False, **kwargs)
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


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    train: bool,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_total = 0
    y_true_all = []
    y_pred_all = []

    phase = "Train" if train else "Val"
    desc = f"{phase} {epoch}/{total_epochs}" if (
        epoch is not None and total_epochs is not None) else f"{phase} Batches"

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
        for weights, coords, labels in iterator:
            weights = weights.to(device)
            coords = coords.to(device)
            labels = labels.to(device)

            logits = model(weights, coords)
            loss = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = torch.argmax(logits, dim=1)

            n = len(labels)
            total_loss += loss.item() * n
            n_total += n

            y_true_all.append(labels.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

            running_acc = (np.concatenate(y_true_all) ==
                           np.concatenate(y_pred_all)).mean()
            iterator.set_postfix(loss=fmt_float(
                total_loss / n_total), acc=fmt_float(running_acc))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    avg_loss = total_loss / max(n_total, 1)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0))

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
    checkpoint_path: str,
    label_smoothing: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=label_smoothing,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    stagnant = 0
    final_epoch = 0

    console_log("")
    console_section("TRAINING", width=100)
    console_log(
        f"{'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>12}  {'TrainAcc':>10}  {'ValAcc':>10}  {'TrainF1':>10}  {'ValF1':>10}  {'LR':>12}"
    )
    console_rule("-", width=100)

    epoch_iterator = tqdm(range(1, num_epochs + 1), desc="Epochs",
                          dynamic_ncols=True, disable=TQDM_DISABLE)

    for epoch in epoch_iterator:
        final_epoch = epoch

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            train=True,
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
            epoch=epoch,
            total_epochs=num_epochs,
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_iterator.set_postfix(val_f1=fmt_float(
            va_f1), val_acc=fmt_float(va_acc), lr=fmt_sci(current_lr))

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
            console_log(
                f"[BEST] epoch={epoch:03d} val_macro_f1={fmt_float(best_val_f1)} checkpoint_saved={checkpoint_path}")
        else:
            stagnant += 1
            if stagnant >= patience:
                console_log(
                    f"[EARLY-STOP] epoch={epoch:03d} no_improvement_for={patience} epochs")
                break

    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    console_log(
        f"[CHECKPOINT] Loaded best checkpoint from epoch {best_epoch:03d} (val_macro_f1={fmt_float(best_val_f1)})")
    console_log(
        f"[SUMMARY] epochs_ran={final_epoch} best_epoch={best_epoch} best_val_macro_f1={fmt_float(best_val_f1)}")

    return history, best_epoch, best_val_f1


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        iterator = tqdm(test_loader, desc="Test Batches",
                        leave=False, dynamic_ncols=True, disable=TQDM_DISABLE)
        for weights, coords, labels in iterator:
            logits = model(weights.to(device), coords.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(labels.numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0))
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
    console_section("TEST RESULTS (CLASSIFICATION)", width=86)
    console_kv("samples", len(test_loader.dataset))
    console_kv("accuracy", fmt_float(acc))
    console_kv("macro_f1", fmt_float(macro_f1))
    console_rule("-", width=86)

    console_log(
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
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
        console_log(
            f"True-{row_name:<5} {cm[i, 0]:>9d} {cm[i, 1]:>9d} {cm[i, 2]:>10d}")
    console_rule("-", width=86)

    return y_true, y_pred, cm


def predict_single_node_class(model, weights_array: np.ndarray, coords_array: np.ndarray, device):
    model.eval()
    w = torch.from_numpy(weights_array.astype(
        np.float32)).unsqueeze(0).to(device)
    c = torch.from_numpy(coords_array.astype(
        np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(w, c)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_class, confidence, probs


def save_metrics_csv(history: dict, csv_path: Path):
    rows = zip(
        range(1, len(history["train_loss"]) + 1),
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        history["train_f1"],
        history["val_f1"],
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss",
                        "train_acc", "val_acc", "train_f1", "val_f1"])
        writer.writerows(rows)


def save_training_curves(history: dict, plot_path: Path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["train_f1"], label="train")
    axes[2].plot(epochs, history["val_f1"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)


def save_confusion_matrix_plot(cm: np.ndarray, class_names, plot_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center", color="black")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)


def main():
    global TQDM_DISABLE

    parser = argparse.ArgumentParser(
        description="Simple gene expression 3-class trainer (Low/Medium/High)")
    parser.add_argument("--label-mode", choices=[
                        "fixed_thresholds", "balanced_rank", "quantile"], default=CONFIG["label_mode"])
    parser.add_argument("--low-max", type=float, default=CONFIG["low_max"])
    parser.add_argument("--med-max", type=float, default=CONFIG["med_max"])
    parser.add_argument("--num-epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--num-sample-infer", type=int, default=10)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.no_tqdm:
        TQDM_DISABLE = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / CONFIG["data_subdir"]
    log_path = init_log_file(base_dir, prefix="train_cls")

    console_section("RUN START", width=86)
    console_kv("device", device)
    console_kv("seed", args.seed)
    console_kv("label_mode", args.label_mode)
    if args.label_mode == "fixed_thresholds":
        console_kv("low_max", args.low_max)
        console_kv("med_max", args.med_max)
    console_kv("log_file", log_path)
    console_kv("data_topk_k", CONFIG["data_k"])
    console_kv("batch_size", args.batch_size)
    console_kv("num_epochs", args.num_epochs)
    console_kv("patience", args.patience)
    console_kv("learning_rate", fmt_sci(args.lr))
    console_kv("weight_decay", fmt_sci(CONFIG["weight_decay"]))
    console_kv("label_smoothing", CONFIG["label_smoothing"])
    console_kv("class_weight_power", CONFIG["class_weight_power"])
    console_kv("use_balanced_sampler", CONFIG["use_balanced_sampler"])
    console_kv("sampler_power", CONFIG["sampler_power"])
    console_kv("balance_loaded_data", CONFIG["balance_loaded_data"])
    console_kv("balance_mode", CONFIG["balance_mode"])

    console_log("\n[1/4] Loading preprocessed data ...")
    weights, coords, targets, data_paths = load_preprocessed_data(
        data_dir=data_dir, k=CONFIG["data_k"])

    labels, boundaries = make_labels(
        targets, mode=args.label_mode, low_max=args.low_max, med_max=args.med_max)
    targets_full = targets.copy()
    original_indices = np.arange(len(targets), dtype=np.int64)

    if CONFIG["balance_loaded_data"]:
        (
            weights,
            coords,
            targets,
            labels,
            original_indices,
            before_counts,
            after_counts,
        ) = balance_loaded_data(
            weights,
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
    console_kv("weights shape", weights.shape)
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

    seq_len = weights.shape[1]

    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, split_stats = build_dataloaders(
        weights=weights,
        coords=coords,
        labels=labels,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        batch_size=args.batch_size,
        use_balanced_sampler=CONFIG["use_balanced_sampler"],
        sampler_power=CONFIG["sampler_power"],
        seed=args.seed,
    )

    console_kv("dataset splits",
               f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    console_kv("split train counts", {CLASS_NAMES[i]: int(
        split_stats["train_counts"][i]) for i in range(3)})
    console_kv("split val counts", {CLASS_NAMES[i]: int(
        split_stats["val_counts"][i]) for i in range(3)})
    console_kv("split test counts", {CLASS_NAMES[i]: int(
        split_stats["test_counts"][i]) for i in range(3)})

    console_log("\n[2/4] Building model ...")
    model = ExpTransformerClassifier(
        num_neighbors=seq_len,
        num_classes=3,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    console_kv("trainable_parameters", f"{total_params:,}")

    console_log("\n[3/4] Training ...")
    history, best_epoch, best_val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=CONFIG["weight_decay"],
        num_epochs=args.num_epochs,
        patience=args.patience,
        checkpoint_path="best_model_cls.pt",
        label_smoothing=CONFIG["label_smoothing"],
    )

    metrics_csv_path = log_path.with_name(f"{log_path.stem}_metrics.csv")
    curves_plot_path = log_path.with_name(f"{log_path.stem}_curves.png")
    save_metrics_csv(history, metrics_csv_path)
    save_training_curves(history, curves_plot_path)

    console_kv("metrics_csv", metrics_csv_path)
    console_kv("training_curves", curves_plot_path)
    console_kv("best_epoch", best_epoch)
    console_kv("best_val_macro_f1", fmt_float(best_val_f1))

    console_log("\n[4/4] Evaluating on test set ...")
    y_true, y_pred, cm = evaluate_model(model, test_loader, device)

    cm_plot_path = log_path.with_name(f"{log_path.stem}_test_cm.png")
    save_confusion_matrix_plot(cm, CLASS_NAMES, cm_plot_path)
    console_kv("test_confusion_matrix_plot", cm_plot_path)

    console_log("")
    console_section("SINGLE-NODE INFERENCE (RANDOM TEST NODES)", width=100)

    test_size = len(test_loader.dataset)
    n_demo = min(max(args.num_sample_infer, 1), test_size)
    rng = np.random.default_rng(args.seed)
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
        sample_weights_t, sample_coords_t, sample_label_t = test_loader.dataset[int(
            local_idx)]

        sample_weights = sample_weights_t.cpu().numpy()
        sample_coords = sample_coords_t.cpu().numpy()
        true_label = int(sample_label_t.item())

        pred_label, confidence, probs = predict_single_node_class(
            model,
            sample_weights,
            sample_coords,
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
