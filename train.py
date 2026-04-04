import torch
import torch.nn as nn
import numpy as np
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm.auto import tqdm
from model import ExpTransformer
from data_loader import build_dataloaders
from plots import plot_training_curves, plot_predictions


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
TQDM_DISABLE = not sys.stdout.isatty()
LOG_FILE_PATH: Optional[Path] = None

CONFIG = {
    "data_k":       512,
    "data_subdir":  "data/preprocessing",
    "log_subdir":   "logs",
    "float_precision": 8,
    "sci_precision":  6,
    "hidden_dim":  128,
    "num_heads":    4,
    "num_layers":   3,
    "dropout":      0.1,
    "batch_size":   1024,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "num_epochs":   500,
    "patience":     20,
    "train_ratio":  0.80,
    "val_ratio":    0.10,
}


def console_log(message: str = ""):
    if TQDM_DISABLE:
        print(message)
    else:
        tqdm.write(message)

    if LOG_FILE_PATH is not None:
        with LOG_FILE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def init_log_file(base_dir: Path) -> Path:
    global LOG_FILE_PATH
    log_dir = base_dir / CONFIG["log_subdir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"
    log_path.write_text("", encoding="utf-8")
    LOG_FILE_PATH = log_path
    return log_path


def fmt_float(value: float) -> str:
    return f"{float(value):.{CONFIG['float_precision']}f}"


def fmt_sci(value: float) -> str:
    return f"{float(value):.{CONFIG['sci_precision']}e}"


def console_rule(char: str = "-", width: int = 72):
    console_log(char * width)


def console_section(title: str, width: int = 72):
    console_rule("=", width)
    console_log(title)
    console_rule("=", width)


def console_kv(key: str, value):
    console_log(f"{key:<22}: {value}")


def load_preprocessed_data(data_dir: Path, k: int):
    weights_path = data_dir / f"contact_matrix_{k}.txt"
    coords_path = data_dir / "coordinates.txt"
    targets_path = data_dir / "gene_exp.txt"

    missing = [
        str(path)
        for path in (weights_path, coords_path, targets_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessed input files:\n" + "\n".join(missing)
        )

    weights = np.loadtxt(weights_path, dtype=np.float32)
    coords = np.loadtxt(coords_path, dtype=np.float32)
    targets = np.loadtxt(targets_path, dtype=np.float32)

    if weights.ndim == 1:
        weights = np.expand_dims(weights, axis=0)

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

    n_weights = len(weights)
    n_coords = len(coords)
    n_targets = len(targets)
    if not (n_weights == n_coords == n_targets):
        raise ValueError(
            "Row mismatch among loaded datasets: "
            f"weights={n_weights}, coords={n_coords}, targets={n_targets}"
        )

    paths = {
        "weights": weights_path,
        "coords": coords_path,
        "targets": targets_path,
    }
    return weights, coords, targets, paths

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
    total_loss, total_mae = 0.0, 0.0
    total_examples = 0

    phase = "Train" if train else "Val"
    if epoch is not None and total_epochs is not None:
        desc = f"{phase} {epoch}/{total_epochs}"
    else:
        desc = f"{phase} Batches"

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
        for weights, coords, targets in iterator:
            weights = weights.to(device)
            coords = coords.to(device)
            targets = targets.to(device)

            preds = model(weights, coords)
            loss = loss_fn(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(targets)
            total_mae += (preds - targets).abs().sum().item()
            total_examples += len(targets)

            running_mse = total_loss / total_examples
            running_mae = total_mae / total_examples
            iterator.set_postfix(mse=fmt_float(running_mse), mae=fmt_float(running_mae))

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    num_epochs:    int = 100,
    patience:      int = 15,
    checkpoint_path: str = "best_model.pt",
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [],
               "train_mae": [], "val_mae": []}
    best_val = float("inf")
    best_epoch = 0
    stagnant = 0
    final_epoch = 0

    console_log("")
    console_section("TRAINING", width=86)
    console_log(f"{'Epoch':>6}  {'Train MSE':>14}  {'Val MSE':>14}  {'Train MAE':>14}  {'Val MAE':>14}  {'LR':>12}")
    console_rule("-", width=86)

    epoch_iterator = tqdm(
        range(1, num_epochs + 1),
        desc="Epochs",
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
    )

    for epoch in epoch_iterator:
        final_epoch = epoch
        tr_loss, tr_mae = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            train=True,
            epoch=epoch,
            total_epochs=num_epochs,
        )
        va_loss, va_mae = run_epoch(
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
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(va_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_iterator.set_postfix(
            val_mse=fmt_float(va_loss),
            val_mae=fmt_float(va_mae),
            lr=fmt_sci(current_lr),
        )
        console_log(
            f"{epoch:>6}  {tr_loss:>14.{CONFIG['float_precision']}f}  "
            f"{va_loss:>14.{CONFIG['float_precision']}f}  "
            f"{tr_mae:>14.{CONFIG['float_precision']}f}  "
            f"{va_mae:>14.{CONFIG['float_precision']}f}  "
            f"{current_lr:>12.{CONFIG['sci_precision']}e}"
        )

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            stagnant = 0
            torch.save(model.state_dict(), checkpoint_path)
            console_log(
                f"[BEST] epoch={epoch:03d} val_mse={fmt_float(best_val)} checkpoint_saved={checkpoint_path}")
        else:
            stagnant += 1
            if stagnant >= patience:
                console_log(
                    f"[EARLY-STOP] epoch={epoch:03d} no_improvement_for={patience} epochs")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    console_log(f"[CHECKPOINT] Loaded best checkpoint from epoch {best_epoch:03d} (val_mse={fmt_float(best_val)})")
    console_log(f"[SUMMARY] epochs_ran={final_epoch} best_epoch={best_epoch} best_val_mse={fmt_float(best_val)}")
    return history


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        test_iterator = tqdm(
            test_loader,
            desc="Test Batches",
            leave=False,
            dynamic_ncols=True,
            disable=TQDM_DISABLE,
        )
        for weights, coords, targets in test_iterator:
            preds = model(weights.to(device), coords.to(device))
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    rmse = mse ** 0.5
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    console_log("")
    console_section("TEST RESULTS", width=44)
    console_kv("samples", len(test_loader.dataset))
    console_kv("mse", fmt_float(mse))
    console_kv("rmse", fmt_float(rmse))
    console_kv("mae", fmt_float(mae))
    console_kv("r2", fmt_float(r2))
    console_rule("-", width=44)

    return preds.numpy(), targets.numpy()


def predict_single_node(model, weights_array: np.ndarray, coords_array: np.ndarray, device):
    model.eval()
    w = torch.from_numpy(weights_array.astype(np.float32)
                         ).unsqueeze(0).to(device)
    c = torch.from_numpy(coords_array.astype(np.float32)
                         ).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(w, c)
    return pred.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / CONFIG["data_subdir"]
    log_path = init_log_file(base_dir)

    console_section("RUN START")
    console_kv("device", device)
    console_kv("seed", SEED)
    console_kv("log_file", log_path)
    console_kv("float_precision", CONFIG["float_precision"])
    console_kv("data_topk_k", CONFIG["data_k"])
    console_kv("preprocess_dir", data_dir)
    console_kv("batch_size", CONFIG["batch_size"])
    console_kv("num_epochs", CONFIG["num_epochs"])
    console_kv("learning_rate", fmt_sci(CONFIG["lr"]))
    console_kv("weight_decay", fmt_sci(CONFIG["weight_decay"]))

    console_log("\n[1/4] Loading preprocessed data ...")
    weights, coords, targets, data_paths = load_preprocessed_data(
        data_dir=data_dir,
        k=CONFIG["data_k"],
    )
    console_kv("weights file", data_paths["weights"])
    console_kv("coords file", data_paths["coords"])
    console_kv("targets file", data_paths["targets"])
    console_kv("weights shape", weights.shape)
    console_kv("coords shape", coords.shape)
    console_kv("targets shape", targets.shape)
    console_kv("target range", f"[{fmt_float(targets.min())}, {fmt_float(targets.max())}]")

    seq_len = weights.shape[1]
    
    train_loader, val_loader, test_loader = build_dataloaders(
        weights, coords, targets,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        batch_size=CONFIG["batch_size"],
    )

    console_log("\n[2/4] Building model ...")
    model = ExpTransformer(
        num_neighbors=seq_len,
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    console_kv("trainable_parameters", f"{total_params:,}")

    console_log("\n[3/4] Training ...")
    history = train_model(
        model, train_loader, val_loader, device,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"],
    )
    plot_training_curves(history)

    console_log("\n[4/4] Evaluating on test set ...")
    preds, true_vals = evaluate_model(model, test_loader, device)
    plot_predictions(preds, true_vals)

    console_log("")
    console_section("SINGLE-NODE INFERENCE (RANDOM TEST NODES)", width=80)

    test_size = len(test_loader.dataset)
    n_demo = min(10, test_size)
    rng = np.random.default_rng(SEED)
    selected_indices = rng.choice(test_size, size=n_demo, replace=False)

    console_kv("sample_source", "random from test split")
    console_kv("num_nodes", n_demo)
    console_kv("indices", selected_indices.tolist())
    console_rule("-", width=80)
    console_log(f"{'Node#':>6}  {'TestIdx':>8}  {'True':>14}  {'Pred':>14}  {'AbsErr':>14}")
    console_rule("-", width=80)

    for i, test_idx in enumerate(selected_indices, start=1):
        sample_weights_t, sample_coords_t, true_value_t = test_loader.dataset[int(test_idx)]
        sample_weights = sample_weights_t.cpu().numpy()
        sample_coords = sample_coords_t.cpu().numpy()
        true_value = float(true_value_t.item())

        predicted = predict_single_node(model, sample_weights, sample_coords, device)
        abs_err = abs(predicted - true_value)

        console_log(
            f"{i:>6}  {int(test_idx):>8}  "
            f"{true_value:>14.{CONFIG['float_precision']}f}  "
            f"{predicted:>14.{CONFIG['float_precision']}f}  "
            f"{abs_err:>14.{CONFIG['float_precision']}f}"
        )

    console_rule("-", width=80)


if __name__ == "__main__":
    main()
