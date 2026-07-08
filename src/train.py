import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import torch
import torch.distributed as dist
import config
import optuna
import os
import sys
from model import VisionModel
from tqdm import tqdm
from data_loader import CustomDataset, GeneExpressionDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchmetrics.functional import pearson_corrcoef, mean_squared_error, mean_absolute_error, r2_score, spearman_corrcoef, mean_absolute_percentage_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = preds.double()
    targets = targets.double()

    pearson = pearson_corrcoef(preds, targets).item()
    spearman = spearman_corrcoef(preds, targets).item()
    r2 = r2_score(preds, targets).item()
    mse = mean_squared_error(preds, targets).item()
    mae = mean_absolute_error(preds, targets).item()
    mape = mean_absolute_percentage_error(preds, targets).item()

    return {"pearson": pearson, "spearman": spearman, "r2": r2, "mse": mse, "mae": mae, "mape": mape}


def make_warmup_then_plateau(optimizer, warmup_steps, base_lr=1e-2,
                             plateau_factor=0.5, plateau_patience=5, min_lr=1e-5):
    def warmup_fn(step):
        if warmup_steps == 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=plateau_factor,
        patience=plateau_patience, min_lr=min_lr,
    )
    return warmup_scheduler, plateau_scheduler


def train_loop(model, loader, device, criterion, optimizer):
    model.train()
    pbar = tqdm(loader, desc="Training", leave=True)

    total_loss = 0.0
    n_samples = 0
    for batch in pbar:
        if batch is None:
            continue
        feature, attention, target = batch
        feature = feature.to(device)
        attention = attention.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(feature, attention)

        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        bs = target.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    avg_loss = total_loss / n_samples
    return avg_loss


def evaluate_loop(model, loader, device, criterion, scheduler=None):
    model.eval()
    pbar = tqdm(loader, desc="Evaluating", leave=True)

    all_preds, all_targets = [], []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            feature, attention, target = batch
            feature = feature.to(device)
            attention = attention.to(device)
            target = target.to(device)

            pred = model(feature, attention)
            loss = criterion(pred, target)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            bs = target.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            all_preds.append(pred.detach().cpu().reshape(-1))
            all_targets.append(target.detach().cpu().reshape(-1))

    avg_loss = total_loss / n_samples
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = avg_loss
    if scheduler is not None:
        scheduler.step(metrics["loss"])

    return metrics


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


def set_seed(seed_v: int = 42):
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def ddp_setup():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requires CUDA/NCCL, but CUDA is not available.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Distributed training requires torchrun. Example: "
            "torchrun --standalone --nproc_per_node=<num_gpus> hicinterpolate.py --distributed --train --config <config>"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 20, shuffle: bool = False, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=set_seed,
            num_workers=20,
            persistent_workers=True,
            sampler=DistributedSampler(ds, shuffle=shuffle)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=set_seed,
            num_workers=20,
            persistent_workers=True
        )


def train(num_epochs=50, warmup_epochs=3, lr=1e-2, weight_decay=0.05,
          device=None, ckpt_path="best_model.pt"):

    num_epochs = config.NUM_EPOCHS
    lr = config.LR
    weight_decay = config.WEIGHT_DECAY
    ckpt_path = config.BEST_MODEL

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log = base_logger(config.LOG_FILENAME)
    batch_size = config.BATCH_SIZE

    train_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_train.csv', feature_dir=config.DATA_DIR,
                              feature_map=config.FEATURE_MAP)
    train_dict = train_cds._get_dataset()
    train_ds = GeneExpressionDataset(feature_list=train_dict)
    train_dl = get_dataloader(
        ds=train_ds, batch_size=batch_size, shuffle=True, isDistributed=config.IS_DISTRIBUTED)

    val_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_val.csv', feature_dir=config.DATA_DIR,
                            feature_map=config.FEATURE_MAP)
    val_dict = val_cds._get_dataset()
    val_ds = GeneExpressionDataset(feature_list=val_dict)
    val_dl = get_dataloader(ds=val_ds, batch_size=batch_size,
                            shuffle=False, isDistributed=config.IS_DISTRIBUTED)

    ftr_size = config.FTR_SIZE
    patch_size = config.PATCH_SIZE
    embed_dim = config.EMBED_DIM
    depth = config.DEPTH
    num_heads = config.NUM_HEAD
    mlp_ratio = config.MLP_RATIO
    dropout = config.DROPOUT
    attn_dropout = config.ATTENTION_DROPOUT
    model = VisionModel(ftr_size=ftr_size, patch_size=patch_size,
                        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, dropout=dropout,
                        attn_dropout=attn_dropout).to(device)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=5)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_loop(
            model, train_dl, device, criterion, optimizer=optimizer)
        val_metrics = evaluate_loop(
            model, val_dl, device, criterion, scheduler)

        current_lr = optimizer.param_groups[0]["lr"]
        status = f"Epoch {epoch:03d} | lr {current_lr:.2e} | train loss={train_loss:.4f} | val loss={val_metrics['loss']:.4f}; Pearson={val_metrics['pearson']:.4f}; Spearman={val_metrics['spearman']:.4f}; R2={val_metrics['r2']:.4f}; MSE={val_metrics['mse']:.4f}; MAE={val_metrics['mae']:.4f}; MAPE={val_metrics['mape']:.4f}"
        print(status)
        log.info(status)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            print(
                f"New best model found at epoch {epoch}, saving to {ckpt_path}")
            log.info(
                f"New best model found at epoch {epoch}, saving to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print(f"Testing on {config.ORGANISM}...")
    log.info(f"Testing on {config.ORGANISM}...")
    test_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_test.csv', feature_dir=config.DATA_DIR,
                             feature_map=config.FEATURE_MAP)
    test_dict = test_cds._get_dataset()
    test_ds = GeneExpressionDataset(feature_list=test_dict)
    test_dl = get_dataloader(
        ds=test_ds, batch_size=batch_size, shuffle=False, isDistributed=config.IS_DISTRIBUTED)

    test_metrics = evaluate_loop(model, test_dl, device, criterion)
    status = f"TEST on {config.ORGANISM} | loss={test_metrics['loss']:.4f} | Pearson={test_metrics['pearson']:.4f}; Spearman={test_metrics['spearman']:.4f}; R2={test_metrics['r2']:.4f}; MSE={test_metrics['mse']:.4f}; MAE={test_metrics['mae']:.4f}; MAPE={test_metrics['mape']:.4f}"
    print(status)
    log.info(status)

    print(f"Cross-organism evaluation on {config.CROSS_ORGANISM}...")
    log.info(f"Cross-organism evaluation on {config.CROSS_ORGANISM}...")
    test_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.CROSS_ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_test.csv', feature_dir=config.DATA_DIR,
                             feature_map=config.FEATURE_MAP)
    test_dict = test_cds._get_dataset()
    test_ds = GeneExpressionDataset(feature_list=test_dict)
    test_dl = get_dataloader(
        ds=test_ds, batch_size=batch_size, shuffle=False, isDistributed=config.IS_DISTRIBUTED)

    test_metrics = evaluate_loop(model, test_dl, device, criterion)
    status = f"TEST on {config.CROSS_ORGANISM} | loss={test_metrics['loss']:.4f} | Pearson={test_metrics['pearson']:.4f}; Spearman={test_metrics['spearman']:.4f}; R2={test_metrics['r2']:.4f}; MSE={test_metrics['mse']:.4f}; MAE={test_metrics['mae']:.4f}; MAPE={test_metrics['mape']:.4f}"
    print(status)
    log.info(status)

    return model


FTR_SIZE = 256
OPTUNA_PATCH_SIZES = [8, 16, 32]
OPTUNA_EMBED_DIMS = [192, 256, 384]
OPTUNA_HEAD_CANDIDATES = [4, 8, 16]
OPTUNA_DIRECTIONS = [
    "maximize", "maximize", "maximize",
    "minimize", "minimize", "minimize", "minimize",
]
OPTUNA_METRIC_NAMES = [
    "pearson", "spearman", "r2", "loss", "mse", "mae", "mape",
]


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _exceeds_gpu_budget(batch_size, num_heads, patch_size, depth, ftr_size=FTR_SIZE):
    if not torch.cuda.is_available():
        return False
    seq_len = (ftr_size // patch_size) ** 2 + 1
    est_bytes = batch_size * num_heads * seq_len * seq_len * depth * 6 * 4
    try:
        _, total = torch.cuda.mem_get_info()
        budget = int(total * 0.75)
    except Exception:
        budget = 100 * (1024 ** 3)
    return est_bytes > budget


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 20, 32])

    patch_size = trial.suggest_categorical("patch_size", OPTUNA_PATCH_SIZES)
    embed_dim = trial.suggest_categorical("embed_dim", OPTUNA_EMBED_DIMS)
    valid_heads = [h for h in OPTUNA_HEAD_CANDIDATES if embed_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", valid_heads)
    depth = trial.suggest_int("depth", 4, 8, step=2)
    mlp_ratio = trial.suggest_categorical("mlp_ratio", [2.0, 4.0])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.2)

    plateau_factor = trial.suggest_float("plateau_factor", 0.2, 0.5)
    plateau_patience = trial.suggest_int("plateau_patience", 4, 8)
    num_epochs = trial.suggest_int("num_epochs", 15, 35)

    if FTR_SIZE % patch_size != 0:
        raise optuna.TrialPruned()
    if _exceeds_gpu_budget(batch_size, num_heads, patch_size, depth):
        raise optuna.TrialPruned()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    model = train_dl = val_dl = optimizer = None
    try:
        train_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_train.csv', feature_dir=config.DATA_DIR,
                                  feature_map=config.FEATURE_MAP)
        train_dict = train_cds._get_dataset()
        train_ds = GeneExpressionDataset(feature_list=train_dict)
        train_dl = get_dataloader(
            ds=train_ds, batch_size=batch_size, shuffle=True, isDistributed=config.IS_DISTRIBUTED)

        val_cds = CustomDataset(feature_filename=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_val.csv', feature_dir=config.DATA_DIR,
                                feature_map=config.FEATURE_MAP)
        val_dict = val_cds._get_dataset()
        val_ds = GeneExpressionDataset(feature_list=val_dict)
        val_dl = get_dataloader(ds=val_ds, batch_size=batch_size,
                                shuffle=False, isDistributed=config.IS_DISTRIBUTED)

        model = VisionModel(ftr_size=FTR_SIZE, patch_size=patch_size,
                            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                            mlp_ratio=mlp_ratio, dropout=dropout,
                            attn_dropout=attn_dropout).to(device)
        criterion = nn.L1Loss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=plateau_factor, patience=plateau_patience)

        best_metrics = {
            "pearson": float("-inf"),
            "spearman": float("-inf"),
            "r2": float("-inf"),
            "loss": float("inf"),
            "mse": float("inf"),
            "mae": float("inf"),
            "mape": float("inf"),
        }

        for epoch in range(1, num_epochs + 1):
            train_loop(model, train_dl, device, criterion, optimizer=optimizer)
            val_metrics = evaluate_loop(
                model, val_dl, device, criterion, scheduler)

            best_metrics["pearson"] = max(
                best_metrics["pearson"], val_metrics["pearson"])
            best_metrics["spearman"] = max(
                best_metrics["spearman"], val_metrics["spearman"])
            best_metrics["r2"] = max(best_metrics["r2"], val_metrics["r2"])
            best_metrics["loss"] = min(
                best_metrics["loss"], val_metrics["loss"])
            best_metrics["mse"] = min(best_metrics["mse"], val_metrics["mse"])
            best_metrics["mae"] = min(best_metrics["mae"], val_metrics["mae"])
            best_metrics["mape"] = min(
                best_metrics["mape"], val_metrics["mape"])

        return (
            best_metrics["pearson"],
            best_metrics["spearman"],
            best_metrics["r2"],
            best_metrics["loss"],
            best_metrics["mse"],
            best_metrics["mae"],
            best_metrics["mape"],
        )
    except torch.cuda.OutOfMemoryError:
        raise optuna.TrialPruned()
    finally:
        del model, optimizer, train_dl, val_dl
        _clear_cuda()


def _objective_safe(trial):
    try:
        return objective(trial)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} error ({type(e).__name__}): {e}")
        raise optuna.TrialPruned()


def optuna_search(n_trials=50, study_name="get_optuna"):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    storage = f"sqlite:///{config.OUTPUT_DIR}/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=OPTUNA_DIRECTIONS,
        load_if_exists=True,
    )
    study.optimize(_objective_safe, n_trials=n_trials)

    print(f"Pareto-optimal trials: {len(study.best_trials)}")
    for trial in study.best_trials:
        metrics = dict(zip(OPTUNA_METRIC_NAMES, trial.values))
        print(f"Trial {trial.number}: {metrics}")
        print(f"  params: {trial.params}")
    return study


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train gene expression vision transformer."
    )
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="get_optuna")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.optuna:
        optuna_search(n_trials=args.n_trials, study_name=args.study_name)
    else:
        train()
