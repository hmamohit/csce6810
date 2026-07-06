import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import config
from data_loader import CustomDataset, GeneExpressionDataset
from model import VisionModel
from tqdm import tqdm
import os
import sys
from torchmetrics.functional import pearson_corrcoef, mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error_using_sliding_window, spearman_corrcoef, mean_absolute_percentage_error

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
    if scheduler is not None:
        scheduler.step(avg_loss)

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = avg_loss
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

    model = VisionModel(ftr_size=200, patch_size=8,
                        embed_dim=256, depth=8, num_heads=8,
                        mlp_ratio=4.0, dropout=0.1,
                        attn_dropout=0.1).to(device)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5)

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


if __name__ == "__main__":
    train()
