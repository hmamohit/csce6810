"""Training and evaluation loops for multimodal TPM model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import MultimodalHiCDataset, collate_multimodal, expand_manifest_to_genes, fit_tpm_scaler
from models.multimodal_transformer import MultimodalGeneExpression
from training.metrics import compute_metrics, scaled_to_raw


class TextLogger:
    def __init__(self, path: Path, tb_writer: SummaryWriter | None = None):
        self.path = path
        self.tb = tb_writer
        self.lines: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, level: str = "INFO", step: int = 0):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        self.lines.append(line)
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.tb is not None:
            self.tb.add_text("log", "\n".join(self.lines[-50:]), step)

    def flush_tb(self):
        if self.tb:
            self.tb.flush()


def _forward_batch(model, batch, device):
    kwargs = {
        "gene_body": batch["gene_body"].to(device),
        "pls": batch["PLS"].to(device),
        "pels": batch["pELS"].to(device),
        "dels": batch["dELS"].to(device),
        **{k: batch[k].to(device) for k in (
            "mask_gene", "mask_pls", "mask_pels", "mask_dels", "has_pls",
        )},
    }
    for bk in ("binary_gene", "binary_pls", "binary_pels", "binary_dels"):
        if bk in batch:
            kwargs[bk] = batch[bk].to(device)
    return model(**kwargs)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tpm_mean: float,
    tpm_std: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    preds_scaled, targets_scaled, raw_true = [], [], []
    for batch in tqdm(loader, desc="Validate", leave=False):
        out = _forward_batch(model, batch, device)
        tgt = batch["tpm_target"].to(device)
        total_loss += criterion(out, tgt).item()
        preds_scaled.append(out.cpu().numpy())
        targets_scaled.append(tgt.cpu().numpy())
        raw_true.append(batch["tpm_raw"].numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    targets_scaled = np.concatenate(targets_scaled, axis=0)
    raw_true = np.concatenate(raw_true, axis=0)
    pred_raw = scaled_to_raw(preds_scaled, tpm_mean, tpm_std)
    metrics = compute_metrics(raw_true, pred_raw)
    metrics["loss"] = total_loss / max(len(loader), 1)
    pr, _ = pearsonr(
        scaled_to_raw(targets_scaled, tpm_mean, tpm_std).flatten(),
        pred_raw.flatten(),
    )
    metrics["pearson_scaled"] = float(pr)
    return metrics


def train(
    organism: str,
    cfg: dict,
    output_dir: Path,
    log_dir: Path,
) -> Path:
    splits_path = Path(cfg["root_path"]) / "processed_tensors" / "splits" / f"{organism}_1000_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    train_genes = expand_manifest_to_genes(splits["train"])
    val_genes = expand_manifest_to_genes(splits["val"])
    tpm_mean, tpm_std = fit_tpm_scaler(splits["train"])

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    train_ds = MultimodalHiCDataset(train_genes, tpm_mean, tpm_std, augment=True, noise_std=tcfg.get("augment_noise_std", 0.02))
    val_ds = MultimodalHiCDataset(val_genes, tpm_mean, tpm_std, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        collate_fn=collate_multimodal, num_workers=tcfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate_multimodal, num_workers=tcfg["num_workers"], pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalGeneExpression(
        num_encoders=mcfg["num_encoders"],
        d_model=mcfg["d_model"],
        d_ff=mcfg["d_ff"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        bias=mcfg["bias"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    warmup_epochs = tcfg.get("warmup_epochs", 5)
    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        ))
        milestones.append(warmup_epochs)
    schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(tcfg["num_epochs"] - warmup_epochs, 1),
        eta_min=tcfg["learning_rate"] * tcfg.get("lr_min_factor", 0.01),
    ))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones,
    )
    scaler = GradScaler(enabled=tcfg.get("use_amp", True) and device.type == "cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir))
    logger = TextLogger(log_dir / "train.log", tb)
    best_path = output_dir / f"{organism}_multimodal_best.pt"
    scaler_path = output_dir / f"{organism}_tpm_scaler.json"

    with open(scaler_path, "w") as f:
        json.dump({"tpm_log_mean": tpm_mean, "tpm_log_std": tpm_std}, f)

    best_pearson = -1.0
    patience_ctr = 0
    logger.log(f"Train {organism}: {len(train_ds)} genes, val {len(val_ds)}")

    for epoch in range(tcfg["num_epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = _forward_batch(model, batch, device)
                loss = criterion(out, batch["tpm_target"].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        val_m = evaluate(model, val_loader, criterion, device, tpm_mean, tpm_std)
        train_loss /= max(len(train_loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        logger.log(
            f"Epoch {epoch+1} train_loss={train_loss:.4f} val_loss={val_m['loss']:.4f} "
            f"val_pearson={val_m['pearson']:.4f} val_mse={val_m['mse']:.2f} lr={lr:.2e}",
            step=epoch + 1,
        )
        tb.add_scalars("loss", {"train": train_loss, "val": val_m["loss"]}, epoch + 1)
        tb.add_scalar("pearson_raw", val_m["pearson"], epoch + 1)
        tb.add_scalar("learning_rate", lr, epoch + 1)

        if val_m["pearson"] > best_pearson:
            best_pearson = val_m["pearson"]
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "tpm_scaler": {"mean": tpm_mean, "std": tpm_std}}, best_path)
            logger.log(f"Saved best checkpoint pearson={best_pearson:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= tcfg["patience"]:
                logger.log("Early stopping")
                break

    logger.flush_tb()
    tb.close()
    return best_path
