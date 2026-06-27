#!/usr/bin/env python3
"""Evaluate checkpoint on validation split; scatter plots + param count."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analysis.plot_style import NATURE_COLORS, apply_nature_style  # noqa: E402
from data.constants import RESOLUTION_BP  # noqa: E402
from data.dataset import MultimodalHiCDataset, collate_multimodal, expand_manifest_to_genes  # noqa: E402
from models.multimodal_transformer import MultimodalGeneExpression, uses_shared_ep_encoder  # noqa: E402
from training.metrics import compute_metrics, scaled_to_raw  # noqa: E402
from training.trainer import _forward_batch  # noqa: E402


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_group: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        by_group[top] = by_group.get(top, 0) + p.numel()
    return {"total": total, "trainable": trainable, "by_group": by_group}


def scatter_log_raw(df: pd.DataFrame, out_dir: Path):
    apply_nature_style()

    fig, ax = plt.subplots()
    x = df["true_tpm"].values
    y = df["pred_tpm"].values
    ax.scatter(x, y, s=6, alpha=0.4, c=NATURE_COLORS[1], edgecolors="none")
    lim = max(x.max(), y.max(), 1e-3)
    ax.plot([0, lim], [0, lim], c=NATURE_COLORS[5], lw=0.8, ls="--")
    ax.set_xlabel("True TPM")
    ax.set_ylabel("Predicted TPM")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(out_dir / "scatter_pred_true_raw.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    x = df["true_log_tpm"].values
    y = df["pred_log_tpm"].values
    ax.scatter(x, y, s=6, alpha=0.4, c=NATURE_COLORS[2], edgecolors="none")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], c=NATURE_COLORS[5], lw=0.8, ls="--")
    ax.set_xlabel("True log1p(TPM)")
    ax.set_ylabel("Predicted log1p(TPM)")
    fig.savefig(out_dir / "scatter_pred_true_log.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", default="hg38")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-train", type=int, default=None, help="Train set size for samples/param ratio")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    with open(Path(__file__).resolve().parents[1] / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["root_path"] = str(ROOT / "data")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "cfg" in ckpt and "model" in ckpt["cfg"]:
        mcfg = ckpt["cfg"]["model"]
    else:
        mcfg = cfg["model"]

    splits_path = Path(cfg["root_path"]) / "processed_tensors" / "splits" / f"{args.organism}_{RESOLUTION_BP}_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    train_genes = expand_manifest_to_genes(splits["train"])
    val_genes = expand_manifest_to_genes(splits["val"])
    n_train = args.n_train if args.n_train is not None else len(train_genes)

    tpm = ckpt["tpm_scaler"]

    ds = MultimodalHiCDataset(val_genes, tpm["mean"], tpm["std"], augment=False)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_ep = uses_shared_ep_encoder(ckpt["model"])
    model = MultimodalGeneExpression(
        num_encoders=mcfg["num_encoders"],
        d_model=mcfg["d_model"],
        d_ff=mcfg["d_ff"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        bias=mcfg["bias"],
        shared_ep_encoder=shared_ep,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_info = count_params(model)
    ratio = n_train / param_info["trainable"]

    preds_raw, truths_raw, rows = [], [], []
    preds_log, truths_log = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val diagnose", unit="batch"):
            out = _forward_batch(model, batch, device)
            pred_scaled = out.cpu().numpy()
            pred_raw = scaled_to_raw(pred_scaled, tpm["mean"], tpm["std"])
            truth_raw = batch["tpm_raw"].numpy()
            pred_log = np.log1p(np.maximum(pred_raw, 0))
            truth_log = np.log1p(truth_raw)
            preds_raw.append(pred_raw)
            truths_raw.append(truth_raw)
            preds_log.append(pred_log)
            truths_log.append(truth_log)
            for i in range(len(batch["gene_id"])):
                rows.append({
                    "gene_id": batch["gene_id"][i],
                    "chrom": batch["chrom"][i],
                    "true_tpm": float(truth_raw[i]),
                    "pred_tpm": float(np.asarray(pred_raw[i]).reshape(-1)[0]),
                    "true_log_tpm": float(truth_log[i]),
                    "pred_log_tpm": float(np.asarray(pred_log[i]).reshape(-1)[0]),
                })

    preds_raw = np.concatenate(preds_raw).flatten()
    truths_raw = np.concatenate(truths_raw).flatten()
    metrics_raw = compute_metrics(truths_raw, preds_raw)
    pr_log, _ = pearsonr(np.concatenate(truths_log).flatten(), np.concatenate(preds_log).flatten())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or ROOT / "data" / "output" / "diagnostics" / f"{args.organism}_val_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "val_predictions.csv", index=False)
    scatter_log_raw(df, out_dir)

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_val": len(df),
        "n_train": n_train,
        "params_total": param_info["total"],
        "params_trainable": param_info["trainable"],
        "samples_per_param": ratio,
        "pearson_raw": metrics_raw["pearson"],
        "pearson_log": float(pr_log),
        "mse_raw": metrics_raw["mse"],
        "r2_raw": metrics_raw["r2"],
        "params_by_group": param_info["by_group"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"# Validation diagnostic ({args.organism})",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Val genes: {len(df)}",
        f"- Train genes: {n_train}",
        f"- Params (trainable): {param_info['trainable']:,}",
        f"- Samples/param: {ratio:.6f}",
        f"- Pearson raw TPM: {metrics_raw['pearson']:.4f}",
        f"- Pearson log1p TPM: {pr_log:.4f}",
        f"- MSE raw TPM: {metrics_raw['mse']:.2f}",
        "",
        "## Params by module",
        "",
    ]
    for k, v in sorted(param_info["by_group"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v:,}")
    lines.extend(["", "## Figures", "", "- `scatter_pred_true_raw.png`", "- `scatter_pred_true_log.png`"])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
