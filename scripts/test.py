#!/usr/bin/env python3
"""Evaluate checkpoint on split modes; write metrics + predictions CSV."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.constants import RESOLUTION_BP  # noqa: E402
from data.dataset import MultimodalHiCDataset, collate_multimodal, expand_manifest_to_genes  # noqa: E402
from models.multimodal_transformer import MultimodalGeneExpression, uses_shared_ep_encoder  # noqa: E402
from training.metrics import compute_metrics, scaled_to_raw  # noqa: E402
from training.trainer import _forward_batch  # noqa: E402


def load_splits(profile: str, root_path: str) -> dict:
    path = Path(root_path) / "processed_tensors" / "splits" / f"{profile}_{RESOLUTION_BP}_splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing splits {path}. Run: python scripts/build_dataset.py --only-manifest")
    with open(path) as f:
        return json.load(f)


def run_test(profile: str, checkpoint: Path, split_mode: str, cfg: dict, out_dir: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tpm = ckpt["tpm_scaler"]
    if "cfg" in ckpt and "model" in ckpt["cfg"]:
        mcfg = ckpt["cfg"]["model"]
    else:
        mcfg = cfg["model"]

    splits = load_splits(profile, cfg["root_path"])
    test_records = splits["test"].get(split_mode, [])
    if not test_records:
        print(f"No records for {split_mode}")
        return

    eval_orgs = sorted({r.get("organism", profile) for r in test_records})
    eval_org = eval_orgs[0] if len(eval_orgs) == 1 else "mixed"

    records = expand_manifest_to_genes(test_records)
    if not records:
        print(f"No gene records for {split_mode}")
        return

    ds = MultimodalHiCDataset(records, tpm["mean"], tpm["std"], augment=False)
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
        num_encoders=mcfg["num_encoders"], d_model=mcfg["d_model"], d_ff=mcfg["d_ff"],
        num_heads=mcfg["num_heads"], dropout=mcfg["dropout"], bias=mcfg["bias"],
        shared_ep_encoder=shared_ep,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds_raw, truths_raw, rows = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Test {split_mode}", unit="batch"):
            out = _forward_batch(model, batch, device)
            pred = scaled_to_raw(out.cpu().numpy(), tpm["mean"], tpm["std"])
            truth = batch["tpm_raw"].numpy()
            preds_raw.append(pred)
            truths_raw.append(truth)
            for i in range(len(batch["gene_id"])):
                rows.append({
                    "organism": batch["organism"][i],
                    "sample": batch["sample"][i],
                    "replicate": batch["replicate"][i],
                    "chrom": batch["chrom"][i],
                    "gene_id": batch["gene_id"][i],
                    "true_tpm": float(truth[i]),
                    "pred_tpm": float(pred[i]),
                    "split": split_mode,
                    "train_profile": profile,
                    "eval_organism": eval_org,
                })

    preds_raw = np.concatenate(preds_raw).flatten()
    truths_raw = np.concatenate(truths_raw).flatten()
    metrics = compute_metrics(truths_raw, preds_raw)
    metrics["split"] = split_mode
    metrics["train_profile"] = profile
    metrics["eval_organism"] = eval_org
    metrics["n_samples"] = len(truths_raw)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{profile}_{split_mode}_{run_id}_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / f"{profile}_{split_mode}_{run_id}_predictions.csv", index=False)
    print(f"{split_mode}: pearson={metrics['pearson']:.4f} mse={metrics['mse']:.2f} n={metrics['n_samples']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", choices=["hg38", "mm10", "multiorganism"], required=True,
                        help="Training profile (hg38, mm10, or multiorganism)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="all",
        help="Test mode from splits JSON (e.g. seen, unseen_rep, unseen_condition, cross_org, unseen) or all",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    with open(Path(__file__).resolve().parents[1] / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["root_path"] = str(ROOT / "data")
    out_dir = ROOT / "data" / "output" / "results"

    splits = load_splits(args.organism, cfg["root_path"])
    available = list(splits["test"].keys())
    modes = available if args.split == "all" else [args.split]
    for m in modes:
        run_test(args.organism, args.checkpoint, m, cfg, out_dir)


if __name__ == "__main__":
    main()
