#!/usr/bin/env python3
"""Analyze test CSVs and produce Nature-style plots + explanation markdown."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analysis.plot_style import NATURE_COLORS, apply_nature_style  # noqa: E402


def scatter_pred_true(pred_df: pd.DataFrame, out_path: Path):
    apply_nature_style()
    fig, ax = plt.subplots()
    x = pred_df["true_tpm"].values
    y = pred_df["pred_tpm"].values
    ax.scatter(x, y, s=4, alpha=0.35, c=NATURE_COLORS[1], edgecolors="none")
    lim = max(x.max(), y.max(), 1e-3)
    ax.plot([0, lim], [0, lim], c=NATURE_COLORS[5], lw=0.8, ls="--")
    ax.set_xlabel("True TPM")
    ax.set_ylabel("Predicted TPM")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def residual_hist(pred_df: pd.DataFrame, out_path: Path):
    apply_nature_style()
    fig, ax = plt.subplots()
    resid = np.log1p(pred_df["pred_tpm"]) - np.log1p(pred_df["true_tpm"])
    ax.hist(resid, bins=50, color=NATURE_COLORS[2], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Residual log1p(pred) - log1p(true)")
    ax.set_ylabel("Count")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def metrics_bar(metrics_df: pd.DataFrame, out_path: Path):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cols = ["pearson", "spearman", "r2"]
    x = np.arange(len(metrics_df))
    w = 0.25
    for i, col in enumerate(cols):
        ax.bar(x + i * w, metrics_df[col], width=w, label=col, color=NATURE_COLORS[i])
    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics_df["split"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_explanation(metrics_df: pd.DataFrame, pred_df: pd.DataFrame, out_md: Path, run_tag: str):
    lines = [
        f"# Analysis: {run_tag}",
        "",
        "## Metrics summary",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Prediction overview",
        "",
        f"- Samples: {len(pred_df)}",
        f"- Mean true TPM: {pred_df['true_tpm'].mean():.4f}",
        f"- Mean pred TPM: {pred_df['pred_tpm'].mean():.4f}",
        "",
        "## Figures",
        "",
        "- `scatter_pred_true.png`: predicted vs true TPM (log-log).",
        "- `residual_hist.png`: distribution of log1p residuals.",
        "- `metrics_bar.png`: Pearson, Spearman, R² by split.",
        "",
        "Style: Nature-inspired sans-serif, Wong colorblind palette, no panel numbers or titles on figures.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    metrics_df = pd.read_csv(args.metrics) if args.metrics and args.metrics.exists() else None
    out_dir = args.out_dir or args.predictions.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.predictions.stem

    scatter_pred_true(pred_df, out_dir / "scatter_pred_true.png")
    residual_hist(pred_df, out_dir / "residual_hist.png")
    if metrics_df is not None and len(metrics_df) > 1:
        metrics_bar(metrics_df, out_dir / "metrics_bar.png")
    elif metrics_df is not None:
        metrics_bar(metrics_df, out_dir / "metrics_bar.png")

    write_explanation(
        metrics_df if metrics_df is not None else pd.DataFrame(),
        pred_df,
        out_dir / "explanation.md",
        tag,
    )
    print(f"Analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
