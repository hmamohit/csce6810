#!/usr/bin/env python3
"""Parse train.log files and plot val Pearson curves across runs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
import sys
sys.path.insert(0, str(SRC))

from analysis.plot_style import NATURE_COLORS, apply_nature_style  # noqa: E402

EPOCH_RE = re.compile(
    r"Epoch (\d+) train_loss=[\d.]+ val_loss=[\d.]+ val_pearson=([-\d.]+) val_mse=[\d.]+"
)


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append({"epoch": int(m.group(1)), "val_pearson": float(m.group(2))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="label=path/to/train.log")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    summary_rows = []

    for i, spec in enumerate(args.logs):
        label, path = spec.split("=", 1)
        df = parse_log(Path(path))
        if df.empty:
            print(f"No epochs parsed from {path}")
            continue
        color = NATURE_COLORS[i % len(NATURE_COLORS)]
        ax.plot(df["epoch"], df["val_pearson"], label=label, color=color, lw=1.0)
        best_idx = df["val_pearson"].idxmax()
        summary_rows.append({
            "run": label,
            "epochs": len(df),
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "best_pearson": float(df.loc[best_idx, "val_pearson"]),
            "final_pearson": float(df["val_pearson"].iloc[-1]),
        })

    ax.axhline(0, color="#888888", lw=0.5, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Pearson (raw TPM)")
    ax.legend(frameon=False, fontsize=5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    summary_path = args.out.with_suffix(".csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(summary_path.read_text())
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
