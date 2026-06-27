#!/usr/bin/env python3
"""Train with configurable yaml and separate output/log dirs."""

import argparse
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.train_utils import check_manifest  # noqa: E402
from training.trainer import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSCE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", default="hg38", choices=["hg38", "mm10", "multiorganism"])
    parser.add_argument("--config", default="default.yaml", help="Config filename under configs/")
    parser.add_argument("--run-suffix", default="", help="Subdir suffix, e.g. ablation_small")
    args = parser.parse_args()

    check_manifest(args.organism)
    cfg_path = CSCE / "configs" / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["root_path"] = str(ROOT / "data")

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    out = ROOT / "data" / "output" / f"{args.organism}_multimodal{suffix}"
    logs = ROOT / "data" / "logs" / f"{args.organism}_multimodal{suffix}"
    train(args.organism, cfg, out, logs)


if __name__ == "__main__":
    main()
