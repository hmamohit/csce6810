#!/usr/bin/env python3
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.train_utils import check_manifest  # noqa: E402
from training.trainer import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    check_manifest("multiorganism")
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["root_path"] = str(ROOT / "data")
    out = ROOT / "data" / "output" / "multiorganism_multimodal"
    logs = ROOT / "data" / "logs" / "multiorganism_multimodal"
    train("multiorganism", cfg, out, logs)


if __name__ == "__main__":
    main()
