#!/usr/bin/env python3
"""Export cooler files to per-chromosome Hi-C square matrices (1kb)."""

import sys
from pathlib import Path

import cooler
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "csce6810" / "src"))

from data.constants import HIC_DICT, MATRIX_PATH, ROOT_PATH, RESOLUTION_BP  # noqa: E402

RAW_PATH = ROOT_PATH / "raw_data"
PROCESSED_PATH = MATRIX_PATH


def main():
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    for organism, hic_dict in HIC_DICT.items():
        for sample_name, filename in hic_dict.items():
            cool_path = RAW_PATH / f"{filename}_{RESOLUTION_BP}_KR.cool"
            if not cool_path.exists():
                print(f"Skip missing: {cool_path}")
                continue
            print(f"Processing {organism} {sample_name}")
            cool_file = cooler.Cooler(str(cool_path))
            for chromosome in cool_file.chromnames:
                chromosome_name = chromosome if str(chromosome).startswith("chr") else f"chr{chromosome}"
                chr_matrix = cool_file.matrix(balance=False).fetch(chromosome)
                out = PROCESSED_PATH / f"{organism}_{sample_name}_{RESOLUTION_BP}_{chromosome_name}.txt"
                np.savetxt(out, chr_matrix, delimiter="\t", fmt="%1.4f")
                print(f"  {chromosome_name} {chr_matrix.shape} -> {out.name}")


if __name__ == "__main__":
    main()
