"""
Build Hi-C / gene-expression .pt datasets for src/train.py.

Each sample is one genomic bin:
  - ftr: normalized Hi-C contact row (length = num_bins)
  - gene_exp: normalized expression scalar
  - tda (optional): cubical PH silhouette vector from a local Hi-C patch
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch

from hic_cubical_tda import (
    TDAConfig,
    extract_local_patch,
    require_gudhi,
    vectorize_patch,
)

ROOT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/csce6810/data"
MATRIX_PATH = f"{ROOT_PATH}/processed_raw_data/hic_matrix"
GENE_EXP_PATH = f"{ROOT_PATH}/processed_raw_data/gene_expression"

RESOLUTION = [5000]

CHROM_SIZES = {
    "hg38": "hg38.chrom.sizes",
    "mm10": "mm10.chrom.sizes",
}

DATA_DICT = {
    "hg38": {
        "dtag": {"rep1", "rep2", "rep3"},
        "dmso": {"rep1", "rep2", "rep3"},
        "auxin_6h": {"rep1", "rep2"},
        "auxin_no_treatment": {"rep1", "rep2"},
    },
    "mm10": {
        "pnd11_mature": {"rep1", "rep2"},
        "pnd11_immature": {"rep1", "rep2"},
        "pnd6_immature": {"rep1", "rep2"},
        "pnd6_precursors": {"rep1", "rep2"},
        "xen": {"rep1", "rep2"},
        "tsc": {"rep1", "rep2"},
        "pnd22": {"rep1", "rep2"},
    },
}


def save_dataset(dataset, save_path: str, tda_config: TDAConfig | None = None) -> None:
    payload = {
        "samples": dataset,
        "tda_enabled": tda_config is not None,
        "tda_feature_dim": int(tda_config.feature_dim) if tda_config else 0,
        "tda_config": {
            "patch_size": tda_config.patch_size,
            "n_samples": tda_config.n_samples,
            "silhouette_p": tda_config.silhouette_p,
        }
        if tda_config
        else None,
    }
    torch.save(payload, save_path)
    print(f"Dataset saved to {save_path}")


def get_balanced_indices(y_new: torch.Tensor) -> np.ndarray:
    y = y_new.view(-1).numpy()

    num_classes = 20
    bins = np.linspace(0, 1, num_classes + 1)

    classes = np.digitize(y, bins) - 1
    classes = np.clip(classes, 0, num_classes - 1)

    counter = Counter(classes)
    unique_counts = sorted(set(counter.values()), reverse=True)
    second_max = unique_counts[1]
    print("Second max count:", second_max)
    balanced_indices = []

    for c in counter.keys():
        idx = np.where(classes == c)[0]
        if len(idx) > second_max:
            selected = np.random.choice(idx, size=second_max, replace=False)
        else:
            selected = idx

        balanced_indices.extend(selected)
    return np.array(balanced_indices)


def compute_tda_vector(
    matrix_log: np.ndarray,
    bin_index: int,
    tda_config: TDAConfig,
) -> torch.Tensor:
    patch = extract_local_patch(
        matrix_log.astype(np.float32),
        int(bin_index),
        tda_config.half_window,
    )
    # Matrix is already log1p; only per-patch normalization inside vectorize_patch.
    cfg = TDAConfig(
        patch_size=tda_config.patch_size,
        n_samples=tda_config.n_samples,
        silhouette_p=tda_config.silhouette_p,
        homology_dims=tda_config.homology_dims,
        use_silhouette=tda_config.use_silhouette,
        use_betti=tda_config.use_betti,
        log1p=False,
        normalize_patch=True,
    )
    vec = vectorize_patch(patch, cfg)
    return torch.tensor(vec, dtype=torch.float32)


def build_datasets(
    root_path: str,
    use_tda: bool,
    tda_config: TDAConfig,
    resolutions: list[int] | None = None,
) -> None:
    if use_tda:
        require_gudhi()

    matrix_path_root = f"{root_path}/processed_raw_data/hic_matrix"
    gene_exp_path_root = f"{root_path}/processed_raw_data/gene_expression"
    res_list = resolutions if resolutions is not None else RESOLUTION

    for res in res_list:
        for organism, sample_dict in DATA_DICT.items():
            chrom_info_path = f"{root_path}/raw_data/{CHROM_SIZES[organism]}"
            chrom_list = np.loadtxt(chrom_info_path, dtype=str)

            dataset = []
            for sample_name, replicates in sample_dict.items():
                for chromosome, chrom_size in chrom_list:
                    chrom_size = int(chrom_size)
                    matrix_path = (
                        f"{matrix_path_root}/{organism}_{sample_name}_{res}_{chromosome}.txt"
                    )
                    if not os.path.exists(matrix_path):
                        continue

                    num_bins = int(torch.ceil(torch.tensor(chrom_size / res)))

                    matrix_np = np.loadtxt(matrix_path, delimiter="\t")
                    matrix_log = np.log1p(matrix_np.astype(np.float64))

                    x_chr_template = torch.tensor(matrix_log, dtype=torch.float32)
                    if x_chr_template.max() > x_chr_template.min():
                        x_chr_template = (x_chr_template - x_chr_template.min()) / (
                            x_chr_template.max() - x_chr_template.min()
                        )

                    abs_pos = torch.arange(
                        1, num_bins + 1, dtype=torch.float32
                    ).unsqueeze(1)
                    abs_pos = torch.log1p(abs_pos)
                    if abs_pos.max() > abs_pos.min():
                        abs_pos = (abs_pos - abs_pos.min()) / (
                            abs_pos.max() - abs_pos.min()
                        )

                    i = torch.arange(num_bins).unsqueeze(1)
                    j = torch.arange(num_bins).unsqueeze(0)
                    rel_pos = num_bins - torch.abs(i - j)
                    rel_pos = torch.log1p(rel_pos.float())
                    if rel_pos.max() > rel_pos.min():
                        rel_pos = (rel_pos - rel_pos.min()) / (
                            rel_pos.max() - rel_pos.min()
                        )

                    for replicate in replicates:
                        gene_exp_path = (
                            f"{gene_exp_path_root}/"
                            f"{organism}_{sample_name}_{replicate}_{res}_{chromosome}.txt"
                        )
                        if not os.path.exists(gene_exp_path):
                            continue

                        gene_exp = pd.read_csv(
                            gene_exp_path, delimiter="\t", header=None
                        )
                        y_values = gene_exp.iloc[:, 3].values.reshape(-1, 1)

                        y_new = torch.tensor(y_values, dtype=torch.float32)
                        y_new = torch.log1p(y_new)
                        if y_new.max() > y_new.min():
                            y_new = (y_new - y_new.min()) / (
                                y_new.max() - y_new.min()
                            )

                        x_new = x_chr_template.clone()
                        sampled_indices = get_balanced_indices(y_new)

                        for bin_idx in sampled_indices:
                            item = {
                                "ftr": x_new[int(bin_idx)],
                                "rel_pos": rel_pos[int(bin_idx)],
                                "abs_pos": abs_pos[int(bin_idx)],
                                "gene_exp": y_new[int(bin_idx)],
                            }
                            if use_tda:
                                item["tda"] = compute_tda_vector(
                                    matrix_log, int(bin_idx), tda_config
                                )
                            dataset.append(item)

                        print(
                            f"Added {organism} {sample_name} {chromosome} {replicate} "
                            f"({len(sampled_indices)} bins)"
                        )

            suffix = "_tda" if use_tda else ""
            save_path = (
                f"{root_path}/processed_tensors/"
                f"hic_{organism}_{res}_norm_select{suffix}.pt"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dataset(
                dataset,
                save_path,
                tda_config=tda_config if use_tda else None,
            )
            print(f"Saved: {save_path} | samples={len(dataset)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create .pt datasets for src/train.py (optional TDA features)."
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=ROOT_PATH,
        help="Project data root (contains processed_raw_data/, processed_tensors/)",
    )
    parser.add_argument(
        "--use-tda",
        action="store_true",
        help="Attach cubical PH silhouette vectors per bin (requires gudhi).",
    )
    parser.add_argument("--tda-patch-size", type=int, default=64)
    parser.add_argument("--tda-n-samples", type=int, default=25)
    parser.add_argument("--tda-silhouette-p", type=float, default=2.0)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Bin resolutions to process (default: 5000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tda_config = TDAConfig(
        patch_size=args.tda_patch_size,
        n_samples=args.tda_n_samples,
        silhouette_p=args.tda_silhouette_p,
    )
    if args.use_tda:
        print(
            f"[TDA] Enabled | patch={tda_config.patch_size} "
            f"| dim={tda_config.feature_dim}"
        )
    build_datasets(
        root_path=args.root_path,
        use_tda=args.use_tda,
        tda_config=tda_config,
        resolutions=args.resolution,
    )


if __name__ == "__main__":
    main()
