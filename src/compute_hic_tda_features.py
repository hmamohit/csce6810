#!/usr/bin/env python3
"""
Precompute local cubical PH features for contact_matrix.txt (text pipeline).


"""

from __future__ import annotations

import argparse
from pathlib import Path

from hic_cubical_tda import (
    TDAConfig,
    compute_features_for_blocks,
    load_contact_matrix_blocks,
    require_gudhi,
    save_tda_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local cubical PH features on Hi-C patches."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/preprocessing"),
        help="Directory containing contact_matrix.txt",
    )
    parser.add_argument(
        "--contact-file",
        type=str,
        default="contact_matrix.txt",
        help="Contact matrix filename inside data-dir",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--silhouette-p", type=float, default=2.0)
    parser.add_argument(
        "--homology-dims",
        type=str,
        default="0,1",
        help="Comma-separated homology dimensions, e.g. 0,1",
    )
    parser.add_argument(
        "--vectorization",
        choices=["silhouette", "betti", "both"],
        default="silhouette",
    )
    parser.add_argument("--no-log1p", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_gudhi()

    data_dir = args.data_dir.resolve()
    contact_path = data_dir / args.contact_file
    if not contact_path.exists():
        raise FileNotFoundError(f"Missing contact matrix: {contact_path}")

    dims = tuple(int(x.strip()) for x in args.homology_dims.split(",") if x.strip())
    use_silhouette = args.vectorization in ("silhouette", "both")
    use_betti = args.vectorization in ("betti", "both")

    config = TDAConfig(
        patch_size=args.patch_size,
        n_samples=args.n_samples,
        silhouette_p=args.silhouette_p,
        homology_dims=dims,
        use_silhouette=use_silhouette,
        use_betti=use_betti,
        log1p=not args.no_log1p,
        normalize_patch=not args.no_normalize,
    )

    print(f"[INFO] Loading contact blocks from {contact_path}")
    blocks = load_contact_matrix_blocks(contact_path)
    n_rows = sum(block.shape[0] for block in blocks)
    print(f"[INFO] Chromosome blocks: {len(blocks)} | total bins: {n_rows}")
    print(f"[INFO] Feature dim per bin: {config.feature_dim}")

    features = compute_features_for_blocks(
        blocks,
        config=config,
        show_progress=not args.no_tqdm,
    )

    paths = save_tda_artifacts(
        features=features,
        config=config,
        output_dir=data_dir,
        contact_path=contact_path,
    )

    print("[DONE] Saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
