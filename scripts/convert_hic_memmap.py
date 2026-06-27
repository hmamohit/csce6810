#!/usr/bin/env python3
"""Convert Hi-C text matrices to float32 memmap caches (one-time, resumable)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.hic_memmap import (  # noqa: E402
    convert_hic_to_memmap,
    discover_hic_matrix_files,
    load_config,
    matrix_ref,
    merge_scalers,
    scaler_key,
)


def _parse_chroms(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def _convert_one(
    organism: str,
    condition: str,
    chrom: str,
    text_path: str,
    force: bool,
) -> tuple[str, str, dict] | None:
    cfg = load_config()
    ref = matrix_ref(organism, condition, chrom, cfg, Path(text_path))
    scaler = convert_hic_to_memmap(ref, cfg, force=force)
    return organism, scaler_key(organism, condition, chrom), scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hi-C .txt matrices to float32 memmap")
    parser.add_argument("--organism", choices=["hg38", "mm10", "all"], default="all")
    parser.add_argument("--chroms", default=None, help="Comma-separated chroms, e.g. chr21,chr22")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (one file each)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if memmap is current")
    args = parser.parse_args()

    chroms = _parse_chroms(args.chroms)
    organism = None if args.organism == "all" else args.organism
    refs = discover_hic_matrix_files(organism=organism, chroms=chroms)
    if not refs:
        print("No Hi-C matrix files matched filters.")
        return

    print(f"Converting {len(refs)} Hi-C matrix file(s)...")
    scaler_batches: dict[str, dict[str, dict]] = {"hg38": {}, "mm10": {}}

    if args.workers <= 1:
        for ref in refs:
            result = _convert_one(
                ref.organism, ref.condition, ref.chrom, str(ref.text_path), args.force
            )
            if result:
                org, key, scaler = result
                scaler_batches[org][key] = scaler
                print(f"Done {ref.text_path.name}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _convert_one,
                    ref.organism,
                    ref.condition,
                    ref.chrom,
                    str(ref.text_path),
                    args.force,
                ): ref
                for ref in refs
            }
            for fut in as_completed(futures):
                ref = futures[fut]
                result = fut.result()
                if result:
                    org, key, scaler = result
                    scaler_batches[org][key] = scaler
                    print(f"Done {ref.text_path.name}")

    from data.hic_memmap import scaler_json_path  # noqa: E402

    for org, entries in scaler_batches.items():
        if entries:
            merge_scalers(org, entries)
            print(f"Updated scalers: {scaler_json_path(org)} ({len(entries)} entries)")

    print("Convert complete.")


if __name__ == "__main__":
    main()
