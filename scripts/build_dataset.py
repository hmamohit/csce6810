#!/usr/bin/env python3
"""Build multimodal dataset from existing Hi-C + gene_expression .txt (no cool export)."""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.build_multimodal_set import BuildOptions, build_all, build_manifest, scan_manifest_from_npz  # noqa: E402
from data.sample_registry import summary_stats  # noqa: E402
from data.splits import write_all  # noqa: E402
from utils.process_ep_regions import process_genome  # noqa: E402
from data.constants import EP_REGIONS_PATH  # noqa: E402


def run_ep_regions(skip_if_exists: bool = True):
    for g in ("hg38", "mm10"):
        if skip_if_exists and (EP_REGIONS_PATH / g).exists() and any((EP_REGIONS_PATH / g).glob("*.json")):
            n = len(list((EP_REGIONS_PATH / g).glob("*.json")))
            print(f"EP regions {g}: skip ({n} json exist)")
            continue
        print(f"EP regions {g}...")
        process_genome(g)


def _parse_chroms(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def main():
    parser = argparse.ArgumentParser(description="Build rep-aware multimodal dataset from .txt")
    parser.add_argument("--skip-ep", action="store_true", help="Skip EP region JSON generation")
    parser.add_argument("--skip-if-exists", action="store_true", default=True,
                        help="Skip EP step if JSON already present (default: true)")
    parser.add_argument("--no-skip-if-exists", action="store_false", dest="skip_if_exists")
    parser.add_argument("--only-manifest", action="store_true", help="Only rebuild manifest/splits from NPZ")
    parser.add_argument("--organism", choices=["hg38", "mm10", "all"], default="all")
    parser.add_argument("--chroms", default=None, help="Comma-separated chroms, e.g. chr21,chr22")
    parser.add_argument("--resume", action="store_true", help="Skip NPZ shards that already exist")
    parser.add_argument("--require-memmap", action="store_true",
                        help="Fail if memmap cache missing (no text fallback)")
    parser.add_argument("--no-convert", action="store_true",
                        help="Do not convert missing memmap on the fly during build")
    args = parser.parse_args()

    build_opts = BuildOptions(
        chroms=_parse_chroms(args.chroms),
        resume=args.resume,
        require_memmap=args.require_memmap,
        convert_if_missing=not args.no_convert,
    )

    print("=== Sample registry ===")
    print(summary_stats())

    if not args.only_manifest:
        if not args.skip_ep:
            run_ep_regions(skip_if_exists=args.skip_if_exists)
        print("=== Multimodal NPZ shards ===")
        if args.organism == "all":
            build_all(options=build_opts)
        else:
            build_manifest(args.organism, options=build_opts)
    else:
        print("=== Manifest only (scan existing NPZ, no rebuild) ===")
        orgs = ("hg38", "mm10") if args.organism == "all" else (args.organism,)
        for org in orgs:
            scan_manifest_from_npz(org)

    print("=== Split manifests ===")
    write_all()
    print("Done.")


if __name__ == "__main__":
    main()
