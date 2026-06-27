#!/usr/bin/env python3
"""Compute PLS / pELS / dELS bp ranges per gene from cCRE BED."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.bin_utils import bin_range_from_bp, bp_to_bin
from data.constants import (
    CHROM_SIZES,
    DELS_WINDOW_BP,
    EP_BED,
    EP_PATH,
    EP_REGIONS_PATH,
    GE_PATH,
    PELS_WINDOW_BP,
    PLS_WINDOW_BP,
    RESOLUTION_BP,
)
from data.sample_registry import discover_paired_shards

CCRE_COLS = ["chrom", "ccre_start", "ccre_end", "d_id", "e_id", "ccre_class"]


def _region_bounds_for_class(
    tss: int,
    ccre_df: pd.DataFrame,
    ccre_class: str,
    window_bp: int,
    pad_bins: int,
    chrom_bins: int,
) -> tuple[int, int, bool]:
    w_start, w_end = tss - window_bp, tss + window_bp
    matched = ccre_df[
        (ccre_df["ccre_class"] == ccre_class)
        & (ccre_df["ccre_end"] >= w_start)
        & (ccre_df["ccre_start"] <= w_end)
    ]
    has_hit = len(matched) > 0
    if has_hit:
        r0 = int(matched["ccre_start"].min())
        r1 = int(matched["ccre_end"].max())
        b0 = bp_to_bin(r0, RESOLUTION_BP) - pad_bins
        b1 = bp_to_bin(r1, RESOLUTION_BP) + pad_bins
    else:
        b0, b1 = bin_range_from_bp(tss, window_bp, pad_bins, RESOLUTION_BP, chrom_bins)
    b0 = max(0, b0)
    b1 = min(chrom_bins - 1, b1)
    return b0, b1, has_hit


def process_genome(
    genome: str,
    pls_pad: int = 15,
    pels_pad: int = 40,
    dels_pad: int = 30,
) -> None:
    ccre = pd.read_csv(
        EP_PATH / EP_BED[genome],
        sep="\t",
        header=None,
        names=CCRE_COLS,
        comment="#",
        usecols=[0, 1, 2, 3, 4, 5],
    )
    out_dir = EP_REGIONS_PATH / genome
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = pd.read_csv(GE_PATH / CHROM_SIZES[genome], sep="\t", names=["chrom", "size"])
    size_map = dict(zip(sizes["chrom"], sizes["size"]))

    paired, _ = discover_paired_shards()
    genome_pairs = [p for p in paired if p.organism == genome]

    for pair in tqdm(genome_pairs, desc=f"EP regions {genome}", unit="file"):
        ge_file = pair.ge_path
        chrom = pair.chrom
        condition, rep = pair.condition, pair.replicate
        genes = pd.read_csv(ge_file, sep="\t", header=None)
        genes.columns = [
            "chrom", "gene_id", "gb_start", "gb_end", "strand",
            "TSS", "TES", "gene_len", "TPM", "EP_start", "EP_end", "EP_len",
        ]
        c_chr = ccre[ccre["chrom"] == chrom]
        chrom_size_bp = int(size_map.get(chrom, genes["gb_end"].max()))
        chrom_bins = bp_to_bin(chrom_size_bp, RESOLUTION_BP) + 1

        records = []
        for _, row in genes.iterrows():
            tss = int(row["TSS"])
            tss_bin = bp_to_bin(tss, RESOLUTION_BP)
            pls_b0, pls_b1, has_pls = _region_bounds_for_class(
                tss, c_chr, "PLS", PLS_WINDOW_BP, pls_pad, chrom_bins
            )
            pels_b0, pels_b1, _ = _region_bounds_for_class(
                tss, c_chr, "pELS", PELS_WINDOW_BP, pels_pad, chrom_bins
            )
            dels_b0, dels_b1, _ = _region_bounds_for_class(
                tss, c_chr, "dELS", DELS_WINDOW_BP, dels_pad, chrom_bins
            )
            records.append({
                "gene_id": row["gene_id"],
                "chrom": chrom,
                "strand": row["strand"],
                "TSS": tss,
                "tss_bin": tss_bin,
                "gb_start": int(row["gb_start"]),
                "gb_end": int(row["gb_end"]),
                "TPM": float(row["TPM"]),
                "pls_bin_start": pls_b0,
                "pls_bin_end": pls_b1,
                "pels_bin_start": pels_b0,
                "pels_bin_end": pels_b1,
                "dels_bin_start": dels_b0,
                "dels_bin_end": dels_b1,
                        # True = PLS cCRE in BED near TSS; False = use TSS±200bp Hi-C bins anyway
                        "has_pls": bool(has_pls),
            })

        out_path = out_dir / f"{condition}_{rep}_{chrom}.json"
        with open(out_path, "w") as f:
            json.dump(records, f)


if __name__ == "__main__":
    for g in ("hg38", "mm10"):
        process_genome(g)
