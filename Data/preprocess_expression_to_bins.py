#!/usr/bin/env python3


#python preprocess_expression_to_bins.py --input datasets/GM12878/ENCFF024TGS.tsv --gene-out gene_expression.tsv --bin-out bin_expression_1Mb.tsv


import argparse
import math


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate TSS-level expression to genes and fixed-size genomic bins."
    )
    p.add_argument("--input", required=True, help="Path to ENCFF024TGS.tsv")
    p.add_argument(
        "--gene-out",
        required=True,
        help="Output TSV with gene-level expression",
    )
    p.add_argument(
        "--bin-out",
        required=True,
        help="Output TSV with bin-level expression (chr1–chr22, chrX)",
    )
    p.add_argument(
        "--bin-size",
        type=int,
        default=1_000_000,
        help="Bin size in bp (default: 1,000,000)",
    )
    return p.parse_args()


def is_autosome_or_X(chr_name: str) -> bool:
    """Return True if chromosome is chr1–chr22 or chrX."""
    if not chr_name.startswith("chr"):
        return False
    tail = chr_name[3:]
    if tail == "X":
        return True
    try:
        n = int(tail)
        return 1 <= n <= 22
    except ValueError:
        return False


def main():
    args = parse_args()

    gene_data = {}
    

    with open(args.input, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")

            
            if len(parts) < 11:
                
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
            except ValueError:
                continue

            gene_id = parts[7]
            gene_name = parts[8]
            expr_str = parts[10]

            try:
                expr_vals = [float(x) for x in expr_str.split(",") if x != ""]
            except ValueError:
                continue

            if gene_id not in gene_data:
                gene_data[gene_id] = {
                    "chr": chrom,
                    "tss": start,
                    "expr_vec": expr_vals,
                    "gene_name": gene_name,
                }
            else:
                g = gene_data[gene_id]
                if start < g["tss"]:
                    g["tss"] = start
                    g["chr"] = chrom
                if len(g["expr_vec"]) == len(expr_vals):
                    g["expr_vec"] = [
                        max(a, b) for a, b in zip(g["expr_vec"], expr_vals)
                    ]

    
    gene_expr_table = []
    for gene_id, g in gene_data.items():
        vec = g["expr_vec"]
        if not vec:
            continue
        mean_expr = sum(vec) / len(vec)
        gene_expr_scalar = math.log2(mean_expr + 1.0)
        gene_expr_table.append(
            (gene_id, g["gene_name"], g["chr"], g["tss"], gene_expr_scalar)
        )

    with open(args.gene_out, "w") as out_g:
        out_g.write("gene_id\tgene_name\tchr\ttss\tgene_expr_log2_mean\n")
        for gene_id, gene_name, chrom, tss, expr in gene_expr_table:
            out_g.write(f"{gene_id}\t{gene_name}\t{chrom}\t{tss}\t{expr:.6f}\n")

    # Aggregate genes into bins
    bin_sums = {}   # (chr, bin_idx) -> sum of gene_expr
    bin_counts = {} # (chr, bin_idx) -> number of genes in bin

    for gene_id, gene_name, chrom, tss, expr in gene_expr_table:
        if not is_autosome_or_X(chrom):
            continue
        bin_idx = tss // args.bin_size
        key = (chrom, bin_idx)
        bin_sums[key] = bin_sums.get(key, 0.0) + expr
        bin_counts[key] = bin_counts.get(key, 0) + 1

    # Build bin-level table
    bin_records = []
    for (chrom, bin_idx), s in bin_sums.items():
        c = bin_counts[(chrom, bin_idx)]
        mean_expr_bin = s / c
        bin_start = bin_idx * args.bin_size
        bin_end = (bin_idx + 1) * args.bin_size
        bin_records.append((chrom, bin_idx, bin_start, bin_end, c, mean_expr_bin))

    # Sort bins by chromosome and index
    bin_records.sort(key=lambda x: (x[0], x[1]))

    with open(args.bin_out, "w") as out_b:
        out_b.write(
            "chr\tbin_index\tbin_start\tbin_end\tgene_count\tbin_expr_log2_mean\n"
        )
        for chrom, bin_idx, b_start, b_end, count, expr in bin_records:
            out_b.write(
                f"{chrom}\t{bin_idx}\t{b_start}\t{b_end}\t{count}\t{expr:.6f}\n"
            )


if __name__ == "__main__":
    main()

