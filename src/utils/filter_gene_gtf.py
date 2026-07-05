from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd


GENE_EXP_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression'
GENE_GTF_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/gene_exp'
OUTPUT_PATH = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression'

GENE_GTF_FILENAMES = {
    "hg38": "hg38.gencode.v49.annotation.gtf",
    "mm10": "mm10.gencode.vM1.annotation.gtf",
}

GENE_EXPRESSION_FILENAMES = {
    "hg38": ["dtag", "dmso", "auxin_6h", "auxin_no_treatment"],
    "mm10": [
        "pnd11_mature",
        "pnd11_immature",
        "pnd6_immature",
        "pnd6_precursors",
        "xen",
        "tsc",
        "pnd22",
        "pnd6",
        "iv_45h_aa",
        "iv_45h",
        "iv_20h_aa",
        "iv_20h",
    ],
}

GTF_COLUMNS = [
    "chrom",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
]


def _normalize_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _extract_attribute(series, key):
    return series.str.extract(rf'{key} "([^"]+)"', expand=False)


def _resolve_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing file: {candidates}")


def load_expression_table(genome, sample):
    exp_path = _resolve_path(
        [
            Path(GENE_EXP_DIR) / f"{genome}_{sample}.tsv",
            Path(GENE_EXP_DIR) / f"{sample}.tsv",
        ]
    )
    df = pd.read_csv(exp_path, sep="\t")
    # df.columns = [str(col).strip() for col in df.columns]

    # transcript_col = None
    # for candidate in ("transcript_ids"):
    #     if candidate in df.columns:
    #         transcript_col = candidate
    #         break
    # if transcript_col is None:
    #     raise ValueError(f"{exp_path} missing transcript_id(s) column")

    # tpm_col = next((col for col in df.columns if col.lower() == "tpm"), None)
    # if tpm_col is None:
    #     raise ValueError(f"{exp_path} missing TPM column")

    # if "gene_id" not in df.columns:
    #     raise ValueError(f"{exp_path} missing gene_id column")

    # df = df.copy()
    # df["gene_id"] = df["gene_id"].map(_normalize_id)
    # df[transcript_col] = df[transcript_col].astype(str)
    # df[tpm_col] = pd.to_numeric(df[tpm_col], errors="coerce")
    # df = df.dropna(subset=[tpm_col])
    # df = df[df[tpm_col] > 0].copy()
    # df["transcript_id"] = df[transcript_col].str.split(r"[;,]", regex=True)
    # df = df.explode("transcript_id", ignore_index=True)
    # df["transcript_id"] = df["transcript_id"].map(_normalize_id)
    # df = df[(df["gene_id"] != "") & (df["transcript_id"] != "")].copy()
    # df = (
    #     df.groupby(["gene_id", "transcript_id"], as_index=False)[tpm_col]
    #     .mean()
    #     .rename(columns={tpm_col: "TPM"})
    # )
    return df, exp_path


def load_transcript_gtf(genome):
    gtf_path = _resolve_path(
        [
            Path(GENE_GTF_DIR) / GENE_GTF_FILENAMES[genome],
            Path(GENE_EXP_DIR) / GENE_GTF_FILENAMES[genome],
        ]
    )
    gtf_df = pd.read_csv(gtf_path, sep="\t", header=None,
                         names=GTF_COLUMNS, comment="#")

    gtf_df["feature"] = gtf_df["feature"].astype(str)

    gtf_df = gtf_df[gtf_df["feature"].isin(["gene", "transcript"])].copy()

    gtf_df["gene_id"] = _extract_attribute(
        gtf_df["attributes"], "gene_id").map(_normalize_id)

    gtf_df["transcript_id"] = _extract_attribute(
        gtf_df["attributes"], "transcript_id").map(_normalize_id)

    gtf_df["gene_name"] = _extract_attribute(
        gtf_df["attributes"], "gene_name").map(_normalize_id)

    gtf_df["start"] = pd.to_numeric(gtf_df["start"], errors="coerce")
    gtf_df["end"] = pd.to_numeric(gtf_df["end"], errors="coerce")
    gtf_df["start"] = gtf_df["start"].astype(int) - 1
    gtf_df["end"] = gtf_df["end"].astype(int)

    gtf_df["tss"] = np.where(gtf_df["strand"] == "+",
                             gtf_df["start"], gtf_df["end"])

    gtf_df = gtf_df[["chrom", "gene_id", "gene_name", "transcript_id", "feature",
                     "start", "end", "strand", "tss"]]
    return gtf_df, gtf_path


# def build_gene_table(exp_df, gtf_df, genome, sample, chromosome, focus_gene_id=None):
#     merged = exp_df.merge(gtf_df, on=["gene_id", "transcript_id"], how="inner")
#     merged = merged[merged["chrom"] == chromosome].copy()
#     if merged.empty:
#         return pd.DataFrame()

#     merged["TPM"] = pd.to_numeric(merged["TPM"], errors="coerce")
#     merged = merged.dropna(subset=["TPM"])
#     merged["organism"] = genome
#     merged["sample"] = sample
#     merged["chr"] = chromosome
#     merged["trans_start"] = merged["start"].astype(int)
#     merged["trans_end"] = merged["end"].astype(int)
#     merged["gene_start"] = merged["gene_start"].astype(int)
#     merged["gene_end"] = merged["gene_end"].astype(int)

#     if focus_gene_id:
#         focus = merged[merged["gene_id"] == focus_gene_id]
#         if focus.empty:
#             warnings.warn(
#                 f"{genome} {sample} {chromosome}: gene_id {focus_gene_id} not found after merge",
#                 RuntimeWarning,
#             )
#         else:
#             print(
#                 f"Flow check {genome} {sample} {chromosome}: "
#                 f"{focus_gene_id} -> {focus['transcript_id'].nunique()} transcript(s)"
#             )

#     merged = merged[[
#         "organism",
#         "sample",
#         "chr",
#         "gene_id",
#         "gene_start",
#         "gene_end",
#         "transcript_id",
#         "chrom",
#         "trans_start",
#         "trans_end",
#         "strand",
#         "tss",
#         "TPM",
#     ]].copy()
#     merged = merged.rename(columns={"TPM": "tpm"})
#     return merged.sort_values(["gene_id", "transcript_id"]).reset_index(drop=True)


def _write_parquet(df, out_file):
    try:
        df.to_parquet(out_file, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Parquet write failed for {out_file}. Install pyarrow or fastparquet."
        ) from exc


def process_gene_expression(genome=None, sample=None, chromosome=None, gene_id=None):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_files = []

    for current_genome, samples in GENE_EXPRESSION_FILENAMES.items():
        if genome is not None and current_genome != genome:
            continue

        gtf_df, gtf_path = load_transcript_gtf(current_genome)
        print(f"Loaded GTF: {gtf_path}")

        for current_sample in samples:
            if sample is not None and current_sample != sample:
                continue

            exp_df, exp_path = load_expression_table(
                current_genome, current_sample)
            print(f"Loaded expression: {exp_path}")

            gene = exp_df.merge(gtf_df[gtf_df["feature"]=="gene"], on=["gene_id"], how="inner")
            gene = gene[["chrom", "gene_id", "gene_name", "mean_tpm", "transcript_ids", "start", "end", "strand", "tss"]].copy()
            gene.sort_values(by=["chrom", "start", "end"], inplace=True)
            gene["start"] = gene["start"].astype(int)
            gene["end"] = gene["end"].astype(int)
            
            transcript = exp_df.merge(gtf_df[gtf_df["feature"]=="transcript"], on=["gene_id"], how="inner")
            transcript = transcript[["chrom", "gene_id", "gene_name", "transcript_id", "start", "end", "strand", "tss"]].copy()
            transcript.sort_values(by=["chrom", "start", "end"], inplace=True)

            chromosomes = gene["chrom"].unique()
            
            output_filename_prefix = f"{current_genome}_{current_sample}"
            gene_output_file = Path(OUTPUT_PATH) / f"{output_filename_prefix}_gene.tsv"
            transcript_output_file = Path(OUTPUT_PATH) / f"{output_filename_prefix}_transcript.tsv"
            gene.to_csv(gene_output_file, sep="\t", index=False)
            transcript.to_csv(transcript_output_file, sep="\t", index=False)
            output_files.append((gene_output_file, transcript_output_file))
            print(f"Saved gene: {gene_output_file}")
            print(f"Saved transcript: {transcript_output_file}")
            for current_chromosome in chromosomes:
                # gene_table = build_gene_table(
                #     exp_df=exp_df,
                #     gtf_df=gtf_df,
                #     genome=current_genome,
                #     sample=current_sample,
                #     chromosome=current_chromosome,
                #     focus_gene_id=gene_id,
                # )
                gene_table = gene[gene["chrom"] == current_chromosome].copy()
                transcript_table = transcript[transcript["chrom"] == current_chromosome].copy()
                
                if gene_table.empty:
                    continue
                output_filename_prefix = f"{current_genome}_{current_sample}_{current_chromosome}"
                gene_output_file = Path(OUTPUT_PATH) / f"{output_filename_prefix}_gene.tsv"
                transcript_output_file = Path(OUTPUT_PATH) / f"{output_filename_prefix}_transcript.tsv"
                gene_table.to_csv(gene_output_file, sep="\t", index=False)
                transcript_table.to_csv(transcript_output_file, sep="\t", index=False)
                output_files.append((gene_output_file, transcript_output_file))
                print(f"Saved gene table: {gene_output_file}")
                print(f"Saved transcript table: {transcript_output_file}")

    return output_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge gene expression TSVs into transcript-level Parquet tables."
    )
    parser.add_argument("--genome", choices=GENE_EXPRESSION_FILENAMES.keys())
    parser.add_argument("--sample")
    parser.add_argument("--chromosome")
    parser.add_argument("--gene_id")
    return parser.parse_args()


def main():
    args = parse_args()
    process_gene_expression(
        genome=args.genome,
        sample=args.sample,
        chromosome=args.chromosome,
        gene_id=args.gene_id,
    )


if __name__ == "__main__":
    main()
