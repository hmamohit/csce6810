from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd


GENE_EXP_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression'
ENHANCER_PROMOTER_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/enhancer_promoter'
OUTPUT_PATH = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/enhancer_promoter'

ENHANCER_PROMOTER_FILENAMES = {
    "hg38": "hg38.encodeCcreCombined.bed",
    "mm10": "mm10.encodeCcreCombined.bed"
}

CHROMOSOME = {
    "hg38": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"],
    "mm10": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chrX", "chrY"]
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
    ]
}

enhancer_promoter_columns = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand", 
    "thickStart", "thickEnd", "itemRgb", "ccRE_type", "ccRE_group", 
    "maxZ", "short_type", "short_id", "description"
]

def process_gene_expression(genome=None, sample=None, chromosome=None, gene_id=None):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for enhancer_promoter_genome, enhancer_promoter_filename in ENHANCER_PROMOTER_FILENAMES.items():
        enhancer_promoter_path = Path(ENHANCER_PROMOTER_DIR) / enhancer_promoter_filename
        if not enhancer_promoter_path.exists():
            raise FileNotFoundError(f"Missing file: {enhancer_promoter_path}")
        print(f"Loaded enhancer/promoter: {enhancer_promoter_path}")
        enhancer_promoter_df = pd.read_csv(enhancer_promoter_path, sep="\t", header=None,
                                           names=enhancer_promoter_columns)
        
        enhancer_promoter_df = enhancer_promoter_df[["chrom", "chromStart", "chromEnd", "ccRE_type", "maxZ", "short_type"]]
        enhancer_promoter_df.sort_values(by=["chrom", "chromStart", "chromEnd", "ccRE_type"], inplace=True)

        for gene_expression_genome, gene_expression_samples in GENE_EXPRESSION_FILENAMES.items():
            for sample in gene_expression_samples:
                for current_chromosome in CHROMOSOME[gene_expression_genome]:

                    gene_promoter_df = pd.DataFrame(columns=["chrom", "gene_id", "gene_name", "start", "end", "strand", "tss", "tpm", "element_start", "element_end", "element_type", "maxz", "element_short_type"])
                    gene_promoter_list = []

                    gene_expression_path = Path(GENE_EXP_DIR) / f"{gene_expression_genome}_{sample}_{current_chromosome}_gene.tsv"
                    gene_expression_df = pd.read_csv(gene_expression_path, sep="\t")
                    transcript_path = Path(GENE_EXP_DIR) / f"{gene_expression_genome}_{sample}_{current_chromosome}_transcript.tsv"
                    transcript_df = pd.read_csv(transcript_path, sep="\t")
                    
                    chrom_enhancer_promoter_df = enhancer_promoter_df[
                        (enhancer_promoter_df["chrom"] == current_chromosome) &
                        (enhancer_promoter_df["ccRE_type"] == "PLS")
                    ]
                    for gene_row in gene_expression_df.itertuples(index=False):
                        gene_id = gene_row.gene_id
                        gene_chrom = gene_row.chrom
                        gene_start = gene_row.start
                        gene_end = gene_row.end
                        gene_strand = gene_row.strand
                        gene_tss = gene_row.tss
                        
                        if gene_strand == "+":
                            promoter_start = gene_tss - 1000
                            promoter_end = gene_tss
                        else:
                            promoter_start = gene_tss
                            promoter_end = gene_tss + 1000

                        overlapping_enhancer_promoter = chrom_enhancer_promoter_df[
                            (chrom_enhancer_promoter_df["chromStart"] <= promoter_end) & 
                            (chrom_enhancer_promoter_df["chromEnd"] >= promoter_start)
                        ]

                        if not overlapping_enhancer_promoter.empty:
                            gene_promoter_list.append({
                                "chrom": overlapping_enhancer_promoter["chrom"].values,
                                "gene_id": [gene_id] * len(overlapping_enhancer_promoter),
                                "gene_name": [gene_row.gene_name] * len(overlapping_enhancer_promoter),
                                "start": [gene_start] * len(overlapping_enhancer_promoter),
                                "end": [gene_end] * len(overlapping_enhancer_promoter),
                                "strand": [gene_strand] * len(overlapping_enhancer_promoter),
                                "tss": [gene_tss] * len(overlapping_enhancer_promoter),
                                "tpm": [gene_row.mean_tpm] * len(overlapping_enhancer_promoter),
                                "element_start": overlapping_enhancer_promoter["chromStart"].values,
                                "element_end": overlapping_enhancer_promoter["chromEnd"].values,
                                "element_type": overlapping_enhancer_promoter["ccRE_type"].values,
                                "maxz": overlapping_enhancer_promoter["maxZ"].values,
                                "element_short_type": overlapping_enhancer_promoter["short_type"].values
                            })
                        else:
                            transcript_rows = transcript_df[transcript_df["gene_id"] == gene_id]
                            for transcript_row in transcript_rows.itertuples(index=False):
                                transcript_tss = transcript_row.tss
                                if gene_strand == "+":
                                    transcript_promoter_start = transcript_tss - 1000
                                    transcript_promoter_end = transcript_tss
                                else:
                                    transcript_promoter_start = transcript_tss
                                    transcript_promoter_end = transcript_tss + 1000

                                overlapping_transcript_enhancer_promoter = chrom_enhancer_promoter_df[
                                    (chrom_enhancer_promoter_df["chromStart"] <= transcript_promoter_end) & 
                                    (chrom_enhancer_promoter_df["chromEnd"] >= transcript_promoter_start)
                                ]
                                if not overlapping_transcript_enhancer_promoter.empty:
                                    gene_promoter_list.append({
                                        "chrom": overlapping_transcript_enhancer_promoter["chrom"].values,
                                        "gene_id": [gene_id] * len(overlapping_transcript_enhancer_promoter),
                                        "gene_name": [gene_row.gene_name] * len(overlapping_transcript_enhancer_promoter),
                                        "start": [gene_start] * len(overlapping_transcript_enhancer_promoter),
                                        "end": [gene_end] * len(overlapping_transcript_enhancer_promoter),
                                        "strand": [gene_strand] * len(overlapping_transcript_enhancer_promoter),
                                        "tss": [gene_tss] * len(overlapping_transcript_enhancer_promoter),
                                        "tpm": [gene_row.mean_tpm] * len(overlapping_transcript_enhancer_promoter),
                                        "element_start": overlapping_transcript_enhancer_promoter["chromStart"].values,
                                        "element_end": overlapping_transcript_enhancer_promoter["chromEnd"].values,
                                        "element_type": overlapping_transcript_enhancer_promoter["ccRE_type"].values,
                                        "maxz": overlapping_transcript_enhancer_promoter["maxZ"].values,
                                        "element_short_type": overlapping_transcript_enhancer_promoter["short_type"].values
                                    })
                                    break

                    gene_promoter_df = pd.DataFrame(gene_promoter_list)
                    count = len(gene_promoter_df)
                    print(f"Processed {count}/{len(gene_expression_df)} genes for genome {gene_expression_genome}, sample {sample}, chromosome {current_chromosome}")


            


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
