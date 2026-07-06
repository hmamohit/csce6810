import math
import os

import pandas as pd
import numpy as np
import cooler

ENHANCER_PROMOTER_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/enhancer_promoter'
COOL_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/cool'
GENE_EXPRESSION_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression'
OUTPUT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression_features_256'
GENE_EXPRESSION_FEATURES_DICT = 'gene_expression_features_dict'
TRAIN_GENE_EXPRESSION_FEATURES_DICT = 'gene_expression_features_dict'
VAL_GENE_EXPRESSION_FEATURES_DICT = 'gene_expression_features_dict'
TEST_GENE_EXPRESSION_FEATURES_DICT = 'gene_expression_features_dict'

RESOLUTION = 1000
WINDOW_SIZE = 256

ENHANCER_PROMOTER_FILENAMES = {
    "hg38": "hg38.encodeCcreCombined.bed",
    "mm10": "mm10.encodeCcreCombined.bed"
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


TRAIN_CHROMOSOME = {
    "hg38": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr12", "chr13", "chr14", "chr17", "chr18", "chr19", "chr22"],
    "mm10": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",  "chr12", "chr13", "chr14", "chr17"]
}

VAL_CHROMOSOME = {
    "hg38": ["chr11", "chr16",  "chr21"],
    "mm10": ["chr11", "chr16", "chr19"]
}

TEST_CHROMOSOME = {
    "hg38": ["chr12", "chr17",  "chr22", "chr10", "chr15", "chr20"],
    "mm10": ["chr12", "chr14", "chr17", "chr10", "chr15", "chr18"]
}

GENE_EXPRESSION_COLUMNS = ["gene_id", "gene_name",
                           "chrom", "start", "end", "strand", "mean_tpm"]

ENHANCER_PROMOTER_COLUMNS = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand",
    "thickStart", "thickEnd", "itemRgb", "ccRE_type", "ccRE_group",
    "maxZ", "short_type", "short_id", "description"
]
GENE_EXPRESSION_FEATURES_COLUMNS = [
    "resolution", "organism", "sample", "chromosome", "gene_id", "gene_name", "features"]

with open(f'{OUTPUT_DIR}/{GENE_EXPRESSION_FEATURES_DICT}_{RESOLUTION}.csv', 'a') as main_dict_file:
    main_dict_file.write('\t'.join(GENE_EXPRESSION_FEATURES_COLUMNS) + '\n')

    for gene_expression_genome, gene_expression_samples in GENE_EXPRESSION_FILENAMES.items():
        with open(f'{OUTPUT_DIR}/{gene_expression_genome}_{TRAIN_GENE_EXPRESSION_FEATURES_DICT}_{RESOLUTION}_train.csv', 'a') as train_dict_file, \
                open(f'{OUTPUT_DIR}/{gene_expression_genome}_{VAL_GENE_EXPRESSION_FEATURES_DICT}_{RESOLUTION}_val.csv', 'a') as val_dict_file, \
                open(f'{OUTPUT_DIR}/{gene_expression_genome}_{TEST_GENE_EXPRESSION_FEATURES_DICT}_{RESOLUTION}_test.csv', 'a') as test_dict_file:

            enhancer_promoter_path = f'{ENHANCER_PROMOTER_DIR}/{ENHANCER_PROMOTER_FILENAMES[gene_expression_genome]}'
            print(f"Loaded enhancer/promoter: {enhancer_promoter_path}")
            enhancer_promoter_df = pd.read_csv(enhancer_promoter_path, sep="\t", header=None,
                                               names=ENHANCER_PROMOTER_COLUMNS)
            enhancer_promoter_df = enhancer_promoter_df[[
                "chrom", "chromStart", "chromEnd", "ccRE_type", "maxZ", "short_type"]]
            enhancer_promoter_df.sort_values(
                by=["chrom", "chromStart", "chromEnd", "ccRE_type"], inplace=True)

            for sample in gene_expression_samples:
                cool_filename = f'{COOL_DIR}/{gene_expression_genome}_{sample}_{RESOLUTION}_KR.cool'
                cool_matrix = cooler.Cooler(cool_filename)
                print(
                    f"Processing cool matrix: {cool_filename} with shape {cool_matrix.shape}")

                for current_chromosome in cool_matrix.chromnames:
                    if current_chromosome in ["X", "Y", "M"]:
                        print(
                            f"Skipping chromosome: {current_chromosome} for sample: {sample} in genome: {gene_expression_genome}")
                        continue
                    print(
                        f"Processing chromosome: {current_chromosome} for sample: {sample} in genome: {gene_expression_genome}")
                    chrom_matrix = cool_matrix.matrix(
                        balance=False).fetch(current_chromosome)

                    current_chromosome = f'chr{current_chromosome}'

                    chrom_enhancer_promoter_df = enhancer_promoter_df[
                        (enhancer_promoter_df["chrom"] == current_chromosome)]

                    gene_expression_filename = f'{GENE_EXPRESSION_DIR}/{gene_expression_genome}_{sample}_{current_chromosome}_gene.tsv'
                    gene_expression_df = pd.read_csv(
                        gene_expression_filename, sep="\t")
                    print(
                        f"Processing gene expression: {gene_expression_filename} with shape {gene_expression_df.shape}")
                    for gene_row in gene_expression_df.itertuples(index=False):
                        attention = np.ones(
                            (WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                        tpm = gene_row.mean_tpm

                        search_window = (gene_row.tss - 100000,
                                         gene_row.tss + 100000)

                        if search_window[0]//RESOLUTION < 0 or search_window[1]//RESOLUTION > chrom_matrix.shape[0]:
                            print(
                                f"Skipping gene {gene_row.gene_name} ({gene_row.gene_id}) on {current_chromosome} due to out-of-bounds search window: {search_window}")
                            continue

                        feature = chrom_matrix[search_window[0] // RESOLUTION: search_window[1] // RESOLUTION,
                                               search_window[0] // RESOLUTION: search_window[1] // RESOLUTION]

                        record = f'{RESOLUTION}/{gene_expression_genome}/{sample}/{current_chromosome}/{gene_row.gene_id}_{gene_row.gene_name}'

                        gene_enhancer_promoter_df = chrom_enhancer_promoter_df[
                            (chrom_enhancer_promoter_df["chromStart"] >= search_window[0]) &
                            (chrom_enhancer_promoter_df["chromEnd"]
                                <= search_window[1])
                        ]

                        if not gene_enhancer_promoter_df.empty:
                            min_bp = gene_enhancer_promoter_df["chromStart"].min(
                            )
                            gene_enhancer_promoter_df["chromStart"] = gene_enhancer_promoter_df["chromStart"] - min_bp
                            gene_enhancer_promoter_df["chromEnd"] = gene_enhancer_promoter_df["chromEnd"] - min_bp

                            for _, enhancer_promoter_row in gene_enhancer_promoter_df.iterrows():
                                enhancer_promoter_start = enhancer_promoter_row["chromStart"]
                                enhancer_promoter_end = enhancer_promoter_row["chromEnd"]

                                enhancer_promoter_start_idx = max(
                                    0, int(math.floor(enhancer_promoter_start / RESOLUTION)))
                                enhancer_promoter_end_idx = min(
                                    WINDOW_SIZE, int(math.ceil(enhancer_promoter_end / RESOLUTION)))

                                attention[enhancer_promoter_start_idx:enhancer_promoter_end_idx,
                                          enhancer_promoter_start_idx:enhancer_promoter_end_idx] += enhancer_promoter_row["maxZ"]

                        os.makedirs(f"{OUTPUT_DIR}/{record}", exist_ok=True)
                        np.save(f"{OUTPUT_DIR}/{record}/feature.npy", feature)
                        np.save(
                            f"{OUTPUT_DIR}/{record}/attention.npy", attention)
                        np.save(f"{OUTPUT_DIR}/{record}/tpm.npy", tpm)
                        row = '\t'.join([str(RESOLUTION), gene_expression_genome, sample,
                                        current_chromosome, gene_row.gene_id, gene_row.gene_name, record]) + '\n'

                        main_dict_file.write(row)
                        if current_chromosome in TRAIN_CHROMOSOME[gene_expression_genome]:
                            train_dict_file.write(row)
                        if current_chromosome in VAL_CHROMOSOME[gene_expression_genome]:
                            val_dict_file.write(row)
                        if current_chromosome in TEST_CHROMOSOME[gene_expression_genome]:
                            test_dict_file.write(row)
