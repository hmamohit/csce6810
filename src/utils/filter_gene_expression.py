import pandas as pd
import os
import argparse
import warnings
import numpy as np


GENE_EXP_DIR = f"/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/gene_exp"
OUTPUT_DIR = f'/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression'

GENE_EXP_FILENAMES = {
    "hg38": {
        "dtag": {
            "rep1": "4DNFIBD1INNK_dtag_v1_tbx5",
            "rep2": "4DNFIA3SEKHI_dtag_v1_tbx5",
            "rep3": "4DNFIKZXF6PK_dtag_v1_tbx5"
        },
        "dmso": {
            "rep1": "4DNFI5FA8O2Q_dmso",
            "rep2": "4DNFIZ35VUEN_dmso",
            "rep3": "4DNFIZ71B2BS_dmso"
        },
        "auxin_6h": {
            "rep1": "4DNFIFHN2AFJ_auxin_6h",
            "rep2": "4DNFINPVMOEX_auxin_6h"
        },
        "auxin_no_treatment": {
            "rep1": "4DNFITEMAVMR_auxin_no_treatment",
            "rep2": "4DNFIIJQ6HA8_auxin_no_treatment"
        }
    },
    "mm10": {
        "pnd11_mature": {
            "rep1": "4DNFIY27UR9B_pnd11_mature",
            "rep2": "4DNFIR7X8G4O_pnd11_mature"
        },
        "pnd11_immature": {
            "rep1": "4DNFIFAPFARG_pnd11_immature",
            "rep2": "4DNFIH4NE916_pnd11_immature"
        },
        "pnd6_immature": {
            "rep1": "4DNFINL42PDZ_pnd6_immature",
            "rep2": "4DNFI3ZX6EPU_pnd6_immature"
        },
        "pnd6_precursors": {
            "rep1": "4DNFIERBERBM_pnd6_precursors",
            "rep2": "4DNFIY21QCB4_pnd6_precursors"
        },
        "xen": {
            "rep1": "4DNFI73DRTLC_xen",
            "rep2": "4DNFIPA6ZI3P_xen"
        },
        "tsc": {
            "rep1": "4DNFIFHKRC2X_tsc",
            "rep2": "4DNFIRDLK7BP_tsc"
        },
        "pnd22": {
            "rep1": "4DNFIVMKQHH6_pnd22",
            "rep2": "4DNFIG76S3MY_pnd22",
            "rep3": "4DNFIVYNBZ2S_pnd22",
            "rep4": "4DNFIRHNWFON_pnd22"
        },
        "pnd6": {
            "rep1": "4DNFIDOGFIEI_pnd6",
            "rep2": "4DNFI94YCVOL_pnd6",
            "rep3": "4DNFIBILX3ED_pnd6",
            "rep4": "4DNFI2JSMLGG_pnd6"
        },
        "iv_45h_aa": {
            "rep1": "4DNFIMIVY89G_iv_45h_aa",
            "rep2": "4DNFIB7S9IW5_iv_45h_aa"
        },
        "iv_45h": {
            "rep1": "4DNFID2EQMXY_iv_45h",
            "rep2": "4DNFIC7D7PJV_iv_45h"
        },
        "iv_20h_aa": {
            "rep1": "4DNFI2E5QYBF_iv_20h_aa",
            "rep2": "4DNFIPSA5HKO_iv_20h_aa"
        },
        "iv_20h": {
            "rep1": "4DNFIU47OVNE_iv_20h",
            "rep2": "4DNFIED8PY69_iv_20h"
        }
    }
}


def resolve_expression_file(genome, condition, file_id, replicate):
    candidates = [
        os.path.join(GENE_EXP_DIR, f"{file_id}.tsv"),
        os.path.join(GENE_EXP_DIR, f"{genome}_{condition}_{replicate}.tsv"),
    ]
    for exp_file in candidates:
        if os.path.exists(exp_file):
            return exp_file
    raise FileNotFoundError(
        f"No expression file found for {genome} {condition} {replicate}: {candidates}")


def read_replicate_expression(genome, condition, file_id, replicate):
    exp_file = resolve_expression_file(genome, condition, file_id, replicate)
    df = pd.read_csv(exp_file, sep="\t")

    missing_cols = {"gene_id", "transcript_id(s)", "TPM"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{exp_file} missing columns: {sorted(missing_cols)}")

    df = df[["gene_id", "transcript_id(s)", "TPM"]]
    df = df[df['TPM'] > 0]
    df["gene_id"] = df["gene_id"].astype(
        str).str.split(".", n=1, regex=False).str[0]
    df["transcript_id(s)"] = df["transcript_id(s)"].astype(str)

    id_cols = {"gene_id", "transcript_id(s)"}
    for col in df.columns:
        if col not in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["replicate"] = replicate
    return df


def warn_if_replicate_ids_differ(replicate_dfs, genome, condition):
    base_df = replicate_dfs[0]
    base_replicate = base_df["replicate"].iloc[0]
    base_gene_ids = set(base_df["gene_id"])
    base_transcript_ids = set(base_df["transcript_id(s)"])

    for df in replicate_dfs[1:]:
        replicate = df["replicate"].iloc[0]
        gene_ids = set(df["gene_id"])
        transcript_ids = set(df["transcript_id(s)"])
        if gene_ids != base_gene_ids:
            warnings.warn(
                f"{genome} {condition}: gene_id mismatch between "
                f"{base_replicate} and {replicate}",
                RuntimeWarning,
            )
        if transcript_ids != base_transcript_ids:
            warnings.warn(
                f"{genome} {condition}: transcript_id mismatch between "
                f"{base_replicate} and {replicate}",
                RuntimeWarning,
            )


def has_no_mismatches(group):
    all_individual_transcripts = set()

    for val in group["transcript_id(s)"].dropna().astype(str):
        transcripts_in_row = [t.strip() for t in val.split(",") if t.strip()]
        all_individual_transcripts.update(transcripts_in_row)

    for val in group["transcript_id(s)"].dropna().astype(str):
        row_set = {t.strip() for t in val.split(",") if t.strip()}
        if row_set != all_individual_transcripts:
            return False

    return True


def summarize_transcript_ids(merged_df, genome, condition):
    filtered_df = merged_df.groupby("gene_id").filter(has_no_mismatches)

    original_genes = set(merged_df["gene_id"])
    remaining_genes = set(filtered_df["gene_id"])
    conflict_count = len(original_genes - remaining_genes)

    if conflict_count > 0:
        warnings.warn(
            f"{genome} {condition}: {conflict_count} gene_id records "
            "removed entirely due to transcript_id mismatches.",
            RuntimeWarning,
        )

    return (
        filtered_df.groupby("gene_id", as_index=False)["transcript_id(s)"]
        .agg(lambda values: ";".join(sorted(set(
            t.strip() for val in values.dropna().astype(str)
            for t in val.split(",") if t.strip()
        ))))
    )


def filter_long_df(df, max_cv=0.30, noise_floor=5.0):
    # Step A: Pivot to wide format temporarily for fast vectorized row math
    # Rows = genes, Columns = replicates, Values = TPM
    wide_df = df.pivot(
        index="gene_id", columns="replicate", values="TPM"
    ).reset_index()

    # Extract replicate columns automatically (handles rep1, rep2, rep3, etc.)
    rep_cols = df["replicate"].unique()
    rep_matrix = wide_df[rep_cols].to_numpy()

    # Step B: Compute metrics across rows (vectorized in C via NumPy)
    row_means = np.nanmean(rep_matrix, axis=1)
    row_stds = np.nanstd(rep_matrix, axis=1, ddof=1)

    # Avoid division by zero
    safe_means = np.where(row_means == 0, np.nan, row_means)
    row_cv = row_stds / safe_means

    # Step C: Build the passing masks
    is_low_noise = row_means <= noise_floor
    is_safe_variance = row_cv <= max_cv
    passing_mask = is_low_noise | is_safe_variance

    # Step D: Filter down to valid gene IDs
    valid_genes = wide_df.loc[passing_mask, "gene_id"]

    # Step E: Keep only the rows in the original DataFrame that belong to valid genes
    filtered_df = df[df["gene_id"].isin(valid_genes)].copy()

    return filtered_df


def merge_replicates(genome, condition, replicates):
    replicate_dfs = [
        read_replicate_expression(genome, condition, file_id, replicate)
        for replicate, file_id in replicates.items()
    ]
    if not replicate_dfs:
        return pd.DataFrame()

    warn_if_replicate_ids_differ(replicate_dfs, genome, condition)
    merged_df = pd.concat(replicate_dfs, ignore_index=True)

    numeric_cols = merged_df.select_dtypes(include="number").columns.tolist()
    if "TPM" not in numeric_cols:
        raise ValueError(f"{genome} {condition}: TPM must be numeric")

    transcript_df = summarize_transcript_ids(merged_df, genome, condition)
    filtered_df = filter_long_df(merged_df, max_cv=0.30, noise_floor=5.0)
    mean_df = (
        filtered_df.groupby("gene_id", as_index=False)[numeric_cols]
        .mean()
        .reset_index(drop=True)
    )
    final_df = transcript_df.merge(mean_df, on="gene_id", how="inner").sort_values("gene_id").reset_index(drop=True).rename(columns={
        "transcript_id(s)": "transcript_ids",
        "TPM": "mean_tpm"
    })

    return final_df


def filter_gene_expression(genome=None, condition=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_files = []

    for current_genome, conditions in GENE_EXP_FILENAMES.items():
        if genome is not None and current_genome != genome:
            continue

        for current_condition, replicates in conditions.items():
            if condition is not None and current_condition != condition:
                continue

            print(f"Processing {current_genome} - {current_condition}")
            final_df = merge_replicates(
                current_genome, current_condition, replicates)
            out_file = os.path.join(
                OUTPUT_DIR, f"{current_genome}_{current_condition}.tsv"
            )
            final_df.to_csv(out_file, sep="\t", index=False)
            output_files.append(out_file)
            print(f"Saved {len(final_df)} genes to {out_file}")

    return output_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge gene expression replicates by averaging numeric values per gene."
    )
    parser.add_argument("--genome", choices=GENE_EXP_FILENAMES.keys())
    parser.add_argument("--condition")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_gene_expression(genome=args.genome, condition=args.condition)
