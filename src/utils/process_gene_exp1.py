import pandas as pd
import numpy as np
import os


ROOT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data"
GE_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/gene_exp"
EP_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/raw_data/enhancer_promoter"
PROCESSED_PATH = f"/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression"
RESOLUTION = [1000]

GENE_GTF = {
    "hg38": "hg38.ensGene.gtf",
    "mm10": "mm10.ensGene.gtf"
}

CHROM_SIZES = {
    "hg38": "hg38.chrom.sizes",
    "mm10": "mm10.chrom.sizes"
}

EP = {
    "hg38": "GRCh38-cCREs.bed",
    "mm10": "mm10-cCREs.bed"
}

GENE_EXP = {
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

PLS_window = 200
pELS_window = 2000
dELS_max_search_limit = 100000
EP_SEARCH_WINDOW = 100000


def get_gtf(genome):
    gtf_cols = ['chrom', 'source', 'feature', 'start',
                'end', 'score', 'strand', 'frame', 'attributes']
    df_gtf = pd.read_csv(
        f'{GE_PATH}/{GENE_GTF[genome]}', sep='\t', header=None, names=gtf_cols, comment="#")
    df_transcripts = df_gtf[df_gtf['feature'] == 'transcript'].copy()

    # Extract gene_id using vectorized string methods
    df_transcripts['gene_id_clean'] = df_transcripts['attributes'].str.extract(
        r'gene_id "([^"]+)"')[0].str.split('.').str[0]

    # Calculate TSS/TES using vectorized np.where
    # Converting to 0-based coordinates
    df_transcripts['start'] = df_transcripts['start'] - 1
    df_transcripts['TSS'] = np.where(
        df_transcripts['strand'] == '+', df_transcripts['start'], df_transcripts['end']) -1
    df_transcripts['TES'] = np.where(
        df_transcripts['strand'] == '+', df_transcripts['end'], df_transcripts['start']) - 1

    df_gene_bodies = df_transcripts.groupby('gene_id_clean').agg(
        chrom=('chrom', 'first'),
        strand=('strand', 'first'),
        gb_start=('start', 'min'),
        gb_end=('end', 'max'),
        TSS=('TSS', 'first'),
        TES=('TES', 'first')
    ).reset_index()

    df_gene_bodies['gene_len'] = df_gene_bodies['gb_end'] - \
        df_gene_bodies['gb_start']
    return df_gene_bodies


def compute_ep_boundaries(genes_df, ep_df, window_size):
    """Optimized interval matching using binary search (searchsorted)."""
    results = []
    for chrom in genes_df['chrom'].unique():
        c_genes = genes_df[genes_df['chrom'] == chrom].copy()
        c_ep = ep_df[ep_df['chrom'] == chrom].sort_values('ccre_start')

        if c_ep.empty:
            c_genes['EP_start'], c_genes['EP_end'] = np.nan, np.nan
            results.append(c_genes)
            continue

        starts, ends = c_ep['ccre_start'].values, c_ep['ccre_end'].values
        tss_vals = c_genes['TSS'].values
        res_starts, res_ends = np.full(
            len(tss_vals), np.nan), np.full(len(tss_vals), np.nan)

        for i, tss in enumerate(tss_vals):
            w_start, w_end = tss - window_size, tss + window_size
            # Find all cCREs where start <= window_end
            idx_end = np.searchsorted(starts, w_end, side='right')
            if idx_end > 0:
                # Filter candidates for end >= window_start
                cand_ends = ends[:idx_end]
                mask = cand_ends >= w_start
                if np.any(mask):
                    res_starts[i] = starts[:idx_end][mask].min()
                    res_ends[i] = cand_ends[mask].max()

        c_genes['EP_start'], c_genes['EP_end'] = res_starts, res_ends
        results.append(c_genes)
    return pd.concat(results) if results else genes_df


for genome, conditions in GENE_EXP.items():
    # Process structural data once per genome
    df_gene_bodies = get_gtf(genome)
    ccre_cols = ['chrom', 'ccre_start',
                 'ccre_end', 'd_id', 'e_id', 'ccre_class']
    df_ccre = pd.read_csv(
        f'{EP_PATH}/{EP[genome]}', sep='\t', header=None, names=ccre_cols, comment="#", usecols=[0, 1, 2, 3, 4, 5])

    print(f"Calculating boundaries for {genome}...")
    df_gene_with_ep = compute_ep_boundaries(
        df_gene_bodies, df_ccre, EP_SEARCH_WINDOW)
    df_gene_with_ep['EP_start'] = df_gene_with_ep['EP_start'].fillna(0).astype(int)
    df_gene_with_ep['EP_end'] = df_gene_with_ep['EP_end'].fillna(0).astype(int)
    df_gene_with_ep['EP_len'] = df_gene_with_ep['EP_end'] - df_gene_with_ep['EP_start']

    chrom_sizes = pd.read_csv(
        f'{GE_PATH}/{CHROM_SIZES[genome]}', sep='\t', names=['chrom', 'size'])

    for condition, replicates in conditions.items():
        for rep, file_id in replicates.items():
            print(f"Processing {genome} - {condition} - {rep}")
            df_expr = pd.read_csv(f'{GE_PATH}/{file_id}.tsv', sep='\t')
            df_expr['gene_id_clean'] = df_expr['gene_id'].str.split('.').str[0]

            # Only keep genes with expression > 0 to optimize downstream storage
            df_expr_filtered = df_expr[df_expr['TPM']
                                       > 0][['gene_id_clean', 'TPM']].copy()

            df_master_genes = pd.merge(
                df_gene_with_ep, df_expr_filtered, on='gene_id_clean', how='inner')

            final_dataframe = df_master_genes[[
                "chrom", "gene_id_clean", "gb_start", "gb_end", "strand",
                "TSS", "TES", "gene_len", "TPM", "EP_start", "EP_end", "EP_len"
            ]]

            for _, row in chrom_sizes.iterrows():
                chrom = row['chrom']
                chrom_genes = final_dataframe[final_dataframe['chrom'] == chrom].copy(
                )
                if chrom_genes.empty:
                    continue
                chrom_genes.sort_values(by=['gb_start', 'gb_end'], inplace=True)
                out_file = os.path.join(
                    f'{PROCESSED_PATH}/{genome}_{condition}_{rep}_{chrom}.txt')
                chrom_genes.to_csv(
                    f'{out_file}', sep='\t', index=False, header=False)

                print(
                    f"Saved processed data for {genome} - {condition} - {rep} - {chrom} to {out_file}")
