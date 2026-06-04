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

# Standard ENCODE-compliant distance windows
PLS_window = 200    # +/- 200 bp around TSS
pELS_window = 2000   # +/- 2 kb around TSS

# Distal elements can loop from massive distances.
# For deep learning model inputs (like 1D CNNs or Transformers),
# researchers typically expand the window up to 50,000 or 100,000 bp
# to ensure dELS elements are captured within the input vector context.
dELS_max_search_limit = 100000

EP_SEARCH_WINDOW = 100000


def get_gtf(genome):
    gtf_cols = ["chrom", "source", "feature", "start",
                "end", "score", "strand", "frame", "attributes"]
    df = pd.read_csv(
        f'{GE_PATH}/{GENE_GTF[genome]}', sep='\t', header=None, names=gtf_cols, comment="#")
    df["gene_id"] = df["attributes"].str.extract(r'gene_id "([^"]+)"')
    df = df[df["feature"] == "transcript"].copy()
    df.drop(["source", "feature", "score", "attributes",
            "frame"], axis=1, inplace=True)
    res = df.groupby("gene_id").agg(
        {"chrom": "first", "start": "min", "end": "max", "strand": "first"}).reset_index()
    res["TSS"] = np.where(res["strand"] == "+", res["start"], res["end"])-1
    res["TES"] = np.where(res["strand"] == "+", res["end"], res["start"])-1
    res["gene_len"] = np.abs(res["end"] - res["start"])

    return res.sort_values(by=["chrom", "start", "end"]).reset_index(drop=True)


for genome, conditions in GENE_EXP.items():
    gtf_df = get_gtf(genome)
    ep_df = pd.read_csv(
        f'{EP_PATH}/{EP[genome]}', sep='\t', header=None, comment="#")
    ep_df = ep_df.iloc[:, [0, 1, 2, 3, 5]]
    ep_df.columns = ["chrom", "start", "end", "id", "ccre_class"]
    ep_df.sort_values(by=["chrom", "start", "end"], inplace=True)
    for condition, replicates in conditions.items():
        for rep, file_id in replicates.items():
            print(f"Processing {genome} - {condition} - {rep}")
            exp_df = pd.read_csv(f'{GE_PATH}/{file_id}.tsv', sep='\t')
            exp_df['gene_id'] = exp_df['gene_id'].astype(
                str).str.split('.').str[0]
            exp_df = exp_df[exp_df["TPM"] > 0].copy()
            tmp_gtf_df = gtf_df.copy()
            merged_df = pd.merge(tmp_gtf_df, exp_df, on='gene_id')
            final_output = merged_df[[
                "chrom", "gene_id", "start", "end", "strand", "TSS", "TES", "gene_len", "TPM", "FPKM"]]
            chrom_sizes = pd.read_csv(
                f'{GE_PATH}/{CHROM_SIZES[genome]}', sep='\t', names=['chrom', 'size'])
            coords_df = final_output.copy()
            for _, row in chrom_sizes.iterrows():
                chrom = row['chrom']
                c_size = row['size']
                chrom_genes = coords_df[coords_df['chrom'] == chrom].copy()
                chrom_ep = ep_df[ep_df['chrom'] == chrom].copy()

                # chrom_genes['EP'] = chrom_genes.apply(
                #     lambda x: chrom_ep[
                #         (chrom_ep['chrom'] == x['chrom']) &
                #         (chrom_ep['start'] <= x['TSS'] + EP_SEARCH_WINDOW) &
                #         (chrom_ep['end'] >= x['TSS'] - EP_SEARCH_WINDOW)
                #     ][['id', "ccre_class", 'start', 'end']].set_index('id').to_dict(orient='index'),
                #     axis=1
                # )

                # chrom_genes['EP'] = chrom_genes.apply(
                #     lambda x: (
                #         lambda matched_df: {
                #             "start": int(matched_df['start'].min()),
                #             "end": int(matched_df['end'].max())
                #         } if not matched_df.empty else {"start": None, "end": None}
                #     )(
                #         chrom_ep[
                #             (chrom_ep['chrom'] == x['chrom']) &
                #             (chrom_ep['start'] <= x['TSS'] + EP_SEARCH_WINDOW) &
                #             (chrom_ep['end'] >= x['TSS'] - EP_SEARCH_WINDOW)
                #         ]
                #     ),
                #     axis=1
                # )

                boundary_df = chrom_genes.apply(
                    lambda x: (
                        lambda matched: pd.Series(
                            [matched['start'].min(), matched['end'].max()], index=['ep_start', 'ep_end'])
                        if not matched.empty
                        else pd.Series([None, None], index=['ep_start', 'ep_end'])
                    )(
                        chrom_ep[
                            (chrom_ep['chrom'] == x['chrom']) &
                            (chrom_ep['start'] <= x['TSS'] + EP_SEARCH_WINDOW) &
                            (chrom_ep['end'] >= x['TSS'] - EP_SEARCH_WINDOW)
                        ]
                    ),
                    axis=1
                )
                chrom_genes = pd.concat([chrom_genes, boundary_df], axis=1)
                chrom_genes['ep_len'] = chrom_genes.apply(
                    lambda x: x['ep_end'] - x['ep_start'] if not pd.isnull(
                        x['ep_start']) and not pd.isnull(x['ep_end']) else None,
                    axis=1
                )

                # chrom_genes['pELS'] = chrom_genes.apply(
                #     lambda x: chrom_ep[
                #         (chrom_ep['ccre_class'] == 'pELS') &
                #         (chrom_ep['chrom'] == x['chrom']) &
                #         (chrom_ep['start'] <= x['TSS'] + PLS_window) &
                #         (chrom_ep['end'] >= x['TSS'] - PLS_window)
                #     ][['id', 'start', 'end']].set_index('id').to_dict(orient='index'),
                #     axis=1
                # )
                # chrom_genes['dELS'] = chrom_genes.apply(
                #     lambda x: chrom_ep[
                #         (chrom_ep['ccre_class'] == 'dELS') &
                #         (chrom_ep['chrom'] == x['chrom']) &
                #         (chrom_ep['start'] <= x['TSS'] + PLS_window) &
                #         (chrom_ep['end'] >= x['TSS'] - PLS_window)
                #     ][['id', 'start', 'end']].set_index('id').to_dict(orient='index'),
                #     axis=1
                # )

                out_file = os.path.join(
                    f'{PROCESSED_PATH}/{genome}_{condition}_{rep}_{chrom}.txt')
                chrom_genes.to_csv(
                    f'{out_file}', sep='\t', index=False, header=False)
