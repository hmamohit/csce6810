import cooler
import numpy as np

ROOT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/csce6810/data"
RAW_PATH = f"{ROOT_PATH}/raw_data"
PROCESSED_PATH = f"{ROOT_PATH}/processed_raw_data/hic_matrix"

HIC_DICT = {
    "hg38": {
        "dtag": "4DNFINQKY5AH_dtag_v1_tbx5",
        "dmso": "4DNFIKMJRM5C_dmso",
        "auxin_6h": "4DNFI35AE3O3_auxin_6h",
        "auxin_no_treatment": "4DNFIXB4O92R_auxin_no_treatment"
    },
    "mm10": {
        "pnd11_mature": "4DNFIVWXJQR1_pnd11_mature",
        "pnd11_immature": "4DNFI14FXOOU_pnd11_immature",
        "pnd6_immature": "4DNFIS1SRPWR_pnd6_immature",
        "pnd6_precursors": "4DNFI8LDZDN9_pnd6_precursors",
        "xen": "4DNFIQEYK87U_xen",
        "tsc": "4DNFIIHXG8KW_tsc",
        "pnd22": "4DNFISZ88WZA_pnd22"
    }
}

RESOLUTION = [5000]

for organism, hic_dict in HIC_DICT.items():
    for sample_name, filename in hic_dict.items():
        print(f"Processing {organism} {sample_name} with ID {filename}")
        for res in RESOLUTION:
            cool_file = cooler.Cooler(f"{RAW_PATH}/{filename}_{res}_KR.cool")
            for chromosome, chr_size in zip(cool_file.chromnames, cool_file.chromsizes):
                chromosome_name = chromosome if chromosome.startswith(
                    "chr") else f"chr{chromosome}"
                chr_matrix = cool_file.matrix(balance=False).fetch(chromosome)
                print(
                    f"Shape of the chromosome {chromosome_name} square matrix: {chr_matrix.shape}")
                np.savetxt(
                    X=chr_matrix, fname=f"{PROCESSED_PATH}/{organism}_{sample_name}_{res}_{chromosome_name}.txt", delimiter="\t", fmt="%1.4f")
