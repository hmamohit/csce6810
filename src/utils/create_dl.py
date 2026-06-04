import numpy as np
from numpy.array_api import unique_counts
import pandas as pd
import os
import torch
from collections import Counter


ROOT_PATH = "/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data"
MATRIX_PATH = f"{ROOT_PATH}/processed_raw_data/hic_matrix"
GENE_EXP_PATH = f"{ROOT_PATH}/processed_raw_data/gene_expression"

RESOLUTION = [5000]

CHROM_SIZES = {
    "hg38": "hg38.chrom.sizes",
    "mm10": "mm10.chrom.sizes"
}

DATA_DICT = {
    "hg38": {
        "dtag": {"rep1", "rep2", "rep3"},
        "dmso": {"rep1", "rep2", "rep3"},
        "auxin_6h": {"rep1", "rep2"},
        "auxin_no_treatment": {"rep1", "rep2"}
    },
    "mm10": {
        "pnd11_mature": {"rep1", "rep2"},
        "pnd11_immature": {"rep1", "rep2"},
        "pnd6_immature": {"rep1", "rep2"},
        "pnd6_precursors": {"rep1", "rep2"},
        "xen": {"rep1", "rep2"},
        "tsc": {"rep1", "rep2"},
        "pnd22": {"rep1", "rep2"}
    }
}


def save_dataset(dataset, save_path):
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")


def get_balanced_indices(y_new):
    y = y_new.view(-1).numpy()

    num_classes = 20
    bins = np.linspace(0, 1, num_classes + 1)

    classes = np.digitize(y, bins) - 1
    classes = np.clip(classes, 0, num_classes - 1)

    counter = Counter(classes)
    unique_counts = sorted(set(counter.values()), reverse=True)
    second_max = unique_counts[1]
    print("Second max count:", second_max)
    balanced_indices = []

    for c in counter.keys():
        idx = np.where(classes == c)[0]
        if len(idx) > second_max:
            selected = np.random.choice(idx, size=second_max, replace=False)
        else:
            selected = idx

        balanced_indices.extend(selected)
    balanced_indices = np.array(balanced_indices)

    return balanced_indices


for res in RESOLUTION:
    for organism, sample_dict in DATA_DICT.items():

        chrom_info_path = f"{ROOT_PATH}/raw_data/{CHROM_SIZES[organism]}"
        chrom_list = np.loadtxt(chrom_info_path, dtype=str)

        dataset = []
        for sample_name, replicates in sample_dict.items():
            for chromosome, chrom_size in chrom_list:
                chrom_size = int(chrom_size)
                matrix_path = f"{MATRIX_PATH}/{organism}_{sample_name}_{res}_{chromosome}.txt"
                if not os.path.exists(matrix_path):
                    continue

                num_bins = int(torch.ceil(torch.tensor(chrom_size / res)))

                matrix_np = np.loadtxt(matrix_path, delimiter="\t")
                x_chr_template = torch.tensor(matrix_np, dtype=torch.float32)
                x_chr_template = torch.log1p(x_chr_template)
                if x_chr_template.max() > x_chr_template.min():
                    x_chr_template = (x_chr_template - x_chr_template.min()) / \
                        (x_chr_template.max() - x_chr_template.min())

                abs_pos = torch.arange(
                    1, num_bins + 1, dtype=torch.float32).unsqueeze(1)
                abs_pos = torch.log1p(abs_pos)
                if abs_pos.max() > abs_pos.min():
                    abs_pos = (abs_pos - abs_pos.min()) / \
                        (abs_pos.max() - abs_pos.min())

                i = torch.arange(num_bins).unsqueeze(1)
                j = torch.arange(num_bins).unsqueeze(0)

                rel_pos = num_bins - torch.abs(i - j)
                rel_pos = torch.log1p(rel_pos)
                if rel_pos.max() > rel_pos.min():
                    rel_pos = (rel_pos - rel_pos.min()) / \
                        (rel_pos.max() - rel_pos.min())

                for replicate in replicates:
                    gene_exp_path = f"{GENE_EXP_PATH}/{organism}_{sample_name}_{replicate}_{res}_{chromosome}.txt"
                    if not os.path.exists(gene_exp_path):
                        continue

                    gene_exp = pd.read_csv(
                        gene_exp_path, delimiter="\t", header=None)
                    y_values = gene_exp.iloc[:, 3].values.reshape(-1, 1)

                    y_new = torch.tensor(y_values, dtype=torch.float32)
                    y_new = torch.log1p(y_new)
                    if y_new.max() > y_new.min():
                        y_new = (y_new - y_new.min()) / \
                            (y_new.max() - y_new.min())

                    x_new = x_chr_template.clone()

                    sampled_indices = get_balanced_indices(y_new)

                    for i in sampled_indices:
                        dataset.append({
                            "ftr": x_new[i],
                            "rel_pos": rel_pos[i],
                            "abs_pos": abs_pos[i],
                            "gene_exp": y_new[i]
                        })

                    print(
                        f"Added {organism} {sample_name} {chromosome} {replicate}")

        save_path = f"{ROOT_PATH}/processed_tensors/hic_{organism}_{res}_norm_select.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dataset(dataset, save_path)
        print(f"Saved: {save_path}")
