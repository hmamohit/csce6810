import cooler
import numpy as np

INPUT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/data/gm12878"
OUTPUT_DIR = "/home/hc0783.unt.ad.unt.edu/workspace/csce6810/data/hic_matrix/gm12878"

ORGANISM = "gm12878"
RES = [100000]


def minmax_scale_with_epsilon(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled = (matrix - min_val) / (max_val - min_val)
    return scaled


for res in RES:
    c = cooler.Cooler(f"{INPUT_DIR}/{ORGANISM}_{res}.cool")
    for chromosome, chr_size in zip(c.chromnames, c.chromsizes):
        chromosome_name = chromosome if chromosome.startswith(
            "chr") else f"chr{chromosome}"
        chr_matrix = c.matrix(balance=False).fetch(chromosome)
        chr_matrix = minmax_scale_with_epsilon(chr_matrix)
        print(
            f"Shape of the chromosome {chromosome_name} square matrix: {chr_matrix.shape}")
        np.savetxt(
            X=chr_matrix, fname=f"{OUTPUT_DIR}/{ORGANISM}_{res}_{chromosome_name}.txt", delimiter=" ")
