from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import re

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ORGANISM = "gm12878"
BIN_SIZE = 100_000
SINGLE_CHROMOSOME: Optional[str] = None

HIC_DIR = DATA_DIR / "hic_matrix" / ORGANISM
HIC_FALLBACK_DIR = DATA_DIR / "hic_matrix"
STRUCTURE_DIR = DATA_DIR / "structures" / f"{ORGANISM}_100kb"
GENE_EXPR_FILE = DATA_DIR / "gene_expression" / f"{ORGANISM}.tsv"
OUTPUT_DIR = DATA_DIR / "preprocessing"

CONTACT_OUT = OUTPUT_DIR / "contact_matrix.txt"
COORD_OUT = OUTPUT_DIR / "coordinates.txt"
GENE_OUT = OUTPUT_DIR / "gene_exp.txt"
BINS_OUT = OUTPUT_DIR / "bins.txt"

CHROMOSOME_PATTERN = re.compile(r"(chr(?:[0-9]+|X|Y|M))", re.IGNORECASE)


def normalize_chromosome(value: str) -> str:
    match = CHROMOSOME_PATTERN.search(value)
    if not match:
        raise ValueError(f"Could not parse chromosome from: {value}")

    token = match.group(1)[3:]
    if token.isdigit():
        return f"chr{int(token)}"
    return f"chr{token.upper()}"


def chromosome_sort_key(chromosome: str) -> Tuple[int, int, str]:
    token = chromosome.replace("chr", "")
    if token.isdigit():
        return 0, int(token), ""

    special = {"X": 23, "Y": 24, "M": 25}
    return 1, special.get(token, 999), token


def pick_hic_files(
        primary_dir: Path,
        fallback_dir: Optional[Path],
        bin_size: int,
) -> Tuple[Dict[str, Path], Path]:
    candidate_dirs: List[Path] = [primary_dir]
    if fallback_dir is not None and fallback_dir != primary_dir:
        candidate_dirs.append(fallback_dir)

    txt_files: List[Path] = []
    used_dir: Optional[Path] = None
    for directory in candidate_dirs:
        if directory.exists():
            files = sorted(directory.glob("*.txt"))
            if files:
                txt_files = files
                used_dir = directory
                break

    if not txt_files or used_dir is None:
        raise FileNotFoundError(
            f"No Hi-C matrix .txt files found in {primary_dir} or {fallback_dir}."
        )

    preferred_prefixes = [
        f"{ORGANISM}_{bin_size}_",
        f"{ORGANISM}_r{bin_size}_",
    ]
    preferred_files = [
        path
        for path in txt_files
        if any(path.name.startswith(prefix) for prefix in preferred_prefixes)
    ]

    if preferred_files:
        selected_files = preferred_files
    else:
        resolution_tag_files = [
            path
            for path in txt_files
            if f"_{bin_size}_" in path.name or f"_r{bin_size}_" in path.name
        ]
        selected_files = resolution_tag_files if resolution_tag_files else txt_files

    chromosome_to_file: Dict[str, Path] = {}
    for path in selected_files:
        try:
            chromosome = normalize_chromosome(path.stem)
        except ValueError:
            continue

        if chromosome in chromosome_to_file:
            continue
        chromosome_to_file[chromosome] = path

    if not chromosome_to_file:
        raise ValueError(
            f"Could not parse any chromosome names from files in {used_dir}."
        )

    ordered = dict(
        sorted(
            chromosome_to_file.items(),
            key=lambda item: chromosome_sort_key(item[0]),
        )
    )
    return ordered, used_dir


def pick_structure_files(structure_dir: Path) -> Dict[str, Path]:
    if not structure_dir.exists():
        raise FileNotFoundError(
            f"Structure directory not found: {structure_dir}")

    chromosome_to_file: Dict[str, Path] = {}
    for path in sorted(structure_dir.glob("*_structure.txt")):
        try:
            chromosome = normalize_chromosome(path.stem)
        except ValueError:
            continue
        chromosome_to_file[chromosome] = path

    if not chromosome_to_file:
        raise FileNotFoundError(
            f"No structure files matching '*_structure.txt' found in {structure_dir}."
        )

    return chromosome_to_file


def load_gene_expression_table(gene_expr_file: Path) -> Dict[str, List[Tuple[int, float]]]:
    if not gene_expr_file.exists():
        raise FileNotFoundError(
            f"Gene expression file not found: {gene_expr_file}")

    chromosome_to_records: Dict[str, List[Tuple[int, float]]] = {}
    with gene_expr_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_cols = {"chr", "tss", "gene_expr_log2_mean"}
        if reader.fieldnames is None or not expected_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Gene expression file is missing required columns: "
                "chr, tss, gene_expr_log2_mean"
            )

        for row in reader:
            try:
                chromosome = normalize_chromosome(row["chr"])
                tss = int(float(row["tss"]))
                expression = float(row["gene_expr_log2_mean"])
            except (KeyError, TypeError, ValueError):
                continue

            if not np.isfinite(expression):
                continue

            chromosome_to_records.setdefault(
                chromosome, []).append((tss, expression))

    return chromosome_to_records


def infer_hic_bin_count(matrix_path: Path) -> int:
    n_rows = 0
    n_cols = None

    with matrix_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if n_cols is None:
                n_cols = len(fields)
            elif len(fields) != n_cols:
                raise ValueError(
                    f"Inconsistent column count in {matrix_path}: "
                    f"expected {n_cols}, found {len(fields)}"
                )
            n_rows += 1

    if n_rows == 0 or n_cols is None:
        raise ValueError(f"Empty Hi-C matrix file: {matrix_path}")

    if n_rows != n_cols:
        raise ValueError(
            f"Expected square Hi-C matrix in {matrix_path}, got ({n_rows}, {n_cols})."
        )

    return n_rows


def load_selected_hic_rows(matrix_path: Path, row_mask: np.ndarray) -> List[str]:
    selected_rows: List[str] = []
    n_rows = 0
    n_cols = None

    with matrix_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if n_cols is None:
                n_cols = len(fields)
            elif len(fields) != n_cols:
                raise ValueError(
                    f"Inconsistent column count in {matrix_path}: "
                    f"expected {n_cols}, found {len(fields)}"
                )

            if n_rows >= len(row_mask):
                raise ValueError(
                    f"Hi-C file {matrix_path} has more rows than expected ({len(row_mask)})."
                )

            if row_mask[n_rows]:
                # Preserve raw row values and allow variable columns across chromosomes.
                selected_rows.append(" ".join(fields))
            n_rows += 1

    if n_rows != len(row_mask):
        raise ValueError(
            f"Hi-C file {matrix_path} row count mismatch: expected {len(row_mask)}, found {n_rows}."
        )

    if n_cols is None:
        raise ValueError(f"Empty Hi-C matrix file: {matrix_path}")

    if n_rows != n_cols:
        raise ValueError(
            f"Expected square Hi-C matrix in {matrix_path}, got ({n_rows}, {n_cols})."
        )

    return selected_rows


def load_structure_coordinates(
        structure_path: Path,
        n_bins: int,
        bin_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    coordinates = np.zeros((n_bins, 3), dtype=np.float32)
    found_mask = np.zeros(n_bins, dtype=bool)

    rows = np.genfromtxt(structure_path, names=True, dtype=None, encoding=None)
    if np.size(rows) == 0:
        return coordinates, found_mask, 0

    rows = np.atleast_1d(rows)
    row_count = len(rows)
    for row in rows:
        start = int(row["start"])
        idx = start // bin_size
        if 0 <= idx < n_bins:
            coordinates[idx, 0] = float(row["x"])
            coordinates[idx, 1] = float(row["y"])
            coordinates[idx, 2] = float(row["z"])
            found_mask[idx] = True

    return coordinates, found_mask, row_count


def build_avg_gene_bins(
        chromosome: str,
        n_bins: int,
        bin_size: int,
        chromosome_to_gene_records: Dict[str, List[Tuple[int, float]]],
) -> np.ndarray:
    values_sum = np.zeros(n_bins, dtype=np.float64)
    values_count = np.zeros(n_bins, dtype=np.int64)

    for tss, expression in chromosome_to_gene_records.get(chromosome, []):
        idx = tss // bin_size
        if 0 <= idx < n_bins:
            values_sum[idx] += expression
            values_count[idx] += 1

    values_avg = np.divide(
        values_sum,
        values_count,
        out=np.zeros(n_bins, dtype=np.float64),
        where=values_count > 0,
    )

    # Keep raw per-bin chromosome-wise averages (no scaling).
    return values_avg.astype(np.float32)


def build_processed_data() -> None:
    hic_files, used_hic_dir = pick_hic_files(
        HIC_DIR, HIC_FALLBACK_DIR, BIN_SIZE)
    structure_files = pick_structure_files(STRUCTURE_DIR)
    gene_records = load_gene_expression_table(GENE_EXPR_FILE)

    print(f"[INFO] Using Hi-C matrices from: {used_hic_dir}")
    print(f"[INFO] Hi-C chromosomes discovered: {len(hic_files)}")
    print(f"[INFO] Structure chromosomes discovered: {len(structure_files)}")
    print("[INFO] Output mode: contact + coordinates + averaged gene expression + bin IDs")
    print("[INFO] Scaling: disabled")
    print("[INFO] Epsilon replacement: disabled")

    hic_chromosomes = set(hic_files.keys())
    structure_chromosomes = set(structure_files.keys())
    common_chromosomes = sorted(
        hic_chromosomes & structure_chromosomes,
        key=chromosome_sort_key,
    )

    if SINGLE_CHROMOSOME is not None:
        target_chromosome = normalize_chromosome(SINGLE_CHROMOSOME)
        common_chromosomes = [
            chrom for chrom in common_chromosomes if chrom == target_chromosome
        ]
        print(f"[INFO] Single-chromosome mode enabled: {target_chromosome}")

    skipped_missing_structure = sorted(
        hic_chromosomes - structure_chromosomes,
        key=chromosome_sort_key,
    )
    skipped_missing_hic = sorted(
        structure_chromosomes - hic_chromosomes,
        key=chromosome_sort_key,
    )

    per_chr_items: List[Tuple[str, List[str], np.ndarray, np.ndarray, np.ndarray]] = []
    skipped_no_coordinates: List[str] = []
    skipped_size_mismatch: List[str] = []

    for chromosome in common_chromosomes:
        hic_path = hic_files[chromosome]
        n_bins = infer_hic_bin_count(hic_path)
        coordinates, coord_mask, structure_row_count = load_structure_coordinates(
            structure_files[chromosome],
            n_bins,
            BIN_SIZE,
        )

        if structure_row_count != n_bins:
            skipped_size_mismatch.append(
                f"{chromosome}(hic_rows={n_bins}, struct_rows={structure_row_count})"
            )

        matched_bins = int(np.sum(coord_mask))
        if matched_bins == 0:
            skipped_no_coordinates.append(
                f"{chromosome}(hic_rows={n_bins}, matched_bins=0)")
            continue

        if matched_bins < n_bins:
            dropped_bins = n_bins - matched_bins
            print(
                f"[FILTER] {chromosome:<5} dropped_bins_without_coordinates={dropped_bins} "
                f"kept_bins={matched_bins}"
            )

        coordinates = coordinates[coord_mask]

        gene_exp = build_avg_gene_bins(
            chromosome, n_bins, BIN_SIZE, gene_records)
        gene_exp = gene_exp[coord_mask]
        bin_ids = np.flatnonzero(coord_mask).astype(np.int64) + 1  # 1-based bin id.

        contact_rows = load_selected_hic_rows(hic_path, coord_mask)
        if len(contact_rows) != matched_bins:
            raise RuntimeError(
                f"Internal mismatch for {chromosome}: "
                f"contact_rows={len(contact_rows)} matched_bins={matched_bins}"
            )

        per_chr_items.append((chromosome, contact_rows, coordinates, gene_exp, bin_ids))

        print(
            f"[ADD] {chromosome:<5} bins_total={n_bins:<6} "
            f"contact_rows={len(contact_rows)} coords_shape={coordinates.shape} "
            f"gene_shape={gene_exp.shape} bins_shape={bin_ids.shape}"
        )

    if skipped_missing_structure:
        print(
            "[WARN] Skipped chromosomes missing structure files (ignored for all outputs): "
            + ", ".join(skipped_missing_structure)
        )

    if skipped_missing_hic:
        print(
            "[WARN] Structure-only chromosomes without Hi-C files (ignored for all outputs): "
            + ", ".join(skipped_missing_hic)
        )

    if skipped_size_mismatch:
        print(
            "[WARN] Chromosomes with structure/Hi-C row-count mismatch "
            "(filtered by available coordinate bins): "
            + ", ".join(skipped_size_mismatch)
        )

    if skipped_no_coordinates:
        print(
            "[WARN] Skipped chromosomes with zero matched coordinates "
            "(ignored for all outputs): "
            + ", ".join(skipped_no_coordinates)
        )

    if not per_chr_items:
        raise RuntimeError(
            "No chromosomes could be processed. Ensure matching Hi-C and structure files exist."
        )

    coordinate_rows: List[np.ndarray] = []
    gene_rows: List[np.ndarray] = []
    bin_rows: List[np.ndarray] = []
    contact_row_count = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CONTACT_OUT.open("w", encoding="utf-8") as handle:
        for chromosome, contact_rows, coordinates, gene_exp, bin_ids in per_chr_items:
            for row_text in contact_rows:
                handle.write(f"{row_text}\n")

            contact_row_count += len(contact_rows)
            coordinate_rows.append(coordinates)
            gene_rows.append(gene_exp)
            bin_rows.append(bin_ids)
            print(f"[PACK] {chromosome:<5} rows={coordinates.shape[0]}")

    coordinates = np.vstack(coordinate_rows)
    gene_exp = np.concatenate(gene_rows)
    bins = np.concatenate(bin_rows)

    if not (contact_row_count == coordinates.shape[0] == gene_exp.shape[0] == bins.shape[0]):
        raise RuntimeError(
            "Row alignment mismatch across outputs: "
            f"contact={contact_row_count}, coords={coordinates.shape[0]}, "
            f"gene_exp={gene_exp.shape[0]}, bins={bins.shape[0]}"
        )

    np.savetxt(COORD_OUT, coordinates, fmt="%.8g")
    np.savetxt(GENE_OUT, gene_exp, fmt="%.8g")
    np.savetxt(BINS_OUT, bins, fmt="%d")

    print("\n[DONE] Saved processed outputs:")
    print(f"       contact matrix : {CONTACT_OUT} rows={contact_row_count} (variable columns)")
    print(f"       coordinates    : {COORD_OUT} shape={coordinates.shape}")
    print(f"       gene expression: {GENE_OUT} shape={gene_exp.shape}")
    print(f"       bins           : {BINS_OUT} shape={bins.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build processed Hi-C, structure, and gene expression files."
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default=None,
        help="Optional chromosome to process, e.g., chr1",
    )
    args = parser.parse_args()

    if args.chromosome:
        SINGLE_CHROMOSOME = args.chromosome

    build_processed_data()
