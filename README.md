# Multimodal Hi-C → TPM

Predict gene expression (TPM) from Hi-C over gene body, PLS, pELS, and dELS.

## Data (already on disk — do not re-export Hi-C)

| Asset | Path | Pattern |
|-------|------|---------|
| Hi-C | `data/processed_raw_data/hic_matrix/` | `{org}_{condition}_1000_{chrom}.txt` — **one matrix per condition** |
| Gene expr | `data/processed_raw_data/gene_expression/` | `{org}_{condition}_{rep}_{chrom}.txt` — **one row = one sample** |

Replicates share the same Hi-C matrix for a condition. GE files without matching Hi-C are skipped (e.g. mm10 `iv_*`, `pnd6`).

## Train / test splits

Splits are **condition × replicate** (not chromosome holdout). Defined in `configs/default.yaml` and written to `data/processed_tensors/splits/{profile}_1000_splits.json`.

| Profile | Train pool | Test modes |
|---------|------------|------------|
| **hg38** | dtag, dmso (rep1–2), auxin_6h | seen (dtag), unseen_rep (dtag rep3), unseen_condition (auxin_no_treatment), cross_org (mm10 xen) |
| **mm10** | pnd11_*, pnd6_immature, pnd6_precursors (rep1), pnd22, tsc | seen (pnd11_immature), unseen_rep (pnd6_precursors rep2), unseen_condition (xen), cross_org (hg38 auxin_no_treatment) |
| **multiorganism** | hg38 + mm10 train pools (excludes hg38 auxin_no_treatment, mm10 xen) | seen (hg38 dtag + mm10 pnd11_immature), unseen (hg38 auxin_no_treatment + mm10 xen) |

Validation = random 90/10 from each profile's train pool (`train_val_ratio: 0.9`, seed 42).

## Pipeline

Use conda env with `numpy`, `pandas`, `torch`, `pyyaml`, `scipy`, `tqdm` (see `environment.yml`).

### Runner (recommended)

Hi-C text matrices are huge (hg38 chr1 ≈ 405 GB). **Convert to memmap once**, then NPZ build is fast with exact bin slicing.

```bash
cd csce6810

# Step 0 — convert text Hi-C → float32 memmap (one-time, resumable; parallel OK)
python scripts/run_pipeline.py convert --workers 4
# Smoke test small chrom first:
python scripts/run_pipeline.py convert --organism hg38 --chroms chr21

# Step 1 — build NPZ from memmap (one matrix load per chrom, all reps). Skip EP if JSON exists.
python scripts/run_pipeline.py build --skip-ep --resume --require-memmap
python scripts/run_pipeline.py build --skip-ep --organism hg38 --chroms chr21 --resume

# Step 2 — after config/code change: refresh manifest + splits only (fast)
python scripts/run_pipeline.py splits

# Step 3–5 — train / test / analyze
python scripts/run_pipeline.py train --profile hg38
python scripts/run_pipeline.py test --profile all
python scripts/run_pipeline.py analyze --profile hg38

# Full chain with convert (smoke chrom)
python scripts/run_pipeline.py all --convert --skip-ep --profile hg38 --chroms chr21 --resume
```

**Time expectations:** chr21 convert ~10–20 min; chr1 convert ~3 hr (once). NPZ build after memmap: minutes–hours. Use `--workers 4` on convert for parallel chromosomes/conditions.

### Manual steps (same as runner)

```bash
cd csce6810

# 0. Convert Hi-C text → memmap
python scripts/convert_hic_memmap.py --organism hg38 --chroms chr21

# 1. Build NPZ + manifest + splits (uses memmap)
python scripts/build_dataset.py --skip-ep --resume --require-memmap
python scripts/build_dataset.py --skip-ep --organism hg38 --chroms chr21 --resume

# 2. Refresh splits only (scans existing NPZ, no Hi-C I/O)
python scripts/build_dataset.py --only-manifest

# 3. Train
python scripts/train_hg38.py
python scripts/train_mm10.py
python scripts/train_multiorganism.py

# 4. Test
python scripts/test.py --organism hg38 \
  --checkpoint ../data/output/hg38_multimodal/hg38_multimodal_best.pt --split all

# 5. Plots
python scripts/analyze_results.py \
  --predictions ../data/output/results/hg38_seen_*_predictions.csv
```

Progress bars (`tqdm`) run during NPZ build, EP region processing, manifest expansion, validation, and test inference.

Optional (only if Hi-C missing): `python scripts/cool_to_sqr.py`

## Sample unit

- **Training sample** = one gene row in a replicate GE file
- **Hi-C** = loaded once per `(organism, condition, chrom)` and reused for rep1/rep2/rep3
- **Target** = raw TPM (log-scaled in loader; metrics on raw TPM)

## Layout

- `src/data/sample_registry.py` — pair GE ↔ Hi-C
- `src/data/hic_memmap.py` — text Hi-C → float32 memmap cache
- `src/data/build_multimodal_set.py` — NPZ shards + manifest (chrom-batch, memmap-first)
- `src/data/splits.py` — condition/replicate train/val/test manifests
- `src/models/multimodal_transformer.py` — multimodal encoder + cross-attn
- `scripts/` — CLI entry points

Legacy: `src/train.py`, `src/model.py`, `src/utils/create_dl.py`
