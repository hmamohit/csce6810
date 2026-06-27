# Multimodal Hi-C → TPM Pipeline Guide

## 1. Data structure for training

### Pipeline stages

```
Raw Hi-C .txt + Gene expression .txt
        ↓ convert (one-time)
float32 memmap cache
        ↓ build
NPZ shards (per organism/condition/rep/chrom)
        ↓ manifest scan
manifest CSV + splits JSON
        ↓ expand_manifest_to_genes
one training sample per gene
        ↓ DataLoader + collate
batched tensors → model
```

### Sample unit

- **One training sample** = one gene in one replicate GE file
- **Hi-C** shared per `(organism, condition, chrom)` — rep1/rep2/rep3 reuse same matrix
- **Target** = raw TPM; model trains on scaled `log1p(TPM)`

### Split logic

Splits are **condition × replicate** holdouts (not chromosome holdout). Defined in `configs/default.yaml`, written to:

`data/processed_tensors/splits/{profile}_1000_splits.json`

| Profile | Train pool | Test modes |
|---------|------------|------------|
| hg38 | dtag, dmso (rep1–2), auxin_6h | seen, unseen_rep, unseen_condition, cross_org |
| mm10 | pnd11_*, pnd6_*, pnd22, tsc, etc. | seen, unseen_rep, unseen_condition, cross_org |
| multiorganism | hg38 + mm10 train pools | seen, unseen |

Val = random 90/10 from train pool (`train_val_ratio: 0.9`, seed 42).

### Four Hi-C regions per gene

| Region | Window | Purpose |
|--------|--------|---------|
| gene_body | gene span + pad | gene body contact profile |
| PLS | TSS ±200 bp | promoter-linked |
| pELS | ±2000 bp | proximal enhancer |
| dELS | ±100 kb | distal enhancer |

---

## 2. Example data structures

### Example A — NPZ shard record

Path pattern:
`data/processed_tensors/multimodal/{organism}/{condition}/{replicate}/{chrom}.npz`

```json
{
  "gene_id": ["ENSG00000277117", "..."],
  "strand": ["+", "..."],
  "TSS": [12345, "..."],
  "TPM": [1.33, "..."],
  "has_pls": [true, "..."],
  "gene_body": ["array(len≈59)", "..."],
  "PLS": ["array(len≈33)", "..."],
  "pELS": ["array(len≈85)", "..."],
  "dELS": ["array(len≈132)", "..."],
  "organism": "hg38",
  "sample": "dtag",
  "replicate": "rep1",
  "chrom": "chr21",
  "hic_path": ".../hg38_dtag_1000_chr21.txt",
  "ge_path": ".../hg38_dtag_rep1_chr21.txt"
}
```

Array shapes inside NPZ:

| Key | Shape | dtype |
|-----|-------|-------|
| gene_id, strand | (N,) | object |
| TSS | (N,) | int64 |
| TPM | (N,) | float32 |
| has_pls | (N,) | bool |
| gene_body, PLS, pELS, dELS | (N,) object of 1D float arrays | object |
| organism, sample, replicate, chrom | scalar | str |

### Example B — Split manifest row → expanded gene record

**Shard-level** (in `splits.json`):

```json
{
  "organism": "hg38",
  "condition": "dtag",
  "replicate": "rep1",
  "chrom": "chr21",
  "shard_path": ".../multimodal/hg38/dtag/rep1/chr21.npz",
  "n_genes": 251
}
```

**Gene-level** (after `expand_manifest_to_genes`):

```json
{
  "organism": "hg38",
  "condition": "dtag",
  "replicate": "rep1",
  "chrom": "chr21",
  "shard_path": ".../chr21.npz",
  "gene_idx": 0
}
```

**Loader output** (one `__getitem__`):

```python
{
  "gene_body": Tensor[L_gb],      # variable length per gene
  "PLS": Tensor[L_pls],
  "pELS": Tensor[L_pels],
  "dELS": Tensor[L_dels],
  "has_pls": bool,
  "tpm_target": float,            # scaled log1p(TPM)
  "tpm_raw": float,               # raw TPM for metrics
  "gene_id": "ENSG00000277117",
  "organism": "hg38",
  "sample": "dtag",
  "replicate": "rep1",
  "chrom": "chr21"
}
```

**Batched** (`collate_multimodal`): pad variable-length vectors → `(B, max_len)` + masks.

---

## 3. Model architecture

Class: `MultimodalGeneExpression` (`src/models/multimodal_transformer.py`)

```
gene_body ──→ HiCProjection + modality_emb[0] ──→ enc_gene ──┐
PLS         ──→ HiCProjection + modality_emb[1] ──→ enc_pls ──┤
pELS        ──→ HiCProjection + modality_emb[2] ──→ enc_pels ─┼→ concat → h_ep
dELS        ──→ HiCProjection + modality_emb[3] ──→ enc_dels ─┘
                                                              │
h_gene ← cross_ep_to_gene(h_gene, h_ep)                       │
h_gene ← cross_pls_to_gene(h_gene, h_pls)                     │
                                                              │
mean-pool(h_gene, h_ep, h_pls) + has_pls flag                 │
        → MLP head → predicted scaled log1p(TPM)
```

| Component | Role |
|-----------|------|
| `HiCProjection` | map 1D Hi-C vector → fixed `d_model` seq |
| `modality_emb` | 4 learnable modality IDs (gene/PLS/pELS/dELS) |
| `enc_gene`, `enc_pls`, `enc_pels`, `enc_dels` | separate TransformerEncoder stacks |
| `cross_ep_to_gene` | gene attends enhancer (pELS+dELS) |
| `cross_pls_to_gene` | gene attends PLS |
| `head` | `Linear(d_model*3+1 → d_model → 1)` |

Default config (`default.yaml`): `d_model=256`, `num_encoders=4`, `num_heads=4`.
Ablation (`ablation_small.yaml`): `num_encoders=2`.

Training loss: **MSE on scaled log1p(TPM)**.
Checkpoint selection: **best val Pearson on raw TPM**.

---

## 4. Validation metrics

Logged each epoch in `train.log`:

| Metric | Space | Meaning |
|--------|-------|---------|
| `train_loss` | scaled log1p | MSE train loss |
| `val_loss` | scaled log1p | MSE val loss |
| `val_pearson` | **raw TPM** | Pearson r — **checkpoint criterion** |
| `val_mse` | **raw TPM** | mean squared error |
| `pearson_scaled` | scaled log1p | logged internally, not used for best ckpt |

**Scaling:**

```
tpm_scaled = (log1p(tpm_raw) - mean) / std    # mean/std fit on train pool
pred_raw   = expm1(pred_scaled * std + mean)
```

**Test metrics** (`scripts/test.py`): same `compute_metrics` on raw TPM — pearson, spearman, mse, rmse, mae, r2.

**Interpretation:**

- `val_pearson` ≈ 0.25 on chr21 smoke = weak but non-zero signal on tiny slice
- `unseen_condition` usually lower than `seen` / `unseen_rep` (harder generalization)
- `cross_org` needs **mm10 manifest built** — empty on hg38-only smoke run

Early stopping: patience=25 epochs without val pearson improvement.

---

## 5. Full run script

### Option A — bash wrapper (recommended)

`scripts/run_full.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CD"
PY="${PY:-python}"
WORKERS="${WORKERS:-4}"

echo "=== FULL: convert all Hi-C text → memmap (parallel) ==="
$PY scripts/run_pipeline.py convert \
  --organism all \
  --workers "$WORKERS"

echo "=== FULL: build all NPZ + manifest + splits ==="
$PY scripts/run_pipeline.py build \
  --skip-ep \
  --resume \
  --require-memmap \
  --organism all

echo "=== FULL: train all profiles ==="
$PY scripts/run_pipeline.py train --profile all

echo "=== FULL: test all profiles, all split modes ==="
$PY scripts/run_pipeline.py test --profile all --split all

echo "=== FULL: analyze prediction CSVs ==="
$PY scripts/run_pipeline.py analyze --profile all

echo "DONE full pipeline"
```

Run:

```bash
cd csce6810
conda activate hicgat   # or your env
WORKERS=4 bash scripts/run_full.sh
```

### Option B — step-by-step CLI

```bash
cd csce6810

# 0. Convert text Hi-C → memmap (one-time, resumable, ~hours)
python scripts/run_pipeline.py convert --organism all --workers 4

# 1. Build NPZ shards + manifest + splits
python scripts/run_pipeline.py build \
  --skip-ep --resume --require-memmap --organism all

# 2. Train (hg38, mm10, multiorganism)
python scripts/run_pipeline.py train --profile all

# Optional: ablation config (num_encoders=2, better on small set)
# python scripts/train_experiment.py --organism hg38 --config ablation_small.yaml --run-suffix ablation

# 3. Test all split modes
python scripts/run_pipeline.py test --profile all --split all

# 4. Plots
python scripts/run_pipeline.py analyze --profile all
```

### Smoke test (chr21 only)

```bash
cd csce6810
bash scripts/run_small.sh
```

### Outputs

| Step | Output |
|------|--------|
| convert | `data/processed_tensors/hic_memmap/` |
| build | `data/processed_tensors/multimodal/{org}/.../*.npz` |
| build | `data/processed_tensors/multimodal/manifest_{org}_1000.csv` |
| build | `data/processed_tensors/splits/{profile}_1000_splits.json` |
| train | `data/output/{profile}_multimodal/{profile}_multimodal_best.pt` |
| train | `data/logs/{profile}_multimodal/train.log` |
| test | `data/output/results/{profile}_{split}_{timestamp}_metrics.csv` |
| test | `data/output/results/{profile}_{split}_{timestamp}_predictions.csv` |

### Time expectations

- chr21 convert: ~10–20 min (smoke)
- chr1 convert: ~3 hr (once)
- Full genome convert: days (parallel with `--workers 4`)
- NPZ build after memmap: minutes–hours per organism
