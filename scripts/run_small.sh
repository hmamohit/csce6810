#!/usr/bin/env bash
set -euo pipefail

CD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CD"
PY="${PY:-python}"

echo "=== SMALL: convert chr21 memmap ==="
$PY scripts/run_pipeline.py convert \
  --organism hg38 \
  --chroms chr21

echo "=== SMALL: build NPZ + manifest + splits ==="
$PY scripts/run_pipeline.py build \
  --skip-ep \
  --resume \
  --require-memmap \
  --organism hg38 \
  --chroms chr21

echo "=== SMALL: train hg38 (ablation config, best so far) ==="
$PY scripts/train_experiment.py \
  --organism hg38 \
  --config ablation_small.yaml \
  --run-suffix chr21_smoke

echo "=== SMALL: test all split modes ==="
$PY scripts/test.py \
  --organism hg38 \
  --checkpoint ../data/output/hg38_multimodal_chr21_smoke/hg38_multimodal_best.pt \
  --split all

echo "=== SMALL: optional val diagnostic + plots ==="
$PY scripts/diagnose_val.py \
  --checkpoint ../data/output/hg38_multimodal_chr21_smoke/hg38_multimodal_best.pt \
  --out-dir ../data/output/diagnostics/hg38_chr21_smoke_val

$PY scripts/analyze_results.py \
  --predictions ../data/output/results/hg38_seen_*_predictions.csv

echo "DONE small pipeline"
