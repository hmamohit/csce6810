#!/usr/bin/env bash
set -euo pipefail

CD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CD"
PY="${PY:-python}"
WORKERS="${WORKERS:-32}"

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
# default config (split enc_pels/enc_dels, cosine scheduler)
$PY scripts/run_pipeline.py train --profile all

# optional: ablation config per profile (better val Pearson on small set)
# $PY scripts/train_experiment.py --organism hg38 --config ablation_small.yaml --run-suffix ablation
# $PY scripts/train_experiment.py --organism mm10 --config ablation_small.yaml --run-suffix ablation
# $PY scripts/train_experiment.py --organism multiorganism --config ablation_small.yaml --run-suffix ablation

echo "=== FULL: test all profiles, all split modes ==="
$PY scripts/run_pipeline.py test --profile all --split all

echo "=== FULL: analyze prediction CSVs ==="
$PY scripts/run_pipeline.py analyze --profile all

echo "DONE full pipeline"
