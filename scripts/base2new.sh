#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/base2new.sh DATA_ROOT DATASET [SEED]" >&2
  exit 2
fi
DATA_ROOT=$1
DATASET=$2
SEED=${3:-1}
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
RUN_DIR="$OUTPUT_ROOT/$DATASET/base/seed$SEED"

"$PYTHON" train.py --dataset "$DATASET" --root "$DATA_ROOT" --shots 16 --seed "$SEED" \
  --config-file configs/base_to_novel.yaml --output-dir "$RUN_DIR"

for SPLIT in base new; do
  "$PYTHON" train.py --dataset "$DATASET" --root "$DATA_ROOT" --shots 16 --seed "$SEED" \
    --config-file configs/base_to_novel.yaml --eval-only --model-dir "$RUN_DIR" \
    --output-dir "$OUTPUT_ROOT/$DATASET/${SPLIT}_eval/seed$SEED" \
    DATASET.SUBSAMPLE_CLASSES "$SPLIT"
done
