#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/cross_domain.sh DATA_ROOT SOURCE TARGET [SEED]" >&2
  exit 2
fi
DATA_ROOT=$1
SOURCE=$2
TARGET=$3
SEED=${4:-1}
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
RUN_DIR="$OUTPUT_ROOT/cross_domain/$SOURCE/seed$SEED"
CKPT_DIR="$RUN_DIR/SaMoEPromptLearner"

if [[ ! -f "$CKPT_DIR/model-best.pth.tar" || ! -f "$CKPT_DIR/model.pth.tar-100" ]]; then
  "$PYTHON" train.py --dataset "$SOURCE" --root "$DATA_ROOT" --shots 16 --seed "$SEED" \
    --config-file configs/cross_domain.yaml --output-dir "$RUN_DIR"
fi

TARGET_ARGS=()
TARGET_SUFFIX="$TARGET"
if [[ -n "${TARGET_MODALITY:-}" ]]; then
  TARGET_ARGS+=(--modality "$TARGET_MODALITY")
  TARGET_SUFFIX="${TARGET}-${TARGET_MODALITY}"
fi
"$PYTHON" train.py --dataset "$TARGET" --root "$DATA_ROOT" --shots -1 --seed "$SEED" \
  --config-file configs/cross_domain.yaml --eval-only --model-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_ROOT/cross_domain/${SOURCE}-to-${TARGET_SUFFIX}/seed$SEED" \
  "${TARGET_ARGS[@]}"
