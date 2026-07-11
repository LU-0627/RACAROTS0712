#!/usr/bin/env bash
# Resume training from checkpoint

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_yaml> <checkpoint_path>"
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Resuming training from $CHECKPOINT..."

$PYTHON_BIN run_rd_carots.py \
    --config "$CONFIG" \
    --mode train \
    --checkpoint "$CHECKPOINT" \
    --resume

echo "✓ Training resumed and completed."
