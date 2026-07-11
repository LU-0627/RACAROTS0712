#!/usr/bin/env bash
set -euo pipefail

echo "=== Resume Experiment ==="

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_path> <checkpoint_dir>"
    exit 1
fi

CONFIG_PATH="$1"
CHECKPOINT_DIR="$2"
DATA_ROOT="${DATA_ROOT:-data}"

python run_rd_carots.py \
    --config "$CONFIG_PATH" \
    --mode train \
    --resume \
    --checkpoint "$CHECKPOINT_DIR/checkpoint_best.pth" \
    --data-root "$DATA_ROOT" \
    --output-root "$CHECKPOINT_DIR" || { echo "Resume failed"; exit 1; }

echo "Experiment resumed"
