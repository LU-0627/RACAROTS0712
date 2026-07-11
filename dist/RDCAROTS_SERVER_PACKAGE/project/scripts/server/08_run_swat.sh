#!/usr/bin/env bash
# Run SWaT experiments

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"

echo "Running SWaT experiments..."

if [ ! -d "$DATA_ROOT/SWaT" ]; then
    echo "ERROR: SWaT data not found at $DATA_ROOT/SWaT"
    exit 1
fi

# RDCAROTS on SWaT
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/swat.yaml \
    --model RDCAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/swat"

echo "✓ SWaT experiments complete."
