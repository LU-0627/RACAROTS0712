#!/usr/bin/env bash
# Run WADI experiments

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"

echo "Running WADI experiments..."

if [ ! -d "$DATA_ROOT/WADI" ]; then
    echo "ERROR: WADI data not found at $DATA_ROOT/WADI"
    exit 1
fi

# RDCAROTS on WADI
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/wadi.yaml \
    --model RDCAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/wadi"

echo "✓ WADI experiments complete."
