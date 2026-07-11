#!/usr/bin/env bash
# Run full synthetic experiments

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"

echo "Running full synthetic experiments..."

# CAROTS baseline
echo "Training CAROTS..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --model CAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/synthetic/carots"

# RDCAROTS full
echo "Training RDCAROTS..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --model RDCAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/synthetic/rdcarots"

echo "✓ Full synthetic experiments complete."
