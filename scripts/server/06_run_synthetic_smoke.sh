#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Synthetic Smoke Test ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

# Train 1 epoch
python run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --mode train \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/synthetic_smoke" \
    --device cpu || { echo "Training failed"; exit 1; }

# Test frozen
python run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --mode frozen \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/synthetic_smoke" \
    --device cpu || { echo "Frozen test failed"; exit 1; }

echo "Smoke test complete"
