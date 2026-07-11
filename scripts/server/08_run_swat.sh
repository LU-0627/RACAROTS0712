#!/usr/bin/env bash
set -euo pipefail

echo "=== Running SWaT Experiment ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

python run_rd_carots.py \
    --config configs/rd_carots/swat.yaml \
    --mode train \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/swat" || { echo "SWaT training failed"; exit 1; }

python run_rd_carots.py \
    --config configs/rd_carots/swat.yaml \
    --mode frozen \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/swat" || { echo "SWaT testing failed"; exit 1; }

echo "SWaT experiment complete"
