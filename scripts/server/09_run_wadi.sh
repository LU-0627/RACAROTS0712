#!/usr/bin/env bash
set -euo pipefail

echo "=== Running WADI Experiment ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

python run_rd_carots.py \
    --config configs/rd_carots/wadi.yaml \
    --mode train \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/wadi" || { echo "WADI training failed"; exit 1; }

python run_rd_carots.py \
    --config configs/rd_carots/wadi.yaml \
    --mode frozen \
    --model RDCAROTS \
    --seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/wadi" || { echo "WADI testing failed"; exit 1; }

echo "WADI experiment complete"
