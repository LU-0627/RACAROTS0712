#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Full Synthetic Experiment ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

for seed in 0 1 2; do
    echo "Training seed $seed"
    
    python run_rd_carots.py \
        --config configs/rd_carots/synthetic.yaml \
        --mode train \
        --model RDCAROTS \
        --seed $seed \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT/synthetic/seed_$seed" || { echo "Training failed for seed $seed"; exit 1; }
    
    python run_rd_carots.py \
        --config configs/rd_carots/synthetic.yaml \
        --mode frozen \
        --model RDCAROTS \
        --seed $seed \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT/synthetic/seed_$seed" || { echo "Testing failed for seed $seed"; exit 1; }
done

echo "Full synthetic experiments complete"
