#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Ablation Studies ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

for model in RDCAROTS-no-regime RDCAROTS-no-delay-negative RDCAROTS-single-prototype; do
    echo "Running ablation: $model"
    
    python run_rd_carots.py \
        --config configs/rd_carots/synthetic.yaml \
        --mode train \
        --model "$model" \
        --seed 0 \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT/ablation_$model" || { echo "Ablation $model failed"; exit 1; }
    
    python run_rd_carots.py \
        --config configs/rd_carots/synthetic.yaml \
        --mode frozen \
        --model "$model" \
        --seed 0 \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT/ablation_$model" || { echo "Ablation $model failed"; exit 1; }
done

echo "Ablations complete"
