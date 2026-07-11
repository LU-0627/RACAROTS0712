#!/usr/bin/env bash
# Run ablation studies

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"

echo "Running ablation studies..."

# No regime
echo "Ablation: no-regime..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/ablation_no_regime.yaml \
    --model RDCAROTS-no-regime \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/ablations/no_regime"

# No delay negative
echo "Ablation: no-delay-negative..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/ablation_no_delay_negative.yaml \
    --model RDCAROTS-no-delay-negative \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/ablations/no_delay_negative"

# Single prototype
echo "Ablation: single-prototype..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/ablation_single_prototype.yaml \
    --model RDCAROTS-single-prototype \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/ablations/single_prototype"

echo "✓ Ablation studies complete."
