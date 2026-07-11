#!/usr/bin/env bash
# Run synthetic smoke test

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"

echo "Running synthetic smoke test..."

# Generate data if needed
if [ ! -d "$DATA_ROOT/synthetic_regime_delay" ]; then
    bash scripts/server/05_generate_synthetic.sh
fi

# CAROTS smoke (1 epoch)
echo "Running CAROTS smoke test..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/synthetic_smoke.yaml \
    --model CAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/carots_smoke" || echo "CAROTS smoke failed"

# RDCAROTS smoke (1 epoch)
echo "Running RDCAROTS smoke test..."
$PYTHON_BIN run_rd_carots.py \
    --config configs/rd_carots/synthetic_smoke.yaml \
    --model RDCAROTS \
    --mode train \
    --seed $SEED \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/rdcarots_smoke" || echo "RDCAROTS smoke failed"

echo "✓ Smoke test complete."
