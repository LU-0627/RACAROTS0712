#!/usr/bin/env bash
set -euo pipefail

echo "=== Generating Synthetic Data ==="

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_DIR="$DATA_ROOT/synthetic_regime_delay"

python -m datasets.synthetic_regime_delay \
    --output-dir "$OUTPUT_DIR" \
    --n-inputs 20 \
    --n-outputs 30 \
    --n-regimes 3 \
    --n-train 10000 \
    --n-val 2000 \
    --n-test 5000 \
    --seed 0 || { echo "Generation failed"; exit 1; }

echo "Synthetic data generated at $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
