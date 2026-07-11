#!/usr/bin/env bash
# Generate synthetic regime-delay dataset

set -euo pipefail

# Default parameters
OUTPUT_DIR="${1:-data/synthetic_regime_delay}"
SEED="${2:-0}"

echo "Generating synthetic regime-delay dataset..."
echo "Output directory: $OUTPUT_DIR"
echo "Seed: $SEED"

python -m datasets.synthetic_regime_delay \
    --output-dir "$OUTPUT_DIR" \
    --n-inputs 20 \
    --n-outputs 30 \
    --n-regimes 3 \
    --n-train 10000 \
    --n-val 2000 \
    --n-test 5000 \
    --seed "$SEED"

echo "Dataset generation complete."
echo "Files created:"
ls -lh "$OUTPUT_DIR"
