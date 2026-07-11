#!/usr/bin/env bash
# Generate synthetic dataset

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-0}"

echo "Generating synthetic regime-delay dataset..."

bash scripts/generate_synthetic_regime_delay.sh \
    "$DATA_ROOT/synthetic_regime_delay" \
    "$SEED"

echo "✓ Synthetic data generated."
