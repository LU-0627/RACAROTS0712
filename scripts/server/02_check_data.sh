#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Data ==="

DATA_ROOT="${DATA_ROOT:-data}"

echo "Data root: $DATA_ROOT"

# Check if data directories exist
for dataset in SWaT WADI synthetic_regime_delay; do
    if [ -d "$DATA_ROOT/$dataset" ]; then
        echo "✓ $dataset directory exists"
        ls -lh "$DATA_ROOT/$dataset" | head -5
    else
        echo "✗ $dataset directory not found"
    fi
done

echo "Data check complete"
