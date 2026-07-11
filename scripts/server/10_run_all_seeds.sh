#!/usr/bin/env bash
# Run experiments with all seeds

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"

echo "Running experiments with seeds 0, 1, 2..."

for SEED in 0 1 2; do
    echo "Seed $SEED..."

    export SEED=$SEED

    bash scripts/server/07_run_synthetic_full.sh || echo "Synthetic seed $SEED failed"

    if [ -d "$DATA_ROOT/SWaT" ]; then
        bash scripts/server/08_run_swat.sh || echo "SWaT seed $SEED failed"
    fi

    if [ -d "$DATA_ROOT/WADI" ]; then
        bash scripts/server/09_run_wadi.sh || echo "WADI seed $SEED failed"
    fi
done

echo "✓ Multi-seed experiments complete."
