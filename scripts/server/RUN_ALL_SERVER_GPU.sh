#!/usr/bin/env bash
set -euo pipefail

echo "=== RDCAROTS Full GPU Server Experiments ==="

# Environment defaults
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
export RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/rd_carots}"
export DEVICE="${DEVICE:-cuda:0}"

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "RESULT_ROOT: ${RESULT_ROOT}"
echo "DEVICE: ${DEVICE}"

# Check environment
bash scripts/server/00_check_environment.sh || exit 1

# Check data
bash scripts/server/02_check_data.sh || exit 1

# Compile check
bash scripts/server/03_run_compile_check.sh || exit 1

# Generate synthetic data
if [ ! -d "${DATA_ROOT}/synthetic_regime_delay" ]; then
    bash scripts/server/05_generate_synthetic.sh || exit 1
fi

# Smoke test
bash scripts/server/06_run_synthetic_gpu_smoke.sh || exit 1

echo "=== All experiments complete ==="
