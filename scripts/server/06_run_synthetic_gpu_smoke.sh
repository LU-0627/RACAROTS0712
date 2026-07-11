#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Synthetic GPU Smoke Test ==="

# Environment
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/rd_carots}"
DEVICE="${DEVICE:-cuda:0}"

# Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "ERROR: CUDA not available but DEVICE=${DEVICE}"
    exit 1
}

# Create output directory
mkdir -p "${RESULT_ROOT}/synthetic_smoke/logs"

# Train 1 epoch
echo "Training 1 epoch on ${DEVICE}..."
python run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --mode train \
    --model RDCAROTS \
    --seed 0 \
    --device "${DEVICE}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${RESULT_ROOT}/synthetic_smoke" \
    2>&1 | tee "${RESULT_ROOT}/synthetic_smoke/logs/train.log"

# Frozen test
echo "Running frozen test..."
python run_rd_carots.py \
    --config configs/rd_carots/synthetic.yaml \
    --mode frozen \
    --model RDCAROTS \
    --seed 0 \
    --device "${DEVICE}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${RESULT_ROOT}/synthetic_smoke" \
    2>&1 | tee "${RESULT_ROOT}/synthetic_smoke/logs/frozen.log"

echo "Smoke test complete"
ls -lh "${RESULT_ROOT}/synthetic_smoke/"
