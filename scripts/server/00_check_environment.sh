#!/usr/bin/env bash
# Check server environment

set -euo pipefail

echo "========================================="
echo "RDCAROTS Server Environment Check"
echo "========================================="
echo ""

# Python version
echo "1. Python version:"
python --version || echo "  ERROR: Python not found"
echo ""

# PyTorch
echo "2. PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}')" || echo "  ERROR: PyTorch not found"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" || true
python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')" 2>/dev/null || true
python -c "import torch; print(f'  Device count: {torch.cuda.device_count()}')" 2>/dev/null || true
echo ""

# Other dependencies
echo "3. Dependencies:"
for pkg in numpy scipy sklearn tensorly pandas yaml tqdm yacs; do
    python -c "import $pkg; print(f'  $pkg: OK')" 2>/dev/null || echo "  $pkg: MISSING"
done
echo ""

# Environment variables
echo "4. Environment variables:"
echo "  PROJECT_ROOT: ${PROJECT_ROOT:-NOT SET}"
echo "  DATA_ROOT: ${DATA_ROOT:-NOT SET}"
echo "  OUTPUT_ROOT: ${OUTPUT_ROOT:-NOT SET}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT SET}"
echo ""

# Disk space
echo "5. Disk space:"
df -h . | tail -1
echo ""

# Memory
echo "6. Memory:"
free -h | grep Mem || echo "  free command not available"
echo ""

echo "Environment check complete."
