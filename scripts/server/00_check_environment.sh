#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Environment ==="

# Check Python
python --version || { echo "ERROR: Python not found"; exit 1; }

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "ERROR: PyTorch not installed"; exit 1; }

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found"
fi

# Check environment variables
echo "PROJECT_ROOT: ${PROJECT_ROOT:-not set}"
echo "DATA_ROOT: ${DATA_ROOT:-not set}"
echo "RESULT_ROOT: ${RESULT_ROOT:-not set}"
echo "DEVICE: ${DEVICE:-not set}"

# Check key modules
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || { echo "ERROR: NumPy not installed"; exit 1; }
python -c "import yaml; print('PyYAML: OK')" || { echo "ERROR: PyYAML not installed"; exit 1; }

echo "Environment check complete"
