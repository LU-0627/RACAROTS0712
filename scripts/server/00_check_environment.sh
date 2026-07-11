#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Environment ==="

# Check Python
python --version || { echo "Python not found"; exit 1; }

# Check required packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "PyTorch not installed"; exit 1; }
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || { echo "NumPy not installed"; exit 1; }
python -c "import yaml; print('PyYAML installed')" || { echo "PyYAML not installed"; exit 1; }
python -c "import sklearn; print('scikit-learn installed')" || { echo "scikit-learn not installed"; exit 1; }

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check disk space
df -h . || echo "Could not check disk space"

echo "Environment check passed"
