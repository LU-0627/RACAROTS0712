#!/usr/bin/env bash
set -euo pipefail

echo "=== Python Compilation Check ==="

python -m compileall -q models/rd_carots || { echo "Compilation failed"; exit 1; }
python -m compileall -q datasets || { echo "Compilation failed"; exit 1; }
python -m compileall -q config.py || { echo "Compilation failed"; exit 1; }

echo "Compilation check passed"
