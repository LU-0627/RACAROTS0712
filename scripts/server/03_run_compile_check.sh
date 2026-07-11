#!/usr/bin/env bash
# Run Python compile check

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"

echo "Running Python syntax check..."

python -m compileall "$PROJECT_ROOT/models/rd_carots" || {
    echo "ERROR: Compilation failed"
    exit 1
}

python -m compileall "$PROJECT_ROOT/datasets" || {
    echo "ERROR: Compilation failed"
    exit 1
}

echo "✓ Syntax check passed."
