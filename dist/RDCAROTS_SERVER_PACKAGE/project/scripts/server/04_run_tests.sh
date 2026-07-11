#!/usr/bin/env bash
# Run pytest

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"

echo "Running pytest..."

cd "$PROJECT_ROOT"

pytest tests/rd_carots/ -v --tb=short || {
    echo "ERROR: Tests failed"
    exit 1
}

echo "✓ All tests passed."
