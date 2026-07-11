#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Tests ==="

# Run pytest on RDCAROTS tests
pytest tests/rd_carots/ -v --tb=short --maxfail=5 || {
    echo "Tests failed"
    exit 1
}

echo "Tests passed"
