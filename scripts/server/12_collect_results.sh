#!/usr/bin/env bash
# Collect all results

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Collecting results..."

$PYTHON_BIN scripts/collect_results.py "$OUTPUT_ROOT"

echo "✓ Results collected."
echo "Summary files:"
ls -lh "$OUTPUT_ROOT"/results_*.csv "$OUTPUT_ROOT"/results_*.md 2>/dev/null || true
