#!/usr/bin/env bash
set -euo pipefail

echo "=== Collecting Results ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"

python scripts/collect_results.py "$OUTPUT_ROOT" || { echo "Result collection failed"; exit 1; }

echo "Results collected at $OUTPUT_ROOT/results_summary.csv"
cat "$OUTPUT_ROOT/results_summary.csv" 2>/dev/null || echo "Summary file not found"
