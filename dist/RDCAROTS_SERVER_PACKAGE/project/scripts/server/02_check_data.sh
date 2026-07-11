#!/usr/bin/env bash
# Check data availability

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"

echo "========================================="
echo "Data Availability Check"
echo "========================================="
echo ""

echo "Data root: $DATA_ROOT"
echo ""

# Synthetic
echo "1. Synthetic data:"
if [ -d "$DATA_ROOT/synthetic_regime_delay" ]; then
    echo "  ✓ Directory exists"
    ls -lh "$DATA_ROOT/synthetic_regime_delay" 2>/dev/null || true
else
    echo "  ✗ NOT FOUND (will be generated)"
fi
echo ""

# SWaT
echo "2. SWaT data:"
if [ -d "$DATA_ROOT/SWaT" ]; then
    echo "  ✓ Directory exists"
    ls -lh "$DATA_ROOT/SWaT"/*.csv 2>/dev/null || echo "  No CSV files found"
else
    echo "  ✗ NOT FOUND"
fi
echo ""

# WADI
echo "3. WADI data:"
if [ -d "$DATA_ROOT/WADI" ]; then
    echo "  ✓ Directory exists"
    ls -lh "$DATA_ROOT/WADI"/*.csv 2>/dev/null || echo "  No CSV files found"
else
    echo "  ✗ NOT FOUND"
fi
echo ""

echo "Data check complete."
