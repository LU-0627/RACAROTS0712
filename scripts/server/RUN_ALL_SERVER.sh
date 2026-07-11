#!/usr/bin/env bash
# Run master server test script

set -euo pipefail

# Default parameters
SKIP_SWAT="${SKIP_SWAT:-false}"
SKIP_WADI="${SKIP_WADI:-false}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

echo "========================================="
echo "RDCAROTS Server Full Test Suite"
echo "========================================="
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-swat)
            SKIP_SWAT=true
            shift
            ;;
        --skip-wadi)
            SKIP_WADI=true
            shift
            ;;
        --smoke-only)
            SMOKE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Step 1: Environment check
echo "Step 1/15: Environment check"
bash scripts/server/00_check_environment.sh || exit 1
echo ""

# Step 2: Data check
echo "Step 2/15: Data check"
bash scripts/server/02_check_data.sh || exit 1
echo ""

# Step 3: Compile check
echo "Step 3/15: Compile check"
bash scripts/server/03_run_compile_check.sh || exit 1
echo ""

# Step 4: Pytest
echo "Step 4/15: Unit tests"
bash scripts/server/04_run_tests.sh || exit 1
echo ""

# Step 5: Generate synthetic
echo "Step 5/15: Generate synthetic data"
bash scripts/server/05_generate_synthetic.sh || exit 1
echo ""

if [ "$SMOKE_ONLY" = "true" ]; then
    echo "Smoke-only mode. Exiting."
    exit 0
fi

# Step 6-15: Full experiments (placeholder)
echo "Step 6/15: Full experiments (not implemented in this script)"
echo ""

echo "========================================="
echo "All steps complete!"
echo "========================================="
