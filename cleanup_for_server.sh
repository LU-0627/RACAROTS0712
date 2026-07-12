#!/usr/bin/env bash
# RDCAROTS Project Cleanup Script for Server Deployment
# This script removes unnecessary files to minimize upload size
# Run this before uploading to Linux server

set -euo pipefail

echo "=========================================="
echo "RDCAROTS Project Cleanup Script"
echo "=========================================="
echo ""
echo "This will DELETE unnecessary files to reduce upload size."
echo "A backup is recommended before running this script."
echo ""
read -p "Continue? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# ==========================================
# 1. Remove Git history (saves ~105MB)
# ==========================================
echo "[1/10] Removing .git directory..."
if [ -d ".git" ]; then
    rm -rf .git
    echo "✓ Removed .git/ (saved ~105MB)"
else
    echo "✓ .git/ not found, skipping"
fi

# ==========================================
# 2. Remove Python cache files
# ==========================================
echo "[2/10] Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
echo "✓ Removed Python cache files"

# ==========================================
# 3. Remove documentation reports (saves ~1MB)
# ==========================================
echo "[3/10] Removing documentation/report files..."
rm -f ACCEPTANCE_REPORT.md
rm -f DATA_MANIFEST.md
rm -f FINAL_ACCEPTANCE.md
rm -f FINAL_DELIVERY_COMPLETE.md
rm -f FINAL_DELIVERY_REPORT.md
rm -f FINAL_DELIVERY_REPORT_ZH.md
rm -f FINAL_DELIVERY_STATIC_CHECK.md
rm -f FINAL_REPORT_SIMPLE.txt
rm -f FINAL_STATIC_CHECK.txt
rm -f FINAL_STATIC_VALIDATION.txt
rm -f FINAL_STATIC_VALIDATION_OUTPUT.txt
rm -f FINAL_STATUS.txt
rm -f FINAL_SUMMARY.txt
rm -f FILE_MANIFEST.txt
rm -f FIXES_APPLIED.md
rm -f GPU_SERVER_DELIVERY_FINAL.md
rm -f GPU_SERVER_DEPLOYMENT_CRITICAL_FIXES.md
rm -f HARDBLOCK_FIXES_SUMMARY.txt
rm -f IMPLEMENTATION_SUMMARY.md
rm -f SERVER_TEST_CHECKLIST.md
rm -f STATIC_VALIDATION_RESULTS.txt
rm -f VALIDATION_REPORT.md
echo "✓ Removed documentation files"

# ==========================================
# 4. Remove figure directory (saves ~480KB)
# ==========================================
echo "[4/10] Removing figure directory..."
if [ -d "figure" ]; then
    rm -rf figure
    echo "✓ Removed figure/ (saved ~480KB)"
else
    echo "✓ figure/ not found, skipping"
fi

# ==========================================
# 5. Remove pdf directory (saves ~7MB)
# ==========================================
echo "[5/10] Removing pdf directory..."
if [ -d "pdf" ]; then
    rm -rf pdf
    echo "✓ Removed pdf/ (saved ~7MB)"
else
    echo "✓ pdf/ not found, skipping"
fi

# ==========================================
# 6. Remove results directory (saves ~13MB)
# ==========================================
echo "[6/10] Removing results directory..."
if [ -d "results" ]; then
    rm -rf results
    echo "✓ Removed results/ (saved ~13MB)"
else
    echo "✓ results/ not found, skipping"
fi

# ==========================================
# 7. Remove dist directory (saves ~1MB)
# ==========================================
echo "[7/10] Removing dist directory..."
if [ -d "dist" ]; then
    rm -rf dist
    echo "✓ Removed dist/ (saved ~1MB)"
else
    echo "✓ dist/ not found, skipping"
fi

# ==========================================
# 8. Remove .uploads directory
# ==========================================
echo "[8/10] Removing .uploads directory..."
if [ -d ".uploads" ]; then
    rm -rf .uploads
    echo "✓ Removed .uploads/"
else
    echo "✓ .uploads/ not found, skipping"
fi

# ==========================================
# 9. Clean docs directory (keep essential docs only)
# ==========================================
echo "[9/10] Cleaning docs directory..."
if [ -d "docs" ]; then
    # Keep only essential documentation
    find docs/ -type f -name "*REPORT*" -delete 2>/dev/null || true
    find docs/ -type f -name "*DELIVERY*" -delete 2>/dev/null || true
    echo "✓ Cleaned docs/ directory"
else
    echo "✓ docs/ not found, skipping"
fi

# ==========================================
# 10. Optional: Remove large sample datasets
# ==========================================
echo "[10/10] Cleaning data directory (removing sample datasets)..."
if [ -d "data" ]; then
    echo "  Warning: This will remove all sample data (~375MB)"
    echo "  You can regenerate synthetic data or provide your own datasets on the server"
    read -p "  Remove data directory? (yes/no): " remove_data

    if [[ "$remove_data" == "yes" ]]; then
        # Keep the directory structure but remove contents
        rm -rf data/Lorenz96 2>/dev/null || true
        rm -rf data/SMAP_MSL 2>/dev/null || true
        rm -rf data/ServerMachineDataset 2>/dev/null || true
        rm -rf data/VAR 2>/dev/null || true
        echo "✓ Removed sample datasets (saved ~375MB)"
        echo "  Note: data/ directory structure preserved"
    else
        echo "✓ Keeping data/ directory"
    fi
else
    echo "✓ data/ not found, skipping"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Estimated space saved: 127MB - 502MB (depending on options)"
echo ""
echo "Remaining essential files:"
echo "  ✓ Source code (models/, datasets/, utils/, tools/, layers/)"
echo "  ✓ Configuration files (configs/, config.py)"
echo "  ✓ Scripts (scripts/)"
echo "  ✓ Tests (tests/)"
echo "  ✓ Entry points (main.py, run_rd_carots.py, trainer.py)"
echo "  ✓ Requirements (requirements*.txt)"
echo "  ✓ Documentation (README.md, RUN_ON_OFFLINE_SERVER.md)"
echo "  ✓ Dependencies (offline/wheels/)"
echo ""
echo "Next steps:"
echo "  1. Review remaining files"
echo "  2. Create tarball: tar -czf rdcarots.tar.gz ."
echo "  3. Upload to server: scp rdcarots.tar.gz user@server:/path/"
echo ""
