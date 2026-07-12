#!/usr/bin/env bash
# Project Organization Script
# This script organizes the messy root directory into a clean structure

set -euo pipefail

echo "=========================================="
echo "RDCAROTS Project Organization Script"
echo "=========================================="
echo ""

# Create archive directory for old reports
mkdir -p docs/archive

echo "Moving redundant files to docs/archive/..."
echo ""

# List of files to move (old reports and validation files)
FILES_TO_ARCHIVE=(
    "ACCEPTANCE_REPORT.md"
    "DATA_MANIFEST.md"
    "FILE_MANIFEST.txt"
    "FINAL_ACCEPTANCE.md"
    "FINAL_DELIVERY_COMPLETE.md"
    "FINAL_DELIVERY_REPORT.md"
    "FINAL_DELIVERY_REPORT_ZH.md"
    "FINAL_DELIVERY_STATIC_CHECK.md"
    "FINAL_REPORT_SIMPLE.txt"
    "FINAL_STATIC_CHECK.txt"
    "FINAL_STATIC_VALIDATION.txt"
    "FINAL_STATIC_VALIDATION_OUTPUT.txt"
    "FINAL_STATUS.txt"
    "FINAL_SUMMARY.txt"
    "FIXES_APPLIED.md"
    "GPU_SERVER_DELIVERY_FINAL.md"
    "GPU_SERVER_DEPLOYMENT_CRITICAL_FIXES.md"
    "HARDBLOCK_FIXES_SUMMARY.txt"
    "IMPLEMENTATION_SUMMARY.md"
    "SERVER_TEST_CHECKLIST.md"
    "STATIC_VALIDATION_RESULTS.txt"
    "VALIDATION_REPORT.md"
)

# Move files to archive
for file in "${FILES_TO_ARCHIVE[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/archive/
        echo "✓ Moved $file"
    fi
done

echo ""
echo "=========================================="
echo "Organization Complete!"
echo "=========================================="
echo ""
echo "Root directory now contains only essential files:"
echo "  ✓ README.md - Project overview"
echo "  ✓ RUN_ON_OFFLINE_SERVER.md - Deployment guide"
echo "  ✓ RDCAROTS_可运行性评估报告.md - Runability assessment"
echo "  ✓ 精简打包指南.md - Packaging guide"
echo "  ✓ 自动路径配置说明.md - Auto-path configuration"
echo "  ✓ requirements*.txt - Dependencies"
echo "  ✓ LICENSE - License file"
echo "  ✓ *.py - Entry point scripts"
echo ""
echo "Old reports archived to: docs/archive/"
echo ""
