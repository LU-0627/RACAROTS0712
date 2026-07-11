#!/usr/bin/env bash
# RDCAROTS Local Development Report Generator

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "RDCAROTS Local Development Report"
echo "========================================"
echo ""

echo "## System Information"
echo "- OS: $(uname -s)"
echo "- Python: $(python --version 2>&1 || echo 'Not found')"
echo "- Timestamp: $(date)"
echo ""

echo "## Project Structure"
find "$PROJECT_ROOT" -type f -name "*.py" | grep -E "(rd_carots|rdcarots)" | head -20
echo ""

echo "## Git Status"
cd "$PROJECT_ROOT"
if [ -d ".git" ]; then
    echo "- Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "- Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "- Modified files: $(git status --short | wc -l)"
else
    echo "- Not a git repository"
fi
echo ""

echo "## Python Syntax Check"
if command -v python &> /dev/null; then
    python -m compileall "$PROJECT_ROOT/models/rd_carots" 2>&1 | tail -5 || echo "Compilation check failed"
else
    echo "Python not available"
fi
echo ""

echo "## File Counts"
echo "- Python files: $(find "$PROJECT_ROOT" -name "*.py" | wc -l)"
echo "- YAML files: $(find "$PROJECT_ROOT" -name "*.yaml" -o -name "*.yml" | wc -l)"
echo "- Shell scripts: $(find "$PROJECT_ROOT" -name "*.sh" | wc -l)"
echo ""

echo "Report complete. Review docs/LOCAL_DEVELOPMENT_REPORT.md for details."
