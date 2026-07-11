#!/usr/bin/env bash
# Download Linux wheels for offline installation

set -euo pipefail

WHEEL_DIR="${1:-offline/wheels}"

echo "Downloading wheels to $WHEEL_DIR..."

mkdir -p "$WHEEL_DIR"

python -m pip download \
    --dest "$WHEEL_DIR" \
    --platform manylinux2014_x86_64 \
    --python-version 310 \
    --only-binary=:all: \
    -r requirements-server-lock.txt

echo "✓ Wheels downloaded to $WHEEL_DIR"
ls -lh "$WHEEL_DIR" | head -20
