#!/usr/bin/env bash
# Install dependencies offline

set -euo pipefail

METHOD="${1:-wheelhouse}"

echo "Installing dependencies (method: $METHOD)..."

case "$METHOD" in
    existing-env)
        echo "Using existing environment. Skipping installation."
        ;;

    wheelhouse)
        if [ ! -d "offline/wheels" ]; then
            echo "ERROR: offline/wheels directory not found"
            exit 1
        fi

        echo "Installing from offline wheelhouse..."
        python -m pip install \
            --no-index \
            --find-links=offline/wheels \
            -r requirements-server-lock.txt

        echo "Installation complete."
        ;;

    conda-pack)
        if [ ! -f "offline/rdcarots_env.tar.gz" ]; then
            echo "ERROR: offline/rdcarots_env.tar.gz not found"
            exit 1
        fi

        echo "Extracting conda environment..."
        mkdir -p rdcarots_env
        tar -xzf offline/rdcarots_env.tar.gz -C rdcarots_env

        echo "Activating environment..."
        source rdcarots_env/bin/activate

        echo "Environment ready."
        ;;

    *)
        echo "ERROR: Unknown method '$METHOD'"
        echo "Valid methods: existing-env, wheelhouse, conda-pack"
        exit 1
        ;;
esac

echo "Done."
