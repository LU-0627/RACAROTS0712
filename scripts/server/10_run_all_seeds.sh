#!/usr/bin/env bash
set -euo pipefail

echo "=== Running All Seeds ==="

OUTPUT_ROOT="${OUTPUT_ROOT:-results/rd_carots}"
DATA_ROOT="${DATA_ROOT:-data}"

for dataset in synthetic swat wadi; do
    for seed in 0 1 2; do
        echo "Running $dataset seed $seed"
        
        config_file="configs/rd_carots/${dataset}.yaml"
        
        if [ ! -f "$config_file" ]; then
            echo "Config $config_file not found, skipping"
            continue
        fi
        
        python run_rd_carots.py \
            --config "$config_file" \
            --mode train \
            --model RDCAROTS \
            --seed $seed \
            --data-root "$DATA_ROOT" \
            --output-root "$OUTPUT_ROOT/${dataset}/seed_$seed" || { echo "Failed $dataset seed $seed"; exit 1; }
        
        python run_rd_carots.py \
            --config "$config_file" \
            --mode frozen \
            --model RDCAROTS \
            --seed $seed \
            --data-root "$DATA_ROOT" \
            --output-root "$OUTPUT_ROOT/${dataset}/seed_$seed" || { echo "Failed $dataset seed $seed"; exit 1; }
    done
done

echo "All seeds complete"
