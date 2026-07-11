#!/usr/bin/env bash
set -euo pipefail

echo "=== RDCAROTS Full Server Experiments ==="

bash scripts/server/00_check_environment.sh
bash scripts/server/02_check_data.sh
bash scripts/server/04_run_tests.sh
bash scripts/server/05_generate_synthetic.sh
bash scripts/server/06_run_synthetic_smoke.sh

if [[ ! " $* " =~ " --smoke-only " ]]; then
    bash scripts/server/07_run_synthetic_full.sh
    
    if [[ ! " $* " =~ " --skip-swat " ]]; then
        bash scripts/server/08_run_swat.sh
    fi
    
    if [[ ! " $* " =~ " --skip-wadi " ]]; then
        bash scripts/server/09_run_wadi.sh
    fi
    
    bash scripts/server/10_run_all_seeds.sh
    bash scripts/server/11_run_ablations.sh
fi

bash scripts/server/12_collect_results.sh

echo "=== All experiments complete ==="
