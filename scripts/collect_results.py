"""Collect and summarize RDCAROTS experiment results"""
import json
import pandas as pd
from pathlib import Path
import sys

def collect_all_results(result_root):
    result_root = Path(result_root)
    all_results = []
    
    for metrics_file in result_root.rglob('metrics.json'):
        with open(metrics_file) as f:
            metrics = json.load(f)
            all_results.append(metrics)
    
    if not all_results:
        print("No results found")
        return
    
    df = pd.DataFrame(all_results)
    
    # Save raw results
    df.to_csv(result_root / 'results_raw.csv', index=False)
    
    # Summary by model and dataset
    summary = df.groupby(['dataset', 'model']).agg({
        'AUROC': ['mean', 'std'],
        'AUPRC': ['mean', 'std'],
        'oracle_best_F1': ['mean', 'std']
    })
    
    summary.to_csv(result_root / 'results_summary.csv')
    print(f"Collected {len(df)} results")
    print(f"Saved to {result_root / 'results_summary.csv'}")

if __name__ == '__main__':
    collect_all_results(sys.argv[1] if len(sys.argv) > 1 else 'results/rd_carots')
