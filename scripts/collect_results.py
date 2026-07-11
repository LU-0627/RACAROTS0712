"""
Collect and summarize all experimental results
"""

import os
import json
import pandas as pd
from pathlib import Path
import numpy as np


def collect_all_results(result_root):
    """
    Collect all experimental results and generate summaries.

    Args:
        result_root: Root directory containing all results
    """
    result_root = Path(result_root)

    # Collect all result files
    results = []

    # Search for result JSON files
    for json_file in result_root.rglob('*_results.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")

    if len(results) == 0:
        print("No results found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    raw_csv_path = result_root / 'results_raw.csv'
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw results to {raw_csv_path}")

    # Generate summary by model and dataset
    if 'model' in df.columns and 'dataset' in df.columns and 'seed' in df.columns:
        summary = df.groupby(['dataset', 'model']).agg({
            'AUROC': ['mean', 'std'],
            'AUPRC': ['mean', 'std'],
            'F1': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).reset_index()

        summary_csv_path = result_root / 'results_summary.csv'
        summary.to_csv(summary_csv_path, index=False)
        print(f"Saved summary to {summary_csv_path}")

        # Generate markdown table
        md_lines = ["# RDCAROTS Results Summary\n"]
        md_lines.append(f"Total experiments: {len(df)}\n")
        md_lines.append(f"Unique models: {df['model'].nunique()}\n")
        md_lines.append(f"Unique datasets: {df['dataset'].nunique()}\n")
        md_lines.append("\n## Results by Model and Dataset\n")
        md_lines.append(summary.to_markdown(index=False))

        md_path = result_root / 'results_summary.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        print(f"Saved markdown summary to {md_path}")

    # Ablation summary
    if 'model' in df.columns:
        ablation_models = [m for m in df['model'].unique() if 'RDCAROTS' in m]
        if len(ablation_models) > 1:
            ablation_df = df[df['model'].isin(ablation_models)]
            ablation_summary = ablation_df.groupby('model').agg({
                'AUROC': 'mean',
                'F1': 'mean',
                'runtime_seconds': 'mean'
            }).reset_index()

            ablation_csv_path = result_root / 'ablation_summary.csv'
            ablation_summary.to_csv(ablation_csv_path, index=False)
            print(f"Saved ablation summary to {ablation_csv_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result_root = sys.argv[1]
    else:
        result_root = 'results/rd_carots'

    collect_all_results(result_root)
