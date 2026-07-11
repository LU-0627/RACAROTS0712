"""
Check dataset schema against actual CSV files
"""

import pandas as pd
import yaml
from pathlib import Path
import argparse


def check_dataset_schema(dataset_name, data_path, schema_path):
    """
    Validate IO schema against actual dataset.

    Args:
        dataset_name: Name of dataset (SWaT, WADI, etc.)
        data_path: Path to CSV file
        schema_path: Path to IO schema YAML
    """
    print(f"Checking {dataset_name} schema...")

    # Load actual data
    try:
        df = pd.read_csv(data_path, nrows=5)
        actual_columns = list(df.columns)
        n_actual = len(actual_columns)
        print(f"✓ Loaded {data_path}")
        print(f"  Columns: {n_actual}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False

    # Load schema
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        print(f"✓ Loaded schema: {schema_path}")
    except Exception as e:
        print(f"✗ Failed to load schema: {e}")
        return False

    # Check mode
    mode = schema.get('mode', 'explicit_io')
    print(f"  Mode: {mode}")

    if mode == 'explicit_io':
        input_indices = schema.get('input_indices', [])
        output_indices = schema.get('output_indices', [])
        ignored_indices = schema.get('ignored_indices', [])

        n_inputs = len(input_indices)
        n_outputs = len(output_indices)
        n_ignored = len(ignored_indices)
        n_total = n_inputs + n_outputs + n_ignored

        print(f"  Schema expects: {n_total} variables")
        print(f"    Inputs: {n_inputs}")
        print(f"    Outputs: {n_outputs}")
        print(f"    Ignored: {n_ignored}")

        if n_total != n_actual:
            print(f"✗ MISMATCH: Schema has {n_total}, data has {n_actual}")
            print("\nActual columns:")
            for i, col in enumerate(actual_columns):
                print(f"  {i}: {col}")
            return False

        # Check indices are valid
        all_indices = input_indices + output_indices + ignored_indices
        for idx in all_indices:
            if idx >= n_actual:
                print(f"✗ Index {idx} out of range (max: {n_actual-1})")
                return False

    print("✓ Schema validation passed")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--schema-path', type=str, required=True, help='Path to schema YAML')
    args = parser.parse_args()

    success = check_dataset_schema(args.dataset, args.data_path, args.schema_path)
    exit(0 if success else 1)
