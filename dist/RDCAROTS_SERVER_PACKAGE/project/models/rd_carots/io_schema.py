"""
IO Schema: Input/Output Variable Specification

Handles three modes:
1. explicit_io: User specifies input and output variable indices
2. metadata_io: Infer from variable metadata (actuators vs sensors)
3. pseudo_io: Use lagged outputs as pseudo-inputs when no explicit inputs
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class IOSchema:
    """Input/output variable specification."""
    mode: str  # 'explicit_io', 'metadata_io', 'pseudo_io'
    input_indices: List[int]
    output_indices: List[int]
    ignored_indices: List[int]
    n_inputs: int
    n_outputs: int
    n_total: int
    pseudo_delay: Optional[int] = None  # For pseudo_io mode
    description: Optional[str] = None


def load_io_schema(schema_path: Path, n_variables: Optional[int] = None) -> IOSchema:
    """
    Load IO schema from YAML file.

    Args:
        schema_path: Path to schema YAML
        n_variables: Total number of variables (for validation)

    Returns:
        IOSchema object
    """
    with open(schema_path, 'r') as f:
        config = yaml.safe_load(f)

    mode = config['mode']
    description = config.get('description', '')

    if mode == 'explicit_io':
        input_indices = config['input_indices']
        output_indices = config['output_indices']
        ignored_indices = config.get('ignored_indices', [])

    elif mode == 'metadata_io':
        # Infer from metadata
        metadata = config.get('metadata', {})
        input_indices = metadata.get('input_variables', [])
        output_indices = metadata.get('output_variables', [])
        ignored_indices = metadata.get('ignored_variables', [])

    elif mode == 'pseudo_io':
        # All variables are outputs, pseudo-inputs are lagged versions
        output_indices = list(range(n_variables or config.get('n_variables', 0)))
        input_indices = []  # Will be constructed dynamically
        ignored_indices = config.get('ignored_indices', [])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Validate
    all_indices = set(input_indices + output_indices + ignored_indices)
    if n_variables is not None:
        expected_indices = set(range(n_variables))
        if all_indices != expected_indices:
            print(f"Warning: IO schema indices {all_indices} don't cover all variables {expected_indices}")

    # Check for overlaps
    if set(input_indices) & set(output_indices):
        raise ValueError(f"Input and output indices overlap: {set(input_indices) & set(output_indices)}")

    schema = IOSchema(
        mode=mode,
        input_indices=input_indices,
        output_indices=output_indices,
        ignored_indices=ignored_indices,
        n_inputs=len(input_indices) if mode != 'pseudo_io' else len(output_indices),
        n_outputs=len(output_indices),
        n_total=len(input_indices) + len(output_indices),
        pseudo_delay=config.get('pseudo_delay', 1) if mode == 'pseudo_io' else None,
        description=description
    )

    return schema


def split_io_variables(
    data: np.ndarray,
    io_schema: IOSchema
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into inputs and outputs according to schema.

    Args:
        data: Full data array (..., n_variables)
        io_schema: IO specification

    Returns:
        (inputs, outputs) arrays
    """
    if io_schema.mode == 'pseudo_io':
        # Use lagged outputs as pseudo-inputs
        outputs = data[..., io_schema.output_indices]

        # Create pseudo-inputs by shifting
        delay = io_schema.pseudo_delay or 1
        if data.shape[-2] < delay:
            raise ValueError(f"Data sequence too short for pseudo_delay={delay}")

        # Inputs are lagged outputs: u(t) = y(t - delay)
        inputs = np.concatenate([
            np.zeros((*data.shape[:-2], delay, len(io_schema.output_indices))),
            outputs[..., :-delay, :]
        ], axis=-2)

    else:
        # Explicit input/output split
        inputs = data[..., io_schema.input_indices]
        outputs = data[..., io_schema.output_indices]

    return inputs, outputs


def create_io_schema_template(
    dataset_name: str,
    n_variables: int,
    output_path: Path,
    mode: str = 'explicit_io'
):
    """
    Create an IO schema template YAML file.

    Args:
        dataset_name: Name of dataset
        n_variables: Total number of variables
        output_path: Where to save template
        mode: IO mode
    """
    if mode == 'explicit_io':
        template = {
            'mode': 'explicit_io',
            'description': f'IO schema for {dataset_name}',
            'input_indices': list(range(0, n_variables // 2)),  # Placeholder
            'output_indices': list(range(n_variables // 2, n_variables)),  # Placeholder
            'ignored_indices': [],
            'notes': 'Please update input_indices and output_indices based on domain knowledge'
        }

    elif mode == 'metadata_io':
        template = {
            'mode': 'metadata_io',
            'description': f'IO schema for {dataset_name}',
            'metadata': {
                'input_variables': [],  # Actuators, control signals
                'output_variables': [],  # Sensors, measurements
                'ignored_variables': []  # Timestamps, labels, etc.
            },
            'notes': 'Specify input (actuators) and output (sensors) variables'
        }

    elif mode == 'pseudo_io':
        template = {
            'mode': 'pseudo_io',
            'description': f'IO schema for {dataset_name} (no explicit inputs)',
            'n_variables': n_variables,
            'pseudo_delay': 1,
            'ignored_indices': [],
            'notes': 'All variables treated as outputs; pseudo-inputs constructed from lagged outputs'
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    print(f"Created IO schema template: {output_path}")


def validate_io_schema(schema: IOSchema, data_shape: Tuple) -> bool:
    """
    Validate IO schema against actual data.

    Args:
        schema: IO schema
        data_shape: Shape of data array

    Returns:
        True if valid
    """
    n_variables = data_shape[-1]

    # Check indices are in range
    all_indices = schema.input_indices + schema.output_indices + schema.ignored_indices

    for idx in all_indices:
        if idx < 0 or idx >= n_variables:
            print(f"Error: Index {idx} out of range [0, {n_variables})")
            return False

    # Check totals match
    if schema.mode != 'pseudo_io':
        if len(set(all_indices)) != n_variables:
            print(f"Warning: Schema covers {len(set(all_indices))} variables but data has {n_variables}")

    return True
