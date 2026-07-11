"""
Test: Configuration Loading
"""

import pytest
from pathlib import Path


def test_load_io_schema_synthetic():
    """Test loading synthetic IO schema."""
    from models.rd_carots.io_schema import load_io_schema

    schema_path = Path("configs/io_schema/synthetic_regime_delay.yaml")

    if schema_path.exists():
        schema = load_io_schema(schema_path, n_variables=50)

        assert schema.mode == 'explicit_io'
        assert len(schema.input_indices) == 20
        assert len(schema.output_indices) == 30


def test_split_io_variables():
    """Test IO variable splitting."""
    from models.rd_carots.io_schema import load_io_schema, split_io_variables
    import numpy as np

    schema_path = Path("configs/io_schema/synthetic_regime_delay.yaml")

    if schema_path.exists():
        schema = load_io_schema(schema_path, n_variables=50)

        # Create fake data
        data = np.random.randn(100, 50)

        inputs, outputs = split_io_variables(data, schema)

        assert inputs.shape == (100, 20)
        assert outputs.shape == (100, 30)


def test_io_schema_modes():
    """Test different IO schema modes."""
    from models.rd_carots.io_schema import IOSchema

    # Explicit mode
    schema1 = IOSchema(
        mode='explicit_io',
        input_indices=[0, 1, 2],
        output_indices=[3, 4, 5],
        ignored_indices=[],
        n_inputs=3,
        n_outputs=3,
        n_total=6
    )

    assert schema1.mode == 'explicit_io'
    assert schema1.n_inputs == 3

    # Pseudo mode
    schema2 = IOSchema(
        mode='pseudo_io',
        input_indices=[],
        output_indices=[0, 1, 2, 3, 4, 5],
        ignored_indices=[],
        n_inputs=6,
        n_outputs=6,
        n_total=6,
        pseudo_delay=1
    )

    assert schema2.mode == 'pseudo_io'
    assert schema2.pseudo_delay == 1
