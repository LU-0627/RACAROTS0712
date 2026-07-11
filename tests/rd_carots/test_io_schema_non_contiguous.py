"""Test IO schema with non-contiguous indices"""
import pytest
import numpy as np
import torch
from models.rd_carots.io_schema import IOSchema, split_io_variables

def test_non_contiguous_indices():
    """Test that IO schema handles non-contiguous variable indices."""
    io_schema = IOSchema(
        mode='explicit_io',
        input_indices=[0, 2, 5, 7],
        output_indices=[1, 3, 4, 6, 8, 9],
        ignored_indices=[],
        n_inputs=4,
        n_outputs=6,
        n_total=10
    )
    
    data = np.random.randn(100, 20, 10)
    inputs, outputs = split_io_variables(data, io_schema)
    
    assert inputs.shape == (100, 20, 4)
    assert outputs.shape == (100, 20, 6)
    assert np.allclose(inputs[..., 0], data[..., 0])
    assert np.allclose(outputs[..., 0], data[..., 1])
