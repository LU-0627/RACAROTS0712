"""
Test: CP Decomposition
"""

import pytest
import numpy as np
import torch


def test_cp_decomposition_known_rank():
    """Test CP decomposition on tensor with known rank."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition

    # Create low-rank tensor
    n_out, n_in, max_lag = 10, 8, 15
    rank = 2

    # Generate known factors
    a = np.random.randn(n_out, rank)
    b = np.random.randn(n_in, rank)
    c = np.random.randn(max_lag, rank)
    weights = np.array([1.0, 0.8])

    # Reconstruct tensor
    tensor = np.zeros((n_out, n_in, max_lag))
    for r in range(rank):
        tensor += weights[r] * np.einsum('i,j,k->ijk', a[:, r], b[:, r], c[:, r])

    # Add small noise
    tensor += np.random.randn(*tensor.shape) * 0.01

    # Decompose
    cp_factors = cp_decomposition(tensor, rank=rank, init='svd', max_iter=100)

    assert cp_factors is not None
    assert cp_factors.n_regimes == rank
    assert cp_factors.output_factors.shape == (n_out, rank)
    assert cp_factors.input_factors.shape == (n_in, rank)
    assert cp_factors.lag_factors.shape == (max_lag, rank)
    assert cp_factors.reconstruction_error < 0.1


def test_cp_decomposition_dimensions():
    """Test CP decomposition output dimensions."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition

    n_out, n_in, max_lag = 20, 15, 10
    rank = 3

    tensor = np.random.randn(n_out, n_in, max_lag)
    cp_factors = cp_decomposition(tensor, rank=rank)

    assert cp_factors.weights.shape == (rank,)
    assert cp_factors.output_factors.shape == (n_out, rank)
    assert cp_factors.input_factors.shape == (n_in, rank)
    assert cp_factors.lag_factors.shape == (max_lag, rank)


def test_cp_decomposition_validation():
    """Test CP decomposition validation."""
    from models.rd_carots.delaymix.cp_decomposition import (
        cp_decomposition,
        validate_cp_decomposition
    )

    tensor = np.random.randn(10, 8, 15)
    cp_factors = cp_decomposition(tensor, rank=2)

    is_valid = validate_cp_decomposition(tensor, cp_factors, error_threshold=1.0)
    assert isinstance(is_valid, bool)


def test_cp_decomposition_no_nan():
    """Test that CP decomposition doesn't produce NaN."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition

    tensor = np.random.randn(10, 8, 15)
    cp_factors = cp_decomposition(tensor, rank=2)

    assert not np.any(np.isnan(cp_factors.weights))
    assert not np.any(np.isnan(cp_factors.output_factors))
    assert not np.any(np.isnan(cp_factors.input_factors))
    assert not np.any(np.isnan(cp_factors.lag_factors))
