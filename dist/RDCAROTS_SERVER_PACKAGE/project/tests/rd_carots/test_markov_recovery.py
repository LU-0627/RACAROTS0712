"""
Test: Markov Parameter Recovery
"""

import pytest
import numpy as np


def test_markov_recovery_basic():
    """Test basic Markov parameter recovery."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition, CPFactors
    from models.rd_carots.delaymix.markov_recovery import recover_markov_parameters

    n_out, n_in, max_lag = 10, 8, 15
    rank = 2

    tensor = np.random.randn(n_out, n_in, max_lag)
    cp_factors = cp_decomposition(tensor, rank=rank)

    markov_list = recover_markov_parameters(cp_factors)

    assert len(markov_list) == rank
    for markov_params in markov_list:
        assert markov_params.markov_sequence.shape == (max_lag, n_out, n_in)
        assert isinstance(markov_params.effective_delay, int)
        assert markov_params.effective_delay >= 0


def test_markov_delay_identification():
    """Test delay identification in Markov parameters."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition
    from models.rd_carots.delaymix.markov_recovery import recover_markov_parameters

    n_out, n_in, max_lag = 10, 8, 20
    rank = 3

    # Create tensor with known delays
    tensor = np.zeros((n_out, n_in, max_lag))

    # Regime 0: no delay (response from lag 0)
    tensor[:, :, 0] += np.random.randn(n_out, n_in) * 2.0

    # Regime 1: delay=2
    tensor[:, :, 2] += np.random.randn(n_out, n_in) * 2.0

    # Regime 2: delay=5
    tensor[:, :, 5] += np.random.randn(n_out, n_in) * 2.0

    cp_factors = cp_decomposition(tensor, rank=rank, max_iter=200)
    markov_list = recover_markov_parameters(cp_factors, threshold_ratio=0.1)

    # Check that delays are identified (approximately)
    delays = sorted([m.effective_delay for m in markov_list])
    assert len(delays) == rank


def test_markov_energy_computation():
    """Test Markov parameter energy computation."""
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition
    from models.rd_carots.delaymix.markov_recovery import recover_markov_parameters

    tensor = np.random.randn(10, 8, 15)
    cp_factors = cp_decomposition(tensor, rank=2)
    markov_list = recover_markov_parameters(cp_factors)

    for markov_params in markov_list:
        assert markov_params.energy > 0
        assert not np.isnan(markov_params.energy)
