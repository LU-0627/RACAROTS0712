"""
Test: Ho-Kalman State Space Realization
"""

import pytest
import numpy as np


def test_ho_kalman_basic():
    """Test basic Ho-Kalman realization."""
    from models.rd_carots.delaymix.ho_kalman import ho_kalman_realization

    n_out, n_in, max_lag = 10, 8, 20

    # Create Markov sequence
    markov_sequence = np.random.randn(max_lag, n_out, n_in) * 0.1

    ss_model = ho_kalman_realization(
        markov_sequence,
        p=10,
        q=10,
        max_states=10
    )

    assert ss_model is not None
    assert ss_model.n_states > 0
    assert ss_model.A.shape == (ss_model.n_states, ss_model.n_states)
    assert ss_model.B.shape == (ss_model.n_states, n_in)
    assert ss_model.C.shape == (n_out, ss_model.n_states)


def test_ho_kalman_stability():
    """Test that Ho-Kalman produces stable systems."""
    from models.rd_carots.delaymix.ho_kalman import ho_kalman_realization

    markov_sequence = np.random.randn(20, 10, 8) * 0.1

    ss_model = ho_kalman_realization(
        markov_sequence,
        stability_margin=0.95
    )

    eigenvalues = np.linalg.eigvals(ss_model.A)
    max_eig = np.max(np.abs(eigenvalues))

    assert max_eig <= 1.0, f"Unstable system: max eigenvalue = {max_eig}"


def test_ho_kalman_no_nan():
    """Test that Ho-Kalman doesn't produce NaN."""
    from models.rd_carots.delaymix.ho_kalman import ho_kalman_realization

    markov_sequence = np.random.randn(20, 10, 8)
    ss_model = ho_kalman_realization(markov_sequence)

    assert not np.any(np.isnan(ss_model.A))
    assert not np.any(np.isnan(ss_model.B))
    assert not np.any(np.isnan(ss_model.C))


def test_ho_kalman_simulation():
    """Test state space simulation."""
    from models.rd_carots.delaymix.ho_kalman import (
        ho_kalman_realization,
        simulate_state_space
    )

    markov_sequence = np.random.randn(20, 5, 3) * 0.1
    ss_model = ho_kalman_realization(markov_sequence)

    T = 50
    inputs = np.random.randn(T, 3)

    outputs, states = simulate_state_space(ss_model, inputs)

    assert outputs.shape == (T, 5)
    assert states.shape == (T, ss_model.n_states)
    assert not np.any(np.isnan(outputs))
