"""
Test: Tensor Shape Validation
"""

import pytest
import torch
import numpy as np


def test_moment_collection_shapes():
    """Test moment collection tensor shapes."""
    from models.rd_carots.delaymix import DynamicMomentCollection

    n_outputs, n_inputs, max_lag = 10, 8, 20

    collector = DynamicMomentCollection(
        n_outputs=n_outputs,
        n_inputs=n_inputs,
        max_lag=max_lag
    )

    # Update
    for _ in range(50):
        outputs = torch.randn(1, n_outputs)
        inputs = torch.randn(1, n_inputs)
        collector.update(outputs, inputs)

    # Get tensor
    tensor = collector.get_tensor()

    assert tensor.shape == (n_outputs, n_inputs, max_lag)


def test_cp_factors_shapes():
    """Test CP decomposition output shapes."""
    from models.rd_carots.delaymix import cp_decomposition

    n_out, n_in, max_lag = 15, 10, 20
    rank = 3

    tensor = np.random.randn(n_out, n_in, max_lag)
    factors = cp_decomposition(tensor, rank=rank)

    assert factors.weights.shape == (rank,)
    assert factors.output_factors.shape == (n_out, rank)
    assert factors.input_factors.shape == (n_in, rank)
    assert factors.lag_factors.shape == (max_lag, rank)


def test_state_space_shapes():
    """Test state space model shapes."""
    from models.rd_carots.delaymix import ho_kalman_realization

    n_out, n_in, max_lag = 10, 8, 20
    markov_seq = np.random.randn(max_lag, n_out, n_in)

    ss_model = ho_kalman_realization(markov_seq, max_states=5)

    n_states = ss_model.n_states
    assert ss_model.A.shape == (n_states, n_states)
    assert ss_model.B.shape == (n_states, n_in)
    assert ss_model.C.shape == (n_out, n_states)


def test_augmentor_output_shapes():
    """Test augmentor output shapes match input."""
    from models.rd_carots.augmentors import RegimeDelayPositiveAugmentor
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.REGIME_DELAY_POSITIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.REGIME_DELAY_POSITIVE_AUGMENTOR.NOISE_LEVEL = 0.1

    augmentor = RegimeDelayPositiveAugmentor(cfg)

    B, T, N = 16, 10, 50
    x = torch.randn(B, T, N)

    # No regime models (fallback)
    x_aug = augmentor(x, regime_models=[], regime_probs=None)

    assert x_aug.shape == x.shape
    assert x_aug.device == x.device
    assert x_aug.dtype == x.dtype


def test_prototype_bank_shapes():
    """Test prototype bank tensor shapes."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 128
    batch_size = 32

    bank = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)

    embeddings = torch.randn(batch_size, embedding_dim)

    distances = bank.get_distances(embeddings)
    assert distances.shape == (batch_size, n_regimes)

    min_distances = bank.get_min_distance(embeddings)
    assert min_distances.shape == (batch_size,)
