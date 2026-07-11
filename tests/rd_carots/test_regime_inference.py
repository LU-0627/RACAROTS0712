"""
Test: Regime Inference
"""

import pytest
import torch
import numpy as np


def test_regime_inference_basic():
    """Test basic regime inference."""
    from models.rd_carots.delaymix import RegimeInference
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel

    n_regimes = 3
    inference = RegimeInference(n_regimes=n_regimes)

    # Create dummy state space models
    ss_models = []
    for _ in range(n_regimes):
        A = np.random.randn(5, 5) * 0.1
        B = np.random.randn(5, 3) * 0.1
        C = np.random.randn(4, 5) * 0.1
        ss_models.append(StateSpaceModel(
            A=A, B=B, C=C,
            n_states=5,
            eigenvalues=np.array([0.5]),
            is_stable=True
        ))

    # Create test data
    B_size, T, n_out, n_in = 8, 10, 4, 3
    outputs = torch.randn(B_size, T, n_out)
    inputs = torch.randn(B_size, T, n_in)

    # Inference
    results = inference(outputs, inputs, ss_models, mode='soft')

    assert 'regime_probs' in results
    assert 'best_regime' in results
    assert 'confidence' in results
    assert results['regime_probs'].shape == (B_size, n_regimes)
    assert results['best_regime'].shape == (B_size,)


def test_regime_inference_soft_vs_hard():
    """Test soft vs hard regime inference."""
    from models.rd_carots.delaymix import RegimeInference
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel

    inference = RegimeInference(n_regimes=2)

    ss_models = []
    for _ in range(2):
        A = np.eye(3) * 0.5
        B = np.random.randn(3, 2)
        C = np.random.randn(5, 3)
        ss_models.append(StateSpaceModel(
            A=A, B=B, C=C, n_states=3,
            eigenvalues=np.array([0.5]), is_stable=True
        ))

    outputs = torch.randn(4, 10, 5)
    inputs = torch.randn(4, 10, 2)

    results_soft = inference(outputs, inputs, ss_models, mode='soft')

    # Check probabilities sum to 1
    prob_sums = results_soft['regime_probs'].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_regime_inference_confidence():
    """Test regime confidence calculation."""
    from models.rd_carots.delaymix import RegimeInference
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel

    inference = RegimeInference(
        n_regimes=3,
        confidence_threshold=0.6
    )

    ss_models = []
    for _ in range(3):
        A = np.eye(2) * 0.8
        B = np.random.randn(2, 3)
        C = np.random.randn(4, 2)
        ss_models.append(StateSpaceModel(
            A=A, B=B, C=C, n_states=2,
            eigenvalues=np.array([0.8]), is_stable=True
        ))

    outputs = torch.randn(10, 15, 4)
    inputs = torch.randn(10, 15, 3)

    results = inference(outputs, inputs, ss_models, mode='soft')

    # Confidence should be max probability
    expected_confidence = results['regime_probs'].max(dim=1)[0]
    assert torch.allclose(results['confidence'], expected_confidence)
