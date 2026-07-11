"""Test regime inference"""
import pytest
import torch
import numpy as np

def test_regime_inference_shapes():
    from models.rd_carots.delaymix.regime_inference import RegimeInference
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel
    
    n_regimes = 3
    batch_size = 8
    window_size = 10
    n_outputs, n_inputs = 5, 4
    
    regime_inference = RegimeInference(n_regimes=n_regimes)
    
    outputs = torch.randn(batch_size, window_size, n_outputs)
    inputs = torch.randn(batch_size, window_size, n_inputs)
    
    # Create dummy state space models
    ss_models = []
    for r in range(n_regimes):
        n_states = 3
        ss_models.append(StateSpaceModel(
            A=np.eye(n_states) * 0.9,
            B=np.random.randn(n_states, n_inputs) * 0.1,
            C=np.random.randn(n_outputs, n_states) * 0.1,
            n_states=n_states,
            eigenvalues=np.ones(n_states) * 0.9,
            is_stable=True
        ))
    
    results = regime_inference(outputs, inputs, ss_models, mode='soft')
    
    assert results['regime_probs'].shape == (batch_size, n_regimes)
    assert results['best_regime'].shape == (batch_size,)
    assert results['confidence'].shape == (batch_size,)
    assert torch.all(results['regime_probs'] >= 0)
    assert torch.all(results['regime_probs'] <= 1)
    assert torch.allclose(results['regime_probs'].sum(dim=1), torch.ones(batch_size), atol=1e-5)
