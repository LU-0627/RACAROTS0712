"""Test Ho-Kalman realization"""
import pytest
import numpy as np

def test_ho_kalman_basic():
    from models.rd_carots.delaymix.ho_kalman import ho_kalman_realization
    
    max_lag = 15
    n_outputs, n_inputs = 4, 3
    markov_seq = np.random.randn(max_lag, n_outputs, n_inputs) * 0.1
    
    ss_model = ho_kalman_realization(markov_seq, p=5, q=5, max_states=10)
    
    assert ss_model is not None
    assert ss_model.A.shape[0] == ss_model.A.shape[1]
    assert ss_model.B.shape[0] == ss_model.A.shape[0]
    assert ss_model.B.shape[1] == n_inputs
    assert ss_model.C.shape[0] == n_outputs
    assert ss_model.C.shape[1] == ss_model.A.shape[0]

def test_ho_kalman_stability():
    from models.rd_carots.delaymix.ho_kalman import ho_kalman_realization
    
    max_lag = 10
    markov_seq = np.random.randn(max_lag, 3, 2) * 0.05
    
    ss_model = ho_kalman_realization(markov_seq, stability_margin=0.95)
    
    eigenvalues = np.linalg.eigvals(ss_model.A)
    max_eig = np.max(np.abs(eigenvalues))
    
    assert max_eig <= 1.0
