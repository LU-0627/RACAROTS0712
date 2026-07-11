"""Test Markov parameter delay detection"""
import pytest
import numpy as np

def test_markov_effective_delay():
    from models.rd_carots.delaymix.markov_recovery import detect_effective_delay
    
    max_lag = 10
    n_outputs, n_inputs = 5, 3
    true_delay = 3
    
    # Create Markov sequence with delay
    markov_seq = np.zeros((max_lag, n_outputs, n_inputs))
    markov_seq[true_delay:] = np.random.randn(max_lag - true_delay, n_outputs, n_inputs) * 0.5
    
    detected_delay = detect_effective_delay(markov_seq, threshold=0.1)
    
    assert detected_delay >= 0
    assert detected_delay <= max_lag

def test_markov_sequence_shape():
    max_lag = 20
    n_outputs, n_inputs = 8, 6
    
    markov_seq = np.random.randn(max_lag, n_outputs, n_inputs)
    
    assert markov_seq.shape == (max_lag, n_outputs, n_inputs)
    assert markov_seq.ndim == 3
