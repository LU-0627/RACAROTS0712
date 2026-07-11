"""Test CP decomposition"""
import pytest
import numpy as np

def test_cp_decomposition_basic():
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition
    
    # Create simple rank-2 tensor
    tensor = np.random.randn(10, 5, 8)
    
    result = cp_decomposition(tensor, rank=2, max_iter=50, tol=1e-4)
    
    assert result is not None
    assert result.rank == 2
    assert result.factors[0].shape == (10, 2)
    assert result.factors[1].shape == (5, 2)
    assert result.factors[2].shape == (8, 2)

def test_cp_reconstruction():
    from models.rd_carots.delaymix.cp_decomposition import cp_decomposition
    
    tensor = np.random.randn(6, 4, 5)
    result = cp_decomposition(tensor, rank=3, max_iter=100)
    
    # Reconstruct
    reconstructed = np.zeros_like(tensor)
    for r in range(result.rank):
        weight = result.weights[r]
        reconstructed += weight * np.outer(result.factors[0][:, r], 
                                           np.outer(result.factors[1][:, r], 
                                                   result.factors[2][:, r]).ravel()).reshape(tensor.shape)
    
    # Check reconstruction error is reasonable
    rel_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
    assert rel_error < 0.5
