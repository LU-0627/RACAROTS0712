"""Test multi-regime prototypes"""
import pytest
import torch

def test_multi_regime_prototypes():
    from models.rd_carots.prototypes import RegimePrototypeBank
    
    n_regimes = 3
    embedding_dim = 128
    
    proto_bank = RegimePrototypeBank(
        n_regimes=n_regimes,
        embedding_dim=embedding_dim,
        device=torch.device('cpu')
    )
    
    assert proto_bank.prototypes.shape == (n_regimes, embedding_dim)
    assert proto_bank.regime_counts.shape == (n_regimes,)

def test_prototype_initialization():
    from models.rd_carots.prototypes import RegimePrototypeBank
    
    n_regimes = 3
    embedding_dim = 64
    
    proto_bank = RegimePrototypeBank(
        n_regimes=n_regimes,
        embedding_dim=embedding_dim
    )
    
    # Generate mock data
    n_samples = 100
    embeddings = torch.randn(n_samples, embedding_dim)
    regime_assignments = torch.randint(0, n_regimes, (n_samples,))
    regime_confidences = torch.rand(n_samples) * 0.5 + 0.5
    
    proto_bank.initialize_from_data(embeddings, regime_assignments, regime_confidences)
    
    assert proto_bank.is_initialized.item()
    assert proto_bank.regime_counts.sum() > 0

def test_prototype_distance_computation():
    from models.rd_carots.prototypes import RegimePrototypeBank
    
    proto_bank = RegimePrototypeBank(n_regimes=3, embedding_dim=64)
    
    embeddings = torch.randn(16, 64)
    distances = proto_bank.get_distances(embeddings, normalize=True)
    
    assert distances.shape == (16, 3)
    assert torch.all(distances >= 0)
