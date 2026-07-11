"""
Test: Prototype Bank
"""

import pytest
import torch


def test_prototype_bank_initialization():
    """Test prototype bank initialization."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 128

    bank = RegimePrototypeBank(
        n_regimes=n_regimes,
        embedding_dim=embedding_dim
    )

    assert bank.prototypes.shape == (n_regimes, embedding_dim)
    assert bank.regime_counts.shape == (n_regimes,)
    assert not bank.is_initialized.item()


def test_prototype_bank_initialize_from_data():
    """Test prototype initialization from data."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 64
    N = 300

    bank = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)

    # Create fake data
    embeddings = torch.randn(N, embedding_dim)
    regime_assignments = torch.randint(0, n_regimes, (N,))
    regime_confidences = torch.rand(N) * 0.5 + 0.5  # 0.5 to 1.0

    bank.initialize_from_data(embeddings, regime_assignments, regime_confidences)

    assert bank.is_initialized.item()
    assert bank.regime_counts.sum() > 0


def test_prototype_bank_update():
    """Test prototype bank online update."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 64

    bank = RegimePrototypeBank(
        n_regimes=n_regimes,
        embedding_dim=embedding_dim,
        update_momentum=0.1,
        min_confidence=0.6
    )

    # Initialize
    N = 100
    embeddings = torch.randn(N, embedding_dim)
    regime_assignments = torch.randint(0, n_regimes, (N,))
    regime_confidences = torch.ones(N) * 0.8
    bank.initialize_from_data(embeddings, regime_assignments, regime_confidences)

    # Update with new batch
    batch_size = 16
    new_embeddings = torch.randn(batch_size, embedding_dim)
    new_regime_probs = torch.softmax(torch.randn(batch_size, n_regimes), dim=1)

    old_prototypes = bank.prototypes.clone()
    bank.update(new_embeddings, new_regime_probs, is_normal=None, use_soft=True)

    # Prototypes should have changed
    assert not torch.allclose(bank.prototypes, old_prototypes)


def test_prototype_bank_distances():
    """Test distance computation."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 64

    bank = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)

    batch_size = 16
    embeddings = torch.randn(batch_size, embedding_dim)

    distances = bank.get_distances(embeddings, normalize=True)

    assert distances.shape == (batch_size, n_regimes)
    assert torch.all(distances >= 0)


def test_prototype_bank_min_distance():
    """Test minimum distance computation."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 64

    bank = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)

    batch_size = 16
    embeddings = torch.randn(batch_size, embedding_dim)

    min_distances = bank.get_min_distance(embeddings)

    assert min_distances.shape == (batch_size,)
    assert torch.all(min_distances >= 0)
