"""
Test: Checkpoint Save/Load Roundtrip
"""

import pytest
import torch
import tempfile
import os


def test_checkpoint_roundtrip_prototype_bank():
    """Test prototype bank checkpoint save/load."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    n_regimes = 3
    embedding_dim = 128

    bank1 = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)

    # Initialize with data
    N = 100
    embeddings = torch.randn(N, embedding_dim)
    assignments = torch.randint(0, n_regimes, (N,))
    confidences = torch.ones(N) * 0.8
    bank1.initialize_from_data(embeddings, assignments, confidences)

    # Save
    state = bank1.state_dict()

    # Load into new bank
    bank2 = RegimePrototypeBank(n_regimes=n_regimes, embedding_dim=embedding_dim)
    bank2.load_state_dict(state)

    # Check equality
    assert torch.allclose(bank1.prototypes, bank2.prototypes)
    assert torch.all(bank1.regime_counts == bank2.regime_counts)
    assert bank1.is_initialized.item() == bank2.is_initialized.item()


def test_checkpoint_roundtrip_model_bank():
    """Test model bank checkpoint save/load."""
    from models.rd_carots.delaymix import RegimeDelayModelBank

    n_outputs, n_inputs = 10, 8
    n_regimes = 2

    bank1 = RegimeDelayModelBank(
        n_outputs=n_outputs,
        n_inputs=n_inputs,
        n_regimes=n_regimes
    )

    # Update some moments
    for _ in range(50):
        outputs = torch.randn(1, n_outputs)
        inputs = torch.randn(1, n_inputs)
        bank1.update_moments(outputs, inputs)

    # Save
    state = bank1.state_dict()

    # Load
    bank2 = RegimeDelayModelBank(
        n_outputs=n_outputs,
        n_inputs=n_inputs,
        n_regimes=n_regimes
    )
    bank2.load_state_dict(state)

    # Check moment collection state preserved
    assert bank1.moment_collection.n_updates == bank2.moment_collection.n_updates


def test_checkpoint_file_io():
    """Test checkpoint save/load to actual file."""
    from models.rd_carots.prototypes import RegimePrototypeBank

    bank1 = RegimePrototypeBank(n_regimes=3, embedding_dim=64)

    # Initialize
    N = 50
    embeddings = torch.randn(N, 64)
    assignments = torch.randint(0, 3, (N,))
    confidences = torch.ones(N) * 0.8
    bank1.initialize_from_data(embeddings, assignments, confidences)

    # Save to file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        filepath = f.name

    try:
        torch.save({'prototype_bank': bank1.state_dict()}, filepath)

        # Load from file
        checkpoint = torch.load(filepath)
        bank2 = RegimePrototypeBank(n_regimes=3, embedding_dim=64)
        bank2.load_state_dict(checkpoint['prototype_bank'])

        # Verify
        assert torch.allclose(bank1.prototypes, bank2.prototypes)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
