"""
Test: Loss Function No NaN/Inf
"""

import pytest
import torch


def test_loss_no_nan_basic():
    """Test loss doesn't produce NaN with normal inputs."""
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    from yacs.config import CfgNode as CN

    # Create config
    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.LOSS = CN()
    cfg.RDCAROTS.LOSS.USE_REGIME_WEIGHTING = True
    cfg.RDCAROTS.SIM_THRESHOLD = 0.5
    cfg.RDCAROTS.TEMPERATURE = 0.1
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE = True

    # Create test data
    B = 32
    D = 128
    n_regimes = 3

    embeddings = torch.randn(B * 3, D)  # orig + pos + neg
    regime_probs = torch.softmax(torch.randn(B, n_regimes), dim=1)

    loss = regime_conditioned_soc_loss(embeddings, regime_probs, cfg)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0


def test_loss_no_nan_small_batch():
    """Test loss with very small batch."""
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.LOSS = CN()
    cfg.RDCAROTS.LOSS.USE_REGIME_WEIGHTING = False
    cfg.RDCAROTS.SIM_THRESHOLD = 0.5
    cfg.RDCAROTS.TEMPERATURE = 0.1
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE = False

    B = 2
    D = 64

    embeddings = torch.randn(B * 2, D)
    regime_probs = None

    loss = regime_conditioned_soc_loss(embeddings, regime_probs, cfg)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_no_nan_identical_embeddings():
    """Test loss with identical embeddings."""
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.LOSS = CN()
    cfg.RDCAROTS.LOSS.USE_REGIME_WEIGHTING = False
    cfg.RDCAROTS.SIM_THRESHOLD = 0.5
    cfg.RDCAROTS.TEMPERATURE = 0.1
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE = False

    B = 8
    D = 64

    # All identical
    embeddings = torch.ones(B * 2, D)

    loss = regime_conditioned_soc_loss(embeddings, None, cfg)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_no_nan_zero_embeddings():
    """Test loss with zero embeddings."""
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.LOSS = CN()
    cfg.RDCAROTS.LOSS.USE_REGIME_WEIGHTING = False
    cfg.RDCAROTS.SIM_THRESHOLD = 0.5
    cfg.RDCAROTS.TEMPERATURE = 0.1
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE = False

    B = 8
    D = 64

    embeddings = torch.zeros(B * 2, D)

    loss = regime_conditioned_soc_loss(embeddings, None, cfg)

    # Loss should handle zero embeddings gracefully
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_components_no_nan():
    """Test loss component computation doesn't produce NaN."""
    from models.rd_carots.loss_rd_carots import compute_loss_components
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE = True

    B = 16
    D = 128
    embeddings = torch.randn(B * 3, D)
    regime_probs = torch.softmax(torch.randn(B, 3), dim=1)

    components = compute_loss_components(embeddings, regime_probs, cfg)

    assert 'orig_sim' in components
    assert 'pos_sim' in components
    assert 'neg_sim' in components
    assert 'regime_entropy' in components

    for key, val in components.items():
        assert not np.isnan(val), f"{key} is NaN"
        assert not np.isinf(val), f"{key} is Inf"


import numpy as np
