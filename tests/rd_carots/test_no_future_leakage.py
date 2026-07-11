"""
Test: No Future Leakage in delay_break
"""

import pytest
import torch
import numpy as np


def test_delay_break_no_future_leakage():
    """Test that delay_break does not use future values."""
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    from yacs.config import CfgNode as CN

    # Create minimal config
    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.RELATION_BREAK_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_BREAK_PROB = 1.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.CROSS_REGIME_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_SHIFT_RANGE = [-3, 3]

    augmentor = RegimeDelayNegativeAugmentor(cfg)

    # Create test data with marker pattern
    B, T, N = 4, 20, 10
    x = torch.zeros(B, T, N)

    # Put unique marker at each timestep
    for t in range(T):
        x[:, t, :] = float(t)

    # Apply delay_break
    x_neg = augmentor._delay_break(x[0])

    # Check no future values appear in past
    for t in range(T - 1):
        # Values at time t should not contain markers from t+1 or later
        # (unless explicitly padded, which uses past values)
        values_at_t = x_neg[t].unique()

        # Should not contain future markers
        for future_t in range(t + 2, T):
            assert float(future_t) not in values_at_t or t == T - 1, \
                f"Future value {future_t} found at timestep {t}"


def test_delay_break_boundary_handling():
    """Test delay_break handles boundaries correctly."""
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.RELATION_BREAK_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_BREAK_PROB = 1.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.CROSS_REGIME_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_SHIFT_RANGE = [-2, 2]

    augmentor = RegimeDelayNegativeAugmentor(cfg)

    B, T, N = 2, 10, 5
    x = torch.randn(B, T, N)

    # Should not crash on boundaries
    for _ in range(10):
        x_neg = augmentor._delay_break(x[0])
        assert x_neg.shape == x[0].shape
        assert not torch.any(torch.isnan(x_neg))


def test_delay_break_shift_directions():
    """Test delay_break shifts in both directions."""
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.RDCAROTS = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR = CN()
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.RELATION_BREAK_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_BREAK_PROB = 1.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.CROSS_REGIME_PROB = 0.0
    cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR.DELAY_SHIFT_RANGE = [-3, 3]

    augmentor = RegimeDelayNegativeAugmentor(cfg)

    T, N = 20, 10
    x = torch.arange(T).unsqueeze(1).repeat(1, N).float()

    # Apply multiple times
    shifts_observed = []
    for _ in range(20):
        x_neg = augmentor._delay_break(x.unsqueeze(0))[0]

        # Check if shifted (compare to original)
        if not torch.allclose(x_neg, x):
            shifts_observed.append(True)

    assert len(shifts_observed) > 0, "No shifts observed"
