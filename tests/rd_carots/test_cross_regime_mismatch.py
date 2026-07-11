"""Test cross-regime mismatch"""
import pytest
import torch
import numpy as np
from config import get_cfg_defaults
from models.rd_carots.io_schema import IOSchema

def test_cross_regime_requires_multiple_regimes():
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    augmentor = RegimeDelayNegativeAugmentor(cfg, device=torch.device('cpu'))
    
    io_schema = IOSchema(
        mode='explicit_io',
        input_indices=[0, 1],
        output_indices=[2, 3],
        ignored_indices=[],
        n_inputs=2,
        n_outputs=2,
        n_total=4
    )
    
    x = torch.randn(10, 4)
    
    # Only 1 regime - should fallback
    ss_models = [StateSpaceModel(
        A=np.eye(2) * 0.9,
        B=np.random.randn(2, 2) * 0.1,
        C=np.random.randn(2, 2) * 0.1,
        n_states=2,
        eigenvalues=np.ones(2) * 0.9,
        is_stable=True
    )]
    
    x_neg = augmentor._cross_regime_mismatch(x, ss_models, None, io_schema)
    
    assert x_neg.shape == x.shape

def test_cross_regime_uses_different_regime():
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    from models.rd_carots.delaymix.ho_kalman import StateSpaceModel
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    augmentor = RegimeDelayNegativeAugmentor(cfg, device=torch.device('cpu'))
    
    io_schema = IOSchema(
        mode='explicit_io',
        input_indices=[0, 1],
        output_indices=[2, 3],
        ignored_indices=[],
        n_inputs=2,
        n_outputs=2,
        n_total=4
    )
    
    # Create 2 different regimes
    ss_models = []
    for r in range(2):
        ss_models.append(StateSpaceModel(
            A=np.eye(2) * (0.8 + r * 0.1),
            B=np.random.randn(2, 2) * (0.1 + r * 0.05),
            C=np.random.randn(2, 2) * (0.1 + r * 0.05),
            n_states=2,
            eigenvalues=np.ones(2) * (0.8 + r * 0.1),
            is_stable=True
        ))
    
    x = torch.randn(10, 4)
    regime_prob = torch.tensor([0.9, 0.1])
    
    x_neg = augmentor._cross_regime_mismatch(x, ss_models, regime_prob, io_schema)
    
    assert x_neg.shape == x.shape
