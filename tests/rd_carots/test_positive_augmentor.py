"""Test positive augmentor"""
import pytest
import torch
from config import get_cfg_defaults

def test_positive_augmentor_output_shape():
    from models.rd_carots.augmentors import RegimeDelayPositiveAugmentor
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    augmentor = RegimeDelayPositiveAugmentor(cfg, device=torch.device('cpu'))
    
    batch_size = 4
    window_size = 10
    n_vars = 50
    x = torch.randn(batch_size, window_size, n_vars)
    
    # No regime models - fallback
    x_pos = augmentor(x, regime_models=[], regime_probs=None)
    
    assert x_pos.shape == x.shape
    assert not torch.allclose(x_pos, x)  # Should be augmented
