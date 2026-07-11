"""Test negative augmentor relation break"""
import pytest
import torch
from config import get_cfg_defaults

def test_relation_break():
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    
    cfg = get_cfg_defaults()
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.RELATION_BREAK_PROB = 1.0
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.DELAY_BREAK_PROB = 0.0
    cfg.RDCAROTS.NEGATIVE_AUGMENTOR.CROSS_REGIME_PROB = 0.0
    cfg.freeze()
    
    augmentor = RegimeDelayNegativeAugmentor(cfg, device=torch.device('cpu'))
    
    x = torch.randn(1, 10, 50)
    causality_mtx = torch.rand(50, 50) > 0.5
    
    x_neg = augmentor._relation_break(x[0], causality_mtx)
    
    assert x_neg.shape == x[0].shape
    assert not torch.allclose(x_neg, x[0])
