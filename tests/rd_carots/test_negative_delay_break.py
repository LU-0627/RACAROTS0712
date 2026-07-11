"""Test delay break does not leak future"""
import pytest
import torch
import numpy as np
from config import get_cfg_defaults
from models.rd_carots.io_schema import IOSchema

def test_delay_break_no_future_leakage():
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    augmentor = RegimeDelayNegativeAugmentor(cfg, device=torch.device('cpu'))
    
    io_schema = IOSchema(
        mode='explicit_io',
        input_indices=[0, 1],
        output_indices=[2, 3, 4],
        ignored_indices=[],
        n_inputs=2,
        n_outputs=3,
        n_total=5
    )
    
    T = 20
    x = torch.randn(T, 5)
    
    # Record original future values
    future_outputs = x[10:, io_schema.output_indices].clone()
    
    x_neg = augmentor._delay_break(x, io_schema)
    
    # Check that outputs at t < 10 don't contain values from t >= 10
    for t in range(10):
        for future_t in range(10, T):
            diff = torch.norm(x_neg[t, io_schema.output_indices] - x[future_t, io_schema.output_indices])
            # Should not be exactly equal (with high probability)
            assert diff > 1e-6 or torch.allclose(x_neg[t, io_schema.output_indices], 
                                                  x[0, io_schema.output_indices], atol=1e-5)
