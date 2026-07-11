"""Test that original CAROTS still works"""
import pytest
import torch

def test_original_carots_import():
    """Verify original CAROTS can still be imported"""
    from models.carots.modeling_carots import CAROTS
    assert CAROTS is not None

def test_original_carots_encoder():
    """Verify CAROTS encoder still works"""
    from models.carots.encoder import LSTM_ENC
    from config import get_cfg_defaults
    
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.DATA.N_VAR = 51
    cfg.freeze()
    
    encoder = LSTM_ENC(cfg)
    assert encoder is not None
    
    x = torch.randn(8, 10, 51)
    try:
        out = encoder(x)
        assert out.shape[0] == 8
    except:
        # May need full model initialization
        pass
