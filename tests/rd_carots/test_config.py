"""Test configuration loading"""
import pytest
from config import get_cfg_defaults

def test_default_config():
    cfg = get_cfg_defaults()
    assert cfg is not None
    assert hasattr(cfg, 'RDCAROTS')

def test_rdcarots_config():
    cfg = get_cfg_defaults()
    assert cfg.RDCAROTS.N_INPUTS == 20
    assert cfg.RDCAROTS.N_OUTPUTS == 30
    assert cfg.RDCAROTS.DELAYMIX.N_REGIMES == 3

def test_config_freeze():
    cfg = get_cfg_defaults()
    cfg.freeze()
    with pytest.raises(AttributeError):
        cfg.SEED = 999
