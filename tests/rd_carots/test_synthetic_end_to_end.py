"""Test synthetic dataset end-to-end"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

def test_synthetic_generation():
    """Test synthetic dataset generation"""
    from datasets.synthetic_regime_delay import RegimeDelaySystemGenerator, save_dataset
    
    generator = RegimeDelaySystemGenerator(
        n_inputs=5,
        n_outputs=8,
        n_regimes=2,
        seed=42
    )
    
    dataset = generator.generate_dataset(
        n_train=100,
        n_val=50,
        n_test=80,
        anomaly_ratio=0.1
    )
    
    assert 'train' in dataset
    assert 'val' in dataset
    assert 'test' in dataset
    assert dataset['train']['u'].shape == (100, 5)
    assert dataset['train']['x'].shape == (100, 8)
    assert dataset['test']['labels'].sum() > 0  # Has anomalies

def test_synthetic_loader():
    """Test synthetic dataset loader"""
    from datasets.synthetic_regime_delay import RegimeDelaySystemGenerator, save_dataset
    from config import get_cfg_defaults
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate and save
        generator = RegimeDelaySystemGenerator(n_inputs=3, n_outputs=5, n_regimes=2, seed=0)
        dataset = generator.generate_dataset(n_train=50, n_val=20, n_test=30)
        save_dataset(dataset, tmpdir)
        
        # Load via config
        cfg = get_cfg_defaults()
        cfg.defrost()
        cfg.DATA.BASE_DIR = tmpdir
        cfg.DATA.NAME = 'synthetic_regime_delay'
        cfg.DATA.N_VAR = 8  # 3 inputs + 5 outputs
        cfg.freeze()
        
        from datasets.build import build_dataset
        train_dataset = build_dataset(cfg, split='train')
        
        assert len(train_dataset) > 0
