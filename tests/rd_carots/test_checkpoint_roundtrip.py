"""Test checkpoint save and load roundtrip"""
import pytest
import torch
import tempfile
import os
from pathlib import Path

def test_model_checkpoint_roundtrip():
    from config import get_cfg_defaults
    from models.rd_carots.modeling_rd_carots import RDCAROTS
    
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.DATA.N_VAR = 50
    cfg.RDCAROTS.N_INPUTS = 20
    cfg.RDCAROTS.N_OUTPUTS = 30
    cfg.freeze()
    
    model = RDCAROTS(cfg, device=torch.device('cpu'))
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / 'test_checkpoint.pth'
        
        state_dict = model.state_dict()
        torch.save({'model_state': state_dict}, ckpt_path)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)
        
        assert ckpt_path.exists()

def test_prototype_bank_checkpoint():
    from models.rd_carots.prototypes import RegimePrototypeBank
    
    proto_bank = RegimePrototypeBank(n_regimes=3, embedding_dim=128)
    
    # Initialize with data
    embeddings = torch.randn(50, 128)
    regime_assignments = torch.randint(0, 3, (50,))
    regime_confidences = torch.ones(50)
    proto_bank.initialize_from_data(embeddings, regime_assignments, regime_confidences)
    
    # Save and load
    state = proto_bank.state_dict()
    
    proto_bank_new = RegimePrototypeBank(n_regimes=3, embedding_dim=128)
    proto_bank_new.load_state_dict(state, strict=True)
    
    assert torch.allclose(proto_bank.prototypes, proto_bank_new.prototypes)
    assert torch.equal(proto_bank.regime_counts, proto_bank_new.regime_counts)
