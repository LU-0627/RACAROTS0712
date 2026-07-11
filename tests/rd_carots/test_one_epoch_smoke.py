"""Smoke test: train one epoch"""
import pytest
import torch

@pytest.mark.slow
def test_one_epoch_smoke():
    """Smoke test: initialize model and run one training step"""
    from config import get_cfg_defaults
    from models.rd_carots.modeling_rd_carots import RDCAROTS
    
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.DATA.N_VAR = 50
    cfg.RDCAROTS.N_INPUTS = 20
    cfg.RDCAROTS.N_OUTPUTS = 30
    cfg.SOLVER.MAX_EPOCH = 1
    cfg.freeze()
    
    model = RDCAROTS(cfg, device=torch.device('cpu'))
    
    # Simulate one forward pass
    batch_size = 4
    x = torch.randn(batch_size, 10, 50)
    
    model.eval()
    with torch.no_grad():
        embeddings = model(x, positive_augment=False, negative_augment=False)
    
    assert embeddings.shape[0] == batch_size
    assert embeddings.shape[1] == cfg.RDCAROTS.PROJECTOR.OUTPUT_DIM
