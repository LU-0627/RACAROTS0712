"""Test loss does not produce NaN"""
import pytest
import torch
from config import get_cfg_defaults

def test_loss_no_nan():
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    batch_size = 8
    embedding_dim = 512
    n_regimes = 3
    
    # Simulate 4B structure
    embeddings = torch.randn(batch_size * 4, embedding_dim)
    regime_probs = torch.softmax(torch.randn(batch_size, n_regimes), dim=1)
    
    batch_metadata = {
        'group_ids': torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.full((batch_size,), 2, dtype=torch.long),
            torch.full((batch_size,), 3, dtype=torch.long)
        ]),
        'source_indices': torch.arange(batch_size).repeat(4),
        'is_positive': torch.cat([
            torch.zeros(batch_size, dtype=torch.bool),
            torch.ones(batch_size, dtype=torch.bool),
            torch.zeros(batch_size * 2, dtype=torch.bool)
        ]),
        'is_negative': torch.cat([
            torch.zeros(batch_size * 2, dtype=torch.bool),
            torch.ones(batch_size * 2, dtype=torch.bool)
        ]),
        'batch_size': batch_size
    }
    
    loss = regime_conditioned_soc_loss(
        embeddings, regime_probs, cfg, 
        batch_metadata=batch_metadata
    )
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0
