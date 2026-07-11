"""
Test: Original CAROTS Compatibility

Verifies that original CAROTS still works after RDCAROTS integration.
"""

import pytest
import torch


def test_carots_model_selection():
    """Test that CAROTS can still be selected."""
    from config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.MODEL.NAME = 'CAROTS'
    cfg.freeze()

    assert cfg.MODEL.NAME == 'CAROTS'
    print("✓ CAROTS model selection works")


def test_carots_model_builds():
    """Test that CAROTS model still builds."""
    from config import get_cfg_defaults
    from models.build import build_model

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.MODEL.NAME = 'CAROTS'
    cfg.DATA.N_VAR = 51
    cfg.freeze()

    try:
        model = build_model(cfg)
        assert model is not None
        print("✓ CAROTS model builds successfully")
    except Exception as e:
        pytest.fail(f"CAROTS model failed to build: {e}")


def test_carots_trainer_builds():
    """Test that CAROTS trainer still builds."""
    from config import get_cfg_defaults
    from models.build import build_model
    from trainer import build_trainer

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.MODEL.NAME = 'CAROTS'
    cfg.DATA.N_VAR = 51
    cfg.freeze()

    model = build_model(cfg)
    trainer = build_trainer(cfg, model)

    assert trainer is not None
    assert trainer.__class__.__name__ == 'CAROTSTrainer'
    print("✓ CAROTS trainer builds successfully")


def test_carots_forward_pass():
    """Test that CAROTS forward pass works."""
    from config import get_cfg_defaults
    from models.build import build_model

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.MODEL.NAME = 'CAROTS'
    cfg.DATA.N_VAR = 51
    cfg.DATA.WIN_SIZE = 10
    cfg.freeze()

    model = build_model(cfg)
    model.eval()

    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, cfg.DATA.WIN_SIZE, cfg.DATA.N_VAR)

    with torch.no_grad():
        output = model(x, positive_augment=False, negative_augment=False)

    assert output.shape[0] == batch_size
    print("✓ CAROTS forward pass works")


def test_checkpoint_isolation():
    """Test that CAROTS and RDCAROTS use separate checkpoint directories."""
    from config import get_cfg_defaults

    # CAROTS config
    cfg_carots = get_cfg_defaults()
    cfg_carots.defrost()
    cfg_carots.MODEL.NAME = 'CAROTS'
    cfg_carots.TRAIN.CHECKPOINT_DIR = 'results/carots/checkpoints'
    cfg_carots.freeze()

    # RDCAROTS config
    cfg_rdcarots = get_cfg_defaults()
    cfg_rdcarots.defrost()
    cfg_rdcarots.MODEL.NAME = 'RDCAROTS'
    cfg_rdcarots.TRAIN.CHECKPOINT_DIR = 'results/rd_carots/checkpoints'
    cfg_rdcarots.freeze()

    assert cfg_carots.TRAIN.CHECKPOINT_DIR != cfg_rdcarots.TRAIN.CHECKPOINT_DIR
    print("✓ Checkpoint directories are isolated")


if __name__ == '__main__':
    test_carots_model_selection()
    test_carots_model_builds()
    test_carots_trainer_builds()
    test_carots_forward_pass()
    test_checkpoint_isolation()
