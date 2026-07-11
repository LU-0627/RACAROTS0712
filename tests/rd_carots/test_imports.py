"""
Test: Import validation
"""

import pytest


def test_import_rdcarots():
    """Test RDCAROTS model import."""
    from models.rd_carots import RDCAROTS
    assert RDCAROTS is not None


def test_import_trainer():
    """Test trainer import."""
    from models.rd_carots import RDCAROTSTrainer
    assert RDCAROTSTrainer is not None


def test_import_scorer():
    """Test scorer import."""
    from models.rd_carots import RDCAROTSScorer
    assert RDCAROTSScorer is not None


def test_import_delaymix():
    """Test DelayMix imports."""
    from models.rd_carots.delaymix import DynamicMomentCollection
    from models.rd_carots.delaymix import cp_decomposition
    from models.rd_carots.delaymix import ho_kalman_realization
    from models.rd_carots.delaymix import RegimeInference
    from models.rd_carots.delaymix import RegimeDelayModelBank

    assert DynamicMomentCollection is not None
    assert cp_decomposition is not None
    assert ho_kalman_realization is not None
    assert RegimeInference is not None
    assert RegimeDelayModelBank is not None


def test_import_augmentors():
    """Test augmentor imports."""
    from models.rd_carots.augmentors import RegimeDelayPositiveAugmentor
    from models.rd_carots.augmentors import RegimeDelayNegativeAugmentor

    assert RegimeDelayPositiveAugmentor is not None
    assert RegimeDelayNegativeAugmentor is not None


def test_import_prototypes():
    """Test prototype bank import."""
    from models.rd_carots.prototypes import RegimePrototypeBank
    assert RegimePrototypeBank is not None


def test_import_loss():
    """Test loss function import."""
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    assert regime_conditioned_soc_loss is not None


def test_import_io_schema():
    """Test IO schema import."""
    from models.rd_carots.io_schema import load_io_schema, split_io_variables
    assert load_io_schema is not None
    assert split_io_variables is not None


def test_import_synthetic():
    """Test synthetic data generator import."""
    from datasets.synthetic_regime_delay import RegimeDelaySystemGenerator
    assert RegimeDelaySystemGenerator is not None


def test_no_import_errors():
    """Test that all imports succeed without errors."""
    try:
        from models.rd_carots import RDCAROTS, RDCAROTSTrainer, RDCAROTSScorer
        from models.rd_carots.delaymix import RegimeDelayModelBank
        from models.rd_carots.augmentors import RegimeDelayPositiveAugmentor
        from models.rd_carots.prototypes import RegimePrototypeBank
        from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
        success = True
    except Exception as e:
        print(f"Import error: {e}")
        success = False

    assert success, "Import failed"
