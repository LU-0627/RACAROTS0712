"""Test all RDCAROTS imports"""
import pytest

def test_import_modeling():
    from models.rd_carots.modeling_rd_carots import RDCAROTS
    assert RDCAROTS is not None

def test_import_trainer():
    from models.rd_carots.trainer_rd_carots import RDCAROTSTrainer
    assert RDCAROTSTrainer is not None

def test_import_augmentors():
    from models.rd_carots.augmentors import RegimeDelayPositiveAugmentor, RegimeDelayNegativeAugmentor
    assert RegimeDelayPositiveAugmentor is not None
    assert RegimeDelayNegativeAugmentor is not None

def test_import_scorer():
    from models.rd_carots.scorer_rd_carots import RDCAROTSScorer
    assert RDCAROTSScorer is not None

def test_import_loss():
    from models.rd_carots.loss_rd_carots import regime_conditioned_soc_loss
    assert regime_conditioned_soc_loss is not None

def test_import_prototypes():
    from models.rd_carots.prototypes import RegimePrototypeBank
    assert RegimePrototypeBank is not None

def test_import_model_bank():
    from models.rd_carots.delaymix.model_bank import RegimeDelayModelBank
    assert RegimeDelayModelBank is not None

def test_import_io_schema():
    from models.rd_carots.io_schema import IOSchema, load_io_schema, split_io_variables
    assert IOSchema is not None
    assert load_io_schema is not None
    assert split_io_variables is not None

def test_import_synthetic_dataset():
    from datasets.synthetic_regime_delay import RegimeDelaySystemGenerator, save_dataset
    assert RegimeDelaySystemGenerator is not None
    assert save_dataset is not None
