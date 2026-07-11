"""
RDCAROTS: Regime- and Delay-aware CAROTS

Multi-regime delayed system identification with causality-aware contrastive learning
for robust multivariate time-series anomaly detection.
"""

from .modeling_rd_carots import RDCAROTS
from .trainer_rd_carots import RDCAROTSTrainer
from .scorer_rd_carots import RDCAROTSScorer

__all__ = ['RDCAROTS', 'RDCAROTSTrainer', 'RDCAROTSScorer']
