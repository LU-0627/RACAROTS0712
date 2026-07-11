"""
RDCAROTS: Regime- and Delay-aware CAROTS

Main model combining:
- CAROTS encoder and projector
- DelayMix regime/delay model bank
- Regime-conditioned augmentation
- Multi-regime prototype bank
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

# Import CAROTS components
from models.carots.encoder import iTransformer_ENC, TimesNet_ENC, LSTM_ENC, GRU_ENC, GATV2_ENC

# Import RDCAROTS components
from .delaymix import RegimeDelayModelBank
from .augmentors import RegimeDelayPositiveAugmentor, RegimeDelayNegativeAugmentor
from .prototypes import RegimePrototypeBank
from .io_schema import load_io_schema


class RDCAROTS(nn.Module):
    """
    RDCAROTS: Regime- and Delay-aware CAROTS for anomaly detection.

    Extends CAROTS with:
    - Multi-regime state space models
    - Input/output variable distinction
    - Delay-aware augmentation
    - Multi-regime prototypes
    """

    def __init__(self, cfg, device: Optional[torch.device] = None):
        super(RDCAROTS, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg.DATA
        self.cfg_rdcarots = cfg.RDCAROTS
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Encoder (same as CAROTS)
        self.encoder = self._init_encoder()

        # Projector (same as CAROTS)
        self.projector = self._init_projector()

        # Causal discoverer (same as CAROTS)
        self.causal_discoverer = self._init_causal_discoverer()

        # DelayMix model bank
        self.model_bank = self._init_model_bank()

        # Regime-aware augmentors
        self.positive_augmentor = self._init_positive_augmentor()
        self.negative_augmentor = self._init_negative_augmentor()

        # Regime prototype bank
        self.prototype_bank = self._init_prototype_bank()

    def _init_encoder(self):
        """Initialize encoder (same as CAROTS)."""
        arch = self.cfg_rdcarots.ENCODER_ARCH
        encoder_classes = {
            'lstm': LSTM_ENC,
            'iTransformer': iTransformer_ENC,
            'GATv2': GATV2_ENC,
            'TimesNet': TimesNet_ENC,
            'gru': GRU_ENC,
        }

        if arch not in encoder_classes:
            raise ValueError(f"Unsupported encoder architecture: {arch}")

        encoder = encoder_classes[arch](self.cfg)
        return encoder.to(self.device)

    def _init_projector(self):
        """Initialize projector (same as CAROTS)."""
        cfg_projector = self.cfg_rdcarots.PROJECTOR
        projector = nn.Sequential(
            nn.Linear(cfg_projector.INPUT_DIM, cfg_projector.HIDDEN_DIM),
            nn.BatchNorm1d(cfg_projector.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg_projector.HIDDEN_DIM, cfg_projector.OUTPUT_DIM),
        )
        return projector.to(self.device)

    def _init_causal_discoverer(self):
        """Initialize causal discoverer (same as CAROTS)."""
        from models.carots.modeling_cuts_plus import CUTS_Plus_Net
        self.cfg_cuts_plus = self.cfg.CUTS_PLUS
        causal_discoverer = CUTS_Plus_Net(self.cfg).to(self.device)
        return causal_discoverer

    def _init_model_bank(self):
        """Initialize DelayMix model bank."""
        # Get IO schema
        n_inputs = self.cfg_rdcarots.N_INPUTS
        n_outputs = self.cfg_rdcarots.N_OUTPUTS
        n_regimes = self.cfg_rdcarots.DELAYMIX.N_REGIMES

        model_bank = RegimeDelayModelBank(
            n_outputs=n_outputs,
            n_inputs=n_inputs,
            n_regimes=n_regimes,
            max_lag=self.cfg_rdcarots.DELAYMIX.MAX_LAG,
            config=dict(self.cfg_rdcarots.DELAYMIX),
            device=self.device
        )

        return model_bank

    def _init_positive_augmentor(self):
        """Initialize regime-aware positive augmentor."""
        augmentor = RegimeDelayPositiveAugmentor(self.cfg, device=self.device)
        return augmentor

    def _init_negative_augmentor(self):
        """Initialize regime-aware negative augmentor."""
        augmentor = RegimeDelayNegativeAugmentor(self.cfg, device=self.device)
        return augmentor

    def _init_prototype_bank(self):
        """Initialize multi-regime prototype bank."""
        prototype_bank = RegimePrototypeBank(
            n_regimes=self.cfg_rdcarots.DELAYMIX.N_REGIMES,
            embedding_dim=self.cfg_rdcarots.PROJECTOR.OUTPUT_DIM,
            update_momentum=self.cfg_rdcarots.PROTOTYPE_UPDATE_MOMENTUM,
            min_confidence=self.cfg_rdcarots.PROTOTYPE_MIN_CONFIDENCE,
            device=self.device
        )
        return prototype_bank

    def forward(
        self,
        x: torch.Tensor,
        positive_augment: bool = True,
        negative_augment: bool = True,
        return_regime_info: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Input windows (batch_size, window_size, n_variables)
            positive_augment: Apply positive augmentation
            negative_augment: Apply negative augmentation
            return_regime_info: Return regime inference results

        Returns:
            embeddings: (B_total, embedding_dim) or dict if return_regime_info=True
        """
        x_all = x

        # Get regime information if model bank is initialized
        regime_probs = None
        regime_models = []
        if self.model_bank.is_initialized:
            # Split into inputs/outputs for regime inference
            n_vars = x.shape[-1]
            n_inputs = n_vars // 2  # Simplified
            outputs = x[:, :, n_inputs:]
            inputs = x[:, :, :n_inputs]

            # Regime inference
            regime_results = self.model_bank.predict(outputs, inputs)
            regime_probs = regime_results.get('regime_probs')
            regime_models = self.model_bank.get_state_space_models()

        # Apply positive augmentation
        if positive_augment and self.cfg_rdcarots.POSITIVE_AUGMENTOR.ENABLE:
            if len(regime_models) > 0:
                positive_samples = self.positive_augmentor(
                    x_all, regime_models, regime_probs
                )
            else:
                # Fallback: use CAROTS-style
                positive_samples = x_all + torch.randn_like(x_all) * 0.1

            x_all = torch.cat([x_all, positive_samples], dim=0)

        # Apply negative augmentation
        if negative_augment and self.cfg_rdcarots.NEGATIVE_AUGMENTOR.ENABLE:
            if len(regime_models) > 0:
                negative_samples = self.negative_augmentor(
                    x_all,
                    regime_models,
                    regime_probs,
                    self.causal_discoverer.causality_mtx
                )
            else:
                # Fallback: use CAROTS-style
                from models.carots.modeling_negative_augmentor import NegativeAugmentor
                neg_aug_fallback = NegativeAugmentor(self.cfg).to(self.device)
                negative_samples = neg_aug_fallback(x_all, self.causal_discoverer.causality_mtx)

            x_all = torch.cat([x_all, negative_samples], dim=0)

        # Encode
        if self.cfg_rdcarots.ENCODER_ARCH == 'GATv2':
            enc_out = self.encoder(x_all, self.causal_discoverer.causality_mtx)
        elif self.cfg_rdcarots.ENCODER_ARCH in ('lstm', 'iTransformer', 'TimesNet', 'gru'):
            enc_out = self.encoder(x_all)
        else:
            raise ValueError(f"Unsupported encoder architecture: {self.cfg_rdcarots.ENCODER_ARCH}")

        # Project
        out = self.projector(enc_out)

        if return_regime_info:
            return {
                'embeddings': out,
                'regime_probs': regime_probs,
                'regime_results': regime_results if self.model_bank.is_initialized else None
            }
        else:
            return out

    def update_model_bank(self, outputs: torch.Tensor, inputs: torch.Tensor):
        """Update moment statistics and potentially refit models."""
        self.model_bank.update_moments(outputs, inputs)

        # Check if should trigger CP decomposition
        if self.model_bank.should_update():
            print("Triggering model bank update (CP decomposition)...")
            success = self.model_bank.fit_models()
            if success:
                print("Model bank updated successfully.")
            else:
                print("Model bank update failed.")
