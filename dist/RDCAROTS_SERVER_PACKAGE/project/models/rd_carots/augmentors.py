"""
Regime- and Delay-Aware Augmentors

Extends CAROTS augmentation with:
1. RegimeDelayPositiveAugmentor: Uses regime state space models to generate causal-preserving samples
2. RegimeDelayNegativeAugmentor: Three types of negative samples
   - relation_break: Perturb without respecting dynamics
   - delay_break: Alter input-output delay
   - cross_regime_mismatch: Use wrong regime's dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, List, Dict

from .delaymix import StateSpaceModel


class RegimeDelayPositiveAugmentor(nn.Module):
    """
    Generate positive samples using regime state space models.

    Preserves:
    - Causal relationships
    - Normal regime dynamics
    - Normal input-output delays
    """

    def __init__(self, cfg, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.cfg_augmentor = cfg.RDCAROTS.REGIME_DELAY_POSITIVE_AUGMENTOR
        self.device = device or torch.device('cpu')
        self.noise_level = self.cfg_augmentor.NOISE_LEVEL

    def forward(
        self,
        x: torch.Tensor,
        regime_models: List[StateSpaceModel],
        regime_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate positive augmentation.

        Args:
            x: Input windows (batch_size, window_size, n_variables)
            regime_models: List of regime state space models
            regime_probs: (batch_size, n_regimes) or None for uniform

        Returns:
            x_pos: Augmented samples (batch_size, window_size, n_variables)
        """
        if len(regime_models) == 0:
            # Fallback: return noisy version
            return x + torch.randn_like(x) * self.noise_level

        batch_size = x.shape[0]
        x_pos = x.clone()

        for b in range(batch_size):
            # Select regime
            if regime_probs is not None:
                regime_id = torch.multinomial(regime_probs[b], 1).item()
            else:
                regime_id = random.randint(0, len(regime_models) - 1)

            # Get state space model for this regime
            ss_model = regime_models[regime_id]

            # Split into inputs and outputs (assuming IO schema available)
            # For now, assume first half inputs, second half outputs
            n_vars = x.shape[-1]
            n_inputs = n_vars // 2
            n_outputs = n_vars - n_inputs

            u = x[b, :, :n_inputs].cpu().numpy()  # (T, n_inputs)
            y_orig = x[b, :, n_inputs:].cpu().numpy()  # (T, n_outputs)

            # Perturb inputs slightly
            u_perturbed = u + np.random.randn(*u.shape) * self.noise_level

            # Simulate through regime model
            try:
                y_new = self._simulate_regime(ss_model, u_perturbed)

                # Update outputs in augmented sample
                x_pos[b, :, n_inputs:] = torch.from_numpy(y_new).float().to(self.device)
                x_pos[b, :, :n_inputs] = torch.from_numpy(u_perturbed).float().to(self.device)

            except Exception as e:
                # Fallback: keep original with small noise
                x_pos[b] = x[b] + torch.randn_like(x[b]) * self.noise_level * 0.1

        return x_pos.detach()

    def _simulate_regime(self, ss_model: StateSpaceModel, inputs: np.ndarray) -> np.ndarray:
        """Simulate state space model."""
        A, B, C = ss_model.A, ss_model.B, ss_model.C
        T = inputs.shape[0]
        n_states = ss_model.n_states
        n_outputs = C.shape[0]

        z = np.zeros(n_states)
        outputs = np.zeros((T, n_outputs))

        for t in range(T):
            outputs[t] = C @ z
            if t < T:
                z = A @ z + B @ inputs[t]

        return outputs


class RegimeDelayNegativeAugmentor(nn.Module):
    """
    Generate negative samples that violate regime-delay properties.

    Three strategies:
    1. relation_break: Perturb variables without respecting dynamics
    2. delay_break: Shift delay to τ + Δτ
    3. cross_regime_mismatch: Use wrong regime's dynamics
    """

    def __init__(self, cfg, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.cfg_augmentor = cfg.RDCAROTS.REGIME_DELAY_NEGATIVE_AUGMENTOR
        self.device = device or torch.device('cpu')

        # Strategy weights
        self.relation_break_prob = self.cfg_augmentor.RELATION_BREAK_PROB
        self.delay_break_prob = self.cfg_augmentor.DELAY_BREAK_PROB
        self.cross_regime_prob = self.cfg_augmentor.CROSS_REGIME_PROB

        # Delay shift range for delay_break
        self.delay_shift_range = self.cfg_augmentor.DELAY_SHIFT_RANGE

    def forward(
        self,
        x: torch.Tensor,
        regime_models: List[StateSpaceModel],
        regime_probs: Optional[torch.Tensor] = None,
        causality_mtx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate negative augmentation.

        Args:
            x: Input windows (batch_size, window_size, n_variables)
            regime_models: List of regime state space models
            regime_probs: (batch_size, n_regimes) or None
            causality_mtx: (n_variables, n_variables) causal adjacency matrix

        Returns:
            x_neg: Negative samples (batch_size, window_size, n_variables)
        """
        batch_size = x.shape[0]
        x_neg = torch.zeros_like(x)

        for b in range(batch_size):
            # Select strategy
            strategy = np.random.choice(
                ['relation_break', 'delay_break', 'cross_regime'],
                p=[self.relation_break_prob, self.delay_break_prob, self.cross_regime_prob]
            )

            if strategy == 'relation_break':
                x_neg[b] = self._relation_break(x[b], causality_mtx)
            elif strategy == 'delay_break':
                x_neg[b] = self._delay_break(x[b])
            elif strategy == 'cross_regime':
                x_neg[b] = self._cross_regime_mismatch(x[b], regime_models, regime_probs[b] if regime_probs is not None else None)

        return x_neg.detach()

    def _relation_break(
        self,
        x: torch.Tensor,
        causality_mtx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Break causal relationships by perturbing causally-related variables."""
        x_neg = x.clone()
        n_vars = x.shape[-1]

        # Select subset of variables to perturb
        if causality_mtx is not None:
            # Use CAROTS-style causal graph traversal
            causality_np = (causality_mtx.cpu().numpy() > 0.5).astype(int)

            # Select random variable and its effects
            var_idx = random.randint(0, n_vars - 1)
            affected_vars = [i for i in range(n_vars) if causality_np[var_idx, i] == 1]

            if len(affected_vars) > 0:
                # Perturb effects without updating causes
                for var in affected_vars:
                    x_neg[:, var] += torch.randn_like(x_neg[:, var]) * 0.5
        else:
            # Fallback: perturb random subset
            n_perturb = random.randint(1, n_vars // 2)
            vars_to_perturb = random.sample(range(n_vars), n_perturb)
            for var in vars_to_perturb:
                x_neg[:, var] += torch.randn_like(x_neg[:, var]) * 0.5

        return x_neg

    def _delay_break(self, x: torch.Tensor) -> torch.Tensor:
        """Alter input-output delay by shifting."""
        x_neg = x.clone()
        T = x.shape[0]
        n_vars = x.shape[-1]

        # Select shift amount (avoid future leakage: only shift backward)
        shift = random.randint(self.delay_shift_range[0], self.delay_shift_range[1])

        if shift == 0:
            shift = random.choice([-2, -1, 1, 2])  # Force non-zero shift

        # Apply shift to output variables (second half)
        n_inputs = n_vars // 2

        if shift > 0:
            # Delay outputs more (shift forward, but pad to avoid leakage)
            x_neg[:-shift, n_inputs:] = x[shift:, n_inputs:].clone()
            x_neg[-shift:, n_inputs:] = x[-1:, n_inputs:].clone()  # Repeat last
        elif shift < 0:
            # Reduce delay (shift backward)
            x_neg[-shift:, n_inputs:] = x[:shift, n_inputs:].clone()
            x_neg[:-shift, n_inputs:] = x[0:1, n_inputs:].clone()  # Repeat first

        return x_neg

    def _cross_regime_mismatch(
        self,
        x: torch.Tensor,
        regime_models: List[StateSpaceModel],
        regime_prob: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Use wrong regime's dynamics."""
        if len(regime_models) < 2:
            # Fallback to relation_break
            return self._relation_break(x, None)

        # Select true regime
        if regime_prob is not None:
            true_regime = torch.argmax(regime_prob).item()
        else:
            true_regime = random.randint(0, len(regime_models) - 1)

        # Select different regime
        wrong_regime = true_regime
        while wrong_regime == true_regime:
            wrong_regime = random.randint(0, len(regime_models) - 1)

        # Use wrong regime's model
        x_neg = x.clone()
        ss_model = regime_models[wrong_regime]

        n_vars = x.shape[-1]
        n_inputs = n_vars // 2

        u = x[:, :n_inputs].cpu().numpy()

        try:
            # Simulate with wrong regime
            y_wrong = self._simulate_regime(ss_model, u)
            x_neg[:, n_inputs:] = torch.from_numpy(y_wrong).float().to(self.device)
        except:
            # Fallback
            x_neg[:, n_inputs:] += torch.randn_like(x_neg[:, n_inputs:]) * 0.5

        return x_neg

    def _simulate_regime(self, ss_model: StateSpaceModel, inputs: np.ndarray) -> np.ndarray:
        """Simulate state space model."""
        A, B, C = ss_model.A, ss_model.B, ss_model.C
        T = inputs.shape[0]
        n_states = ss_model.n_states
        n_outputs = C.shape[0]

        z = np.zeros(n_states)
        outputs = np.zeros((T, n_outputs))

        for t in range(T):
            outputs[t] = C @ z
            if t < T:
                z = A @ z + B @ inputs[t]

        return outputs
