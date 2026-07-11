"""
DelayMix: Regime Inference

Estimate regime probabilities p(r|x,u) based on prediction error and model likelihood.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from dataclasses import dataclass

from .ho_kalman import StateSpaceModel


@dataclass
class RegimeInferenceResult:
    """Result of regime inference."""
    regime_probs: np.ndarray  # (n_regimes,) probability distribution
    best_regime: int  # Most likely regime
    confidence: float  # max(regime_probs)
    prediction_errors: np.ndarray  # (n_regimes,) error for each regime
    is_uncertain: bool  # True if confidence below threshold


class RegimeInference(nn.Module):
    """
    Infer current regime from prediction errors.

    Args:
        n_regimes: Number of regimes
        inverse_temperature: β in softmax (higher = sharper distribution)
        confidence_threshold: Threshold for uncertain detection
        device: Torch device
    """

    def __init__(
        self,
        n_regimes: int,
        inverse_temperature: float = 1.0,
        confidence_threshold: float = 0.6,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.inverse_temperature = inverse_temperature
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cpu')

    def compute_prediction_errors(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        state_space_models: List[StateSpaceModel],
        window_size: int
    ) -> torch.Tensor:
        """
        Compute prediction error for each regime.

        Args:
            outputs: Current output window (batch_size, window_size, n_outputs)
            inputs: Current input window (batch_size, window_size, n_inputs)
            state_space_models: List of regime state space models
            window_size: Length of window

        Returns:
            errors: (batch_size, n_regimes) prediction errors
        """
        batch_size = outputs.shape[0]
        errors = torch.zeros(batch_size, self.n_regimes, device=self.device)

        for regime_id, ss_model in enumerate(state_space_models):
            for b in range(batch_size):
                # Simulate this regime's model
                y_true = outputs[b].cpu().numpy()  # (window_size, n_outputs)
                u = inputs[b].cpu().numpy()  # (window_size, n_inputs)

                # Predict using state space model
                y_pred, _ = self._simulate_ss(ss_model, u)

                # Compute MSE
                mse = np.mean((y_true - y_pred) ** 2)
                errors[b, regime_id] = mse

        return errors

    def _simulate_ss(
        self,
        ss_model: StateSpaceModel,
        inputs: np.ndarray
    ) -> tuple:
        """Simulate state space model."""
        A, B, C = ss_model.A, ss_model.B, ss_model.C
        T = inputs.shape[0]
        n_states = ss_model.n_states
        n_outputs = C.shape[0]

        z = np.zeros(n_states)
        outputs = np.zeros((T, n_outputs))

        for t in range(T):
            outputs[t] = C @ z
            z = A @ z + B @ inputs[t]

        return outputs, None

    def infer_soft(
        self,
        prediction_errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft regime assignment via softmax.

        Args:
            prediction_errors: (batch_size, n_regimes)

        Returns:
            regime_probs: (batch_size, n_regimes)
        """
        # Convert errors to probabilities: p(r) ∝ exp(-β * error_r)
        log_probs = -self.inverse_temperature * prediction_errors

        # Numerical stability: subtract max
        log_probs = log_probs - torch.max(log_probs, dim=1, keepdim=True)[0]

        # Softmax
        probs = torch.softmax(log_probs, dim=1)

        return probs

    def infer_hard(
        self,
        prediction_errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard regime assignment (argmin error).

        Args:
            prediction_errors: (batch_size, n_regimes)

        Returns:
            regime_ids: (batch_size,)
        """
        return torch.argmin(prediction_errors, dim=1)

    def forward(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        state_space_models: List[StateSpaceModel],
        mode: str = 'soft'
    ) -> Dict:
        """
        Infer regime distribution.

        Args:
            outputs: (batch_size, window_size, n_outputs)
            inputs: (batch_size, window_size, n_inputs)
            state_space_models: List of regime models
            mode: 'soft' or 'hard'

        Returns:
            Dictionary with regime inference results
        """
        batch_size = outputs.shape[0]
        window_size = outputs.shape[1]

        # Compute prediction errors
        errors = self.compute_prediction_errors(
            outputs, inputs, state_space_models, window_size
        )

        # Soft probabilities
        probs = self.infer_soft(errors)

        # Hard assignment
        best_regimes = self.infer_hard(errors)

        # Confidence
        confidence, _ = torch.max(probs, dim=1)
        is_uncertain = confidence < self.confidence_threshold

        results = {
            'regime_probs': probs,  # (batch_size, n_regimes)
            'best_regime': best_regimes,  # (batch_size,)
            'confidence': confidence,  # (batch_size,)
            'is_uncertain': is_uncertain,  # (batch_size,)
            'prediction_errors': errors  # (batch_size, n_regimes)
        }

        return results


class SmoothRegimeInference:
    """
    Smooth regime inference with temporal consistency.

    Requires regime to be confident for multiple consecutive windows before switching.
    """

    def __init__(
        self,
        n_regimes: int,
        switch_threshold: float = 0.7,
        consecutive_required: int = 3
    ):
        self.n_regimes = n_regimes
        self.switch_threshold = switch_threshold
        self.consecutive_required = consecutive_required

        # State
        self.current_regime = 0
        self.candidate_regime = None
        self.consecutive_count = 0

    def update(self, regime_probs: np.ndarray) -> int:
        """
        Update regime with smoothing.

        Args:
            regime_probs: (n_regimes,) probability distribution

        Returns:
            current_regime: Smoothed regime assignment
        """
        best_regime = np.argmax(regime_probs)
        best_prob = regime_probs[best_regime]

        if best_prob >= self.switch_threshold:
            if best_regime == self.current_regime:
                # Stay in current regime
                self.consecutive_count = 0
                self.candidate_regime = None
            elif best_regime == self.candidate_regime:
                # Accumulating evidence for switch
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_required:
                    # Switch regime
                    self.current_regime = best_regime
                    self.consecutive_count = 0
                    self.candidate_regime = None
            else:
                # New candidate regime
                self.candidate_regime = best_regime
                self.consecutive_count = 1
        else:
            # Low confidence, stay in current regime
            self.consecutive_count = 0
            self.candidate_regime = None

        return self.current_regime

    def reset(self, initial_regime: int = 0):
        """Reset to initial regime."""
        self.current_regime = initial_regime
        self.candidate_regime = None
        self.consecutive_count = 0
