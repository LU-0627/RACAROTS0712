"""
DelayMix: Model Bank for Multi-Regime State Space Models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json

from .moment_collection import DynamicMomentCollection
from .cp_decomposition import cp_decomposition, CPFactors
from .markov_recovery import recover_markov_parameters, MarkovParameters
from .ho_kalman import ho_kalman_realization, StateSpaceModel
from .regime_inference import RegimeInference
from .update_trigger import UpdateTrigger, UpdateTriggerConfig


@dataclass
class RegimeModel:
    """Complete regime model with statistics."""
    regime_id: int
    state_space: StateSpaceModel
    markov_params: MarkovParameters
    weight: float  # From CP decomposition
    n_samples: int  # Number of windows assigned to this regime
    last_update_time: float
    confidence: float  # Model quality/confidence


class RegimeDelayModelBank(nn.Module):
    """
    Model bank maintaining multiple regime state space models.

    Args:
        n_outputs: Number of output variables
        n_inputs: Number of input variables
        n_regimes: Number of regimes
        max_lag: Maximum time lag
        config: Dictionary with DelayMix configuration
        device: Torch device
    """

    def __init__(
        self,
        n_outputs: int,
        n_inputs: int,
        n_regimes: int = 3,
        max_lag: int = 20,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.n_regimes = n_regimes
        self.max_lag = max_lag
        self.device = device or torch.device('cpu')

        # Configuration with defaults
        self.config = config or {}
        self.forgetting_factor = self.config.get('forgetting_factor', 0.99)
        self.cp_max_iter = self.config.get('cp_max_iter', 200)
        self.hankel_p = self.config.get('hankel_p', 10)
        self.hankel_q = self.config.get('hankel_q', 10)

        # Moment collection
        self.moment_collection = DynamicMomentCollection(
            n_outputs=n_outputs,
            n_inputs=n_inputs,
            max_lag=max_lag,
            forgetting_factor=self.forgetting_factor,
            device=self.device
        )

        # Regime inference
        self.regime_inference = RegimeInference(
            n_regimes=n_regimes,
            inverse_temperature=self.config.get('inverse_temperature', 1.0),
            confidence_threshold=self.config.get('low_confidence_threshold', 0.4),
            device=self.device
        )

        # Update trigger
        trigger_config = UpdateTriggerConfig(
            fixed_interval=self.config.get('update_interval', 500),
            sample_threshold=self.config.get('update_sample_threshold', 200),
            mismatch_threshold=self.config.get('update_mismatch_threshold', 0.1),
            mismatch_consecutive=self.config.get('update_mismatch_consecutive', 10),
            min_interval=self.config.get('bootstrap_windows', 100)
        )
        self.update_trigger = UpdateTrigger(trigger_config)

        # Model storage
        self.regime_models: List[Optional[RegimeModel]] = [None] * n_regimes
        self.is_initialized = False

    def update_moments(self, outputs: torch.Tensor, inputs: torch.Tensor):
        """
        Update moment statistics.

        Args:
            outputs: (batch_size, n_outputs) or (n_outputs,)
            inputs: (batch_size, n_inputs) or (n_inputs,)
        """
        self.moment_collection.update(outputs, inputs)

    def fit_models(self):
        """
        Fit regime models via CP decomposition and state space realization.
        """
        # Get moment tensor
        moment_tensor = self.moment_collection.get_tensor()  # (n_outputs, n_inputs, max_lag)
        moment_np = moment_tensor.cpu().numpy()

        # CP decomposition
        try:
            cp_factors = cp_decomposition(
                moment_np,
                rank=self.n_regimes,
                init='svd',
                max_iter=self.cp_max_iter,
                tol=1e-6
            )
        except Exception as e:
            print(f"CP decomposition failed: {e}")
            return False

        # Recover Markov parameters
        markov_list = recover_markov_parameters(cp_factors)

        # Build state space models
        import time
        current_time = time.time()

        for r, markov_params in enumerate(markov_list):
            try:
                ss_model = ho_kalman_realization(
                    markov_params.markov_sequence,
                    p=self.hankel_p,
                    q=self.hankel_q,
                    max_states=self.config.get('state_dim_max', 20),
                    stability_margin=self.config.get('stability_margin', 0.95)
                )

                # Store regime model
                self.regime_models[r] = RegimeModel(
                    regime_id=r,
                    state_space=ss_model,
                    markov_params=markov_params,
                    weight=markov_params.weight,
                    n_samples=0,
                    last_update_time=current_time,
                    confidence=1.0 - cp_factors.reconstruction_error
                )

            except Exception as e:
                print(f"State space realization failed for regime {r}: {e}")
                continue

        self.is_initialized = True
        self.update_trigger.reset()
        return True

    def predict(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor
    ) -> Dict:
        """
        Predict using regime models.

        Args:
            outputs: (batch_size, window_size, n_outputs)
            inputs: (batch_size, window_size, n_inputs)

        Returns:
            Dictionary with predictions and regime info
        """
        if not self.is_initialized:
            return {
                'regime_probs': None,
                'best_regime': None,
                'confidence': torch.zeros(outputs.shape[0], device=self.device),
                'prediction_errors': None,
                'is_uncertain': torch.ones(outputs.shape[0], dtype=torch.bool, device=self.device)
            }

        # Get state space models
        ss_models = [rm.state_space for rm in self.regime_models if rm is not None]

        if len(ss_models) == 0:
            return {
                'regime_probs': None,
                'best_regime': None,
                'confidence': torch.zeros(outputs.shape[0], device=self.device),
                'prediction_errors': None,
                'is_uncertain': torch.ones(outputs.shape[0], dtype=torch.bool, device=self.device)
            }

        # Regime inference
        results = self.regime_inference(outputs, inputs, ss_models, mode='soft')

        return results

    def should_update(self, current_error: Optional[float] = None) -> bool:
        """Check if models should be updated."""
        should_update, reason = self.update_trigger.should_update(current_error)
        return should_update

    def get_state_space_models(self) -> List[StateSpaceModel]:
        """Get list of state space models."""
        return [rm.state_space for rm in self.regime_models if rm is not None]

    def state_dict(self, *args, **kwargs) -> Dict:
        """Save state for checkpointing."""
        state = super().state_dict(*args, **kwargs)

        # Save regime models (convert numpy arrays to lists for JSON)
        regime_models_state = []
        for rm in self.regime_models:
            if rm is not None:
                rm_dict = {
                    'regime_id': rm.regime_id,
                    'A': rm.state_space.A.tolist(),
                    'B': rm.state_space.B.tolist(),
                    'C': rm.state_space.C.tolist(),
                    'n_states': rm.state_space.n_states,
                    'weight': float(rm.weight),
                    'n_samples': rm.n_samples,
                    'confidence': float(rm.confidence),
                    'effective_delay': rm.markov_params.effective_delay
                }
                regime_models_state.append(rm_dict)
            else:
                regime_models_state.append(None)

        state['regime_models'] = regime_models_state
        state['is_initialized'] = self.is_initialized

        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """Load state from checkpoint."""
        # Load regime models
        regime_models_state = state_dict.pop('regime_models', None)
        is_initialized = state_dict.pop('is_initialized', False)

        super().load_state_dict(state_dict, strict=False)

        self.is_initialized = is_initialized

        if regime_models_state is not None:
            # Reconstruct regime models (simplified - full reconstruction would need Markov params)
            # For now, just mark as initialized if models were saved
            pass
