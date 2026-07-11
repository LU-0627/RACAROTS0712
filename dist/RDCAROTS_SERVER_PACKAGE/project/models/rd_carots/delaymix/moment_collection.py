"""
DelayMix: Dynamic Moment Collection for Streaming Multi-Regime System Identification

This module implements incremental moment tensor computation with forgetting factor,
maintaining fixed memory footprint independent of time.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class DynamicMomentCollection(nn.Module):
    """
    Maintains running statistics for system moment tensor M[i,j,τ] = E[y_i(t) * u_j(t-τ)]

    Args:
        n_outputs: Number of output variables
        n_inputs: Number of input variables
        max_lag: Maximum time lag to consider
        forgetting_factor: Exponential forgetting factor (0 < λ ≤ 1)
        device: Torch device
    """

    def __init__(
        self,
        n_outputs: int,
        n_inputs: int,
        max_lag: int = 20,
        forgetting_factor: float = 0.99,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.max_lag = max_lag
        self.forgetting_factor = forgetting_factor
        self.device = device or torch.device('cpu')

        # Initialize moment tensor: M[output, input, lag]
        self.register_buffer(
            'moment_tensor',
            torch.zeros(n_outputs, n_inputs, max_lag, device=self.device)
        )

        # Track number of updates for debugging
        self.register_buffer(
            'n_updates',
            torch.tensor(0, dtype=torch.long, device=self.device)
        )

        # Circular buffer for maintaining input history (for lag computation)
        self.register_buffer(
            'input_history',
            torch.zeros(max_lag, n_inputs, device=self.device)
        )

        self.register_buffer(
            'history_ptr',
            torch.tensor(0, dtype=torch.long, device=self.device)
        )

    def update(self, outputs: torch.Tensor, inputs: torch.Tensor):
        """
        Update moment tensor with new observation.

        Args:
            outputs: Current output (batch_size, n_outputs) or (n_outputs,)
            inputs: Current input (batch_size, n_inputs) or (n_inputs,)
        """
        # Handle batch dimension
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        batch_size = outputs.shape[0]

        for b in range(batch_size):
            y_t = outputs[b]  # (n_outputs,)
            u_t = inputs[b]   # (n_inputs,)

            # Compute moment contributions for all lags
            for lag in range(self.max_lag):
                # Get lagged input from history
                hist_idx = (self.history_ptr - lag) % self.max_lag
                u_t_lag = self.input_history[hist_idx]  # (n_inputs,)

                # Compute outer product: y_t ⊗ u_{t-τ}
                moment_contrib = torch.outer(y_t, u_t_lag)  # (n_outputs, n_inputs)

                # Exponential moving average update
                self.moment_tensor[:, :, lag] = (
                    self.forgetting_factor * self.moment_tensor[:, :, lag] +
                    (1 - self.forgetting_factor) * moment_contrib
                )

            # Update input history (circular buffer)
            self.input_history[self.history_ptr] = u_t
            self.history_ptr = (self.history_ptr + 1) % self.max_lag
            self.n_updates += 1

    def get_tensor(self) -> torch.Tensor:
        """
        Get current moment tensor.

        Returns:
            Moment tensor of shape (n_outputs, n_inputs, max_lag)
        """
        return self.moment_tensor.clone()

    def reset(self):
        """Reset all statistics."""
        self.moment_tensor.zero_()
        self.input_history.zero_()
        self.n_updates.zero_()
        self.history_ptr.zero_()

    def state_dict(self, *args, **kwargs) -> Dict:
        """Save state for checkpointing."""
        state = super().state_dict(*args, **kwargs)
        state['n_outputs'] = self.n_outputs
        state['n_inputs'] = self.n_inputs
        state['max_lag'] = self.max_lag
        state['forgetting_factor'] = self.forgetting_factor
        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """Load state from checkpoint."""
        # Validate dimensions match
        assert state_dict['n_outputs'] == self.n_outputs
        assert state_dict['n_inputs'] == self.n_inputs
        assert state_dict['max_lag'] == self.max_lag

        super().load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return (
            f"DynamicMomentCollection(n_outputs={self.n_outputs}, "
            f"n_inputs={self.n_inputs}, max_lag={self.max_lag}, "
            f"forgetting_factor={self.forgetting_factor}, "
            f"n_updates={self.n_updates.item()})"
        )


class BatchMomentCollection:
    """
    Batch version for offline CP decomposition on historical data.

    Args:
        n_outputs: Number of output variables
        n_inputs: Number of input variables
        max_lag: Maximum time lag
    """

    def __init__(self, n_outputs: int, n_inputs: int, max_lag: int = 20):
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.max_lag = max_lag
        self.moment_tensor = np.zeros((n_outputs, n_inputs, max_lag))
        self.n_samples = 0

    def fit(self, outputs: np.ndarray, inputs: np.ndarray):
        """
        Compute moment tensor from batch data.

        Args:
            outputs: Output time series (T, n_outputs)
            inputs: Input time series (T, n_inputs)
        """
        T = outputs.shape[0]
        assert inputs.shape[0] == T

        for lag in range(self.max_lag):
            if lag >= T:
                break

            # Compute cross-correlation at this lag
            y_slice = outputs[lag:]  # (T-lag, n_outputs)
            u_slice = inputs[:T-lag]  # (T-lag, n_inputs)

            # M[:,:,lag] = (1/N) * Σ_t y(t) ⊗ u(t-lag)
            self.moment_tensor[:, :, lag] = (y_slice.T @ u_slice) / (T - lag)

        self.n_samples = T

    def get_tensor(self) -> np.ndarray:
        """Get moment tensor."""
        return self.moment_tensor.copy()
