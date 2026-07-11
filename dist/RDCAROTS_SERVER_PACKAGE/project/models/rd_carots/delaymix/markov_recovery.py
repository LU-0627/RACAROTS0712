"""
DelayMix: Markov Parameter Recovery from CP Factors

Reconstructs regime-specific Markov parameters and identifies input delays.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .cp_decomposition import CPFactors


@dataclass
class MarkovParameters:
    """Markov parameters for a single regime."""
    regime_id: int
    markov_sequence: np.ndarray  # (max_lag, n_outputs, n_inputs)
    effective_delay: int  # First lag where response becomes significant
    weight: float  # Regime weight from CP
    energy: float  # Total energy ||H||_F


def recover_markov_parameters(
    cp_factors: CPFactors,
    threshold_ratio: float = 0.05
) -> List[MarkovParameters]:
    """
    Recover regime-specific Markov parameters from CP factors.

    Args:
        cp_factors: CP decomposition factors
        threshold_ratio: Ratio of max norm to determine effective delay

    Returns:
        List of MarkovParameters for each regime
    """
    n_outputs, R = cp_factors.output_factors.shape
    n_inputs = cp_factors.input_factors.shape[0]
    max_lag = cp_factors.lag_factors.shape[0]

    markov_list = []

    for r in range(R):
        # Extract factors for regime r
        weight_r = cp_factors.weights[r]
        a_r = cp_factors.output_factors[:, r]  # (n_outputs,)
        b_r = cp_factors.input_factors[:, r]   # (n_inputs,)
        c_r = cp_factors.lag_factors[:, r]     # (max_lag,)

        # Reconstruct Markov sequence: H_r[τ] = λ_r * c_r[τ] * (a_r ⊗ b_r)
        markov_sequence = np.zeros((max_lag, n_outputs, n_inputs))

        for lag in range(max_lag):
            # Outer product scaled by lag factor and weight
            markov_sequence[lag] = weight_r * c_r[lag] * np.outer(a_r, b_r)

        # Identify effective delay: first lag where ||H[τ]||_F exceeds threshold
        norms = np.array([np.linalg.norm(markov_sequence[lag], 'fro') for lag in range(max_lag)])
        max_norm = np.max(norms)
        threshold = threshold_ratio * max_norm

        effective_delay = 0
        for lag in range(max_lag):
            if norms[lag] >= threshold:
                effective_delay = lag
                break

        # Total energy
        energy = np.linalg.norm(markov_sequence, 'fro')

        markov_list.append(MarkovParameters(
            regime_id=r,
            markov_sequence=markov_sequence,
            effective_delay=effective_delay,
            weight=weight_r,
            energy=energy
        ))

    return markov_list


def validate_markov_causality(markov_params: MarkovParameters) -> bool:
    """
    Validate that Markov parameters respect causality (no significant pre-delay response).

    Args:
        markov_params: Markov parameters to validate

    Returns:
        True if causal (low energy before effective delay)
    """
    delay = markov_params.effective_delay
    if delay == 0:
        return True  # No delay to validate

    # Energy before delay should be small
    pre_delay_energy = np.linalg.norm(markov_params.markov_sequence[:delay], 'fro')
    total_energy = markov_params.energy

    if total_energy < 1e-8:
        return True  # Near-zero system

    ratio = pre_delay_energy / total_energy

    return ratio < 0.1  # Less than 10% energy before delay


def get_impulse_response(
    markov_params: MarkovParameters,
    input_idx: int,
    output_idx: int
) -> np.ndarray:
    """
    Get impulse response from specific input to specific output.

    Args:
        markov_params: Markov parameters
        input_idx: Input channel index
        output_idx: Output channel index

    Returns:
        Impulse response sequence (max_lag,)
    """
    max_lag = markov_params.markov_sequence.shape[0]
    response = np.zeros(max_lag)

    for lag in range(max_lag):
        response[lag] = markov_params.markov_sequence[lag, output_idx, input_idx]

    return response


def compare_markov_parameters(
    markov_a: MarkovParameters,
    markov_b: MarkovParameters,
    metric: str = 'frobenius'
) -> float:
    """
    Compare two Markov parameter sequences.

    Args:
        markov_a: First Markov parameters
        markov_b: Second Markov parameters
        metric: Comparison metric ('frobenius', 'cosine')

    Returns:
        Distance or dissimilarity measure
    """
    H_a = markov_a.markov_sequence
    H_b = markov_b.markov_sequence

    if metric == 'frobenius':
        # Frobenius norm of difference
        return np.linalg.norm(H_a - H_b, 'fro')

    elif metric == 'cosine':
        # Cosine distance between flattened sequences
        h_a_flat = H_a.flatten()
        h_b_flat = H_b.flatten()

        norm_a = np.linalg.norm(h_a_flat)
        norm_b = np.linalg.norm(h_b_flat)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0  # Maximum distance for zero vectors

        cos_sim = np.dot(h_a_flat, h_b_flat) / (norm_a * norm_b)
        return 1.0 - cos_sim

    else:
        raise ValueError(f"Unknown metric: {metric}")
