"""
DelayMix: Ho-Kalman State Space Realization

Converts Markov parameters to minimal state space representation:
    z(t+1) = A z(t) + B u(t)
    y(t) = C z(t)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class StateSpaceModel:
    """State space representation of a regime."""
    A: np.ndarray  # State transition matrix (n_states, n_states)
    B: np.ndarray  # Input matrix (n_states, n_inputs)
    C: np.ndarray  # Output matrix (n_outputs, n_states)
    n_states: int
    eigenvalues: np.ndarray  # Eigenvalues of A for stability check
    is_stable: bool


def build_hankel_matrix(
    markov_sequence: np.ndarray,
    p: int,
    q: int
) -> np.ndarray:
    """
    Build Hankel matrix from Markov parameters.

    Args:
        markov_sequence: Markov sequence (max_lag, n_outputs, n_inputs)
        p: Number of block columns
        q: Number of block rows

    Returns:
        Hankel matrix H of shape (q*n_outputs, p*n_inputs)
    """
    max_lag, n_outputs, n_inputs = markov_sequence.shape

    if p + q > max_lag:
        warnings.warn(f"Hankel size {p}+{q} exceeds available lags {max_lag}, truncating")
        available = max_lag - 1
        p = min(p, available // 2)
        q = min(q, available - p)

    # Initialize Hankel matrix
    H = np.zeros((q * n_outputs, p * n_inputs))

    for i in range(q):
        for j in range(p):
            lag = i + j
            if lag < max_lag:
                H[i*n_outputs:(i+1)*n_outputs, j*n_inputs:(j+1)*n_inputs] = markov_sequence[lag]

    return H


def ho_kalman_realization(
    markov_sequence: np.ndarray,
    p: int = 10,
    q: int = 10,
    sv_threshold: Optional[float] = None,
    max_states: int = 20,
    stability_margin: float = 0.95
) -> StateSpaceModel:
    """
    Compute state space realization using Ho-Kalman algorithm.

    Args:
        markov_sequence: Markov parameters (max_lag, n_outputs, n_inputs)
        p: Number of Hankel block columns
        q: Number of Hankel block rows
        sv_threshold: Singular value threshold (absolute or None for auto)
        max_states: Maximum state dimension
        stability_margin: Scale A if unstable to have max |eig| = stability_margin

    Returns:
        StateSpaceModel
    """
    max_lag, n_outputs, n_inputs = markov_sequence.shape

    # Build Hankel matrices H0 and H1
    H0 = build_hankel_matrix(markov_sequence, p, q)
    H1 = build_hankel_matrix(markov_sequence[1:], p, q)  # Shifted by one lag

    # Add small regularization for numerical stability
    reg = 1e-8 * np.eye(min(H0.shape))
    H0_reg = H0 + reg[:H0.shape[0], :H0.shape[1]]

    # SVD of H0
    U, S, Vt = np.linalg.svd(H0_reg, full_matrices=False)

    # Determine state dimension
    if sv_threshold is None:
        # Auto: keep singular values capturing 95% of energy
        cumsum = np.cumsum(S ** 2)
        total = cumsum[-1]
        n_states = np.searchsorted(cumsum, 0.95 * total) + 1
    else:
        # Use threshold
        n_states = np.sum(S > sv_threshold)

    # Enforce bounds
    n_states = max(1, min(n_states, max_states, len(S)))

    # Truncate to n_states
    U_n = U[:, :n_states]
    S_n = S[:n_states]
    Vt_n = Vt[:n_states, :]

    # Compute state space matrices
    Sigma_sqrt = np.diag(np.sqrt(S_n))
    Sigma_inv_sqrt = np.diag(1.0 / np.sqrt(S_n))

    # Observability matrix: O = U_n @ Sigma_sqrt
    # Controllability matrix: C_mat = Sigma_sqrt @ Vt_n

    # Extract C (first n_outputs rows of observability matrix)
    O = U_n @ Sigma_sqrt
    C = O[:n_outputs, :]

    # Extract B (first n_inputs columns of controllability matrix)
    C_mat = Sigma_sqrt @ Vt_n
    B = C_mat[:, :n_inputs]

    # Compute A from shifted Hankel
    # A = Sigma_inv_sqrt @ U_n^T @ H1 @ Vt_n^T @ Sigma_inv_sqrt
    try:
        A = Sigma_inv_sqrt @ U_n.T @ H1 @ Vt_n.T @ Sigma_inv_sqrt
    except Exception as e:
        warnings.warn(f"A computation failed: {e}, using identity")
        A = np.eye(n_states)

    # Check stability
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    is_stable = max_eig <= 1.0

    # Enforce stability by scaling if needed
    if not is_stable and stability_margin < 1.0:
        scale_factor = stability_margin / max_eig
        A = A * scale_factor
        eigenvalues = eigenvalues * scale_factor
        is_stable = True
        warnings.warn(f"Scaled A by {scale_factor:.3f} to enforce stability")

    return StateSpaceModel(
        A=A,
        B=B,
        C=C,
        n_states=n_states,
        eigenvalues=eigenvalues,
        is_stable=is_stable
    )


def validate_state_space(
    ss_model: StateSpaceModel,
    markov_sequence: np.ndarray,
    n_check: int = 10,
    rtol: float = 0.1
) -> Tuple[bool, float]:
    """
    Validate state space model by checking if it reproduces Markov parameters.

    Args:
        ss_model: State space model
        markov_sequence: Original Markov parameters
        n_check: Number of lags to check
        rtol: Relative tolerance

    Returns:
        (is_valid, relative_error)
    """
    A, B, C = ss_model.A, ss_model.B, ss_model.C
    max_lag = min(n_check, markov_sequence.shape[0])

    # Reconstruct Markov parameters from state space
    # H[0] = C @ B
    # H[k] = C @ A^k @ B
    errors = []

    for k in range(max_lag):
        if k == 0:
            H_reconstructed = C @ B
        else:
            A_power = np.linalg.matrix_power(A, k)
            H_reconstructed = C @ A_power @ B

        H_true = markov_sequence[k]

        error = np.linalg.norm(H_reconstructed - H_true, 'fro')
        true_norm = np.linalg.norm(H_true, 'fro')

        if true_norm > 1e-10:
            errors.append(error / true_norm)
        else:
            errors.append(error)

    avg_error = np.mean(errors)
    is_valid = avg_error < rtol

    return is_valid, avg_error


def simulate_state_space(
    ss_model: StateSpaceModel,
    inputs: np.ndarray,
    initial_state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate state space model given input sequence.

    Args:
        ss_model: State space model
        inputs: Input sequence (T, n_inputs)
        initial_state: Initial state (n_states,), defaults to zero

    Returns:
        (outputs, states):
            outputs: (T, n_outputs)
            states: (T, n_states)
    """
    A, B, C = ss_model.A, ss_model.B, ss_model.C
    T = inputs.shape[0]
    n_states = ss_model.n_states
    n_outputs = C.shape[0]

    if initial_state is None:
        z = np.zeros(n_states)
    else:
        z = initial_state.copy()

    outputs = np.zeros((T, n_outputs))
    states = np.zeros((T, n_states))

    for t in range(T):
        # Output
        outputs[t] = C @ z

        # Store state
        states[t] = z

        # Update state
        z = A @ z + B @ inputs[t]

    return outputs, states
