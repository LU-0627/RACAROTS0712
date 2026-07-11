"""
DelayMix: Tensor Decomposition for Multi-Regime Identification

CP (CANDECOMP/PARAFAC) decomposition to extract regime components from moment tensor.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CPFactors:
    """Storage for CP decomposition factors."""
    weights: np.ndarray  # (R,) regime weights
    output_factors: np.ndarray  # (n_outputs, R)
    input_factors: np.ndarray  # (n_inputs, R)
    lag_factors: np.ndarray  # (max_lag, R)
    n_regimes: int
    reconstruction_error: float


def cp_decomposition(
    moment_tensor: np.ndarray,
    rank: int,
    init: str = 'svd',
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 1,
    random_state: Optional[int] = None
) -> CPFactors:
    """
    Perform CP decomposition on moment tensor.

    Args:
        moment_tensor: Tensor of shape (n_outputs, n_inputs, max_lag)
        rank: Number of components (regimes) to extract
        init: Initialization method ('svd', 'random')
        max_iter: Maximum iterations
        tol: Convergence tolerance
        n_restarts: Number of random restarts (for 'random' init)
        random_state: Random seed

    Returns:
        CPFactors object containing decomposition results
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import parafac

        # Set backend to numpy
        tl.set_backend('numpy')

        if random_state is not None:
            np.random.seed(random_state)

        best_factors = None
        best_error = float('inf')

        n_tries = n_restarts if init == 'random' else 1

        for trial in range(n_tries):
            try:
                # Run CP decomposition
                factors = parafac(
                    moment_tensor,
                    rank=rank,
                    init=init,
                    n_iter_max=max_iter,
                    tol=tol,
                    random_state=random_state + trial if random_state is not None else None
                )

                # Extract weights and factor matrices
                if isinstance(factors, tuple):
                    weights, factor_matrices = factors
                else:
                    # Older tensorly versions return differently
                    weights = factors.weights
                    factor_matrices = factors.factors

                # Compute reconstruction
                reconstructed = tl.cp_to_tensor((weights, factor_matrices))
                error = np.linalg.norm(moment_tensor - reconstructed) / np.linalg.norm(moment_tensor)

                if error < best_error:
                    best_error = error
                    best_factors = (weights, factor_matrices)

            except Exception as e:
                print(f"CP decomposition trial {trial} failed: {e}")
                continue

        if best_factors is None:
            raise RuntimeError("All CP decomposition trials failed")

        weights, factor_matrices = best_factors

        # Ensure weights are positive and normalized
        weights = np.abs(weights)
        weights = weights / np.sum(weights)

        return CPFactors(
            weights=weights,
            output_factors=factor_matrices[0],  # (n_outputs, R)
            input_factors=factor_matrices[1],   # (n_inputs, R)
            lag_factors=factor_matrices[2],     # (max_lag, R)
            n_regimes=rank,
            reconstruction_error=best_error
        )

    except ImportError:
        raise ImportError(
            "tensorly is required for CP decomposition. "
            "Install with: pip install tensorly"
        )


def validate_cp_decomposition(
    moment_tensor: np.ndarray,
    cp_factors: CPFactors,
    error_threshold: float = 0.5
) -> bool:
    """
    Validate CP decomposition quality.

    Args:
        moment_tensor: Original tensor
        cp_factors: Decomposition result
        error_threshold: Maximum acceptable relative error

    Returns:
        True if decomposition is valid
    """
    if cp_factors.reconstruction_error > error_threshold:
        return False

    # Check for NaN or Inf
    if (np.any(np.isnan(cp_factors.weights)) or
        np.any(np.isnan(cp_factors.output_factors)) or
        np.any(np.isnan(cp_factors.input_factors)) or
        np.any(np.isnan(cp_factors.lag_factors))):
        return False

    if (np.any(np.isinf(cp_factors.weights)) or
        np.any(np.isinf(cp_factors.output_factors)) or
        np.any(np.isinf(cp_factors.input_factors)) or
        np.any(np.isinf(cp_factors.lag_factors))):
        return False

    return True


def select_rank_aic(
    moment_tensor: np.ndarray,
    rank_candidates: List[int],
    **cp_kwargs
) -> int:
    """
    Select optimal rank using Akaike Information Criterion (AIC).

    Args:
        moment_tensor: Moment tensor
        rank_candidates: List of ranks to try
        **cp_kwargs: Additional arguments for cp_decomposition

    Returns:
        Optimal rank
    """
    n_outputs, n_inputs, max_lag = moment_tensor.shape
    n_obs = n_outputs * n_inputs * max_lag

    best_rank = rank_candidates[0]
    best_aic = float('inf')

    for rank in rank_candidates:
        try:
            cp_factors = cp_decomposition(moment_tensor, rank=rank, **cp_kwargs)

            # Number of parameters
            n_params = rank + rank * (n_outputs + n_inputs + max_lag)

            # Residual sum of squares
            rss = (cp_factors.reconstruction_error ** 2) * np.sum(moment_tensor ** 2)

            # AIC = n * log(RSS/n) + 2k
            aic = n_obs * np.log(rss / n_obs) + 2 * n_params

            if aic < best_aic:
                best_aic = aic
                best_rank = rank

        except Exception as e:
            print(f"Rank {rank} failed: {e}")
            continue

    return best_rank


class SystemTensor:
    """
    Wrapper for system moment tensor with regime extraction.
    """

    def __init__(
        self,
        n_outputs: int,
        n_inputs: int,
        max_lag: int = 20,
        rank: int = 3,
        auto_rank: bool = False
    ):
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.max_lag = max_lag
        self.rank = rank
        self.auto_rank = auto_rank
        self.cp_factors: Optional[CPFactors] = None

    def fit(
        self,
        moment_tensor: np.ndarray,
        rank_candidates: Optional[List[int]] = None,
        **cp_kwargs
    ):
        """
        Fit CP decomposition to moment tensor.

        Args:
            moment_tensor: Moment tensor (n_outputs, n_inputs, max_lag)
            rank_candidates: Ranks to try if auto_rank=True
            **cp_kwargs: Additional CP decomposition arguments
        """
        if self.auto_rank:
            if rank_candidates is None:
                rank_candidates = [2, 3, 4, 5]

            optimal_rank = select_rank_aic(moment_tensor, rank_candidates, **cp_kwargs)
            print(f"Auto-selected rank: {optimal_rank}")
            self.rank = optimal_rank

        # Perform CP decomposition
        self.cp_factors = cp_decomposition(moment_tensor, rank=self.rank, **cp_kwargs)

        # Validate
        is_valid = validate_cp_decomposition(moment_tensor, self.cp_factors)
        if not is_valid:
            print(f"Warning: CP decomposition quality is poor (error={self.cp_factors.reconstruction_error:.4f})")

    def get_factors(self) -> CPFactors:
        """Get CP factors."""
        if self.cp_factors is None:
            raise ValueError("Must call fit() before get_factors()")
        return self.cp_factors
