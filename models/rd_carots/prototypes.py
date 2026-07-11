"""
Regime-Conditioned Prototype Bank

Maintains separate prototypes for each normal operating regime.
Supports exponential moving average updates during training and guarded online updates during testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class RegimePrototypeBank(nn.Module):
    """
    Multi-regime prototype bank for anomaly detection.

    Each regime has its own prototype (centroid) in embedding space.
    Prototypes are updated online using high-confidence normal samples.

    Args:
        n_regimes: Number of regimes
        embedding_dim: Dimension of embedding space
        update_momentum: EMA momentum for updates (0 = no update, 1 = replace)
        min_confidence: Minimum regime confidence for update
        device: Torch device
    """

    def __init__(
        self,
        n_regimes: int,
        embedding_dim: int,
        update_momentum: float = 0.1,
        min_confidence: float = 0.7,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.embedding_dim = embedding_dim
        self.update_momentum = update_momentum
        self.min_confidence = min_confidence
        self.device = device or torch.device('cpu')

        # Prototypes: (n_regimes, embedding_dim)
        self.register_buffer(
            'prototypes',
            torch.randn(n_regimes, embedding_dim, device=self.device)
        )

        # Normalize prototypes
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

        # Track statistics
        self.register_buffer(
            'regime_counts',
            torch.zeros(n_regimes, dtype=torch.long, device=self.device)
        )

        self.register_buffer(
            'is_initialized',
            torch.tensor(False, dtype=torch.bool, device=self.device)
        )

    @torch.no_grad()
    def initialize_from_data(
        self,
        embeddings: torch.Tensor,
        regime_assignments: torch.Tensor,
        regime_confidences: torch.Tensor
    ):
        """
        Initialize prototypes from training data.

        Args:
            embeddings: (N, embedding_dim) embeddings
            regime_assignments: (N,) hard regime assignments
            regime_confidences: (N,) confidence scores
        """
        # Filter high-confidence samples
        high_conf_mask = regime_confidences >= self.min_confidence
        embeddings_hc = embeddings[high_conf_mask]
        assignments_hc = regime_assignments[high_conf_mask]

        if len(embeddings_hc) == 0:
            print("Warning: No high-confidence samples for prototype initialization")
            return

        # Compute prototype for each regime
        for r in range(self.n_regimes):
            regime_mask = assignments_hc == r
            regime_embeddings = embeddings_hc[regime_mask]

            if len(regime_embeddings) > 0:
                # Compute mean
                prototype = torch.mean(regime_embeddings, dim=0)
                self.prototypes[r] = F.normalize(prototype, p=2, dim=0)
                self.regime_counts[r] = len(regime_embeddings)
            else:
                print(f"Warning: No samples for regime {r}, keeping random initialization")

        self.is_initialized.fill_(True)

    @torch.no_grad()
    def update(
        self,
        embeddings: torch.Tensor,
        regime_probs: torch.Tensor,
        is_normal: Optional[torch.Tensor] = None,
        use_soft: bool = True
    ):
        """
        Update prototypes with new samples.

        Args:
            embeddings: (batch_size, embedding_dim) new embeddings
            regime_probs: (batch_size, n_regimes) regime probabilities
            is_normal: (batch_size,) normal/anomaly flags (if None, assume all normal)
            use_soft: Use soft regime assignment (weighted update)
        """
        if is_normal is None:
            is_normal = torch.ones(embeddings.shape[0], dtype=torch.bool, device=self.device)

        # Filter normal samples
        normal_mask = is_normal
        embeddings_normal = embeddings[normal_mask]
        regime_probs_normal = regime_probs[normal_mask]

        if len(embeddings_normal) == 0:
            return

        # Get confidence (max prob)
        confidence, _ = torch.max(regime_probs_normal, dim=1)
        high_conf_mask = confidence >= self.min_confidence

        embeddings_hc = embeddings_normal[high_conf_mask]
        regime_probs_hc = regime_probs_normal[high_conf_mask]

        if len(embeddings_hc) == 0:
            return

        # Normalize embeddings
        embeddings_hc = F.normalize(embeddings_hc, p=2, dim=1)

        if use_soft:
            # Soft update: weighted by regime probability
            for r in range(self.n_regimes):
                weights = regime_probs_hc[:, r].unsqueeze(1)  # (N, 1)
                weighted_embeddings = embeddings_hc * weights  # (N, D)

                total_weight = weights.sum()
                if total_weight > 0:
                    new_contribution = weighted_embeddings.sum(dim=0) / total_weight
                    self.prototypes[r] = (
                        (1 - self.update_momentum) * self.prototypes[r] +
                        self.update_momentum * new_contribution
                    )
                    self.prototypes[r] = F.normalize(self.prototypes[r], p=2, dim=0)
                    self.regime_counts[r] += int(total_weight.item())
        else:
            # Hard update: assign to best regime
            regime_assignments = torch.argmax(regime_probs_hc, dim=1)

            for r in range(self.n_regimes):
                regime_mask = regime_assignments == r
                regime_embeddings = embeddings_hc[regime_mask]

                if len(regime_embeddings) > 0:
                    new_contribution = torch.mean(regime_embeddings, dim=0)
                    self.prototypes[r] = (
                        (1 - self.update_momentum) * self.prototypes[r] +
                        self.update_momentum * new_contribution
                    )
                    self.prototypes[r] = F.normalize(self.prototypes[r], p=2, dim=0)
                    self.regime_counts[r] += len(regime_embeddings)

    def get_distances(self, embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute distances from embeddings to all prototypes.

        Args:
            embeddings: (batch_size, embedding_dim)
            normalize: Normalize embeddings before computing distance

        Returns:
            distances: (batch_size, n_regimes)
        """
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Euclidean distance
        distances = torch.cdist(embeddings, self.prototypes)  # (batch_size, n_regimes)

        return distances

    def get_min_distance(self, embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute minimum distance to any prototype.

        Args:
            embeddings: (batch_size, embedding_dim)
            normalize: Normalize embeddings

        Returns:
            min_distances: (batch_size,)
        """
        distances = self.get_distances(embeddings, normalize=normalize)
        min_distances, _ = torch.min(distances, dim=1)
        return min_distances

    def state_dict(self, *args, **kwargs) -> Dict:
        """Save state."""
        state = super().state_dict(*args, **kwargs)
        state['n_regimes'] = self.n_regimes
        state['embedding_dim'] = self.embedding_dim
        state['update_momentum'] = self.update_momentum
        state['min_confidence'] = self.min_confidence
        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """Load state."""
        # Validate dimensions
        assert state_dict['n_regimes'] == self.n_regimes
        assert state_dict['embedding_dim'] == self.embedding_dim

        super().load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return (
            f"RegimePrototypeBank(n_regimes={self.n_regimes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"initialized={self.is_initialized.item()}, "
            f"total_samples={self.regime_counts.sum().item()})"
        )
