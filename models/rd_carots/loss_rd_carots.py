"""
Regime-Conditioned Similarity-filtered One-class Contrastive Loss

Extends CAROTS SOC loss with regime awareness:
- Same-regime normal samples are primary positives
- Cross-regime samples have reduced positive weight
- Soft regime weighting: w_ij = Σ_r p(r|x_i)p(r|x_j)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def regime_conditioned_soc_loss(
    embeddings: torch.Tensor,
    regime_probs: Optional[torch.Tensor],
    cfg,
    is_positive: Optional[torch.Tensor] = None,
    is_negative: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute regime-conditioned SOC loss.

    Args:
        embeddings: (B_total, D) embeddings [original, positives, negatives]
        regime_probs: (B_original, n_regimes) regime probabilities for original samples
        cfg: Configuration with RDCAROTS.* parameters
        is_positive: (B_total,) boolean mask for positive samples
        is_negative: (B_total,) boolean mask for negative samples

    Returns:
        loss: Scalar loss
    """
    cfg_loss = cfg.RDCAROTS.LOSS

    # Determine split points
    if is_positive is not None:
        n_orig = (~is_positive & ~is_negative).sum().item()
    else:
        # Assume: [original, positives, negatives]
        B_total = embeddings.shape[0]
        n_orig = B_total // 3 if cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE else B_total // 2

    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix
    sim_mtx = torch.matmul(embeddings_norm, embeddings_norm.T)  # (B_total, B_total)

    # Compute regime weight matrix for original samples
    if regime_probs is not None and cfg_loss.USE_REGIME_WEIGHTING:
        # w_ij = Σ_r p(r|x_i) * p(r|x_j)
        regime_weights = torch.matmul(regime_probs, regime_probs.T)  # (n_orig, n_orig)

        # Expand to full batch (positives/negatives get weight from their source)
        regime_weights_full = torch.ones(embeddings.shape[0], embeddings.shape[0], device=embeddings.device)
        regime_weights_full[:n_orig, :n_orig] = regime_weights
    else:
        regime_weights_full = torch.ones_like(sim_mtx)

    # Apply similarity threshold filtering (from CAROTS)
    sim_threshold = cfg.RDCAROTS.SIM_THRESHOLD if hasattr(cfg.RDCAROTS, 'SIM_THRESHOLD') else cfg.CAROTS.SIM_THRESHOLD

    # Positive pairs: original samples with high similarity
    sim_mtx_pos = sim_mtx[:n_orig, :n_orig]
    positive_mask = sim_mtx_pos >= sim_threshold

    # Apply regime weighting to positive pairs
    positive_mask_weighted = positive_mask.float() * regime_weights[:n_orig, :n_orig] if regime_probs is not None else positive_mask.float()

    # Mask low-similarity positives
    sim_mtx_masked = sim_mtx.clone()

    # Mask out self-similarity
    sim_mtx_masked.fill_diagonal_(float('-inf'))

    # Mask out low-similarity positives and their augmentations
    indices_low_sim = torch.where(~positive_mask)
    sim_mtx_masked[indices_low_sim[0], indices_low_sim[1]] = float('-inf')

    # If positives exist, also mask their low-sim counterparts
    if embeddings.shape[0] > n_orig:
        for i in range(n_orig):
            for j in range(n_orig):
                if not positive_mask[i, j]:
                    # Also mask augmented versions
                    if embeddings.shape[0] > 2 * n_orig:  # Has negatives
                        sim_mtx_masked[i, j + n_orig] = float('-inf')  # Positive augmentation
                        sim_mtx_masked[i, j + 2 * n_orig] = float('-inf')  # Negative augmentation

    # Exponential and temperature scaling
    temperature = cfg.RDCAROTS.TEMPERATURE if hasattr(cfg.RDCAROTS, 'TEMPERATURE') else cfg.CAROTS.TEMPERATURE
    sim_mtx_exp = torch.exp(sim_mtx_masked / temperature)

    # Compute denominator: sum over all non-masked samples
    # For original samples, include positives but emphasize negatives
    if embeddings.shape[0] > 2 * n_orig:
        # Has negative samples
        neg_start = 2 * n_orig
        neg_similarities = sim_mtx_exp[:n_orig, neg_start:]
        denominator = sim_mtx_exp[:n_orig, :].sum(dim=1, keepdim=True)
    else:
        denominator = sim_mtx_exp[:n_orig, :].sum(dim=1, keepdim=True)

    # Normalize
    sim_mtx_normalized = sim_mtx_exp / (denominator + 1e-12)

    # Apply negative log
    sim_mtx_log = -torch.log(sim_mtx_normalized[:n_orig, :n_orig] + 1e-12)

    # Compute loss: weighted average over positive pairs
    positive_weights = positive_mask_weighted

    if positive_weights.sum() > n_orig:  # Has valid positive pairs
        loss = (sim_mtx_log * positive_weights).sum() / (positive_weights.sum() - n_orig + 1e-8)
    else:
        # Fallback: no valid positives, return small penalty
        loss = torch.tensor(0.0, device=embeddings.device)

    # Ensure no NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Invalid loss detected. Setting to 0.")
        loss = torch.tensor(0.0, device=embeddings.device)

    return loss


def compute_loss_components(
    embeddings: torch.Tensor,
    regime_probs: Optional[torch.Tensor],
    cfg
) -> dict:
    """
    Compute and return individual loss components for logging.

    Returns:
        Dictionary with loss components
    """
    n_orig = embeddings.shape[0] // 3 if cfg.RDCAROTS.NEGATIVE_AUGMENTOR.ENABLE else embeddings.shape[0] // 2

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    sim_mtx = torch.matmul(embeddings_norm, embeddings_norm.T)

    # Component 1: Average similarity among originals
    orig_sim = sim_mtx[:n_orig, :n_orig].mean()

    # Component 2: Average similarity between originals and positives
    if embeddings.shape[0] > n_orig:
        pos_sim = sim_mtx[:n_orig, n_orig:2*n_orig].mean()
    else:
        pos_sim = torch.tensor(0.0)

    # Component 3: Average similarity between originals and negatives
    if embeddings.shape[0] > 2 * n_orig:
        neg_sim = sim_mtx[:n_orig, 2*n_orig:].mean()
    else:
        neg_sim = torch.tensor(0.0)

    # Component 4: Regime entropy
    if regime_probs is not None:
        regime_entropy = -(regime_probs * torch.log(regime_probs + 1e-12)).sum(dim=1).mean()
    else:
        regime_entropy = torch.tensor(0.0)

    return {
        'orig_sim': orig_sim.item(),
        'pos_sim': pos_sim.item(),
        'neg_sim': neg_sim.item(),
        'regime_entropy': regime_entropy.item()
    }
