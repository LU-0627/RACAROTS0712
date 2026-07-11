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
    is_negative: Optional[torch.Tensor] = None,
    current_sim_threshold: Optional[float] = None
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

    # Extract G1 and G2 (normal samples)
    normal_mask = ~is_negative
    embeddings_normal = embeddings[normal_mask]
    n_normal = embeddings_normal.shape[0]

    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix
    sim_mtx = torch.matmul(embeddings_norm, embeddings_norm.T)  # (B_total, B_total)

    # Compute regime weight matrix for G1 samples
    if regime_probs is not None and cfg_loss.USE_REGIME_WEIGHTING:
        # w_ij = Σ_r p(r|x_i) * p(r|x_j) for i,j in G1
        regime_weights_g1 = torch.matmul(regime_probs, regime_probs.T)  # (B, B)

        # Expand to G1+G2
        if cfg.RDCAROTS.POSITIVE_AUGMENTOR.ENABLE:
            regime_weights = torch.zeros(n_normal, n_normal, device=embeddings.device)
            regime_weights[:batch_size, :batch_size] = regime_weights_g1
            regime_weights[batch_size:, :batch_size] = regime_weights_g1
            regime_weights[:batch_size, batch_size:] = regime_weights_g1
            regime_weights[batch_size:, batch_size:] = regime_weights_g1
        else:
            regime_weights = regime_weights_g1
    else:
        regime_weights = torch.ones(n_normal, n_normal, device=embeddings.device)

    # Apply similarity threshold
    if current_sim_threshold is not None:
        sim_threshold = current_sim_threshold
    elif hasattr(cfg, 'RDCAROTS') and hasattr(cfg.RDCAROTS, 'SIM_THRESHOLD'):
        sim_threshold = cfg.RDCAROTS.SIM_THRESHOLD
    else:
        sim_threshold = 0.5

    # Positive pairs: normal samples with high similarity
    sim_mtx_normal = sim_mtx[normal_mask][:, normal_mask]
    positive_mask = sim_mtx_normal >= sim_threshold

    # Apply regime weighting
    positive_mask_weighted = positive_mask.float() * regime_weights

    # Mask self-similarity
    sim_mtx_masked = sim_mtx.clone()
    sim_mtx_masked.fill_diagonal_(float('-inf'))

    # Mask low-similarity pairs
    mask_matrix = torch.ones_like(sim_mtx, dtype=torch.bool)
    mask_matrix[normal_mask, :][:, normal_mask] = positive_mask
    mask_matrix.fill_diagonal_(False)
    sim_mtx_masked[~mask_matrix] = float('-inf')

    # Temperature scaling
    temperature = cfg.RDCAROTS.TEMPERATURE if hasattr(cfg.RDCAROTS, 'TEMPERATURE') else 0.1
    sim_mtx_exp = torch.exp(sim_mtx_masked / temperature)

    # Denominator
    denominator = sim_mtx_exp[normal_mask].sum(dim=1, keepdim=True)

    # Normalize
    sim_mtx_normalized = sim_mtx_exp / (denominator + 1e-12)

    # Negative log
    sim_mtx_log = -torch.log(sim_mtx_normalized[normal_mask][:, normal_mask] + 1e-12)

    # Weighted loss
    if positive_mask_weighted.sum() > n_normal:
        loss = (sim_mtx_log * positive_mask_weighted).sum() / (positive_mask_weighted.sum() - n_normal + 1e-8)
    else:
        loss = torch.tensor(0.0, device=embeddings.device)

    # NaN/Inf check
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
