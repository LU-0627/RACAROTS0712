"""
RDCAROTS Scorer: Multi-Component Anomaly Scoring

Combines four scoring components:
1. A_embed: Distance to nearest regime prototype
2. A_pred: Minimum prediction error across regime models
3. A_delay: Markov parameter consistency
4. A_uncertainty: Regime posterior uncertainty
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import copy

from .prototypes import RegimePrototypeBank
from .delaymix import RegimeDelayModelBank


class RDCAROTSScorer:
    """
    Multi-component anomaly scorer for RDCAROTS.

    Args:
        cfg: Configuration
        model: RDCAROTS model
        prototype_bank: Regime prototype bank
        model_bank: Regime delay model bank
    """

    def __init__(
        self,
        cfg,
        model,
        prototype_bank: RegimePrototypeBank,
        model_bank: Optional[RegimeDelayModelBank] = None
    ):
        self.cfg = cfg
        self.model = model
        self.prototype_bank = prototype_bank
        self.model_bank = model_bank

        # Score weights
        cfg_scorer = cfg.RDCAROTS.SCORER
        self.lambda_embed = cfg_scorer.LAMBDA_EMBED
        self.lambda_pred = cfg_scorer.LAMBDA_PRED if model_bank is not None else 0.0
        self.lambda_delay = cfg_scorer.LAMBDA_DELAY if model_bank is not None else 0.0
        self.lambda_uncertainty = cfg_scorer.LAMBDA_UNCERTAINTY

        # Normalization statistics (fitted on training data)
        self.score_normalizers = {
            'embed': {'mean': 0.0, 'std': 1.0},
            'pred': {'mean': 0.0, 'std': 1.0},
            'delay': {'mean': 0.0, 'std': 1.0},
            'uncertainty': {'mean': 0.0, 'std': 1.0}
        }

        self.is_fitted = False

    def fit_normalizers(self, loader, device: torch.device):
        """
        Fit score normalizers on training data.

        Args:
            loader: Training data loader
            device: Device to use
        """
        print("Fitting score normalizers on training data...")

        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        scores_embed = []
        scores_pred = []
        scores_delay = []
        scores_uncertainty = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                # Get scores
                score_dict = self._compute_raw_scores(inputs)

                scores_embed.append(score_dict['embed'].cpu())
                if 'pred' in score_dict:
                    scores_pred.append(score_dict['pred'].cpu())
                if 'delay' in score_dict:
                    scores_delay.append(score_dict['delay'].cpu())
                if 'uncertainty' in score_dict:
                    scores_uncertainty.append(score_dict['uncertainty'].cpu())

        # Concatenate
        scores_embed = torch.cat(scores_embed, dim=0).numpy()

        # Fit normalizers (robust: use median and IQR)
        self.score_normalizers['embed'] = {
            'mean': float(np.median(scores_embed)),
            'std': float(np.percentile(scores_embed, 75) - np.percentile(scores_embed, 25) + 1e-8)
        }

        if len(scores_pred) > 0:
            scores_pred = torch.cat(scores_pred, dim=0).numpy()
            self.score_normalizers['pred'] = {
                'mean': float(np.median(scores_pred)),
                'std': float(np.percentile(scores_pred, 75) - np.percentile(scores_pred, 25) + 1e-8)
            }

        if len(scores_delay) > 0:
            scores_delay = torch.cat(scores_delay, dim=0).numpy()
            self.score_normalizers['delay'] = {
                'mean': float(np.median(scores_delay)),
                'std': float(np.percentile(scores_delay, 75) - np.percentile(scores_delay, 25) + 1e-8)
            }

        if len(scores_uncertainty) > 0:
            scores_uncertainty = torch.cat(scores_uncertainty, dim=0).numpy()
            self.score_normalizers['uncertainty'] = {
                'mean': float(np.median(scores_uncertainty)),
                'std': float(np.percentile(scores_uncertainty, 75) - np.percentile(scores_uncertainty, 25) + 1e-8)
            }

        self.is_fitted = True

        if is_training:
            self.model.train()

        print("Score normalizers fitted.")

    @torch.no_grad()
    def get_anomaly_scores(self, x: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        """
        Compute anomaly scores.

        Args:
            x: Input windows (batch_size, window_size, n_variables)
            return_components: If True, return dict with all components

        Returns:
            scores: (batch_size,) anomaly scores or dict if return_components=True
        """
        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        # Get raw scores
        score_dict = self._compute_raw_scores(x)

        # Normalize
        score_embed_norm = self._normalize_score(score_dict['embed'], 'embed')

        if self.lambda_pred > 0 and 'pred' in score_dict:
            score_pred_norm = self._normalize_score(score_dict['pred'], 'pred')
        else:
            score_pred_norm = torch.zeros_like(score_embed_norm)

        if self.lambda_delay > 0 and 'delay' in score_dict:
            score_delay_norm = self._normalize_score(score_dict['delay'], 'delay')
        else:
            score_delay_norm = torch.zeros_like(score_embed_norm)

        if self.lambda_uncertainty > 0 and 'uncertainty' in score_dict:
            score_uncertainty_norm = self._normalize_score(score_dict['uncertainty'], 'uncertainty')
        else:
            score_uncertainty_norm = torch.zeros_like(score_embed_norm)

        # Weighted combination
        score_final = (
            self.lambda_embed * score_embed_norm +
            self.lambda_pred * score_pred_norm +
            self.lambda_delay * score_delay_norm +
            self.lambda_uncertainty * score_uncertainty_norm
        )

        if is_training:
            self.model.train()

        if return_components:
            return {
                'score_embed': score_dict['embed'],
                'score_pred': score_dict.get('pred', torch.zeros_like(score_dict['embed'])),
                'score_delay': score_dict.get('delay', torch.zeros_like(score_dict['embed'])),
                'score_uncertainty': score_dict.get('uncertainty', torch.zeros_like(score_dict['embed'])),
                'score_final': score_final,
                'regime_probs': score_dict.get('regime_probs'),
                'best_regime': score_dict.get('best_regime')
            }
        else:
            return score_final

    def _compute_raw_scores(self, x: torch.Tensor) -> Dict:
        """Compute raw (unnormalized) score components."""
        # Get embeddings
        embeddings = self.model(x, positive_augment=False, negative_augment=False)

        # A_embed: Distance to nearest prototype
        distances = self.prototype_bank.get_distances(embeddings, normalize=True)
        score_embed, best_regime = torch.min(distances, dim=1)

        score_dict = {
            'embed': score_embed,
            'best_regime': best_regime
        }

        # A_pred, A_delay, A_uncertainty: Require model bank
        if self.model_bank is not None and self.model_bank.is_initialized:
            # Get IO schema
            from .io_schema import load_io_schema, split_io_variables
            from pathlib import Path
            io_schema_path = Path(self.cfg.RDCAROTS.IO_SCHEMA_PATH)
            io_schema = load_io_schema(io_schema_path, n_variables=x.shape[-1])

            # Split using IO schema
            inputs_np, outputs_np = split_io_variables(x.cpu().numpy(), io_schema)
            inputs = torch.from_numpy(inputs_np).to(x.device)
            outputs = torch.from_numpy(outputs_np).to(x.device)

            # Regime inference
            regime_results = self.model_bank.predict(outputs, inputs)

            if regime_results['regime_probs'] is not None:
                score_dict['regime_probs'] = regime_results['regime_probs']

                # A_pred: Minimum prediction error
                score_pred = torch.min(regime_results['prediction_errors'], dim=1)[0]
                score_dict['pred'] = score_pred

                # A_uncertainty: Entropy of regime distribution
                regime_probs = regime_results['regime_probs']
                score_uncertainty = -(regime_probs * torch.log(regime_probs + 1e-12)).sum(dim=1)
                score_dict['uncertainty'] = score_uncertainty

                # A_delay: Markov parameter-based delay score
                score_delay = self._compute_delay_score(
                    inputs, outputs, regime_results, io_schema
                )
                score_dict['delay'] = score_delay

        return score_dict

    def _normalize_score(self, score: torch.Tensor, component: str) -> torch.Tensor:
        """Normalize score component."""
        if not self.is_fitted:
            return score

        norm_params = self.score_normalizers[component]
        return (score - norm_params['mean']) / (norm_params['std'] + 1e-8)

    def state_dict(self) -> Dict:
        """Save scorer state."""
        return {
            'score_normalizers': self.score_normalizers,
            'is_fitted': self.is_fitted
        }

    def _compute_delay_score(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        regime_results: Dict,
        io_schema
    ) -> torch.Tensor:
        """
        Compute delay-aware anomaly score.

        Combines:
        1. Delay position error: difference in effective delay
        2. Markov response error: mismatch in lagged response patterns

        Args:
            inputs: (batch_size, window_size, n_inputs)
            outputs: (batch_size, window_size, n_outputs)
            regime_results: Results from model_bank.predict
            io_schema: IOSchema object

        Returns:
            delay_scores: (batch_size,) delay anomaly scores
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        regime_probs = regime_results['regime_probs']
        best_regimes = regime_results['best_regime']

        delay_scores = torch.zeros(batch_size, device=device)

        # Get regime models
        regime_models = self.model_bank.regime_models

        for b in range(batch_size):
            regime_id = best_regimes[b].item()
            regime_model = regime_models[regime_id]

            if regime_model is None:
                continue

            # Get expected delay for this regime
            expected_delay = regime_model.markov_params.effective_delay
            markov_seq = regime_model.markov_params.markov_sequence  # (L, n_outputs, n_inputs)

            # Extract window data
            u_window = inputs[b].cpu().numpy()  # (T, n_inputs)
            y_window = outputs[b].cpu().numpy()  # (T, n_outputs)
            T = u_window.shape[0]

            # Compute Markov-based response error
            markov_error = 0.0
            n_valid = 0

            for lag in range(min(expected_delay + 3, len(markov_seq))):
                if lag >= T:
                    break

                # Expected response at this lag
                H_lag = markov_seq[lag]  # (n_outputs, n_inputs)

                # Compute expected vs actual response
                for t in range(lag, T):
                    if t - lag >= 0:
                        u_lagged = u_window[t - lag]  # (n_inputs,)
                        y_expected = H_lag @ u_lagged  # (n_outputs,)
                        y_actual = y_window[t]  # (n_outputs,)

                        # Residual
                        residual = np.linalg.norm(y_actual - y_expected)
                        markov_error += residual
                        n_valid += 1

            if n_valid > 0:
                markov_error /= n_valid

            # Delay position error: check if responses align with expected delay
            delay_position_error = 0.0
            if expected_delay < T:
                H_main = markov_seq[expected_delay]  # (n_outputs, n_inputs)
                for t in range(expected_delay, T):
                    u_delayed = u_window[t - expected_delay]
                    y_expected = H_main @ u_delayed
                    y_actual = y_window[t]
                    delay_position_error += np.linalg.norm(y_actual - y_expected)

                delay_position_error /= (T - expected_delay)

            # Combined delay score
            delay_scores[b] = markov_error + delay_position_error

        return delay_scores

    def load_state_dict(self, state_dict: Dict):
        """Load scorer state."""
        self.score_normalizers = state_dict['score_normalizers']
        self.is_fitted = state_dict['is_fitted']
