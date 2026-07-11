# Delay Score Implementation for RDCAROTS

## Overview

The A_delay component measures consistency between observed input-output relationships and expected normal regime delay patterns.

## Implementation Strategy

### 1. Local Markov Parameter Estimation

For each test window (u[t], y[t]):
1. Extract recent history: u[t-L:t], y[t-L:t]
2. Estimate local Markov parameters H_local via least-squares:
   y[t] ≈ Σ_τ H_local[τ] @ u[t-τ]

### 2. Comparison with Normal Regime Markov

For each regime r:
1. Get stored normal Markov parameters H_r[τ]
2. Compute weighted residual:
   e_r = Σ_τ w[τ] * ||H_local[τ] - H_r[τ]||_F
   where w[τ] emphasizes the effective delay

### 3. Minimum Delay Discrepancy

A_delay = min_r (e_r * (1 - p(r|x)))

Combines:
- Markov parameter mismatch
- Regime confidence weighting

## Current Implementation

Located in: `models/rd_carots/scorer_rd_carots.py`

Line 234 placeholder:
```python
score_dict['delay'] = torch.zeros_like(score_pred)
```

## Full Implementation (to be added)

```python
def compute_delay_score(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    regime_models: List,
    window_length: int = 20
) -> torch.Tensor:
    \"\"\"
    Compute delay consistency score.
    
    Args:
        outputs: (B, T, n_out)
        inputs: (B, T, n_in)
        regime_models: List of regime models with Markov parameters
        window_length: History length for local estimation
        
    Returns:
        delay_scores: (B,)
    \"\"\"
    B, T, n_out = outputs.shape
    n_in = inputs.shape[2]
    
    delay_scores = torch.zeros(B, device=outputs.device)
    
    for b in range(B):
        # Local Markov estimation (simplified)
        y_local = outputs[b].cpu().numpy()
        u_local = inputs[b].cpu().numpy()
        
        # For each lag
        residuals = []
        for r_model in regime_models:
            H_r = r_model.markov_params.markov_sequence  # (L, n_out, n_in)
            
            # Predict using stored Markov
            y_pred = np.zeros_like(y_local)
            for tau in range(min(H_r.shape[0], T)):
                if tau < T:
                    y_pred[tau:] += H_r[tau] @ u_local[:-tau if tau > 0 else None].T
                    
            # Residual
            residual = np.linalg.norm(y_local - y_pred)
            residuals.append(residual)
            
        delay_scores[b] = min(residuals)
        
    return delay_scores
```

## Limitations

- Current: Placeholder zeros
- Full implementation requires stable Markov parameter storage
- Needs careful handling of:
  - Insufficient history at boundaries
  - Ill-conditioned local estimation
  - Regime switching during window

## Integration Point

In `_compute_raw_scores()` at line ~230, replace:
```python
score_dict['delay'] = torch.zeros_like(score_pred)
```

With:
```python
if self.cfg.RDCAROTS.SCORER.LAMBDA_DELAY > 0:
    score_dict['delay'] = compute_delay_score(
        outputs, inputs, 
        self.model_bank.regime_models,
        window_length=self.cfg.RDCAROTS.DELAYMIX.MAX_LAG
    )
else:
    score_dict['delay'] = torch.zeros_like(score_pred)
```
