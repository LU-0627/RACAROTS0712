# DelayMix Implementation Notes

## Overview
This document describes the implementation of DelayMix-style multi-regime delayed system identification for RDCAROTS, based on the paper "Multi-Regime System Identification with Tensor Decomposition and Bayesian Inference".

## Core Concepts

### 1. System Tensor Construction
DelayMix represents a multi-regime delayed input-output system as a 3rd-order tensor.

**System Representation:**
```
x(t) = Σ_r Σ_τ A_r^τ x(t-τ) + B_r^τ u(t-τ) + noise
```

Where:
- x(t): output variables at time t (dimension: n_y)
- u(t): input variables at time t (dimension: n_u)
- r: regime index
- τ: time lag
- A_r^τ, B_r^τ: regime-specific Markov parameters

**Moment Tensor Construction:**
Given window size T and maximum lag L, construct tensor M ∈ R^(n_y × n_u × L):
```
M[i,j,τ] = E[x_i(t) * u_j(t-τ)]
```

For streaming data, use exponential moving average:
```
M_new = λ * M_old + (1-λ) * m_current
```

Where λ is forgetting factor (e.g., 0.99).

### 2. CP Decomposition
Apply Canonical Polyadic (CP) decomposition to extract regime components:
```
M ≈ Σ_r λ_r (a_r ⊗ b_r ⊗ c_r)
```

Where:
- R: number of regimes (typically 2-5)
- λ_r: weight of regime r
- a_r ∈ R^n_y: output factor
- b_r ∈ R^n_u: input factor
- c_r ∈ R^L: lag factor

**Implementation Choice:**
Use `tensorly.decomposition.parafac()` with:
- Rank R determined by cross-validation or AIC/BIC
- Initialization: 'svd' or 'random' with multiple restarts
- Max iterations: 100-200
- Convergence tolerance: 1e-6

### 3. Markov Parameter Recovery
From CP factors, reconstruct regime-specific Markov parameters:
```
H_r[τ] = λ_r * c_r[τ] * (a_r ⊗ b_r)
```

Reshape to matrix form:
```
H_r[τ] ∈ R^(n_y × n_u)
```

**Delay Identification:**
Find effective delay τ_r for regime r:
```
τ_r = argmin_τ { ||H_r[τ]||_F < threshold }
```

Interpretation: First lag where Markov parameter becomes significant.

### 4. State Space Realization
Use Ho-Kalman algorithm or ERA (Eigensystem Realization Algorithm) to convert Markov parameters to state space form:

**Input:** Markov sequence {H_r[0], H_r[1], ..., H_r[L-1]}

**Output:** State space model
```
z(t+1) = A_r z(t) + B_r u(t)
x(t) = C_r z(t)
```

**Ho-Kalman Steps:**
1. Construct Hankel matrix:
   ```
   H_0 = [H[0]  H[1]  ...  H[p]  ]
         [H[1]  H[2]  ...  H[p+1]]
         [  ⋮     ⋮    ⋱     ⋮   ]
         [H[q]  H[q+1] ... H[p+q]]
   ```

2. Compute SVD and select state dimension:
   ```
   H_0 = U Σ V^T
   Keep top n_s singular values
   ```

3. Extract state space matrices:
   ```
   A = Σ^{-1/2} U^T H_1 V Σ^{-1/2}
   B = Σ^{1/2} V^T e_1
   C = e_1^T U Σ^{1/2}
   ```

Where H_1 is H_0 shifted by one lag.

**Implementation Choice:**
- Use reduced Hankel matrix (p=q=10) for computational efficiency
- State dimension n_s selected by singular value threshold (e.g., 0.01 of max singular value)
- Validate stability: check eigenvalues of A are inside unit circle

### 5. Regime Inference
Estimate regime probability p(r|x,u) based on model fit:

**Prediction Error Method:**
```
ε_r(t) = ||x(t) - C_r z_r(t)||^2
p(r|data) ∝ exp(-β * ε_r(t))
```

Where β is inverse temperature parameter.

**Kalman Filter Likelihood:**
For each regime, run Kalman filter and compute log-likelihood:
```
log p(x_{1:T}|u_{1:T}, r) = Σ_t log N(x(t); C_r z_r(t|t-1), R_r)
```

**Soft Assignment:**
```
p(r|x,u) = exp(log_lik_r) / Σ_r' exp(log_lik_r')
```

**Hard Assignment:**
```
r* = argmax_r p(r|x,u)
```

**Low Confidence Detection:**
If max_r p(r|x,u) < threshold (e.g., 0.6), flag as uncertain → potential anomaly or new regime.

### 6. Update Triggering
CP decomposition is expensive, so don't run every batch.

**Trigger Conditions:**
1. **Fixed Interval:** Every N windows (e.g., N=500)
2. **Sample Accumulation:** After collecting M new windows (e.g., M=200)
3. **Model Mismatch:** When max_r ε_r(t) > threshold for K consecutive windows
4. **Drift Detection:** When moving average of residuals increases significantly

**Warm Start:**
Use previous CP factors as initialization for next decomposition.

**Memory Management:**
- Store only moment statistics, not raw data
- Moment tensor size: O(n_y * n_u * L) - independent of time
- Typical: 50 variables, lag 20 → 50*50*20 = 50K floats ≈ 200KB per regime

## Implementation Decisions

### What the Paper Doesn't Specify

1. **Rank Selection:**
   - Paper assumes rank R is known
   - **Our choice:** Use model selection criteria (AIC/BIC) or cross-validation on synthetic data
   - Default: R=3 regimes
   - Allow user configuration

2. **Forgetting Factor:**
   - Paper uses exponential weighting
   - **Our choice:** λ=0.99 for slow adaptation, λ=0.95 for faster adaptation
   - Configurable per dataset

3. **State Dimension Selection:**
   - Paper doesn't specify how to choose n_s
   - **Our choice:** Select n_s to capture 95% of singular value energy
   - Bound: 2 ≤ n_s ≤ min(n_y, n_u, 20)

4. **Initial Bootstrapping:**
   - Need sufficient data before first CP decomposition
   - **Our choice:** Collect at least 100 windows before first decomposition
   - Use simple VAR model as fallback until then

5. **Regime Switching Penalty:**
   - Avoid spurious regime switches
   - **Our choice:** Require regime probability > 0.7 for 3 consecutive windows before switching

6. **New Regime Creation:**
   - When all existing models fit poorly
   - **Our choice:** 
     - If max_r p(r|x,u) < 0.4 for 10 consecutive windows → create new regime
     - Maximum K_max regimes (e.g., 5)
     - Prune regimes with < 1% usage

7. **Numerical Stability:**
   - Hankel matrix can be ill-conditioned
   - **Our choice:** 
     - Add small regularization (1e-6) to diagonal
     - Clip singular values below 1e-8
     - Validate A eigenvalues are inside unit circle; if not, scale down

8. **Pseudo-IO Mode:**
   - When no explicit input variables
   - **Our choice:** 
     - Use lagged outputs as pseudo-inputs: u(t) = x(t-d) for some delay d
     - Clearly mark this mode in logs and configs

## Fallback Strategies

### Scenario 1: Insufficient Data
**Problem:** < 100 windows collected
**Fallback:** Simple VAR(p) model on outputs only
**Indicator:** Set regime_confidence = 0.0

### Scenario 2: CP Decomposition Fails
**Problem:** tensorly.parafac() doesn't converge or returns NaN
**Fallback:** 
- Retry with different initialization
- If still fails, use previous model bank
- Log warning

### Scenario 3: Unstable State Space Model
**Problem:** |eigenvalue(A_r)| > 1.0
**Fallback:** 
- Scale down A by factor γ such that max|eig(γA)| = 0.95
- Log warning about instability

### Scenario 4: All Models Misfit
**Problem:** All regime prediction errors are high
**Options:**
1. Create new regime if below max count
2. Use worst-case model for anomaly scoring
3. Flag as anomaly candidate

## Testing Strategy

### Unit Tests
1. **test_moment_collection.py:**
   - Test incremental moment updates
   - Test forgetting factor application
   - Test state_dict save/load

2. **test_cp_decomposition.py:**
   - Test on synthetic multi-regime data with known ground truth
   - Verify recovered regimes match true regimes (up to permutation)

3. **test_markov_recovery.py:**
   - Test delay identification
   - Test that first τ_r lags have near-zero response

4. **test_ho_kalman.py:**
   - Test on known SISO system
   - Verify reconstructed A,B,C reproduce original Markov parameters

5. **test_regime_inference.py:**
   - Test soft vs hard assignment
   - Test low-confidence detection

6. **test_update_trigger.py:**
   - Test trigger conditions
   - Test that memory doesn't grow unbounded

### Integration Test
**test_synthetic_end_to_end.py:**
- Generate 3-regime switching system
- Run full pipeline: moment collection → CP → Ho-Kalman → inference
- Verify regime recovery accuracy > 80%

## Configuration Schema

```yaml
rd_carots:
  delaymix:
    # Tensor construction
    max_lag: 20
    forgetting_factor: 0.99
    
    # CP decomposition
    n_regimes: 3  # or 'auto' for model selection
    cp_init: 'svd'  # or 'random'
    cp_max_iter: 200
    cp_tol: 1e-6
    cp_n_restarts: 3  # for random init
    
    # State space realization
    hankel_p: 10
    hankel_q: 10
    sv_threshold: 0.01  # or 'auto' for 95% energy
    state_dim_max: 20
    stability_margin: 0.95
    
    # Regime inference
    inverse_temperature: 1.0
    regime_switch_threshold: 0.7
    regime_switch_consecutive: 3
    low_confidence_threshold: 0.4
    
    # Update triggering
    bootstrap_windows: 100
    update_interval: 500
    update_sample_threshold: 200
    update_mismatch_threshold: 0.1
    update_mismatch_consecutive: 10
    
    # Regime management
    max_regimes: 5
    new_regime_threshold: 0.4
    new_regime_consecutive: 10
    prune_usage_threshold: 0.01
```

## Memory Complexity

**Per Regime:**
- Moment tensor: O(n_y * n_u * L)
- Markov params: O(n_y * n_u * L)
- State space (A,B,C): O(n_s^2 + n_s*n_u + n_y*n_s) ≈ O(n_s^2)
- Statistics: O(1)

**Total:** O(K * (n_y*n_u*L + n_s^2))

**Example:** K=3, n_y=30, n_u=20, L=20, n_s=10
- Moment: 3 * 30*20*20 = 36K floats
- Markov: 36K floats
- State space: 3 * (100 + 200 + 300) = 1.8K floats
- **Total: ~300KB** - very manageable

## Limitations and Caveats

1. **Linearity Assumption:** DelayMix assumes linear dynamics. Highly nonlinear regimes may not be captured well.

2. **Delay Identifiability:** Input delay is only observable if input variation is sufficient. Constant inputs → delay is unidentifiable.

3. **Regime Overlap:** If two regimes have very similar dynamics, CP may merge them.

4. **Computational Cost:** CP decomposition scales as O(iterations * rank * n_y * n_u * L). For large systems (>100 variables), consider dimension reduction.

5. **Non-stationary Systems:** If system parameters drift continuously, regime model may become a poor approximation.

## References
- Papalexakis et al., "Tensor Decompositions for Signal Processing Applications"
- Ho & Kalman, "Effective construction of linear state-variable models from input/output functions"
- DelayMix paper (2605.26191v1.pdf)
