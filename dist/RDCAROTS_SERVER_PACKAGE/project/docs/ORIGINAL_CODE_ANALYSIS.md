# CAROTS Original Code Analysis

## Overview
This document analyzes the original CAROTS codebase to understand its architecture, key components, and identify integration points for RDCAROTS.

## Key Components

### 1. Model Architecture (models/carots/modeling_carots.py)
**CAROTS Class:**
- Encoder: Multiple architectures supported (LSTM, iTransformer, GATv2, TimesNet, GRU)
- Projector: MLP with BatchNorm and GELU activation (3 layers)
- Causal Discoverer: CUTS_Plus_Net for learning causal graphs
- Positive Augmentor: Generates causal-preserving augmentations
- Negative Augmentor: Generates causal-breaking augmentations

**Forward Pass Flow:**
```
Input x (B, T, N) 
  → Positive Augmentation (if enabled): x_pos using causal model
  → Negative Augmentation (if enabled): x_neg by disturbing causal relations
  → Concatenate: [x, x_pos, x_neg] 
  → Encoder: x_all → enc_out (B, hidden_dim)
  → Projector: enc_out → z (B, output_dim)
  → Output: z for contrastive loss
```

**Tensor Shapes:**
- Input: (B, WIN_SIZE, N_VAR)
- After augmentation: (2B or 3B, WIN_SIZE, N_VAR) depending on augmentation flags
- Encoder output: (B_total, HIDDEN_DIM) where HIDDEN_DIM=512
- Projector output: (B_total, OUTPUT_DIM) where OUTPUT_DIM=512

### 2. Training Pipeline (models/carots/trainer_carots.py)
**CAROTSTrainer Class:**
- Inherits from base Trainer
- Two-stage training:
  1. Train/load causal discoverer (CUTS_Plus)
  2. Train encoder+projector with frozen causal discoverer
- Similarity threshold scheduling: linear interpolation from START to END over epochs
- Checkpoint management: saves best model based on validation metric

**Key Methods:**
- `train_causal_discoverer()`: Trains CUTS_Plus_Net
- `load_causal_discoverer()`: Loads pretrained causal graph
- `train_epoch()`: Processes batches, computes loss, updates parameters
- `eval_epoch()`: Validation without gradient updates

### 3. Loss Function (models/carots/loss.py)
**Similarity-filtered One-class Contrastive Loss:**
```python
# Normalize embeddings: z = F.normalize(z, p=2, dim=1)
# Compute similarity matrix: S = z @ z.T
# Filter by threshold: mask out pairs with sim < threshold
# InfoNCE-style loss with filtered negatives
```

**Key Mechanism:**
- Only high-similarity positive pairs (sim >= threshold) contribute to loss
- Low-similarity pairs are masked with -inf before exp()
- Threshold gradually increases during training (0.5 → 0.9)

### 4. Augmentors

**PositiveAugmentor (models/carots/modeling_positive_augmentor.py):**
- Randomly selects a variable (start_node)
- Adds Gaussian noise to that variable at INPUT_STEP
- Uses causal discoverer to propagate effect to causally-related variables
- Preserves causal dynamics

**NegativeAugmentor (models/carots/modeling_negative_augmentor.py):**
- Selects subgraph via DFS with cutoff probability
- Applies transforms (AddBias) to disturb causal relations
- Breaks normal dynamics

### 5. Scoring (models/carots/scorer_carots.py)
**Scorer Class:**
- Computes centroid from training data embeddings
- Three scoring modes:
  - `l2`: Euclidean distance to centroid
  - `cos`: Cosine distance to centroid
  - `causal_discoverer`: Prediction error from CUTS_Plus

**Centroid Initialization:**
- Pass all training data through model
- Compute mean of embeddings (normalized and unnormalized versions)
- Store for test-time scoring

### 6. Data Loading (datasets/build.py)
**TSDataset Base Class:**
- Window-based data loading
- Normalization: StandardScaler, MinMaxScaler, or none
- Train/val/test split
- Downsampling support

**Dataset Loaders:**
- SWaTSegLoader: 51 variables, normal training data
- WADISegLoader: Variable count depends on preprocessing
- Synthetic: Lorenz96SegLoader, VARSegLoader

**Key Properties:**
- Window size: configurable (default 10)
- Step: configurable (train_step, test_step)
- Returns: (window_data, binary_label) where label=1 if any anomaly in window

### 7. Configuration System (config.py)
**YACS-based configuration:**
- Hierarchical config nodes (CN)
- Dataset config: DATA (name, path, window size, normalization)
- Model config: CAROTS, LSTM, ITRANSFORMER, GATV2, TIMESNET
- Training config: TRAIN, VAL, TEST, SOLVER
- Causal discoverer: CUTS_PLUS

**Critical Issues Found:**
1. **Hardcoded CUDA:**
   - Line 50 (modeling_carots.py): `.cuda()`
   - Line 56 (modeling_carots.py): `.cuda()`
   - Line 62 (modeling_carots.py): `.cuda()`
   - Line 23 (modeling_positive_augmentor.py): `.cuda()`
   - Line 26 (trainer_carots.py): `.cuda()`
   - Line 37 (trainer_carots.py): `.cuda()`

2. **Fixed Tensor Dimensions:**
   - N_VAR hardcoded in config (51 for SWaT)
   - Must be configured per dataset

3. **No Input/Output Variable Distinction:**
   - All variables treated uniformly
   - No notion of control inputs vs. sensor outputs

4. **Single Normal Prototype:**
   - Only one centroid for all normal data
   - No regime awareness

5. **No Online Update Mechanism:**
   - Test phase is purely inference
   - Cannot adapt to regime changes

## Integration Points for RDCAROTS

### 1. Model Extension
**Location:** `models/rd_carots/modeling_rd_carots.py`
- Inherit from or parallel to CAROTS
- Add RegimeDelayModelBank
- Replace single centroid with RegimePrototypeBank

### 2. Augmentor Enhancement
**Location:** `models/rd_carots/augmentors.py`
- Extend PositiveAugmentor → RegimeDelayPositiveAugmentor
- Extend NegativeAugmentor → RegimeDelayNegativeAugmentor
- Add delay_break and cross_regime_mismatch modes

### 3. Loss Function Enhancement
**Location:** `models/rd_carots/loss_rd_carots.py`
- Extend loss_fn → regime_conditioned_soc_loss
- Add soft regime weighting: w_ij = Σ_r p(r|x_i)p(r|x_j)

### 4. Scorer Enhancement
**Location:** `models/rd_carots/scorer_rd_carots.py`
- Replace single centroid with multi-regime prototypes
- Add A_pred (prediction error from DelayMix models)
- Add A_delay (Markov parameter consistency)
- Add A_uncertainty (regime confidence)

### 5. Trainer Enhancement
**Location:** `models/rd_carots/trainer_rd_carots.py`
- Add model bank update logic
- Add guarded online update for test phase
- Save regime inference history

### 6. Data Schema
**Location:** `configs/io_schema/`
- Create YAML files specifying input/output variable indices
- Support explicit_io, metadata_io, pseudo_io modes

## Checkpoint Structure
Original CAROTS checkpoint contains:
```python
{
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': current_epoch,
    # ... other training metadata
}
```

RDCAROTS must extend to include:
```python
{
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': current_epoch,
    'model_bank': {
        'regime_models': [...],
        'markov_params': [...],
        'delays': [...],
        'statistics': [...]
    },
    'prototype_bank': {
        'prototypes': [...],
        'regime_stats': [...]
    },
    'moment_collection_state': {...},
    'score_normalizers': {...}
}
```

## Device Compatibility Plan
Replace all `.cuda()` calls with:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

Use `prepare_inputs()` helper to move data to appropriate device.

## Path Compatibility Plan
Replace all hardcoded paths with:
```python
from pathlib import Path
data_path = Path(cfg.DATA.BASE_DIR) / cfg.DATA.NAME
```

## Variable Dimension Flexibility
Current code assumes fixed N_VAR at initialization. RDCAROTS must:
- Infer N_VAR from data if not specified
- Support variable input/output dimension split
- Handle datasets with different channel counts

## Summary
The original CAROTS provides a solid foundation with:
- Flexible encoder architectures
- Causal graph learning via CUTS_Plus
- Causal-aware augmentation
- Similarity-filtered contrastive learning

RDCAROTS will extend this by adding:
- Multi-regime state space models (DelayMix)
- Input/output variable distinction
- Delay-aware augmentation
- Multi-prototype representation
- Online adaptation capability
- Device and path agnostic code

All original CAROTS functionality will be preserved to allow direct comparison.
