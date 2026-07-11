# Server Test Checklist

Use this checklist when deploying RDCAROTS on the server.

## Pre-Deployment

- [ ] ZIP file transferred to server
- [ ] SHA256 checksum verified
- [ ] Extraction successful
- [ ] File permissions OK

## Environment Setup

- [ ] PROJECT_ROOT set
- [ ] DATA_ROOT set
- [ ] OUTPUT_ROOT set
- [ ] PYTHON_BIN set
- [ ] CUDA_VISIBLE_DEVICES set

## Environment Validation

- [ ] Python 3.10 available
- [ ] All dependencies installed (or existing-env validated)
- [ ] PyTorch available
- [ ] CUDA available (if using GPU)
- [ ] tensorly installed

## Data Validation

- [ ] Synthetic data directory exists or can be generated
- [ ] SWaT data available (if running SWaT experiments)
- [ ] WADI data available (if running WADI experiments)
- [ ] IO schema YAMLs present

## Code Validation

- [ ] Python syntax check passes (compileall)
- [ ] All imports successful
- [ ] No Windows paths in code
- [ ] No hardcoded .cuda() calls

## Unit Tests

- [ ] test_imports.py passes
- [ ] test_config.py passes
- [ ] test_tensor_shapes.py passes
- [ ] test_cp_decomposition.py passes
- [ ] test_markov_recovery.py passes
- [ ] test_ho_kalman.py passes
- [ ] test_regime_inference.py passes
- [ ] test_prototype_bank.py passes
- [ ] test_loss_no_nan.py passes
- [ ] test_no_future_leakage.py passes
- [ ] test_checkpoint_roundtrip.py passes

## Synthetic Data

- [ ] Data generation completes
- [ ] train.npz created
- [ ] val.npz created
- [ ] test.npz created
- [ ] metadata.json created

## Smoke Tests

- [ ] CAROTS 1-epoch smoke test completes
- [ ] RDCAROTS 1-epoch smoke test completes
- [ ] Checkpoint saved
- [ ] Checkpoint loadable

## Full Experiments

- [ ] Synthetic CAROTS training completes
- [ ] Synthetic RDCAROTS training completes
- [ ] SWaT experiments complete (if data available)
- [ ] WADI experiments complete (if data available)

## Multi-Seed

- [ ] Seed 0 completes
- [ ] Seed 1 completes
- [ ] Seed 2 completes

## Ablations

- [ ] RDCAROTS-no-regime completes
- [ ] RDCAROTS-no-delay-negative completes
- [ ] RDCAROTS-single-prototype completes

## Results

- [ ] results_raw.csv generated
- [ ] results_summary.csv generated
- [ ] results_summary.md generated
- [ ] ablation_summary.csv generated
- [ ] No fabricated metrics

## Issues Encountered

Document any issues below:

```
(List issues here)
```

## Performance Notes

Document performance observations:

```
(Training time, memory usage, etc.)
```
