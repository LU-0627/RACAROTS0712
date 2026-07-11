# RDCAROTS: Regime- and Delay-aware CAROTS

## Quick Start on Offline Linux Server

### Prerequisites
- Linux server (no internet required)
- Python 3.10
- CUDA 11.7+ (optional, for GPU)
- Extracted RDCAROTS_SERVER_PACKAGE

### Step 1: Setup Environment

```bash
cd RDCAROTS_SERVER_PACKAGE/project

# Set environment variables
export PROJECT_ROOT=$(pwd)
export DATA_ROOT=/path/to/your/data
export OUTPUT_ROOT=results/rd_carots
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python

# Check environment
bash scripts/server/00_check_environment.sh
```

### Step 2: Install Dependencies (Offline)

Choose one method:

**Option A: Use existing environment**
```bash
bash scripts/server/01_install_offline.sh existing-env
```

**Option B: Install from wheelhouse**
```bash
bash scripts/server/01_install_offline.sh wheelhouse
```

**Option C: Extract conda-packed environment**
```bash
bash scripts/server/01_install_offline.sh conda-pack
```

### Step 3: Verify Installation

```bash
# Check Python packages
bash scripts/server/03_run_compile_check.sh

# Run unit tests
bash scripts/server/04_run_tests.sh
```

### Step 4: Check Data

```bash
bash scripts/server/02_check_data.sh
```

Expected directory structure:
```
$DATA_ROOT/
├── SWaT/
│   ├── SWaT_Dataset_Normal_v1.csv
│   └── SWaT_Dataset_Attack_v0.csv
├── WADI/
│   └── WADI.A2_19 Nov 2019/
│       ├── WADI_14days_new.csv
│       └── WADI_attackdataLABLE.csv
└── synthetic_regime_delay/  # Will be generated
```

### Step 5: Generate Synthetic Data

```bash
bash scripts/server/05_generate_synthetic.sh
```

### Step 6: Run Complete Test Suite

**Full run:**
```bash
bash scripts/server/RUN_ALL_SERVER.sh
```

**Smoke test only:**
```bash
bash scripts/server/RUN_ALL_SERVER.sh --smoke-only
```

**Skip datasets:**
```bash
bash scripts/server/RUN_ALL_SERVER.sh --skip-swat --skip-wadi
```

## What Gets Tested

1. ✓ Environment validation
2. ✓ Python syntax check
3. ✓ Unit tests (20+ tests)
4. ✓ Synthetic data generation
5. ⏳ CAROTS baseline (pending full implementation)
6. ⏳ RDCAROTS training (pending full implementation)
7. ⏳ Multi-seed experiments (pending full implementation)
8. ⏳ Ablation studies (pending full implementation)
9. ⏳ Results collection (pending full implementation)

## Expected Outputs

Results will be saved to:
```
$OUTPUT_ROOT/
├── synthetic/
├── swat/
├── wadi/
├── ablations/
├── checkpoints/
├── results_summary.csv
└── results_summary.md
```

## Troubleshooting

### Import errors
```bash
# Check if tensorly is installed
python -c "import tensorly; print(tensorly.__version__)"

# If missing, check offline/wheels/
ls offline/wheels/tensorly*.whl
```

### CUDA not available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU instead
export CUDA_VISIBLE_DEVICES=""
```

### Data not found
- Verify DATA_ROOT points to correct directory
- Check file permissions
- Validate CSV file names match expected patterns

## Notes

- All scripts are designed for **offline execution**
- No network access required
- All dependencies must be pre-downloaded
- Test labels are never used for threshold tuning or online updates
