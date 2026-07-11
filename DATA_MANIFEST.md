# Data Manifest for RDCAROTS Server Deployment

## Expected Data Directory Structure

```
$DATA_ROOT/
├── SWaT/
│   ├── SWaT_Dataset_Normal_v1.csv
│   └── SWaT_Dataset_Attack_v0.csv
├── WADI/
│   └── WADI.A2_19 Nov 2019/
│       ├── WADI_14days_new.csv
│       └── WADI_attackdataLABLE.csv
└── synthetic_regime_delay/  (generated)
    ├── train.npz
    ├── val.npz
    ├── test.npz
    ├── metadata.json
    └── system_parameters.json
```

## Dataset Details

### Synthetic Regime-Delay Dataset

**Status:** Generated on server  
**Command:** `bash scripts/server/05_generate_synthetic.sh`

**Contents:**
- 3 normal regimes with different dynamics
- Variable input delays per regime
- Train/val/test splits
- Multiple anomaly types

**Variables:**
- Inputs: 20
- Outputs: 30
- Total: 50

### SWaT (Secure Water Treatment)

**Status:** Must be provided by user  
**Source:** iTrust dataset  
**Size:** ~500K rows

**Expected columns:** 51 variables
- Mix of actuators (pumps, valves) and sensors (flow, level, pressure)

**IO Schema:** `configs/io_schema/SWaT.yaml` (TEMPLATE - verify columns)

**Notes:**
- Schema is a template and must be validated against actual CSV column names
- Run `python scripts/check_dataset_schema.py --dataset SWaT` to validate

### WADI (Water Distribution)

**Status:** Must be provided by user  
**Source:** iTrust dataset  
**Size:** ~1.2M rows

**Expected columns:** ~127 variables  
- Mix of actuators and sensors across water distribution network

**IO Schema:** `configs/io_schema/WADI.yaml` (TEMPLATE - verify columns)

**Notes:**
- Schema is a template and must be validated against actual CSV column names
- Variable count depends on preprocessing
- Run `python scripts/check_dataset_schema.py --dataset WADI` to validate

## Data Validation

### Before Running Experiments

1. **Check data availability:**
   ```bash
   bash scripts/server/02_check_data.sh
   ```

2. **Validate SWaT schema (if data available):**
   ```bash
   python scripts/check_dataset_schema.py \
       --dataset SWaT \
       --data-path $DATA_ROOT/SWaT/SWaT_Dataset_Normal_v1.csv \
       --schema-path configs/io_schema/SWaT.yaml
   ```

3. **Validate WADI schema (if data available):**
   ```bash
   python scripts/check_dataset_schema.py \
       --dataset WADI \
       --data-path $DATA_ROOT/WADI/.../WADI_14days_new.csv \
       --schema-path configs/io_schema/WADI.yaml
   ```

### Schema Mismatch Handling

If column names don't match IO schema:
1. Script will print actual column names
2. Update corresponding `configs/io_schema/*.yaml`
3. Classify columns as inputs (actuators) vs outputs (sensors)
4. Re-run validation

## Missing Data Handling

### If SWaT not available:
```bash
bash scripts/server/RUN_ALL_SERVER.sh --skip-swat
```

### If WADI not available:
```bash
bash scripts/server/RUN_ALL_SERVER.sh --skip-wadi
```

### Synthetic only:
```bash
bash scripts/server/05_generate_synthetic.sh
bash scripts/server/06_run_synthetic_smoke.sh
bash scripts/server/07_run_synthetic_full.sh
```

## Data Licensing

**Important:** SWaT and WADI are licensed datasets from iTrust.
- Cannot be redistributed in this package
- Users must obtain separately from iTrust
- Follow iTrust terms of use

**Synthetic data:** Generated on-the-fly, no license restrictions.
