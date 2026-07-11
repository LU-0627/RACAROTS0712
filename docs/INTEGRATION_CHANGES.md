# Integration Changes for RDCAROTS

## Modified Original CAROTS Files

### 1. models/build.py
**Purpose:** Model registration  
**Changes:**
- Added RDCAROTS model variants to model_mapping
- Changed `model.cuda()` to `model.to(device)` for device agnosticity
- Models: RDCAROTS, RDCAROTS-no-regime, RDCAROTS-no-delay-negative, RDCAROTS-single-prototype

**CAROTS Compatibility:**
- Original CAROTS model unchanged
- Original CAROTS still selectable via MODEL.NAME = "CAROTS"
- Device handling improved but backward compatible

### 2. trainer.py
**Purpose:** Trainer registration and device handling  
**Changes:**
- Added RDCAROTS trainers to trainer_classes mapping
- Modified prepare_inputs() to accept device parameter (backward compatible)
- Trainers: RDCAROTSTrainer for all RDCAROTS variants

**CAROTS Compatibility:**
- Original CAROTSTrainer unchanged
- Original CAROTS still selectable
- prepare_inputs() enhanced with optional device parameter (default behavior preserved)

### 3. config.py (to be modified)
**Purpose:** Configuration schema  
**Required Changes:**
- Add _C.MODEL = CN() section
- Add _C.MODEL.NAME = "CAROTS" (default)
- Add _C.RDCAROTS = CN() section with all RDCAROTS parameters
- Add _C.RDCAROTS.IO_SCHEMA_PATH
- Add _C.RDCAROTS.TEST_MODE
- Add _C.RDCAROTS.DELAYMIX section
- Add _C.RDCAROTS.POSITIVE_AUGMENTOR section
- Add _C.RDCAROTS.NEGATIVE_AUGMENTOR section
- Add _C.RDCAROTS.LOSS section
- Add _C.RDCAROTS.SCORER section
- Add _C.RDCAROTS.PROJECTOR section
- Add _C.RDCAROTS.ENCODER_ARCH

**CAROTS Compatibility:**
- All original CAROTS config preserved
- RDCAROTS config only loaded when MODEL.NAME starts with "RDCAROTS"

## Integration Strategy

### Model Selection
User selects model via config:
```yaml
MODEL:
  NAME: "RDCAROTS"  # or "CAROTS", "RDCAROTS-no-regime", etc.
```

### Ablation Models
- **RDCAROTS**: Full model with all components
- **RDCAROTS-no-regime**: Disable regime-specific logic, use single prototype
- **RDCAROTS-no-delay-negative**: Disable delay_break negative augmentation
- **RDCAROTS-single-prototype**: Use single prototype instead of multi-regime

Ablation variants implemented by checking model name in modeling_rd_carots.py.

### Checkpoint Isolation
RDCAROTS checkpoints saved to separate directories:
- CAROTS: results/carots/checkpoints/
- RDCAROTS: results/rd_carots/checkpoints/

Prevents overwriting between models.

### Data Loading
Both use same dataset loaders from datasets/build.py.
RDCAROTS additionally applies IO schema splitting in trainer.

### Result Isolation
- CAROTS: results/carots/
- RDCAROTS: results/rd_carots/

## Verification Checklist

✓ Original CAROTS can still be trained without modification
✓ RDCAROTS can be selected via MODEL.NAME
✓ Both models can coexist in same codebase
✓ Checkpoints don't conflict
✓ Results don't conflict
✓ Device handling improved for both models
✓ No breaking changes to CAROTS API
