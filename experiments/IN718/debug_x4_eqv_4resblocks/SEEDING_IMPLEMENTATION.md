# Seeding Implementation Summary

## Changes Made

### 1. Created Seeding Utilities (`training/seed_utils.py`)

**Purpose**: Centralized seeding functions to ensure reproducibility

**Functions**:
- `set_seed(seed: int)`: Sets all random seeds (Python, NumPy, PyTorch CPU/CUDA, CuDNN)
- `get_seed_from_config(cfg, default=42)`: Extracts seed from configuration

**Features**:
- Seeds all RNG sources in one call
- Enables CuDNN deterministic mode for GPU reproducibility
- Provides detailed logging of what was seeded

### 2. Updated Configuration System (`training/config_utils.py`)

**Added to DEFAULT_CONFIG**:
```json
"seed": 42  // Random seed for reproducibility
```

This ensures every experiment has a seed, even if not explicitly specified.

### 3. Updated Data Loading (`training/data_loading.py`)

**Changes**:
- Added `seed` parameter to `build_dataloader()` function
- Seed is used for:
  - DataLoader's generator for shuffling
  - DistributedSampler for DDP training
  - Worker initialization function

**Before**:
```python
g.manual_seed(42)  # Hardcoded
```

**After**:
```python
g.manual_seed(seed)  # Configurable
```

### 4. Updated Single-GPU Training (`training/train_sr.py`)

**Changes**:
```python
# Import seeding utilities
from training.seed_utils import set_seed, get_seed_from_config

# In main():
seed = get_seed_from_config(cfg)
set_seed(seed)

# Pass seed to dataloaders
build_dataloader(..., seed=seed)
```

### 5. Updated DDP Training (`training/train_sr_ddp.py`)

**Changes**:
```python
# Import seeding utilities
from training.seed_utils import set_seed, get_seed_from_config

# In train_worker():
seed = get_seed_from_config(cfg)
set_seed(seed + rank)  # Each GPU gets offset seed

# Pass seed to dataloaders
build_dataloader(..., seed=seed)
```

**Note**: Each DDP process uses `seed + rank` to ensure:
- Different data distribution per GPU
- Reproducible behavior when using same GPU count
- Deterministic but diverse behavior across GPUs

### 6. Updated Experiment Config

**Added seed to config.json**:
```json
{
  "seed": 42,
  ...
}
```

## Usage

### Setting Seed in Config

```json
{
  "dataset_root": "...",
  "epochs": 2000,
  "batch_size": 16,
  "seed": 42,  // <-- Add this line
  ...
}
```

### Running Reproducible Training

```bash
# Single-GPU training
python training/train_sr.py --exp_dir experiments/IN718/debug_x4_eqv_4resblocks/

# Multi-GPU DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/
```

### Testing Reproducibility

Run the test script:
```bash
python scripts/test_seeding.py
```

## What's Guaranteed

✅ **Same Configuration → Same Results**
- Same model initialization
- Same data order
- Same optimization trajectory
- Same final model

✅ **Different Seeds → Different Results**
- Useful for ensemble methods
- Helps verify model stability

✅ **Full Control**
- Seed specified in config file
- Easy to change for experiments
- Logged in run_config.json

## What's Seeded

1. **Python's `random` module**
   - Used for: Any Python-level randomness
   
2. **NumPy's random generator**
   - Used for: Data preprocessing, augmentation
   
3. **PyTorch CPU operations**
   - Used for: Model initialization, CPU tensors
   
4. **PyTorch CUDA operations**
   - Used for: GPU computations, dropout, etc.
   
5. **CuDNN backend**
   - Deterministic mode enabled
   - May reduce performance ~10-20%

6. **DataLoader operations**
   - Shuffling order
   - Worker processes
   - Distributed sampling (DDP)

## Files Modified

```
training/
├── seed_utils.py              [NEW] Core seeding utilities
├── config_utils.py            [MODIFIED] Added seed to defaults
├── data_loading.py            [MODIFIED] Accept seed parameter
├── train_sr.py                [MODIFIED] Use seeding at startup
└── train_sr_ddp.py            [MODIFIED] Use seeding per-process

experiments/IN718/debug_x4_eqv_4resblocks/
├── config.json                [MODIFIED] Added seed field
└── REPRODUCIBILITY.md         [NEW] Detailed documentation

scripts/
└── test_seeding.py            [NEW] Test reproducibility
```

## Verification

To verify your setup is reproducible:

1. Run training twice with same config → Should get identical results
2. Change seed in config → Should get different results
3. Run test script → All tests should pass

## Notes

- **Performance Impact**: CuDNN determinism may slow training by 10-20%
- **Hardware**: Results may vary slightly across different GPU architectures
- **PyTorch Version**: Ensure same PyTorch/CUDA version for exact reproducibility
- **DDP**: Using different GPU counts will produce different results (but still reproducible for that count)

## Migration for Existing Experiments

To make existing experiments reproducible:

1. Add `"seed": 42` to your config.json
2. Re-run training from scratch (checkpoints don't store RNG state)
3. Results should now be reproducible across runs

## Future Improvements (Optional)

- [ ] Save/load RNG states in checkpoints for exact resume
- [ ] Add seed to TensorBoard logging
- [ ] Create convenience function for multi-seed experiments
- [ ] Add validation that checks reproducibility automatically
