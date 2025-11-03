# Reproducibility Guide

## Overview

This training setup ensures full reproducibility across experiments by controlling all sources of randomness.

## Seeding Implementation

### Global Seed Setting

All experiments use a consistent random seed that controls:

1. **Python's built-in random module**
2. **NumPy's random number generator**
3. **PyTorch's CPU operations**
4. **PyTorch's CUDA operations** (all GPUs)
5. **CuDNN backend** (deterministic mode enabled)

### Configuration

The seed is specified in `config.json`:

```json
{
  "seed": 42
}
```

If not specified, the default seed of `42` will be used.

### How It Works

#### Single-GPU Training (`train_sr.py`)

```python
# At the start of training
seed = get_seed_from_config(cfg)  # Read from config
set_seed(seed)                     # Set all random seeds
```

This ensures:
- Same model initialization
- Same data shuffling order
- Same optimizer initialization
- Same dropout masks (if using dropout)

#### Multi-GPU DDP Training (`train_sr_ddp.py`)

```python
# Each process gets a slightly offset seed
seed = get_seed_from_config(cfg)
set_seed(seed + rank)  # rank 0 uses seed, rank 1 uses seed+1, etc.
```

This ensures:
- Different but reproducible behavior per GPU
- Data is distributed differently across GPUs
- Results are still fully reproducible when using the same number of GPUs

### Data Loading

DataLoaders use the same seed for:
- Shuffling training data
- Worker process initialization
- DistributedSampler (for DDP)

```python
build_dataloader(
    ...,
    seed=seed  # Passed explicitly
)
```

## Verification

To verify reproducibility:

1. **Same Configuration → Same Results**
   ```bash
   # Run 1
   python training/train_sr.py --exp_dir experiments/IN718/debug_x4_eqv_4resblocks/
   
   # Run 2 (should give identical results)
   python training/train_sr.py --exp_dir experiments/IN718/debug_x4_eqv_4resblocks/
   ```

2. **Different Seeds → Different Results**
   ```json
   # config1.json
   {"seed": 42}
   
   # config2.json
   {"seed": 123}
   ```

3. **DDP Reproducibility**
   ```bash
   # Same seed + same GPU count = identical results
   CUDA_VISIBLE_DEVICES=0,1,2 ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/
   ```

## Important Notes

### Performance Impact

- **CuDNN Determinism**: Setting `torch.backends.cudnn.deterministic = True` may reduce performance by 10-20%
- This is necessary for full reproducibility
- Can be disabled if approximate reproducibility is acceptable

### Limitations

1. **Hardware Differences**: Results may vary slightly across different GPU models due to floating-point precision differences
2. **PyTorch Version**: Different PyTorch versions may produce slightly different results even with the same seed
3. **CUDA Version**: Different CUDA versions can affect results
4. **Distributed Training**: Must use the same number of GPUs for exact reproducibility

### Best Practices

1. **Document Your Environment**:
   ```bash
   python --version
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
   ```

2. **Fix Your Seed**: Choose a seed and stick with it for a series of experiments

3. **Save Your Config**: The resolved config is saved to `logs/run_config.json` automatically

4. **Checkpoint Compatibility**: Checkpoints contain model state but not RNG states, so resuming may not be bit-exact

## Code Structure

```
training/
├── seed_utils.py          # Core seeding utilities
│   ├── set_seed()         # Set all random seeds
│   └── get_seed_from_config()  # Extract seed from config
├── train_sr.py            # Single-GPU training (uses seeding)
├── train_sr_ddp.py        # Multi-GPU training (uses seeding)
├── data_loading.py        # DataLoader creation (accepts seed parameter)
└── config_utils.py        # Config management (includes seed in defaults)
```

## Troubleshooting

### "Results are not reproducible"

1. Check that seed is set in config.json
2. Verify CuDNN determinism is enabled
3. Ensure same PyTorch/CUDA version
4. Check for any random augmentations without seeding

### "Training is slower than before"

- This is expected due to CuDNN deterministic mode
- Typical slowdown: 10-20%
- To disable (loses exact reproducibility):
  ```python
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True
  ```

### "DDP results differ from single-GPU"

- This is expected! DDP uses `seed + rank` per process
- To get similar results, you'd need to ensure identical data distribution
- For exact comparison, use single-GPU mode

## References

- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [CuDNN Determinism](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.deterministic)
