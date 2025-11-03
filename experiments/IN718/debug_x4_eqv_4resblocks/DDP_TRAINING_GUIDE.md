# DDP Training with Dynamic Batch Size Scheduling

This document explains how to use the Distributed Data Parallel (DDP) training setup with dynamic batch size scheduling.

## Overview

The DDP training system automatically:
- Distributes training across all available GPUs
- Adjusts batch size dynamically during training according to a schedule
- Maintains consistent effective batch size across GPUs

## Batch Size Schedule

The training follows this batch size schedule:

- **Epochs 0-499**: batch_size = 64 (constant for 500 epochs)
- **Epochs 500-599**: batch_size = 32 (÷2)
- **Epochs 600-699**: batch_size = 16 (÷2)
- **Epochs 700-799**: batch_size = 8 (÷2)
- **Epochs 800-899**: batch_size = 4 (÷2, reached minimum)
- **Epochs 900-999**: batch_size = 1 (alternation starts)
- **Epochs 1000-1099**: batch_size = 4
- **Epochs 1100-1199**: batch_size = 1
- ... (continues alternating between 4 and 1 every 100 epochs until epoch 2000)

## Usage

### Start New Training

```bash
./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/
```

### Resume Training

```bash
./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/ --resume
```

### Specify GPUs

Use `CUDA_VISIBLE_DEVICES` to control which GPUs are used:

```bash
# Use GPUs 0 and 1 only
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/

# Use GPUs 2, 3, 4, 5
CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/train_ddp.sh experiments/IN718/debug_x4_eqv_4resblocks/
```

## Configuration

The config file (`experiments/IN718/debug_x4_eqv_4resblocks/config.json`) should include:

```json
{
    "epochs": 2000,
    "batch_size": 64,
    "num_workers": 4,
    "preload": false,
    "preload_torch": false,
    "pin_memory": true,
    "use_multi_gpu": true,
    "save_every": 10
}
```

**Important notes:**
- `batch_size: 64` is the **initial** batch size (will change automatically)
- `preload: false` is recommended for DDP to avoid excessive memory usage
- `num_workers: 4` provides good balance between data loading and training
- `save_every: 10` controls how often visualizations are generated

## How It Works

### DDP Initialization

The training script (`training/train_sr_ddp.py`) uses `torch.multiprocessing.spawn()` to:
1. Create one process per GPU
2. Initialize process groups with NCCL backend
3. Wrap the model with `DistributedDataParallel`

### Dynamic Batch Size

The `BatchSizeScheduler` class automatically:
1. Calculates the appropriate batch size for each epoch
2. Rebuilds dataloaders when batch size changes
3. Ensures all processes use the same batch size

### Data Loading

The `build_dataloader` function (in `training/data_loading.py`) supports DDP through:
- `DistributedSampler` for splitting data across GPUs
- Proper seed initialization for reproducibility
- Synchronized sampling across all processes

### Checkpointing

Only the main process (rank 0) saves checkpoints to avoid conflicts:
- `best_model.pt`: Best validation loss model
- `last_checkpoint.pt`: Most recent epoch (for resuming)

### Visualizations

Visualizations are generated only on the main process (rank 0) at intervals specified by `save_every`.

## Effective Batch Size

The **effective batch size** is `batch_size × num_gpus`:

| Epoch Range | Per-GPU Batch Size | Effective Batch Size (4 GPUs) |
|-------------|-------------------|------------------------------|
| 0-499       | 64                | 256                         |
| 500-599     | 32                | 128                         |
| 600-699     | 16                | 64                          |
| 700-799     | 8                 | 32                          |
| 800-899     | 4                 | 16                          |
| 900-999     | 1                 | 4                           |
| 1000-1099   | 4                 | 16                          |

## Monitoring

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir experiments/IN718/debug_x4_eqv_4resblocks/runs
```

### Progress Bar

The main process shows a progress bar with:
- Current epoch
- Training loss
- Validation loss
- Current batch size

Example:
```
Training Epochs:  25%|████▌             | 500/2000 [2:15:30<6:46:30, 16.26s/it, train_loss=0.456789, val_loss=0.512345, bs=32]
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. The batch size may be too large for your GPU memory
2. Consider modifying the `BatchSizeScheduler` to start with a smaller initial batch size
3. Reduce `n_feats` or `n_resblocks` in the model config

### Slow Data Loading

If training is bottlenecked by data loading:
1. Increase `num_workers` (try 8 or 16)
2. Enable `preload: true` if you have sufficient RAM
3. Use faster storage (NVMe SSD) for the dataset

### Uneven GPU Utilization

If GPUs have uneven utilization:
1. Check that `DistributedSampler` is being used (automatic in DDP mode)
2. Verify batch size is divisible by number of GPUs
3. Monitor with `nvidia-smi` during training

## Files

- `training/train_sr_ddp.py`: Main DDP training script
- `training/batch_scheduler.py`: Dynamic batch size scheduler
- `training/data_loading.py`: DDP-aware dataloader builder
- `scripts/train_ddp.sh`: Launch script
- `experiments/IN718/debug_x4_eqv_4resblocks/config.json`: Experiment configuration

## Performance Tips

1. **Pin Memory**: Keep `pin_memory: true` for faster GPU transfers
2. **Persistent Workers**: Use `persistent_workers: true` in dataloader if `num_workers > 0`
3. **Mixed Precision**: Set `amp: true` in config to use automatic mixed precision (FP16)
4. **Gradient Checkpointing**: For very large models, enable gradient checkpointing to reduce memory

## Example Output

```
==========================================
  DDP Training Configuration
==========================================
Experiment dir: experiments/IN718/debug_x4_eqv_4resblocks/
Resume: false
Dynamic batch size: Enabled
  - Epochs 0-499: batch_size=64
  - Epochs 500+: Divide by 2 every 100 epochs
  - After reaching 4: Alternate 4/1 every 100 epochs

==========================================

================================================================================
DISTRIBUTED DATA PARALLEL TRAINING
================================================================================
World size (total GPUs): 4
Using dynamic batch size scheduling
================================================================================

================================================================================
BATCH SIZE SCHEDULE
================================================================================
Epochs    0- 499: batch_size = 64
Epochs  500- 599: batch_size = 32
Epochs  600- 699: batch_size = 16
...
================================================================================

Training Epochs:   0%|          | 0/2000 [00:00<?, ?it/s]
```
