#!/usr/bin/env python
"""
DDP Training with Alternating Batch Size Schedule
Alternates between batch_size=1 and batch_size=4 every 100 epochs
"""

import os
import subprocess
import sys
from pathlib import Path

def run_training_segment(exp_dir, num_gpus, start_epoch, end_epoch, batch_size, master_port=29600):
    """Run training for a segment with specified batch size"""
    
    # Update config with current batch size and epoch range
    import json
    config_path = Path(exp_dir) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['batch_size'] = batch_size
    config['epochs'] = end_epoch
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Training Segment: Epochs {start_epoch} → {end_epoch}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "/data/home/umang/miniconda3/envs/material/bin/python",
        "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_port={master_port}",
        "train_ddp.py",
        "--exp_dir", exp_dir,
        "--resume"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    
    # Run training
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"Error in training segment (epochs {start_epoch}-{end_epoch})", file=sys.stderr)
        sys.exit(1)
    
    return result.returncode

def main():
    exp_dir = "experiments/IN718/debug_x4_eqv_res/"
    num_gpus = 6
    
    # Training schedule: alternating batch sizes
    # Starting from epoch 2000, going to epoch 3000 (1000 epochs total)
    # Batch size 1: epochs 2000-2100 (first 100)
    # Batch size 4: epochs 2100-2200 (next 100)
    # Batch size 1: epochs 2200-2300 (next 100)
    # ... and so on
    
    start_epoch = 2000
    total_epochs = 1000
    segment_length = 100
    
    current_epoch = start_epoch
    end_epoch = start_epoch + total_epochs
    
    batch_sizes = [1, 4]  # Alternating pattern
    batch_idx = 0
    
    while current_epoch < end_epoch:
        segment_end = min(current_epoch + segment_length, end_epoch)
        batch_size = batch_sizes[batch_idx % len(batch_sizes)]
        
        run_training_segment(
            exp_dir=exp_dir,
            num_gpus=num_gpus,
            start_epoch=current_epoch,
            end_epoch=segment_end,
            batch_size=batch_size
        )
        
        current_epoch = segment_end
        batch_idx += 1
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Total epochs trained: {start_epoch} → {end_epoch}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
