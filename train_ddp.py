#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) training wrapper for multi-GPU training.
This provides better GPU utilization than DataParallel.

Usage:
    # Train on GPUs 0-5
    torchrun --nproc_per_node=6 train_ddp.py --exp_dir experiments/IN718/debug_x4_eqv_res/
    
    # Or with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 train_ddp.py --exp_dir experiments/IN718/debug_x4_eqv_res/
"""

import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Project imports ---
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.trainer import Trainer
from training.loss_functions import build_loss
from models import build_model
from post_processing.post_process import run_postprocess_from_config

torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_ddp():
    """Initialize DDP training."""
    # These environment variables are set by torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with Distributed Data Parallel (DDP)"
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="Path to experiment directory containing config.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from best checkpoint",
    )
    return parser.parse_args()


def plot_loss(train_losses, val_losses, learning_rates, save_path, start_epoch=1):
    """Create and save loss and learning rate plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Create epoch array starting from the correct epoch
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))
    
    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_rate(learning_rates, save_path, start_epoch=1):
    """Create and save standalone learning rate plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create epoch array starting from the correct epoch
    epochs = list(range(start_epoch, start_epoch + len(learning_rates)))
    
    # Plot learning rate
    ax.plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='green', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # --- Setup DDP ---
    rank, local_rank, world_size = setup_ddp()
    is_main_process = rank == 0
    
    # --- Set seed for reproducibility ---
    # All processes use the same seed 42 for consistent results
    set_seed(42)
    
    # --- CLI ---
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)
    
    # --- Config ---
    config_path = exp_dir / "config.json"
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    learning_rates = []
    start_epoch = 1  # Will be updated if resuming from checkpoint
    
    # --- Device ---
    device = torch.device(f"cuda:{local_rank}")
    cfg.device = str(device)
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED DATA PARALLEL (DDP) TRAINING")
        print(f"{'='*80}")
        print(f"World size (total GPUs): {world_size}")
        print(f"Batch size per GPU: {cfg.batch_size}")
        print(f"Effective global batch size: {cfg.batch_size * world_size}")
        print(f"{'='*80}\n")
    
    # --- DataLoaders with DDP sampler ---
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset_split = split.capitalize()
        
        # Build dataset using the existing quaternion_dataset
        from training.quaternion_dataset import QuaternionDataset
        dataset = QuaternionDataset(
            dataset_root=cfg.dataset_root,
            split=dataset_split,
            preload=cfg.preload,
            preload_torch=cfg.preload_torch,
            take_first=8 if cfg.smoke_test else None,
        )
        
        # Use DistributedSampler for train, regular sampler for val/test
        if split == "train":
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
        else:
            sampler = None
        
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=(split == "train"),
        )
    
    # --- Model ---
    model = build_model(cfg).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"MODEL WRAPPED WITH DDP")
        print(f"{'='*80}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*80}\n")
    
    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # --- Loss ---
    loss_fn = build_loss(cfg)
    # Move loss function to device if it's a nn.Module
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    
    # --- TensorBoard (only on main process) ---
    writer = None
    if is_main_process:
        log_dir = exp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    
    # --- Trainer ---
    trainer = Trainer(cfg, model, optimizer, scheduler, loaders, loss_fn, writer)
    
    # --- Resume if requested ---
    if args_cli.resume:
        # Try last_checkpoint.pt first (most recent), then best_model.pt
        last_ckpt = Path(cfg["checkpoints_dir"]) / "last_checkpoint.pt"
        best_ckpt = Path(cfg["checkpoints_dir"]) / "best_model.pt"
        
        if last_ckpt.exists():
            if is_main_process:
                print(f"Resuming from {last_ckpt}")
            trainer.load_checkpoint(last_ckpt, load_optimizer=True)
            start_epoch = trainer.epoch + 1
        elif best_ckpt.exists():
            if is_main_process:
                print(f"Resuming from {best_ckpt}")
            trainer.load_checkpoint(best_ckpt, load_optimizer=True)
            start_epoch = trainer.epoch + 1
        else:
            if is_main_process:
                print(f"No checkpoint found, starting from scratch")
    
    # --- Training Loop ---
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING")
        print(f"{'='*80}\n")
    
    epochs_iter = range(trainer.epoch + 1, cfg.epochs + 1)
    if is_main_process:
        epochs_iter = tqdm(epochs_iter, desc="Training Epochs", unit="epoch")
    
    for epoch in epochs_iter:
        trainer.epoch = epoch
        
        # Set epoch for DistributedSampler (important for shuffling)
        if hasattr(loaders["train"].sampler, 'set_epoch'):
            loaders["train"].sampler.set_epoch(epoch)
        
        # Train (all processes)
        train_loss = trainer.train()
        
        # All processes need to participate in validation to avoid deadlock
        # But only main process saves checkpoints and logs
        val_loss = trainer.validate()
        
        if is_main_process:
            # Track losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Track learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Logging
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("LR", current_lr, epoch)
            
            # Save checkpoints
            trainer.maybe_save_best(val_loss)
            if epoch % 10 == 0 or epoch == cfg.epochs:
                trainer.save_last_checkpoint()
            
            # Periodic visualizations every 100 epochs
            save_every = getattr(cfg, "save_every", 100)
            if save_every > 0 and epoch % save_every == 0:
                try:
                    viz_dir = exp_dir / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save loss plot (single file, overwritten each time with cumulative data)
                    print(f"🖼️ Generating visualizations at epoch {epoch}...")
                    plot_loss(
                        train_losses,
                        val_losses,
                        learning_rates,
                        save_path=str(viz_dir / "loss_plot.png"),
                        start_epoch=start_epoch,
                    )
                    
                    # Save standalone learning rate plot
                    plot_learning_rate(
                        learning_rates,
                        save_path=str(viz_dir / "learning_rate.png"),
                        start_epoch=start_epoch,
                    )
                    
                    # Generate SR/HR/LR comparisons and IPF images with epoch-specific folder
                    epoch_viz_dir = viz_dir / f"epoch_{epoch:04d}"
                    epoch_viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    run_postprocess_from_config(
                        str(exp_dir),
                        max_samples=4 if getattr(cfg, "smoke_test", False) else 8,
                        output_dir=str(epoch_viz_dir),
                    )
                    
                    # List generated files
                    ipf_files = sorted(epoch_viz_dir.glob('fz_ipf_sr_hr_*.png'))
                    comp_files = sorted(epoch_viz_dir.glob('sr_hr_lr_comparison_*.png'))
                    print(f"✅ Visualizations saved to: {epoch_viz_dir}")
                    print(f"   Loss plot: {viz_dir / 'loss_plot.png'}")
                    print(f"   Learning rate plot: {viz_dir / 'learning_rate.png'}")
                    print(f"   Epoch {epoch} samples: Comparisons: {len(comp_files)}, IPF: {len(ipf_files)}")
                    if ipf_files:
                        print(f"   Example: {ipf_files[0].name}")
                except Exception as e:
                    print(f"⚠️ Visualization failed at epoch {epoch}: {e}")
        
        # Scheduler step (all processes to keep in sync)
        if scheduler is not None:
            scheduler.step()
        
        # Synchronize all processes before next epoch
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}\n")
        if writer is not None:
            writer.close()
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
