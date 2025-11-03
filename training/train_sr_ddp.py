import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project imports ---
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.trainer import Trainer
from training.loss_functions import build_loss
from training.batch_scheduler import BatchSizeScheduler
from training.seed_utils import set_seed, get_seed_from_config
from models import build_model
from post_processing.post_process import run_postprocess_from_config

torch.autograd.set_detect_anomaly(True)


# ----------------------------------------------------------------------
# DDP Setup and Cleanup
# ----------------------------------------------------------------------
def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


# ----------------------------------------------------------------------
# CLI Argument Parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Quaternion Super-Resolution Model with DDP"
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
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Plot Loss Function
# ----------------------------------------------------------------------
def plot_loss(train_losses, val_losses, save_path=None):
    """Plot training and validation losses"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Plot saved to {save_path}")
        plt.close()


# ----------------------------------------------------------------------
# Main Training Function (per process)
# ----------------------------------------------------------------------
def train_worker(rank, world_size, args_cli, exp_dir, cfg):
    """Training function for each DDP process"""
    
    # Setup DDP for this process
    setup_ddp(rank, world_size)
    
    # Set seed for reproducibility (different for each rank to ensure different data augmentation if used)
    seed = get_seed_from_config(cfg)
    set_seed(seed + rank)  # Each process gets a slightly different seed
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    cfg.device = str(device)
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED DATA PARALLEL TRAINING")
        print(f"{'='*80}")
        print(f"World size (total GPUs): {world_size}")
        print(f"Using dynamic batch size scheduling")
        print(f"{'='*80}\n")
    
    # Initialize batch size scheduler with initial batch size from config
    initial_batch_size = getattr(cfg, 'batch_size', 64)  # Read from config, default to 64
    batch_scheduler = BatchSizeScheduler(initial_batch_size=initial_batch_size, min_batch_size=4)
    
    if is_main_process:
        batch_scheduler.get_schedule_summary(max_epochs=cfg.epochs)
    
    # Start with initial batch size
    current_batch_size = batch_scheduler.get_batch_size(0)
    
    # Build initial dataloaders
    def build_loaders(batch_size):
        return {
            split: build_dataloader(
                dataset_root=cfg.dataset_root,
                split=split.capitalize(),
                batch_size=batch_size,
                num_workers=cfg.num_workers,
                preload=cfg.preload,
                preload_torch=cfg.preload_torch,
                pin_memory=cfg.pin_memory,
                take_first=8 if getattr(cfg, 'smoke_test', False) else None,
                distributed=True,  # Enable DDP-aware sampling
                rank=rank,
                world_size=world_size,
                seed=seed,  # Pass seed for reproducibility
            )
            for split in ["train", "val", "test"]
        }
    
    loaders = build_loaders(current_batch_size)
    
    # Build model and wrap with DDP
    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    if is_main_process:
        print("\n" + "="*80)
        print("MODEL WRAPPED WITH DDP")
        print("="*80)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("="*80 + "\n")
    
    # Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # Loss function
    loss_fn = build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    
    # TensorBoard (only on main process)
    writer = None
    if is_main_process:
        writer = SummaryWriter(log_dir=exp_dir / "runs")
    
    # Trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        loss_fn=loss_fn,
        writer=writer if is_main_process else None,
    )
    
    # Track losses (only on main process)
    train_losses, val_losses = [], []
    
    # Resume logic
    last_ckpt = Path(cfg.checkpoints_dir) / "last_checkpoint.pt"
    best_ckpt = Path(cfg.checkpoints_dir) / "best_model.pt"
    start_epoch = 0
    
    if args_cli.resume and is_main_process:
        ckpt_to_load = None
        if last_ckpt.exists():
            ckpt_to_load = last_ckpt
            reason = "last checkpoint"
        elif best_ckpt.exists():
            ckpt_to_load = best_ckpt
            reason = "best checkpoint"
        
        if ckpt_to_load is not None:
            try:
                trainer.load_checkpoint(ckpt_to_load)
                start_epoch = trainer.epoch + 1
                print(f"Resuming training from {reason} at epoch {start_epoch}")
            except Exception as e:
                print(f"Warning: failed to resume from checkpoint: {e}")
    
    # Synchronize start epoch across all processes
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
    
    # Training loop with dynamic batch size
    epoch_bar = None
    if is_main_process:
        epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Training Epochs", dynamic_ncols=True)
    else:
        epoch_bar = range(start_epoch, cfg.epochs)
    
    for epoch in epoch_bar:
        trainer.epoch = epoch
        
        # Check if batch size needs to change
        new_batch_size = batch_scheduler.get_batch_size(epoch)
        if new_batch_size != current_batch_size:
            if is_main_process:
                print(f"\n📦 Batch size changing from {current_batch_size} to {new_batch_size} at epoch {epoch}")
            current_batch_size = new_batch_size
            
            # Rebuild dataloaders with new batch size
            loaders = build_loaders(current_batch_size)
            trainer.loaders = loaders
            
            # Synchronize before continuing
            if world_size > 1:
                dist.barrier()
        
        # Train and validate
        train_loss = trainer.train()
        val_loss = trainer.validate()
        
        if is_main_process:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update progress bar
            if hasattr(epoch_bar, 'set_postfix'):
                epoch_bar.set_postfix(
                    train_loss=f"{train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                    bs=current_batch_size
                )
            
            # Save checkpoints
            trainer.maybe_save_best(val_loss)
            try:
                trainer.save_last_checkpoint()
            except Exception as e:
                print(f"Warning: failed to save last checkpoint: {e}")
            
            # Periodic visualizations
            try:
                save_every = getattr(cfg, "save_every", 100)
                try:
                    save_every = int(save_every)
                except Exception:
                    save_every = 100
                
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    viz_dir = exp_dir / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save intermediate loss plot
                    plot_loss(
                        train_losses,
                        val_losses,
                        save_path=str(viz_dir / f"loss_plot_epoch_{epoch+1:04d}.png"),
                    )
                    
                    # Run postprocessing
                    print(f"🖼️  Generating visualizations at epoch {epoch+1}...")
                    run_postprocess_from_config(
                        str(exp_dir),
                        max_samples=4 if getattr(cfg, "smoke_test", False) else 8,
                    )
                    
                    # List generated files
                    from pathlib import Path as _P
                    viz_dir_p = _P(viz_dir)
                    ipf_files = sorted(viz_dir_p.glob('fz_ipf_sr_hr_*.png'))
                    comp_files = sorted(viz_dir_p.glob('sr_hr_lr_comparison_*.png'))
                    loss_files = sorted(viz_dir_p.glob('loss_plot_epoch_*.png'))
                    print(f"🖼️  Visualizations saved to: {viz_dir} (loss plots: {len(loss_files)}, comparisons: {len(comp_files)}, ipf: {len(ipf_files)})")
                    if ipf_files:
                        print(f"  Example IPF file: {ipf_files[0]}")
            except Exception as e:
                print(f"⚠️  Visualization step failed at epoch {epoch+1}: {e}")
        
        # Synchronize all processes before next epoch
        if world_size > 1:
            dist.barrier()
    
    # Post-training (only on main process)
    if is_main_process:
        print(f"\n✅ Training complete. Outputs saved in: {exp_dir}")
        
        plot_loss(
            train_losses,
            val_losses,
            save_path=str(exp_dir / "visualizations" / "loss_plot.png"),
        )
        
        run_postprocess_from_config(
            str(exp_dir),
            max_samples=8 if getattr(cfg, 'smoke_test', False) else 20,
        )
    
    # Cleanup
    cleanup_ddp()


# ----------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------
def main():
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)
    
    # Load config (do this once before spawning)
    config_path = exp_dir / "config.json"
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)
    
    # Determine number of GPUs - use available GPUs but max 6
    available_gpus = torch.cuda.device_count()
    world_size = min(available_gpus, 6)  # Cap at 6 GPUs
    
    if available_gpus < 1:
        raise RuntimeError("No CUDA devices available for DDP training")
    
    if available_gpus > 6:
        print(f"Note: {available_gpus} GPUs detected but will only use 6 (as configured)")
    
    print(f"Launching DDP training on {world_size} GPUs...")
    
    # Spawn one process per GPU
    mp.spawn(
        train_worker,
        args=(world_size, args_cli, exp_dir, cfg),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
