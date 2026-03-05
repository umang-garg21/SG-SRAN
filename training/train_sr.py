import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm  # ✅ tqdm at epoch level

# --- Project imports ---
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.trainer import Trainer
from training.loss_functions import build_loss
from training.seed_utils import set_seed, get_seed_from_config
from models import build_model
from post_processing.post_process import run_postprocess_from_config

torch.autograd.set_detect_anomaly(True)

# ----------------------------------------------------------------------
# CLI Argument Parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Quaternion Super-Resolution Model"
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="Path to experiment directory containing config.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional comma-separated list of GPU ids to make visible (e.g. '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES before training starts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume training from the best checkpoint found in the experiment checkpoints directory",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main Training Function
# ----------------------------------------------------------------------
def main():
    # --- CLI ---
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)

    # --- Config ---
    config_path = exp_dir / args_cli.config
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    # --- Set Random Seed for Reproducibility ---
    seed = 42  # Always use seed 42
    set_seed(seed)

    # If user supplied GPU ids, restrict visible GPUs via env var before torch picks device
    if args_cli.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args_cli.gpu_ids}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.cuda.current_device()

    # --- DataLoaders ---
    loaders = {
        split: build_dataloader(
            dataset_root=cfg.dataset_root,
            split=split.capitalize(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            preload=cfg.preload,
            preload_torch=cfg.preload_torch,
            pin_memory=cfg.pin_memory,
            take_first=8 if cfg.smoke_test else None,
            seed=seed,  # Pass seed for reproducibility
        )
        for split in ["train", "val", "test"]
    }

    # --- Model ---
    model = build_model(cfg).to(device)
    
    # --- Multi-GPU Support ---
    use_multi_gpu = getattr(cfg, 'use_multi_gpu', True)  # default to True
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and use_multi_gpu:
        print(f"\n{'='*80}")
        print(f"MULTI-GPU TRAINING ENABLED")
        print(f"{'='*80}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        
        # Get GPU IDs to use (either from config or all available)
        gpu_ids = getattr(cfg, 'gpu_ids', None)
        if gpu_ids is not None:
            # Convert string or list to list of ints
            if isinstance(gpu_ids, str):
                gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
            elif not isinstance(gpu_ids, list):
                gpu_ids = [gpu_ids]
            print(f"Using GPUs: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            print(f"Using all available GPUs: {list(range(torch.cuda.device_count()))}")
            model = torch.nn.DataParallel(model)
        
        # Adjust effective batch size
        effective_batch_size = cfg.batch_size * torch.cuda.device_count()
        print(f"Batch size per GPU: {cfg.batch_size}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"{'='*80}\n")
    elif torch.cuda.is_available() and use_multi_gpu:
        print(f"Only 1 GPU available, using single GPU training.")
    else:
        print(f"Multi-GPU disabled or not available.")

    # print model summary for verification
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE & PARAMETER BREAKDOWN")
    print("="*80)
    
    # Print model structure
    print("\nModel Structure:")
    print(model)
    
    # Detailed parameter breakdown
    print("\n" + "="*80)
    print("PARAMETER BREAKDOWN BY LAYER")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer Name':<50} {'Parameters':>15} {'Trainable':>12}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        trainable_str = "Yes" if param.requires_grad else "No"
        print(f"{name:<50} {num_params:>15,} {trainable_str:>12}")
    
    print("-" * 80)
    print(f"{'TOTAL':<50} {total_params:>15,}")
    print(f"{'TRAINABLE':<50} {trainable_params:>15,}")
    print(f"{'NON-TRAINABLE':<50} {total_params - trainable_params:>15,}")
    
    # Group by module type
    print("\n" + "="*80)
    print("PARAMETER SUMMARY BY MODULE TYPE")
    print("="*80)
    
    module_params = {}
    for name, param in model.named_parameters():
        # Extract module type (first part of name before first dot or number)
        parts = name.split('.')
        if len(parts) > 0:
            module_type = parts[0]
            if module_type not in module_params:
                module_params[module_type] = 0
            module_params[module_type] += param.numel()
    
    for module_name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (params / total_params) * 100
        print(f"{module_name:<30} {params:>15,} ({percentage:>5.1f}%)")
    
    # Memory estimation
    print("\n" + "="*80)
    print("MEMORY ESTIMATES")
    print("="*80)
    param_size_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32
    print(f"Model parameters size: {param_size_mb:.2f} MB (float32)")
    print(f"Approx. training memory: {param_size_mb * 4:.2f} MB (params + gradients + optimizer states)")
    print("="*80 + "\n")

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Loss ---
    loss_fn = build_loss(cfg)
    # If the loss is a torch.nn.Module instance, move it to the device so its
    # registered buffers (e.g. finite-difference kernels) live on the same
    # device as model inputs and avoid implicit device transfers per-batch.
    try:
        if isinstance(loss_fn, torch.nn.Module):
            loss_fn = loss_fn.to(device)
    except Exception:
        # Be conservative: if moving the loss fails, continue with function API
        # (some loss builders return plain functions).
        pass

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=exp_dir / "runs")

    # --- Trainer ---
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        loss_fn=loss_fn,
        writer=writer,
    )

    train_losses, val_losses = [], []
    learning_rates = []
    start_epoch = 0

    # ----------------------------------------------------------------------
    # 🏋️ Epoch-level tqdm progress bar
    # ----------------------------------------------------------------------
    # Prefer resuming from an exact last checkpoint (most recent epoch) if available.
    last_ckpt = Path(cfg.checkpoints_dir) / "last_checkpoint.pt"
    best_ckpt = Path(cfg.checkpoints_dir) / "best_model.pt"
    start_epoch = 0
    if args_cli.resume:
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
                print(f"Resuming training from {reason} at epoch {start_epoch} (loaded {ckpt_to_load})")
            except Exception as e:
                print(f"Warning: failed to resume from checkpoint {ckpt_to_load}: {e}. Starting from scratch.")
        else:
            print("No checkpoint found to resume from; starting from scratch.")

    epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Training Epochs", dynamic_ncols=True)

    for epoch in epoch_bar:
        trainer.epoch = epoch
        trainer.epoch = epoch

        train_loss = trainer.train()
        val_loss = trainer.validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Track learning rate
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Update tqdm bar with current loss
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}", lr=f"{current_lr:.2e}"
        )

        trainer.maybe_save_best(val_loss)
        # save a 'last checkpoint' every epoch so resume can continue exactly
        try:
            trainer.save_last_checkpoint()
        except Exception as e:
            print(f"Warning: failed to save last checkpoint: {e}")
        
        # ------------------------------------------------------------------
        # Periodic visualizations: create loss plot and run postprocessing
        # visualizations based on `save_every` setting in the resolved config
        # (e.g. every 100 epochs, every 5 epochs, etc.). Uses current
        # checkpoint (best_model.pt preferred, falls back to last_checkpoint.pt).
        # ------------------------------------------------------------------
        try:
            save_every = getattr(cfg, "save_every", 100)
            # allow int-like strings in configs
            try:
                save_every = int(save_every)
            except Exception:
                save_every = 100

            if save_every > 0 and (epoch + 1) % save_every == 0:
                viz_dir = exp_dir / "visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)

                # Save loss plot (single file, overwritten each time with cumulative data)
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

                # Create epoch-specific subfolder for SR/HR/LR comparisons and IPF images
                epoch_viz_dir = viz_dir / f"epoch_{epoch+1:04d}"
                epoch_viz_dir.mkdir(parents=True, exist_ok=True)

                # run postprocess (renders sr/hr/lr comparisons) using best/last checkpoint
                # keep sample size modest to avoid long pauses
                import traceback
                print(f"🖼️ Generating visualizations at epoch {epoch+1}...")
                # Call postprocessing and explicitly pass string path to avoid type issues
                run_postprocess_from_config(
                    str(exp_dir),
                    max_samples=4 if getattr(cfg, "smoke_test", False) else 8,
                    output_dir=str(epoch_viz_dir),
                )

                # After postprocess returns, list produced visualization files so user can see IPF images
                from pathlib import Path as _P
                viz_dir_p = _P(epoch_viz_dir)
                ipf_files = sorted(viz_dir_p.glob('fz_ipf_sr_hr_*.png'))
                comp_files = sorted(viz_dir_p.glob('sr_hr_lr_comparison_*.png'))
                print(f"🖼️ Visualizations saved to: {epoch_viz_dir}")
                print(f"   Loss plot: {viz_dir / 'loss_plot.png'}")
                print(f"   Learning rate plot: {viz_dir / 'learning_rate.png'}")
                print(f"   Epoch {epoch+1} samples: Comparisons: {len(comp_files)}, IPF: {len(ipf_files)}")
                if ipf_files:
                    print(f"  Example IPF file: {ipf_files[0].name}")
        except Exception as e:
            # Don't crash training for visualization errors; log and continue
            print(f"⚠️ Visualization step failed at epoch {epoch+1}: {e}")
    # ----------------------------------------------------------------------
    # ✅ Post-training
    # ----------------------------------------------------------------------
    print(f"✅ Training complete. Outputs saved in: {exp_dir}")

    plot_loss(
        train_losses,
        val_losses,
        learning_rates,
        save_path=str(exp_dir / "visualizations" / "loss_plot.png"),
        start_epoch=start_epoch,
    )
    
    plot_learning_rate(
        learning_rates,
        save_path=str(exp_dir / "visualizations" / "learning_rate.png"),
        start_epoch=start_epoch,
    )

    run_postprocess_from_config(
        exp_dir,
        max_samples=8 if cfg.smoke_test else 20,
        output_dir=str(exp_dir / "visualizations" / "final"),
    )

# ----------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------
def plot_loss(train_losses, val_losses, learning_rates=None, save_path=None, start_epoch=1):
    """
    Plot training and validation losses, and optionally save the plot to a file.
    """
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))

    if learning_rates is not None and len(learning_rates) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot losses
        ax1.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
        ax1.plot(epochs, val_losses, label="Validation Loss", color="orange", marker="o")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss over Epochs")
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, learning_rates, label="Learning Rate", color="green", marker="o", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
    else:
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


def plot_learning_rate(learning_rates, save_path=None, start_epoch=1):
    """
    Plot learning rate schedule and save to file.
    """
    epochs = list(range(start_epoch, start_epoch + len(learning_rates)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, label="Learning Rate", color="green", marker="o", linewidth=2, markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule over Epochs")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Learning rate plot saved to {save_path}")
        plt.close()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

# # -*- coding:utf-8 -*-
# """
# File:        train_sr.py
# Author:      Warren Zamudio
# Description: Main training entrypoint for Reynolds-QSR model.
# """

# import argparse
# import os
# from pathlib import Path
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt

# # --- Project imports ---
# from training.config_utils import load_and_prepare_config
# from training.data_loading import build_dataloader
# from training.optimizer_utils import build_optimizer
# from training.schedulers import build_scheduler
# from training.trainer import Trainer
# from training.loss_functions import build_loss
# from models import build_model
# from post_processing.post_process import run_postprocess_from_config
# from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)


# # ----------------------------------------------------------------------
# # CLI Argument Parsing
# # ----------------------------------------------------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Train Quaternion Super-Resolution Model"
#     )
#     parser.add_argument(
#         "--exp_dir",
#         required=True,
#         type=str,
#         help="Path to experiment directory containing config.json",
#     )
#     return parser.parse_args()


# # ----------------------------------------------------------------------
# # Main Training Function
# # ----------------------------------------------------------------------
# def main():
#     # --- CLI ---
#     args_cli = parse_args()
#     exp_dir = Path(args_cli.exp_dir)

#     # --- Config ---
#     config_path = exp_dir / "config.json"
#     run_config_path = exp_dir / "logs" / "run_config.json"
#     cfg = load_and_prepare_config(config_path, run_config_path)

#     # --- Device ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cfg.device = str(device)
#     print(f"Using device: {device}")

#     # Warm up CUDA context before spawning DataLoader workers
#     if torch.cuda.is_available():
#         torch.cuda.init()
#         _ = torch.cuda.current_device()

#     # --- DataLoaders ---
#     loaders = {
#         split: build_dataloader(
#             dataset_root=cfg.dataset_root,
#             split=split.capitalize(),
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             preload=cfg.preload,
#             preload_torch=cfg.preload_torch,
#             pin_memory=cfg.pin_memory,
#             take_first=128 if cfg.smoke_test else None,
#         )
#         for split in ["train", "val", "test"]
#     }

#     # --- Model ---
#     model = build_model(cfg).to(device)

#     # --- Optimizer & Scheduler ---
#     optimizer = build_optimizer(model, cfg)
#     scheduler = build_scheduler(optimizer, cfg)

#     # --- Loss ---
#     loss_fn = build_loss(cfg)

#     # --- TensorBoard ---
#     writer = SummaryWriter(log_dir=exp_dir / "runs")

#     # --- Trainer ---
#     trainer = Trainer(
#         cfg=cfg,
#         model=model,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         loaders=loaders,
#         loss_fn=loss_fn,
#         writer=writer,
#     )

#     # Store loss values for plotting
#     train_losses = []
#     val_losses = []

#     # --- Training loop ---
#     for epoch in range(cfg.epochs):
#         trainer.epoch = epoch
#         train_loss = trainer.train()
#         train_losses.append(train_loss)

#         val_loss = trainer.validate()
#         val_losses.append(val_loss)

#         trainer.maybe_save_best(val_loss)

#     print(f"✅ Training complete. Outputs saved in: {exp_dir}")

#     plot_loss(
#         train_losses,
#         val_losses,
#         save_path=str(exp_dir / "visualizations" / "loss_plot.png"),
#     )

#     run_postprocess_from_config(
#         exp_dir,
#         max_samples=8 if cfg.smoke_test else 20,
#     )


# # ----------------------------------------------------------------------
# # Plotting helper
# # ----------------------------------------------------------------------
# def plot_loss(train_losses, val_losses, save_path=None):
#     """
#     Plot training and validation losses, and optionally save the plot to a file.
#     """
#     epochs = range(1, len(train_losses) + 1)

#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         epochs,
#         train_losses,
#         label="Training Loss",
#         color="blue",
#         linestyle="-",
#         marker="o",
#     )
#     plt.plot(
#         epochs,
#         val_losses,
#         label="Validation Loss",
#         color="orange",
#         linestyle="-",
#         marker="o",
#     )
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss over Epochs")
#     plt.legend()
#     plt.grid(True)

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#         print(f"📈 Plot saved to {save_path}")


# # ----------------------------------------------------------------------
# # Entry point
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     main()

# # # -*- coding:utf-8 -*-
# # """
# # File:        train_sr.py
# # Author:      Warren Zamudio
# # Description: Main training entrypoint for Reynolds-QSR model.
# # """

# # import argparse
# # from pathlib import Path
# # import torch
# # from torch.utils.tensorboard import SummaryWriter
# # import os

# # # --- Project imports ---
# # from training.config_utils import load_and_prepare_config

# # # from training.symmetry_utils import prepare_symmetry_files
# # from training.data_loading import build_dataloader
# # from training.optimizer_utils import build_optimizer
# # from training.schedulers import build_scheduler
# # from training.trainer import Trainer
# # from training.loss_functions import build_loss
# # from models import build_model
# # from post_processing.post_process import run_postprocess_from_config
# # import matplotlib.pyplot as plt  # For plotting

# # torch.autograd.set_detect_anomaly(True)


# # # ----------------------------------------------------------------------
# # # CLI Argument Parsing
# # # ----------------------------------------------------------------------
# # def parse_args():
# #     parser = argparse.ArgumentParser(
# #         description="Train Quaternion Super-Resolution Model"
# #     )
# #     parser.add_argument(
# #         "--exp_dir",
# #         required=True,
# #         type=str,
# #         help="Path to experiment directory containing config.json, see README.md for details.",
# #     )
# #     return parser.parse_args()


# # # ----------------------------------------------------------------------
# # # Main Training Function
# # # ----------------------------------------------------------------------
# # def main():
# #     # --- CLI ---
# #     args_cli = parse_args()
# #     exp_dir = Path(args_cli.exp_dir)

# #     # --- Config ---
# #     config_path = exp_dir / "config.json"
# #     run_config_path = exp_dir / "logs" / "run_config.json"

# #     cfg = load_and_prepare_config(config_path, run_config_path)

# #     # --- Device ---
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     cfg.device = str(device)
# #     print(f"Using device: {device}")

# #     # Warm up CUDA context before spawning DataLoader workers
# #     if torch.cuda.is_available():
# #         torch.cuda.init()
# #         _ = torch.cuda.current_device()

# #     # --- DataLoaders ---
# #     loaders = {
# #         split: build_dataloader(
# #             dataset_root=cfg["dataset_root"],
# #             split=split.capitalize(),
# #             batch_size=cfg["batch_size"],
# #             num_workers=cfg["num_workers"],
# #             preload=cfg["preload"],
# #             preload_torch=cfg["preload_torch"],
# #             pin_memory=cfg["pin_memory"],
# #             take_first=64 if cfg["smoke_test"] else None,
# #         )
# #         for split in ["train", "val", "test"]
# #     }

# #     # --- Model ---
# #     model = build_model(cfg).to(device)

# #     # --- Optimizer & Scheduler ---
# #     optimizer = build_optimizer(model, cfg)
# #     scheduler = build_scheduler(optimizer, cfg)
# #     loss_fn = build_loss()

# #     # --- TensorBoard ---
# #     writer = SummaryWriter(log_dir=exp_dir / "runs")

# #     # --- Trainer ---
# #     trainer = Trainer(
# #         cfg=cfg,
# #         model=model,
# #         optimizer=optimizer,
# #         scheduler=scheduler,
# #         loaders=loaders,
# #         loss_fn=loss_fn,
# #         writer=writer,
# #     )

# #     # Store loss values for plotting
# #     train_losses = []
# #     val_losses = []
# #     # --- Training loop ---
# #     for epoch in range(cfg["epochs"]):
# #         trainer.epoch = epoch
# #         train_loss = trainer.train()
# #         train_losses.append(train_loss)

# #         val_loss = trainer.validate()
# #         val_losses.append(val_loss)

# #         trainer.maybe_save_best(val_loss)

# #     print(f"Training complete, outputs saved in --> {exp_dir}")
# #     plot_loss(
# #         train_losses,
# #         val_losses,
# #         save_path=str(exp_dir / "visualizations" / f"loss_plot.png"),
# #     )

# #     run_postprocess_from_config(
# #         exp_dir,
# #         max_samples=8 if cfg["smoke_test"] else None,
# #     )


# # def plot_loss(train_losses, val_losses, save_path=None):
# #     """
# #     Plot training and validation losses, and optionally save the plot to a file.

# #     Parameters:
# #     - train_losses (list): List of training losses for each epoch.
# #     - val_losses (list): List of validation losses for each epoch.
# #     - save_path (str or Path, optional): Path to save the plot. If None, the plot is shown but not saved.
# #     """
# #     epochs = range(1, len(train_losses) + 1)

# #     # Create the plot
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(
# #         epochs,
# #         train_losses,
# #         label="Training Loss",
# #         color="blue",
# #         linestyle="-",
# #         marker="o",
# #     )
# #     plt.plot(
# #         epochs,
# #         val_losses,
# #         label="Validation Loss",
# #         color="orange",
# #         linestyle="-",
# #         marker="o",
# #     )
# #     plt.xlabel("Epoch")
# #     plt.ylabel("Loss")
# #     plt.title("Training and Validation Loss over Epochs")
# #     plt.legend()
# #     plt.grid(True)

# #     # Save the plot if save_path is provided
# #     if save_path:
# #         # Ensure the directory exists
# #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
# #         plt.savefig(save_path)
# #         print(f"Plot saved to {save_path}")

# #     # Show the plot


# # # ----------------------------------------------------------------------
# # # Entry point
# # # ----------------------------------------------------------------------
# # if __name__ == "__main__":
# #     main()

# # from post_processing.post_process import run_postprocess_from_config

# # # from postprocess.run_postprocess_from_config import run_postprocess_from_config

# # run_postprocess_from_config("experiments/IN718/debug_x4", max_samples=8)

# # # -*- coding:utf-8 -*-
# # import argparse
# # from pathlib import Path
# # import torch
# # from torch.utils.tensorboard import SummaryWriter

# # from training.config_utils import load_config
# # from training.symmetry_utils import prepare_symmetry_files
# # from training.data_loading import build_dataloader
# # from training.optimizer_utils import build_optimizer
# # from training.schedulers import build_scheduler
# # from training.trainer import Trainer
# # from training.loss_functions import build_loss
# # from models.reynolds_qsr import Reynolds_QSR


# # def parse_args():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--exp_dir", required=True)
# #     return p.parse_args()


# # def main():
# #     # Load config
# #     args_cli = parse_args()
# #     exp_dir = Path(args_cli.exp_dir)
# #     cfg = load_config(exp_dir / "config.json")

# #     # Prepare symmetry
# #     prepare_symmetry_files(cfg)

# #     # Dataloaders
# #     loaders = {
# #         split.lower(): build_dataloader(
# #             dataset_root=cfg["dataset_root"],
# #             split=split.capitalize(),
# #             batch_size=cfg["batch_size"],
# #             num_workers=cfg["num_workers"],
# #             preload=cfg["preload"],
# #             preload_torch=cfg["preload_torch"],
# #         )
# #         for split in ["train", "val", "test"]
# #     }

# #     # Model, optimizer, loss, scheduler
# #     model = Reynolds_QSR(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
# #     optimizer = build_optimizer(model, cfg)
# #     scheduler = build_scheduler(optimizer, cfg)
# #     loss_fn = build_loss()

# #     # Logging + trainer
# #     writer = SummaryWriter(log_dir=str(exp_dir / "runs"))
# #     trainer = Trainer(cfg, model, optimizer, scheduler, loaders, loss_fn, writer)

# #     for epoch in range(cfg["epochs"]):
# #         trainer.epoch = epoch
# #         trainer.train()
# #         trainer.validate()

# #     print(f"✅ Training complete in {exp_dir}")


# # if __name__ == "__main__":
# #     main()

# # # -*- coding:utf-8 -*-
# # import argparse
# # from pathlib import Path
# # import torch
# # from torch.utils.tensorboard import SummaryWriter

# # from training.config_utils import load_config
# # from training.symmetry_utils import prepare_symmetry_files
# # from training.data_loading import build_dataloader
# # from training.optimizer_utils import build_optimizer
# # from training.schedulers import build_scheduler
# # from training.trainer import Trainer
# # from training.loss_functions import build_loss
# # from models.reynolds_qsr import Reynolds_QSR


# # def parse_args():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--exp_dir", required=True)
# #     return p.parse_args()


# # def main():
# #     # Load config
# #     args_cli = parse_args()
# #     exp_dir = Path(args_cli.exp_dir)
# #     cfg = load_config(exp_dir / "config.json")

# #     # Prepare symmetry
# #     prepare_symmetry_files(cfg)

# #     # Dataloaders
# #     loaders = {
# #         split.lower(): build_dataloader(
# #             dataset_root=cfg["dataset_root"],
# #             split=split.capitalize(),
# #             batch_size=cfg["batch_size"],
# #             num_workers=cfg["num_workers"],
# #             preload=cfg["preload"],
# #             preload_torch=cfg["preload_torch"],
# #         )
# #         for split in ["train", "val", "test"]
# #     }

# #     # Model, optimizer, loss, scheduler
# #     model = Reynolds_QSR(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
# #     optimizer = build_optimizer(model, cfg)
# #     scheduler = build_scheduler(optimizer, cfg)
# #     loss_fn = build_loss()

# #     # Logging + trainer
# #     writer = SummaryWriter(log_dir=str(exp_dir / "runs"))
# #     trainer = Trainer(cfg, model, optimizer, scheduler, loaders, loss_fn, writer)

# #     for epoch in range(cfg["epochs"]):
# #         trainer.epoch = epoch
# #         trainer.train()
# #         trainer.validate()

# #     print(f"✅ Training complete in {exp_dir}")


# # if __name__ == "__main__":
# #     main()
