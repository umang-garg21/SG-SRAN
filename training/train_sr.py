import argparse
import os
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
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main Training Function
# ----------------------------------------------------------------------
def main():
    # --- CLI ---
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)

    # --- Config ---
    config_path = exp_dir / "config.json"
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

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
        )
        for split in ["train", "val", "test"]
    }

    # --- Model ---
    model = build_model(cfg).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Loss ---
    loss_fn = build_loss(cfg)

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

    # ----------------------------------------------------------------------
    # 🏋️ Epoch-level tqdm progress bar
    # ----------------------------------------------------------------------
    epoch_bar = tqdm(range(cfg.epochs), desc="Training Epochs", dynamic_ncols=True)

    for epoch in epoch_bar:
        trainer.epoch = epoch

        train_loss = trainer.train()
        val_loss = trainer.validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update tqdm bar with current loss
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}"
        )

        trainer.maybe_save_best(val_loss)

    # ----------------------------------------------------------------------
    # ✅ Post-training
    # ----------------------------------------------------------------------
    print(f"✅ Training complete. Outputs saved in: {exp_dir}")

    plot_loss(
        train_losses,
        val_losses,
        save_path=str(exp_dir / "visualizations" / "loss_plot.png"),
    )

    run_postprocess_from_config(
        exp_dir,
        max_samples=8 if cfg.smoke_test else 20,
    )


# ----------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------
def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses, and optionally save the plot to a file.
    """
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

# #     # --- Loss ---
# #     loss_fn = build_loss(cfg)

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
