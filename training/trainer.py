# -*-coding:utf-8 -*-
"""
File:        trainer.py
Created at:  2025/10/18 14:00:04
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import torch
from tqdm import tqdm
from pathlib import Path
import time


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, loaders, loss_fn, writer):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.writer = writer
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.device = torch.device(cfg["device"])
        # Mixed precision scaler (enabled when cfg['amp'] is truthy)
        try:
            amp_enabled = bool(self.cfg.get("amp", False))
        except Exception:
            amp_enabled = False
        self.amp_enabled = amp_enabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def train(self):
        self.model.train()
        total_loss = 0.0

        # Optional lightweight profiling: measures time spent on data transfer,
        # forward, backward and optimizer step for the first N batches.
        debug_profile = bool(self.cfg.get("debug_profile", False))
        profile_steps = int(self.cfg.get("profile_steps", 5)) if debug_profile else None
        prof_stats = []

        if debug_profile:
            # Profiling mode: instrument timings and synchronize to get accurate
            # per-step timings. This adds overhead and should only be used when
            # debugging performance.
            for batch_idx, (lr, hr) in enumerate(self.loaders["train"]):
                # --- Data transfer to device ---
                t0 = time.perf_counter()
                lr = lr.to(self.device, non_blocking=True)
                hr = hr.to(self.device, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                # --- Forward + loss ---
                self.optimizer.zero_grad()
                t_fwd_start = time.perf_counter()
                sr = self.model(lr)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_end = time.perf_counter()

                loss = self.loss_fn(sr, hr)

                # --- Backward ---
                t_bwd_start = time.perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd_end = time.perf_counter()

                # --- Optimizer step ---
                t_opt_start = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                self.optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_opt_end = time.perf_counter()

                total_loss += loss.item()

                prof_stats.append(
                    {
                        "batch_idx": batch_idx,
                        "transfer": t1 - t0,
                        "forward": t_fwd_end - t_fwd_start,
                        "backward": t_bwd_end - t_bwd_start,
                        "opt": t_opt_end - t_opt_start,
                        "loss": loss.item(),
                    }
                )

                print(
                    f"[PROFILE] batch={batch_idx} transfer={prof_stats[-1]['transfer']:.4f}s "
                    f"fwd={prof_stats[-1]['forward']:.4f}s bwd={prof_stats[-1]['backward']:.4f}s "
                    f"opt={prof_stats[-1]['opt']:.4f}s loss={prof_stats[-1]['loss']:.6f}"
                )

                # Stop after requested number of steps
                if batch_idx + 1 >= profile_steps:
                    break
        else:
            # Normal training loop (no extra synchronizations or timing)
            for lr, hr in self.loaders["train"]:
                lr = lr.to(self.device, non_blocking=True)
                hr = hr.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                # Use AMP autocast during forward to speed up and reduce memory
                if self.amp_enabled:
                    with torch.cuda.amp.autocast():
                        sr = self.model(lr)
                        loss = self.loss_fn(sr, hr)
                    # scale the loss and call backward on scaled loss
                    self.scaler.scale(loss).backward()
                    # unscale and clip grads, then step via scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    sr = self.model(lr)
                    loss = self.loss_fn(sr, hr)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.loaders["train"])
        self.scheduler.step()
        self.writer.add_scalar("Loss/Train", avg_loss, self.epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for lr, hr in self.loaders["val"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            # Use AMP autocast during validation forward for speed/memory
            if self.amp_enabled:
                with torch.cuda.amp.autocast():
                    sr = self.model(lr)
                    loss = self.loss_fn(sr, hr)
            else:
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr)
            total_loss += loss.item()

        avg_val_loss = total_loss / len(self.loaders["val"])
        self.writer.add_scalar("Loss/Val", avg_val_loss, self.epoch)
        return avg_val_loss

    def maybe_save_best(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            ckpt = Path(self.cfg["checkpoints_dir"]) / "best_model.pt"
            # Save a full checkpoint with optimizer and scheduler state so training
            # can be resumed exactly from this point.
            ckpt_data = {
                "epoch": int(self.epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "best_val_loss": float(self.best_val_loss),
            }
            torch.save(ckpt_data, ckpt)
            print(f"Saved best checkpoint (epoch={self.epoch}, val_loss={val_loss:.6f}) to {ckpt}")

    def save_last_checkpoint(self):
        """Save the most recent checkpoint (overwrites each epoch).

        This is useful for resuming exactly where training left off even if the
        best checkpoint was saved earlier in training.
        """
        ckpt = Path(self.cfg["checkpoints_dir"]) / "last_checkpoint.pt"
        ckpt_data = {
            "epoch": int(self.epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "best_val_loss": float(self.best_val_loss),
        }
        torch.save(ckpt_data, ckpt)
        # Noisy but helpful to know last checkpoint updated
        print(f"Saved last checkpoint (epoch={self.epoch}) to {ckpt}")

    def load_checkpoint(self, ckpt_path: str | Path, load_optimizer: bool = True):
        """Load a checkpoint into the trainer.

        This will load model weights and, if present and load_optimizer=True,
        optimizer/scheduler state and epoch/best_val_loss.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Backwards-compat: ckpt may be a raw state_dict
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            # assume ckpt is a model state_dict
            self.model.load_state_dict(ckpt)
            print(f"Loaded model state_dict from {ckpt_path} (no optimizer state present)")
            return

        # Load model weights
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Optionally load optimizer and scheduler
        if load_optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: failed to load optimizer state: {e}")

        if load_optimizer and self.scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: failed to load scheduler state: {e}")

        # Restore epoch and best loss
        if "epoch" in ckpt:
            self.epoch = int(ckpt["epoch"])
        if "best_val_loss" in ckpt:
            self.best_val_loss = float(ckpt["best_val_loss"])

        print(f"Resumed from checkpoint {ckpt_path} (epoch={self.epoch}, best_val_loss={self.best_val_loss:.6f})")
