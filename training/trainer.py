# -*- coding:utf-8 -*-
"""
File:        trainer.py
Created at:  2025/10/18 14:00:04
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Quaternion SR Trainer with AMP support
"""

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path


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
        self.use_amp = cfg.get("amp", True)

        if self.use_amp:
            self.scaler = GradScaler()

    def train(self):
        self.model.train()
        total_loss = 0.0

        for lr, hr in self.loaders["train"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
                    sr = self.model(lr)
                    loss = self.loss_fn(sr, hr)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["clip"]
                )
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

            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
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

    def save_last_checkpoint(self):
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

    def load_checkpoint(self, ckpt_path, load_optimizer: bool = True):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Backward compatibility: raw state_dict checkpoint
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            self.model.load_state_dict(ckpt)
            return

        self.model.load_state_dict(ckpt["model_state_dict"])

        if load_optimizer and ckpt.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

        if (
            load_optimizer
            and self.scheduler is not None
            and ckpt.get("scheduler_state_dict") is not None
        ):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass

        if "epoch" in ckpt:
            self.epoch = int(ckpt["epoch"])
        if "best_val_loss" in ckpt:
            self.best_val_loss = float(ckpt["best_val_loss"])