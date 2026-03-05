# -*- coding:utf-8 -*-
"""
File:        trainer.py
Created at:  2025/10/18 14:00:04
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Quaternion SR Trainer with AMP support
"""

import torch
import torch.nn.functional as F
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

    def _is_invariant_sr(self) -> bool:
        model_type = str(getattr(self.cfg, "model_type", "")).lower()
        return model_type == "invariant_sr"

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    @staticmethod
    def _flatten_quat_chw(q_chw: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if q_chw.dim() != 3 or q_chw.shape[0] != 4:
            raise ValueError(f"Expected CHW quaternion tensor (4,H,W), got {tuple(q_chw.shape)}")
        h, w = int(q_chw.shape[1]), int(q_chw.shape[2])
        q_flat = q_chw.permute(1, 2, 0).reshape(-1, 4)
        return q_flat, (h, w)

    def _compute_invariant_sr_irrep_loss(self, lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        core = self._unwrap_model()

        lambda_f4 = float(getattr(self.cfg, "irrep_lambda_f4", 1.0))
        lambda_f6 = float(getattr(self.cfg, "irrep_lambda_f6", 1.0))

        batch_losses = []
        for b in range(lr.shape[0]):
            lr_flat, lr_shape = self._flatten_quat_chw(lr[b])
            hr_flat, _ = self._flatten_quat_chw(hr[b])

            out = core._forward_flat(
                quats=lr_flat,
                img_shape=lr_shape,
                decode=False,
            )
            f4_sr, f6_sr = out["hr_convolved_irreps"]

            hr_flat = core.normalize_quaternions(hr_flat)
            with torch.no_grad():
                f4_hr, f6_hr = core.encoder(hr_flat)

            loss_f4 = F.mse_loss(f4_sr, f4_hr)
            loss_f6 = F.mse_loss(f6_sr, f6_hr)
            total_loss = lambda_f4 * loss_f4 + lambda_f6 * loss_f6

            decode_lambda = float(getattr(self.cfg, "invariant_decode_lambda", 0.0))
            decode_start_epoch = int(getattr(self.cfg, "invariant_decode_start_epoch", 0))
            decode_mode = str(getattr(self.cfg, "invariant_decode_mode", "learnable")).lower()

            if decode_lambda > 0.0 and int(self.epoch) >= decode_start_epoch:
                decode_out = core._forward_flat(
                    quats=lr_flat,
                    img_shape=lr_shape,
                    decode=True,
                    decode_mode=decode_mode,
                )
                q_pred = core.normalize_quaternions(decode_out["output"])
                q_tgt = core.normalize_quaternions(hr_flat)

                match_symmetry = bool(getattr(self.cfg, "invariant_decode_match_symmetry", True))
                if match_symmetry and hasattr(core, "reduce_to_fz"):
                    q_pred = core.reduce_to_fz(q_pred)

                decode_loss = self.loss_fn(q_pred, q_tgt)
                total_loss = total_loss + decode_lambda * decode_loss

                smooth_lambda = float(getattr(self.cfg, "invariant_smooth_lambda", 0.0))
                if smooth_lambda > 0.0:
                    hr_h, hr_w = decode_out["hr_shape"]
                    q_img = q_pred.reshape(hr_h, hr_w, 4).permute(2, 0, 1).unsqueeze(0)
                    tv_h = (q_img[:, :, 1:, :] - q_img[:, :, :-1, :]).abs().mean()
                    tv_w = (q_img[:, :, :, 1:] - q_img[:, :, :, :-1]).abs().mean()
                    total_loss = total_loss + smooth_lambda * (tv_h + tv_w)

            batch_losses.append(total_loss)

        return torch.stack(batch_losses).mean()

    def train(self):
        self.model.train()
        total_loss = 0.0

        for lr, hr in self.loaders["train"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self._is_invariant_sr():
                if self.use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        loss = self._compute_invariant_sr_irrep_loss(lr, hr)

                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["clip"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self._compute_invariant_sr_irrep_loss(lr, hr)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["clip"]
                    )
                    self.optimizer.step()
            elif self.use_amp:
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

            if self._is_invariant_sr():
                if self.use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        loss = self._compute_invariant_sr_irrep_loss(lr, hr)
                else:
                    loss = self._compute_invariant_sr_irrep_loss(lr, hr)
            elif self.use_amp:
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