# -*- coding:utf-8 -*-
"""
Autoencoder-specific trainer.
"""

from pathlib import Path

import torch
from torch.amp import GradScaler, autocast


class AutoencoderTrainer:
    METRIC_KEYS = ("error_mean", "error_max", "mis_deg_mean", "mis_deg_max")

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
        self.use_amp = bool(cfg.get("amp", False))
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.log_recon_metrics = bool(cfg.get("log_recon_metrics", True))
        self.canonical_target_for_loss = bool(cfg.get("canonical_target_for_loss", False))
        self.canonicalize_pred_for_loss = bool(cfg.get("canonicalize_pred_for_loss", False))
        self.last_train_metrics = {}
        self.last_val_metrics = {}
        self.has_optimizer = self.optimizer is not None

    def _compute_loss(self, pred_flat, target_flat):
        return self.loss_fn(pred_flat, target_flat)

    def _prepare_loss_quaternions(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.canonical_target_for_loss and not self.canonicalize_pred_for_loss:
            return pred_flat, target_flat

        canonicalize = getattr(self.model, "canonicalize_for_metrics", None)
        if not callable(canonicalize):
            raise ValueError(
                "canonical_target_for_loss/canonicalize_pred_for_loss requested, "
                "but model does not expose canonicalize_for_metrics()."
            )

        pred_loss = pred_flat
        tgt_loss = target_flat
        if self.canonicalize_pred_for_loss:
            pred_loss = canonicalize(pred_loss)
        if self.canonical_target_for_loss:
            tgt_loss = canonicalize(tgt_loss)
        return pred_loss, tgt_loss

    @staticmethod
    def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return quats / torch.norm(quats, dim=1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                quats[:, 0],
                -quats[:, 1],
                -quats[:, 2],
                -quats[:, 3],
            ],
            dim=1,
        )

    @classmethod
    def _empty_metric_accum(cls) -> dict[str, float]:
        return {k: 0.0 for k in cls.METRIC_KEYS}

    @classmethod
    def _accumulate_metrics(cls, accum: dict[str, float], metrics: dict[str, float]) -> None:
        for key in cls.METRIC_KEYS:
            accum[key] += metrics[key]

    def _finalize_and_log_metrics(
        self,
        metric_accum: dict[str, float],
        metric_steps: int,
        split_name: str,
    ) -> dict[str, float]:
        if not self.log_recon_metrics or metric_steps == 0:
            return {}

        out = {k: metric_accum[k] / metric_steps for k in self.METRIC_KEYS}
        if self.writer is not None:
            cap = split_name.capitalize()
            self.writer.add_scalar(f"Recon/{cap}ErrorMean", out["error_mean"], self.epoch)
            self.writer.add_scalar(f"Recon/{cap}MisDegMean", out["mis_deg_mean"], self.epoch)
        return out

    def _compute_recon_metrics(self, pred_flat, target_flat):
        # Reconstruction metrics with optional model-provided canonicalization.
        if not hasattr(self.model, "quat_mul"):
            return {}

        with torch.no_grad():
            if hasattr(self.model, "_normalize_quaternions"):
                pred_norm = self.model._normalize_quaternions(pred_flat)
                target_norm = self.model._normalize_quaternions(target_flat)
            else:
                pred_norm = self._normalize_quaternions(pred_flat)
                target_norm = self._normalize_quaternions(target_flat)

            canonicalize = getattr(self.model, "canonicalize_for_metrics", None)
            if callable(canonicalize):
                pred_norm = canonicalize(pred_norm)
                target_norm = canonicalize(target_norm)

            if hasattr(self.model, "_to_active_convention") and hasattr(self.model, "_quat_conjugate"):
                qA = self.model._to_active_convention(pred_norm)
                qB = self.model._to_active_convention(target_norm)
                delta = self.model.quat_mul(qA, self.model._quat_conjugate(qB))
            elif hasattr(self.model, "_quat_conjugate"):
                delta = self.model.quat_mul(pred_norm, self.model._quat_conjugate(target_norm))
            else:
                delta = self.model.quat_mul(pred_norm, self._quat_conjugate(target_norm))

            w_errors = delta[:, 0].abs().clamp(max=1.0)
            errors = 2.0 * torch.acos(w_errors)
            mis_deg = errors * 180.0 / torch.pi

            return {
                "error_mean": float(errors.mean().item()),
                "error_max": float(errors.max().item()),
                "mis_deg_mean": float(mis_deg.mean().item()),
                "mis_deg_max": float(mis_deg.max().item()),
            }

    @staticmethod
    def _flatten_hr(hr: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        # (B,4,H,W) -> (B*H*W,4)
        b, c, h, w = hr.shape
        if c != 4:
            raise ValueError(f"Expected quaternion channels=4, got shape {tuple(hr.shape)}")
        flat = hr.permute(0, 2, 3, 1).reshape(-1, 4)
        return flat, (b, h, w)

    def train(self):
        self.model.train()
        total_loss = 0.0
        metric_accum = self._empty_metric_accum()
        metric_steps = 0

        for _, hr in self.loaders["train"]:
            hr = hr.to(self.device, non_blocking=True)
            q_target, _ = self._flatten_hr(hr)
            if self.has_optimizer:
                self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.has_optimizer:
                with autocast("cuda", dtype=torch.float16):
                    q_pred = self.model(q_target, normalize_input=True)
                    q_pred_loss, q_target_loss = self._prepare_loss_quaternions(q_pred, q_target)
                    loss = self._compute_loss(q_pred_loss, q_target_loss)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.has_optimizer:
                q_pred = self.model(q_target, normalize_input=True)
                q_pred_loss, q_target_loss = self._prepare_loss_quaternions(q_pred, q_target)
                loss = self._compute_loss(q_pred_loss, q_target_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                self.optimizer.step()
            else:
                q_pred = self.model(q_target, normalize_input=True)
                q_pred_loss, q_target_loss = self._prepare_loss_quaternions(q_pred, q_target)
                loss = self._compute_loss(q_pred_loss, q_target_loss)

            total_loss += float(loss.item())

            if self.log_recon_metrics:
                m = self._compute_recon_metrics(q_pred.detach(), q_target.detach())
                if m:
                    self._accumulate_metrics(metric_accum, m)
                    metric_steps += 1

        avg_loss = total_loss / max(1, len(self.loaders["train"]))
        if self.scheduler is not None:
            self.scheduler.step()
        if self.writer is not None:
            self.writer.add_scalar("Loss/Train", avg_loss, self.epoch)

        self.last_train_metrics = self._finalize_and_log_metrics(metric_accum, metric_steps, "train")

        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        metric_accum = self._empty_metric_accum()
        metric_steps = 0

        for _, hr in self.loaders["val"]:
            hr = hr.to(self.device, non_blocking=True)
            q_target, _ = self._flatten_hr(hr)

            if self.use_amp:
                with autocast("cuda", dtype=torch.float16):
                    q_pred = self.model(q_target, normalize_input=True)
                    q_pred_loss, q_target_loss = self._prepare_loss_quaternions(q_pred, q_target)
                    loss = self._compute_loss(q_pred_loss, q_target_loss)
            else:
                q_pred = self.model(q_target, normalize_input=True)
                q_pred_loss, q_target_loss = self._prepare_loss_quaternions(q_pred, q_target)
                loss = self._compute_loss(q_pred_loss, q_target_loss)

            total_loss += float(loss.item())

            if self.log_recon_metrics:
                m = self._compute_recon_metrics(q_pred.detach(), q_target.detach())
                if m:
                    self._accumulate_metrics(metric_accum, m)
                    metric_steps += 1

        avg_val = total_loss / max(1, len(self.loaders["val"]))
        if self.writer is not None:
            self.writer.add_scalar("Loss/Val", avg_val, self.epoch)

        self.last_val_metrics = self._finalize_and_log_metrics(metric_accum, metric_steps, "val")

        return avg_val

    def maybe_save_best(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            ckpt = Path(self.cfg["checkpoints_dir"]) / "best_model.pt"
            torch.save(
                {
                    "epoch": int(self.epoch),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "best_val_loss": float(self.best_val_loss),
                },
                ckpt,
            )

    def save_last_checkpoint(self):
        ckpt = Path(self.cfg["checkpoints_dir"]) / "last_checkpoint.pt"
        torch.save(
            {
                "epoch": int(self.epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_val_loss": float(self.best_val_loss),
            },
            ckpt,
        )

    def load_checkpoint(self, ckpt_path, load_optimizer: bool = True):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            self.model.load_state_dict(ckpt)
            return

        self.model.load_state_dict(ckpt["model_state_dict"])

        if load_optimizer and self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

        if load_optimizer and self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass

        if "epoch" in ckpt:
            self.epoch = int(ckpt["epoch"])
        if "best_val_loss" in ckpt:
            self.best_val_loss = float(ckpt["best_val_loss"])
