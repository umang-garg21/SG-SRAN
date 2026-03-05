# -*- coding:utf-8 -*-
"""
Autoencoder-specific trainer.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class Autoencoder:
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
        self.last_train_metrics = {}
        self.last_val_metrics = {}
        self.has_optimizer = self.optimizer is not None
        # When True, loop over images in the batch and pass img_shape so that
        # spatial conv layers (e.g. EquivariantSpatialConv) can use the 2-D grid.
        self.use_img_shape = bool(cfg.get("use_img_shape", False))
        # When True, expect (lr, hr) pairs from the loader and run the SR pipeline.
        self.use_sr = bool(cfg.get("use_sr", False))

    def _compute_loss(self, pred_flat, target_flat):
        return self.loss_fn(pred_flat, target_flat)

    @staticmethod
    def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return quats / torch.norm(quats, dim=1, keepdim=True).clamp_min(eps)

    @staticmethod
    def misorientation_bunge(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Geodesic angle (radians) between two FZ-reduced Bunge passive quaternions.

        IMPORTANT: both q1 and q2 must already be reduced to the fundamental zone
        (i.e. the representative with maximum |w| among all symmetry equivalents)
        before calling this function.  Without FZ reduction, symmetry-equivalent
        orientations (e.g. a 90° rotation in O) will return a non-zero angle instead
        of 0°.  See debug_quats_90_degree_Rotation.ipynb for a concrete example.

        Formula (Bunge convention, unit quaternions):
            cos(θ/2) = |q1 · q2|   →   θ = 2 * acos(|q1 · q2|)

        Args:
            q1, q2: (N, 4) FZ-reduced quaternions in [w, x, y, z] order.
        Returns:
            (N,) misorientation angles in [0, π] radians.
        """
        n1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
        n2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_half = (n1 * n2).sum(dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(cos_half)

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
        # Reconstruction metrics: FZ-reduce both sides, then compute misorientation.
        _core = getattr(self.model, "core", self.model)
        reduce_to_fz = getattr(_core, "reduce_to_fz", None)
        if reduce_to_fz is None:
            return {}

        with torch.no_grad():
            pred_fz   = reduce_to_fz(pred_flat)
            target_fz = reduce_to_fz(target_flat)
            errors    = self.misorientation_bunge(pred_fz, target_fz)
            mis_deg   = errors * 180.0 / torch.pi

            return {
                "error_mean":   float(errors.mean().item()),
                "error_max":    float(errors.max().item()),
                "mis_deg_mean": float(mis_deg.mean().item()),
                "mis_deg_max":  float(mis_deg.max().item()),
            }

    @staticmethod
    def _flatten_hr(img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        # (B,4,H,W) -> (B*H*W,4)
        b, c, h, w = img.shape
        if c != 4:
            raise ValueError(f"Expected quaternion channels=4, got shape {tuple(img.shape)}")
        flat = img.permute(0, 2, 3, 1).reshape(-1, 4)
        return flat, (b, h, w)

    def _sr_feature_loss(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        b: int,
        h_lr: int,
        w_lr: int,
    ) -> torch.Tensor:
        """
        Feature-space SR loss (FCCAutoEncoderSR).

        Calls FCCAutoEncoderSR.feature_loss_sr() on the full batch, comparing
        the model's HR-resolution irreps against the irreps obtained by encoding
        the ground-truth HR quaternions.
        """
        _core = getattr(self.model, "core", self.model)
        lr_quats = lr.permute(0, 2, 3, 1).reshape(b, -1, 4)  # (B, N_lr, 4)
        hr_quats = hr.permute(0, 2, 3, 1).reshape(b, -1, 4)  # (B, N_hr, 4)
        return _core.feature_loss_sr(lr_quats, hr_quats, lr_shape=(h_lr, w_lr))

    def _conv_feature_loss(self, hr: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        """
        Feature-space loss for conv models.

        The decoder (FastLookupFCCDecoder) is a non-differentiable lookup table,
        so gradients cannot flow through it.  Instead we train the conv layer by
        minimising MSE between its output and the encoder output in (f4, f6) space.

        Calls FCCAutoEncoderWithConv.feature_loss() per image in the batch.
        """
        _core = getattr(self.model, "core", self.model)
        per_img = hr.permute(0, 2, 3, 1)  # (B, H, W, 4)
        losses = [
            _core.feature_loss(per_img[i].reshape(-1, 4), img_shape=(h, w))
            for i in range(b)
        ]
        return torch.stack(losses).mean()

    def train(self):
        self.model.train()
        total_loss = 0.0
        metric_accum = self._empty_metric_accum()
        metric_steps = 0
        n_batches = len(self.loaders["train"])

        pbar = tqdm(
            self.loaders["train"],
            desc=f"Train e{self.epoch}",
            total=n_batches,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            if self.use_sr:
                lr_batch, hr_batch = batch
                lr_batch = lr_batch.to(self.device, non_blocking=True)
                hr_batch = hr_batch.to(self.device, non_blocking=True)
                b, _, h_lr, w_lr = lr_batch.shape
                q_target, _ = self._flatten_hr(hr_batch)

                if self.has_optimizer:
                    self.optimizer.zero_grad(set_to_none=True)
                loss = self._sr_feature_loss(lr_batch, hr_batch, b, h_lr, w_lr)
                if self.has_optimizer:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.optimizer.step()

                total_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                continue

            _, lr = batch
            lr = lr.to(self.device, non_blocking=True)
            q_target, (b, h, w) = self._flatten_hr(lr)
            if self.has_optimizer:
                self.optimizer.zero_grad(set_to_none=True)

            if self.use_img_shape:
                # Conv model: differentiable feature-space loss (bypasses decoder)
                loss = self._conv_feature_loss(lr, b, h, w)
                if self.has_optimizer:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.optimizer.step()
                # Reconstruction metrics are computed in quaternion space; skip them
                # here because they require the decoder which is expensive per-step.
                q_pred = q_target  # dummy so metric block is a no-op
            else:
                if self.use_amp and self.has_optimizer:
                    with autocast("cuda", dtype=torch.float16):
                        q_pred = self.model(q_target, normalize_input=True)
                        loss = self._compute_loss(q_pred, q_target)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif self.has_optimizer:
                    q_pred = self.model(q_target, normalize_input=True)
                    loss = self._compute_loss(q_pred, q_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                    self.optimizer.step()
                else:
                    q_pred = self.model(q_target, normalize_input=True)
                    loss = self._compute_loss(q_pred, q_target)

                if self.log_recon_metrics:
                    m = self._compute_recon_metrics(q_pred.detach(), q_target.detach())
                    if m:
                        self._accumulate_metrics(metric_accum, m)
                        metric_steps += 1

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        avg_loss = total_loss / max(1, n_batches)
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
        n_batches = len(self.loaders["val"])

        pbar = tqdm(
            self.loaders["val"],
            desc=f"Val   e{self.epoch}",
            total=n_batches,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            if self.use_sr:
                lr_batch, hr_batch = batch
                lr_batch = lr_batch.to(self.device, non_blocking=True)
                hr_batch = hr_batch.to(self.device, non_blocking=True)
                b, _, h_lr, w_lr = lr_batch.shape
                q_hr_target, _ = self._flatten_hr(hr_batch)

                loss = self._sr_feature_loss(lr_batch, hr_batch, b, h_lr, w_lr)

                # Orientation-space metrics: decode SR output per image, compare
                # with HR ground truth after FZ reduction.
                if self.log_recon_metrics:
                    _core = getattr(self.model, "core", self.model)
                    per_img_lr = lr_batch.permute(0, 2, 3, 1)
                    q_preds = [
                        _core.forward_sr(
                            per_img_lr[i].reshape(-1, 4),
                            lr_shape=(h_lr, w_lr),
                        )
                        for i in range(b)
                    ]
                    q_pred = torch.cat(q_preds, dim=0)
                    m = self._compute_recon_metrics(q_pred, q_hr_target)
                    if m:
                        self._accumulate_metrics(metric_accum, m)
                        metric_steps += 1

                total_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                continue

            _, lr = batch
            lr = lr.to(self.device, non_blocking=True)
            q_target, (b, h, w) = self._flatten_hr(lr)

            if self.use_img_shape:
                # Conv model: feature-space loss for val (no_grad context already active)
                loss = self._conv_feature_loss(lr, b, h, w)
                # Orientation-space metrics: decode per-image with img_shape so the
                # conv layer sees the correct 2-D grid, then FZ-reduce both pred and
                # target before computing misorientation (matches notebook reference).
                if self.log_recon_metrics:
                    per_img = lr.permute(0, 2, 3, 1)  # (B, H, W, 4)
                    q_preds = [
                        self.model(per_img[i].reshape(-1, 4), img_shape=(h, w), normalize_input=True)
                        for i in range(b)
                    ]
                    q_pred = torch.cat(q_preds, dim=0)
                    m = self._compute_recon_metrics(q_pred, q_target)
                    if m:
                        self._accumulate_metrics(metric_accum, m)
                        metric_steps += 1
            else:
                if self.use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        q_pred = self.model(q_target, normalize_input=True)
                        loss = self._compute_loss(q_pred, q_target)
                else:
                    q_pred = self.model(q_target, normalize_input=True)
                    loss = self._compute_loss(q_pred, q_target)

                if self.log_recon_metrics:
                    m = self._compute_recon_metrics(q_pred.detach(), q_target.detach())
                    if m:
                        self._accumulate_metrics(metric_accum, m)
                        metric_steps += 1

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        avg_val = total_loss / max(1, n_batches)
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
