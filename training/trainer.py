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
import numpy as np
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from utils.symmetry_utils import resolve_symmetry


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
        self.last_train_metrics = {}
        self.last_val_metrics = {}

        self.metric_symmetry_enabled = bool(cfg.get("metric_symmetry_enabled", True))
        self.metric_compute_train = bool(cfg.get("metric_compute_train", False))
        self.metric_compute_val = bool(cfg.get("metric_compute_val", True))
        self.metric_symmetry_chunk_size = int(cfg.get("metric_symmetry_chunk_size", 65536))
        self.quat_norm_reg_weight = float(cfg.get("quat_norm_reg_weight", 0.0))
        self.sym_quats_inv = None

        quat_conv = str(getattr(cfg, "quaternion_convention", "bunge_passive_wxyz")).strip().lower()
        if quat_conv not in {"bunge_passive_wxyz", "bunge", "bunge_passive"}:
            self.metric_symmetry_enabled = False

        if self.metric_symmetry_enabled:
            try:
                sym_name = cfg.get("symmetry_group", cfg.get("symmetry", "O"))
                sym = resolve_symmetry(sym_name)
                sym_quats = np.asarray(getattr(sym, "data", sym), dtype=np.float32)
                if sym_quats.ndim != 2 or sym_quats.shape[1] != 4:
                    raise ValueError(f"Unexpected symmetry quaternion shape: {sym_quats.shape}")
                syms = torch.as_tensor(sym_quats, dtype=torch.float32, device=self.device)
                syms = syms / syms.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                syms_inv = syms.clone()
                syms_inv[:, 1:] = -syms_inv[:, 1:]
                self.sym_quats_inv = syms_inv
            except Exception as exc:
                self.metric_symmetry_enabled = False
                print(f"[metrics] symmetry-aware metric setup failed, disabling symmetry metrics: {exc}")

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

    @staticmethod
    def _flatten_quats(q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 2 and q.shape[-1] == 4:
            return q
        if q.dim() == 4 and q.shape[1] == 4:
            return q.permute(0, 2, 3, 1).reshape(-1, 4)
        if q.dim() == 4 and q.shape[-1] == 4:
            return q.reshape(-1, 4)
        if q.dim() == 3 and q.shape[0] == 4:
            return q.permute(1, 2, 0).reshape(-1, 4)
        raise ValueError(f"Unsupported quaternion shape for metrics: {tuple(q.shape)}")

    @staticmethod
    def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack(
            [
                wa * wb - xa * xb - ya * yb - za * zb,
                wa * xb + xa * wb + ya * zb - za * yb,
                wa * yb - xa * zb + ya * wb + za * xb,
                wa * zb + xa * yb - ya * xb + za * wb,
            ],
            dim=-1,
        )

    @staticmethod
    def _new_stat() -> dict:
        return {"count": 0, "sum": 0.0, "sumsq": 0.0, "max": 0.0}

    @staticmethod
    def _update_stat(stat: dict, values: torch.Tensor):
        if values is None:
            return
        v = values.reshape(-1)
        if v.numel() == 0:
            return
        stat["count"] += int(v.numel())
        stat["sum"] += float(v.sum().item())
        stat["sumsq"] += float((v * v).sum().item())
        vmax = float(v.max().item())
        stat["max"] = vmax if stat["count"] == int(v.numel()) else max(stat["max"], vmax)

    @staticmethod
    def _finalize_stat(stat: dict, prefix: str) -> dict:
        if stat["count"] <= 0:
            return {}
        mean = stat["sum"] / stat["count"]
        var = max(stat["sumsq"] / stat["count"] - mean * mean, 0.0)
        return {
            f"{prefix}_mean": float(mean),
            f"{prefix}_std": float(np.sqrt(var)),
            f"{prefix}_max": float(stat["max"]),
        }

    def _new_metric_state(self) -> dict:
        state = {
            "raw_deg": self._new_stat(),
            "pred_norm_err": self._new_stat(),
        }
        if self.metric_symmetry_enabled and self.sym_quats_inv is not None:
            state["sym_deg"] = self._new_stat()
        return state

    def _symmetry_min_misorientation_deg(self, q_pred_flat: torch.Tensor, q_tgt_flat: torch.Tensor) -> torch.Tensor:
        if self.sym_quats_inv is None:
            return None

        n = q_pred_flat.shape[0]
        if n == 0:
            return q_pred_flat.new_zeros((0,))

        out = torch.empty((n,), dtype=torch.float32, device=q_pred_flat.device)
        chunk = max(int(self.metric_symmetry_chunk_size), 1)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            qp = q_pred_flat[start:end]
            qt = q_tgt_flat[start:end]

            orbit = self._quat_mul(self.sym_quats_inv.unsqueeze(0), qt.unsqueeze(1))  # (Nc,G,4)
            dots = (qp.unsqueeze(1) * orbit).sum(dim=-1).abs()
            dots = torch.clamp(dots, min=0.0, max=1.0 - 1e-8)
            min_ang = 2.0 * torch.acos(dots).min(dim=1).values
            out[start:end] = min_ang * (180.0 / float(np.pi))

        return out

    def _compute_metric_tensors(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> dict:
        qp = self._flatten_quats(q_pred).to(dtype=torch.float32)
        qt = self._flatten_quats(q_target).to(dtype=torch.float32)
        if qp.shape != qt.shape:
            raise ValueError(
                f"Metric flatten mismatch: q_pred {tuple(qp.shape)} vs q_target {tuple(qt.shape)}"
            )

        pred_norm = qp.norm(dim=-1)
        pred_norm_err = (pred_norm - 1.0).abs()  # before normalization

        qp = qp / pred_norm.unsqueeze(-1).clamp_min(1e-12)
        qt = qt / qt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        dots = (qp * qt).sum(dim=-1).abs()
        dots = torch.clamp(dots, min=0.0, max=1.0 - 1e-8)
        raw_deg = 2.0 * torch.acos(dots) * (180.0 / float(np.pi))

        out = {
            "raw_deg": raw_deg,
            "pred_norm_err": pred_norm_err,
        }
        if self.metric_symmetry_enabled and self.sym_quats_inv is not None:
            out["sym_deg"] = self._symmetry_min_misorientation_deg(qp, qt)
        else:
            out["sym_deg"] = None
        return out

    def _quat_norm_regularizer(self, q_pred: torch.Tensor) -> torch.Tensor | None:
        if self.quat_norm_reg_weight <= 0.0:
            return None
        if q_pred.dim() == 4 and q_pred.shape[1] == 4:
            n = q_pred.norm(dim=1)
        elif q_pred.dim() == 4 and q_pred.shape[-1] == 4:
            n = q_pred.norm(dim=-1)
        elif q_pred.dim() == 2 and q_pred.shape[-1] == 4:
            n = q_pred.norm(dim=-1)
        elif q_pred.dim() == 3 and q_pred.shape[0] == 4:
            n = q_pred.norm(dim=0)
        else:
            return None
        n = torch.nan_to_num(n, nan=0.0, posinf=10.0, neginf=0.0)
        return ((n - 1.0) ** 2).mean()

    def _accumulate_metrics(self, state: dict, q_pred: torch.Tensor, q_target: torch.Tensor):
        t = self._compute_metric_tensors(q_pred, q_target)
        self._update_stat(state["raw_deg"], t["raw_deg"])
        self._update_stat(state["pred_norm_err"], t["pred_norm_err"])
        if "sym_deg" in state:
            self._update_stat(state["sym_deg"], t["sym_deg"])

    def _finalize_metrics(self, state: dict) -> dict:
        metrics = {}
        metrics.update(self._finalize_stat(state["raw_deg"], "raw_deg"))
        metrics.update(self._finalize_stat(state["pred_norm_err"], "pred_norm_err"))
        if "sym_deg" in state:
            metrics.update(self._finalize_stat(state["sym_deg"], "sym_deg"))
        return metrics

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
        total_norm_reg = 0.0
        norm_reg_count = 0
        metric_state = self._new_metric_state() if self.metric_compute_train else None

        for lr, hr in self.loaders["train"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            norm_reg = None

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
                    norm_reg = self._quat_norm_regularizer(sr)
                    if norm_reg is not None:
                        loss = loss + self.quat_norm_reg_weight * norm_reg

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr)
                norm_reg = self._quat_norm_regularizer(sr)
                if norm_reg is not None:
                    loss = loss + self.quat_norm_reg_weight * norm_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["clip"]
                )
                self.optimizer.step()

            if metric_state is not None and not self._is_invariant_sr():
                with torch.no_grad():
                    self._accumulate_metrics(metric_state, sr.detach(), hr.detach())
            if norm_reg is not None:
                total_norm_reg += float(norm_reg.detach().item())
                norm_reg_count += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(self.loaders["train"])
        self.scheduler.step()
        self.writer.add_scalar("Loss/Train", avg_loss, self.epoch)
        metrics = {"loss": float(avg_loss)}
        if metric_state is not None:
            metrics.update(self._finalize_metrics(metric_state))
        if norm_reg_count > 0:
            metrics["norm_reg_loss"] = float(total_norm_reg / norm_reg_count)
        if metric_state is not None or norm_reg_count > 0:
            for key, val in metrics.items():
                if key == "loss":
                    continue
                self.writer.add_scalar(f"Metrics/Train/{key}", val, self.epoch)
        self.last_train_metrics = metrics
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_norm_reg = 0.0
        norm_reg_count = 0
        metric_state = self._new_metric_state() if self.metric_compute_val else None

        for lr, hr in self.loaders["val"]:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            norm_reg = None

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
                    norm_reg = self._quat_norm_regularizer(sr)
                    if norm_reg is not None:
                        loss = loss + self.quat_norm_reg_weight * norm_reg
            else:
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr)
                norm_reg = self._quat_norm_regularizer(sr)
                if norm_reg is not None:
                    loss = loss + self.quat_norm_reg_weight * norm_reg

            if metric_state is not None and not self._is_invariant_sr():
                self._accumulate_metrics(metric_state, sr.detach(), hr.detach())
            if norm_reg is not None:
                total_norm_reg += float(norm_reg.detach().item())
                norm_reg_count += 1

            total_loss += loss.item()

        avg_val_loss = total_loss / len(self.loaders["val"])
        self.writer.add_scalar("Loss/Val", avg_val_loss, self.epoch)
        metrics = {"loss": float(avg_val_loss)}
        if metric_state is not None:
            metrics.update(self._finalize_metrics(metric_state))
        if norm_reg_count > 0:
            metrics["norm_reg_loss"] = float(total_norm_reg / norm_reg_count)
        if metric_state is not None or norm_reg_count > 0:
            for key, val in metrics.items():
                if key == "loss":
                    continue
                self.writer.add_scalar(f"Metrics/Val/{key}", val, self.epoch)
        self.last_val_metrics = metrics
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
