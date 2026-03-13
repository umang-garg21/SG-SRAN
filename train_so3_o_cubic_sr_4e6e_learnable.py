#!/usr/bin/env python3
"""
Train SO(3)/O cubic-quotient SR model (4e/6e hidden fibers) on quaternion SR datasets.

This script is config-driven and uses:
  - model/codec from `so3_o_cubic_sr.py`
  - dataset loader from `training.data_loading.build_dataloader`
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from so3_o_cubic_sr import (
    descriptor_to_local_affinity,
    normalize_quaternion,
    standardize_quaternion_sign,
)
from training.data_loading import build_dataloader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    # data
    dataset_root: str = ""
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    preload: bool = False
    preload_torch: bool = False
    train_take_first: Optional[int] = None
    val_take_first: Optional[int] = None

    # optimization
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    min_lr: float = 1e-6

    # loss
    lam_shell: float = 1e-2
    lam_boundary_bce: float = 1.0
    lam_boundary_dice: float = 0.5
    lam_affinity: float = 0.25
    lam_affinity0: float = 0.1
    affinity_target_tau: float = 0.0

    # model
    hidden_mul4: int = 16
    hidden_mul6: int = 8
    num_blocks_lr: int = 3
    num_blocks_hr: int = 3
    upsample_mode: str = "masked_learnable"
    sr_scale: int = 4
    passive_input: bool = True
    affinity_kernel_size: int = 3
    use_boundary_prior: bool = True
    learnable_version: str = "v1"  # "v1" (so3_o_cubic_sr_4e6e_learnable) or "v2"

    # misc
    seed: int = 42
    dtype: str = "float32"  # "float32" or "float64"
    checkpoint_dir: str = ""
    save_every: int = 1
    log_every: int = 1
    resume_from: str = ""
    use_tqdm: bool = True
    print_model_summary: bool = True

    # optional val decode
    val_decode_k: int = 0
    decode_dict_size: int = 20000
    decode_chunk: int = 2048

    # periodic visualization
    viz_every: int = 5
    viz_split: str = "Test"
    viz_max_samples: int = 10
    viz_batch_size: int = 1
    viz_num_workers: int = 0
    viz_ref_dir: str = "ALL"
    viz_decode_dict_size: int = 20000
    viz_decode_chunk: int = 2048
    viz_decode_topk: int = 1
    viz_decode_refine_steps: int = 0
    viz_decode_refine_lr: float = 1e-2
    viz_dict_sampling: str = "fz"
    viz_dict_fz_resolution: int = 3
    viz_dict_fz_method: str = "cubochoric"
    viz_dict_fz_point_group: str = "O"


def get_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_learnable_api(version: str) -> Dict[str, Any]:
    v = str(version).strip().lower()
    if v in ("", "v1", "learnable", "baseline"):
        module_name = "so3_o_cubic_sr_4e6e_learnable"
    elif v in ("v2", "nextgen", "learnable_v2"):
        module_name = "so3_o_cubic_sr_4e6e_learnable_v2"
    else:
        raise ValueError(
            f"Unknown learnable_version '{version}'. Use one of: v1, v2."
        )

    module = importlib.import_module(module_name)
    required = (
        "SO3OQuotientSRNet4e6e",
        "affinity_to_boundary",
        "combined_boundary_aware_quotient_loss",
    )
    for name in required:
        if not hasattr(module, name):
            raise AttributeError(
                f"Module '{module_name}' is missing required symbol '{name}'."
            )

    return {
        "module_name": module_name,
        "model_class": getattr(module, "SO3OQuotientSRNet4e6e"),
        "affinity_to_boundary": getattr(module, "affinity_to_boundary"),
        "combined_loss": getattr(module, "combined_boundary_aware_quotient_loss"),
    }


def print_model_summary(model: nn.Module) -> None:
    """
    Print leaf-module parameter summary and global parameter totals.
    """
    header = f"{'Layer':<58} {'Type':<30} {'Trainable':>12} {'Params':>12}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for name, module in model.named_modules():
        if name == "":
            continue
        if any(True for _ in module.children()):
            continue

        n_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        n_total = sum(p.numel() for p in module.parameters(recurse=False))
        if n_total == 0:
            continue

        layer_name = name if len(name) <= 58 else f"...{name[-55:]}"
        print(f"{layer_name:<58} {module.__class__.__name__:<30} {n_trainable:>12,d} {n_total:>12,d}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    print("-" * len(header))
    print(f"{'Trainable params':<58} {'':<30} {trainable_params:>12,d} {trainable_params:>12,d}")
    print(f"{'Non-trainable params':<58} {'':<30} {0:>12,d} {non_trainable:>12,d}")
    print(f"{'Total params':<58} {'':<30} {trainable_params:>12,d} {total_params:>12,d}")
    print("=" * len(header))


def _direct_config_override(cfg: TrainConfig, raw: Dict[str, Any]) -> None:
    field_names = {f.name for f in fields(TrainConfig)}
    for k, v in raw.items():
        if k in field_names:
            setattr(cfg, k, v)


def _apply_aliases(cfg: TrainConfig, raw: Dict[str, Any]) -> None:
    aliases = {
        "scale": "sr_scale",
        "clip": "grad_clip",
        "checkpoints_dir": "checkpoint_dir",
    }
    field_names = {f.name for f in fields(TrainConfig)}

    for src, dst in aliases.items():
        if src in raw and dst in field_names and dst not in raw:
            setattr(cfg, dst, raw[src])


def _apply_nested_overrides(cfg: TrainConfig, raw: Dict[str, Any]) -> None:
    model_cfg = raw.get("model", {})
    if isinstance(model_cfg, dict):
        for name in (
            "hidden_mul4",
            "hidden_mul6",
            "num_blocks_lr",
            "num_blocks_hr",
            "upsample_mode",
            "sr_scale",
            "passive_input",
            "affinity_kernel_size",
            "use_boundary_prior",
            "learnable_version",
        ):
            if name in model_cfg:
                setattr(cfg, name, model_cfg[name])

    loss_cfg = raw.get("loss", {})
    if isinstance(loss_cfg, dict):
        for name in (
            "lam_shell",
            "lam_boundary_bce",
            "lam_boundary_dice",
            "lam_affinity",
            "lam_affinity0",
            "affinity_target_tau",
        ):
            if name in loss_cfg and name not in raw:
                setattr(cfg, name, loss_cfg[name])

    optim_cfg = raw.get("optimizer", {})
    if isinstance(optim_cfg, dict):
        if "lr" in optim_cfg and "lr" not in raw:
            cfg.lr = optim_cfg["lr"]
        if "weight_decay" in optim_cfg and "weight_decay" not in raw:
            cfg.weight_decay = optim_cfg["weight_decay"]

    sched_cfg = raw.get("scheduler", {})
    if isinstance(sched_cfg, dict):
        if "min_lr" in sched_cfg and "min_lr" not in raw:
            cfg.min_lr = sched_cfg["min_lr"]

    viz_cfg = raw.get("visualization", {})
    if isinstance(viz_cfg, dict):
        for name in (
            "viz_every",
            "viz_split",
            "viz_max_samples",
            "viz_batch_size",
            "viz_num_workers",
            "viz_ref_dir",
            "viz_decode_dict_size",
            "viz_decode_chunk",
            "viz_decode_topk",
            "viz_decode_refine_steps",
            "viz_decode_refine_lr",
            "viz_dict_sampling",
            "viz_dict_fz_resolution",
            "viz_dict_fz_method",
            "viz_dict_fz_point_group",
        ):
            if name in viz_cfg and name not in raw:
                setattr(cfg, name, viz_cfg[name])


def load_train_config(config_path: Path, exp_dir: Path) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = TrainConfig()
    _direct_config_override(cfg, raw)
    _apply_aliases(cfg, raw)
    _apply_nested_overrides(cfg, raw)

    if not cfg.dataset_root:
        raise ValueError(
            "Config must set `dataset_root` for dataset-backed training."
        )

    if not cfg.checkpoint_dir:
        if str(cfg.learnable_version).strip().lower() in ("v2", "nextgen", "learnable_v2"):
            cfg.checkpoint_dir = str(exp_dir / "checkpoints_so3o_sr_4e6e_learnable_v2")
        else:
            cfg.checkpoint_dir = str(exp_dir / "checkpoints_so3o_sr_4e6e_learnable")

    return cfg


def _extract_lr_hr_from_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        lr = batch.get("lr_q", batch.get("lr"))
        hr = batch.get("hr_q", batch.get("hr"))
        if lr is None or hr is None:
            raise KeyError(
                "Batch dict must contain lr/hr tensors under keys "
                "`lr_q`/`hr_q` or `lr`/`hr`."
            )
        return lr, hr

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _to_channel_first_quat_map(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 4:
        raise ValueError(f"Expected [B,4,H,W] or [B,H,W,4], got {tuple(q.shape)}")

    if q.shape[1] == 4:
        return q.contiguous()
    if q.shape[-1] == 4:
        return q.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"Quaternion channel dimension not found in shape {tuple(q.shape)}")


def _normalize_map_channel_first(q: torch.Tensor) -> torch.Tensor:
    q_last = q.permute(0, 2, 3, 1).contiguous()
    q_last = standardize_quaternion_sign(normalize_quaternion(q_last))
    return q_last.permute(0, 3, 1, 2).contiguous()


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lr_q, hr_q = _extract_lr_hr_from_batch(batch)

    lr_q = _to_channel_first_quat_map(lr_q).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    hr_q = _to_channel_first_quat_map(hr_q).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )

    # Normalize and fix q ~ -q branch for both LR and HR maps.
    lr_q = _normalize_map_channel_first(lr_q)
    hr_q = _normalize_map_channel_first(hr_q)
    return lr_q, hr_q


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split="Train",
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        preload=cfg.preload,
        preload_torch=cfg.preload_torch,
        take_first=cfg.train_take_first,
        seed=cfg.seed,
    )

    val_loader: Optional[DataLoader] = None
    try:
        val_loader = build_dataloader(
            dataset_root=cfg.dataset_root,
            split="Val",
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            preload=cfg.preload,
            preload_torch=cfg.preload_torch,
            take_first=cfg.val_take_first,
            seed=cfg.seed,
        )
    except Exception as exc:
        print(f"[warn] Validation split unavailable, continuing without validation: {exc}")

    return train_loader, val_loader


@torch.no_grad()
def build_hr_descriptor_target(model: nn.Module, hr_q: torch.Tensor) -> torch.Tensor:
    return model.codec.encode_map(hr_q)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    best_val: float,
    cfg: TrainConfig,
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    strict_model_load: bool = False,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    incompatible = model.load_state_dict(
        ckpt["model_state_dict"],
        strict=strict_model_load,
    )
    if not strict_model_load:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            print(f"[warn] Missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading checkpoint: {len(unexpected)}")

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


@torch.no_grad()
def descriptor_metrics(x_pred: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((x_pred - x_gt) ** 2).item()

    r_pred = torch.sqrt((x_pred ** 2).sum(dim=1).clamp_min(1e-12))
    r_gt = torch.sqrt((x_gt ** 2).sum(dim=1).clamp_min(1e-12))

    shell_mae = torch.mean(torch.abs(r_pred - r_gt)).item()
    desc_l2 = torch.mean(torch.norm(x_pred - x_gt, dim=1)).item()

    return {
        "mse": mse,
        "shell_mae": shell_mae,
        "desc_l2": desc_l2,
    }


def _compute_step_losses(
    model: nn.Module,
    pred: Dict[str, torch.Tensor],
    hr_q_gt: torch.Tensor,
    cfg: TrainConfig,
    affinity_to_boundary_fn,
    combined_loss_fn,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        x_gt = model.codec.encode_map(hr_q_gt)

    k2 = int(pred["affinity_hr"].shape[1])
    kernel_size = int(round(k2 ** 0.5))
    if kernel_size * kernel_size != k2:
        raise ValueError(f"Invalid affinity channel count {k2}, expected perfect square.")
    center_idx = k2 // 2
    tau_tgt = cfg.affinity_target_tau if cfg.affinity_target_tau > 0 else 0.6

    with torch.no_grad():
        aff_gt = descriptor_to_local_affinity(
            x_desc=x_gt,
            kernel_size=kernel_size,
            tau=tau_tgt,
            eps=1e-6,
        )
        boundary_gt = affinity_to_boundary_fn(aff_gt, center_idx=center_idx)

    losses = combined_loss_fn(
        model=model,
        pred=pred,
        hr_q_gt=hr_q_gt,
        gt_boundary_hr=boundary_gt,
        lam_shell=cfg.lam_shell,
        lam_boundary_bce=cfg.lam_boundary_bce,
        lam_boundary_dice=cfg.lam_boundary_dice,
        lam_affinity=cfg.lam_affinity,
        lam_affinity0=cfg.lam_affinity0,
    )

    out: Dict[str, torch.Tensor] = {
        "loss_total": losses["total"],
        "loss_desc": losses["descriptor"],
        "loss_shell": losses["shell"],
        "metric_boundary_hr_gt_mean": boundary_gt.mean(),
        "metric_boundary_hr_pred_mean": pred["boundary_hr"].mean(),
    }

    if "boundary_bce" in losses:
        out["loss_boundary_bce"] = losses["boundary_bce"]
    if "boundary_dice" in losses:
        out["loss_boundary_dice"] = losses["boundary_dice"]
    if "affinity" in losses:
        out["loss_affinity_hr"] = losses["affinity"]
    if "affinity0" in losses:
        out["loss_affinity0_hr"] = losses["affinity0"]

    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainConfig,
    model_api: Dict[str, Any],
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_shell = 0.0
    total_bdry = 0.0
    total_aff = 0.0
    total_couple = 0.0
    total_bdry_gt_mean = 0.0
    total_bdry_pred_mean = 0.0
    n_aux_batches = 0
    n_batches = 0

    iterator = loader
    pbar = None
    if cfg.use_tqdm:
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Train {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = pbar

    for batch in iterator:
        lr_q, hr_q = move_batch_to_device(batch, device, dtype)

        with torch.no_grad():
            x_gt = build_hr_descriptor_target(model, hr_q)

        pred = model(lr_q)
        x_pred = pred["descriptor_hr"]
        if x_pred.shape != x_gt.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch: pred={tuple(x_pred.shape)} "
                f"gt={tuple(x_gt.shape)}. Check sr_scale vs dataset LR/HR scale."
            )

        loss_terms = _compute_step_losses(
            model=model,
            pred=pred,
            hr_q_gt=hr_q,
            cfg=cfg,
            affinity_to_boundary_fn=model_api["affinity_to_boundary"],
            combined_loss_fn=model_api["combined_loss"],
        )

        loss = loss_terms["loss_total"]
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            print("[warn] Non-finite loss detected, skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(loss_terms["loss_desc"].item())
        total_shell += float(loss_terms["loss_shell"].item())

        if "loss_boundary_bce" in loss_terms:
            total_bdry += float(loss_terms["loss_boundary_bce"].item())
            total_aff += float(loss_terms["loss_affinity_hr"].item())
            total_couple += float(loss_terms["loss_affinity0_hr"].item())
            total_bdry_gt_mean += float(loss_terms["metric_boundary_hr_gt_mean"].item())
            total_bdry_pred_mean += float(loss_terms["metric_boundary_hr_pred_mean"].item())
            n_aux_batches += 1
        n_batches += 1

        if pbar is not None:
            if n_aux_batches > 0:
                pbar.set_postfix(
                    loss=f"{(total_loss / n_batches):.3e}",
                    mse=f"{(total_mse / n_batches):.3e}",
                    bdry=f"{(total_bdry / n_aux_batches):.3e}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{(total_loss / n_batches):.3e}",
                    mse=f"{(total_mse / n_batches):.3e}",
                )

    denom = max(n_batches, 1)
    out = {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "shell": total_shell / denom,
    }
    if n_aux_batches > 0:
        aux_denom = float(n_aux_batches)
        out["boundary_hr"] = total_bdry / aux_denom
        out["affinity_hr"] = total_aff / aux_denom
        out["affinity_couple"] = total_couple / aux_denom
        out["boundary_hr_gt_mean"] = total_bdry_gt_mean / aux_denom
        out["boundary_hr_pred_mean"] = total_bdry_pred_mean / aux_denom
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainConfig,
    model_api: Dict[str, Any],
    epoch: int = 0,
    q_dict: Optional[torch.Tensor] = None,
    x_dict: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_shell = 0.0
    total_desc_l2 = 0.0
    total_bdry = 0.0
    total_aff = 0.0
    total_couple = 0.0
    total_bdry_gt_mean = 0.0
    total_bdry_pred_mean = 0.0
    n_aux_batches = 0
    n_batches = 0

    decode_runs = 0
    total_decode_nn = 0.0
    remaining_decode = cfg.val_decode_k

    iterator = loader
    pbar = None
    if cfg.use_tqdm:
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Val   {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = pbar

    for batch in iterator:
        lr_q, hr_q = move_batch_to_device(batch, device, dtype)

        x_gt = build_hr_descriptor_target(model, hr_q)
        pred = model(lr_q)
        x_pred = pred["descriptor_hr"]

        loss_terms = _compute_step_losses(
            model=model,
            pred=pred,
            hr_q_gt=hr_q,
            cfg=cfg,
            affinity_to_boundary_fn=model_api["affinity_to_boundary"],
            combined_loss_fn=model_api["combined_loss"],
        )
        loss = loss_terms["loss_total"]
        mets = descriptor_metrics(x_pred, x_gt)

        total_loss += float(loss.item())
        total_mse += float(mets["mse"])
        total_shell += float(loss_terms["loss_shell"].item())
        total_desc_l2 += float(mets["desc_l2"])
        if "loss_boundary_bce" in loss_terms:
            total_bdry += float(loss_terms["loss_boundary_bce"].item())
            total_aff += float(loss_terms["loss_affinity_hr"].item())
            total_couple += float(loss_terms["loss_affinity0_hr"].item())
            total_bdry_gt_mean += float(loss_terms["metric_boundary_hr_gt_mean"].item())
            total_bdry_pred_mean += float(loss_terms["metric_boundary_hr_pred_mean"].item())
            n_aux_batches += 1
        n_batches += 1

        if pbar is not None:
            if n_aux_batches > 0:
                pbar.set_postfix(
                    loss=f"{(total_loss / n_batches):.3e}",
                    mse=f"{(total_mse / n_batches):.3e}",
                    bdry=f"{(total_bdry / n_aux_batches):.3e}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{(total_loss / n_batches):.3e}",
                    mse=f"{(total_mse / n_batches):.3e}",
                )

        if remaining_decode > 0 and q_dict is not None:
            b = min(remaining_decode, lr_q.shape[0])
            x_pred_small = x_pred[:b].permute(0, 2, 3, 1).contiguous()  # [b,sH,sW,9]
            dec = model.codec.decode_by_dictionary(
                x_pred_small,
                q_dict=q_dict,
                x_dict=x_dict,
                chunk=cfg.decode_chunk,
            )
            total_decode_nn += float(dec.distances.mean().item())
            decode_runs += 1
            remaining_decode -= b

    denom = max(n_batches, 1)
    out = {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "shell": total_shell / denom,
        "desc_l2": total_desc_l2 / denom,
    }
    if n_aux_batches > 0:
        aux_denom = float(n_aux_batches)
        out["boundary_hr"] = total_bdry / aux_denom
        out["affinity_hr"] = total_aff / aux_denom
        out["affinity_couple"] = total_couple / aux_denom
        out["boundary_hr_gt_mean"] = total_bdry_gt_mean / aux_denom
        out["boundary_hr_pred_mean"] = total_bdry_pred_mean / aux_denom
    if decode_runs > 0:
        out["decode_nn"] = total_decode_nn / decode_runs

    return out


def _run_epoch_visualization(
    exp_dir: Path,
    cfg: TrainConfig,
    epoch: int,
    checkpoint_path: Path,
) -> None:
    if cfg.viz_every <= 0 or (epoch % cfg.viz_every) != 0:
        return

    script_path = Path(__file__).resolve().parent / "scripts" / "visualize_so3o_cubic_sr_results.py"
    out_dir = exp_dir / "visualizations" / f"epoch_{epoch:04d}"

    cmd = [
        sys.executable,
        str(script_path),
        "--exp_dir",
        str(exp_dir),
        "--ckpt",
        str(checkpoint_path.resolve()),
        "--split",
        str(cfg.viz_split),
        "--max_samples",
        str(int(cfg.viz_max_samples)),
        "--batch_size",
        str(int(cfg.viz_batch_size)),
        "--num_workers",
        str(int(cfg.viz_num_workers)),
        "--decode_dict_size",
        str(int(cfg.viz_decode_dict_size)),
        "--decode_chunk",
        str(int(cfg.viz_decode_chunk)),
        "--decode_topk",
        str(int(cfg.viz_decode_topk)),
        "--decode_refine_steps",
        str(int(cfg.viz_decode_refine_steps)),
        "--decode_refine_lr",
        str(float(cfg.viz_decode_refine_lr)),
        "--dict_sampling",
        str(cfg.viz_dict_sampling),
        "--dict_fz_resolution",
        str(int(cfg.viz_dict_fz_resolution)),
        "--dict_fz_method",
        str(cfg.viz_dict_fz_method),
        "--dict_fz_point_group",
        str(cfg.viz_dict_fz_point_group),
        "--ref_dir",
        str(cfg.viz_ref_dir),
        "--output_dir",
        str(out_dir),
    ]
    print(f"[viz] Epoch {epoch}: rendering {cfg.viz_split} samples to {out_dir}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[warn] Visualization failed at epoch {epoch}: {exc}")


def _resolve_resume_path(
    checkpoint_dir: Path,
    resume: bool,
    resume_from: str = "",
) -> Optional[Path]:
    if resume_from:
        p = Path(resume_from)
        if not p.is_absolute():
            p = checkpoint_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {p}")
        return p

    if not resume:
        return None

    for candidate in (checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"):
        if candidate.exists():
            return candidate
    return None


def train_from_dataset(
    cfg: TrainConfig,
    exp_dir: Path,
    resume: bool = False,
    resume_from: str = "",
) -> Tuple[nn.Module, Dict[str, list]]:
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.dtype)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_dataloaders(cfg)
    model_api = resolve_learnable_api(cfg.learnable_version)
    model_cls = model_api["model_class"]

    model = model_cls(
        hidden_mul4=cfg.hidden_mul4,
        hidden_mul6=cfg.hidden_mul6,
        num_blocks_lr=cfg.num_blocks_lr,
        num_blocks_hr=cfg.num_blocks_hr,
        upsample_mode=cfg.upsample_mode,
        sr_scale=cfg.sr_scale,
        passive_input=cfg.passive_input,
        affinity_kernel_size=cfg.affinity_kernel_size,
        use_boundary_prior=cfg.use_boundary_prior,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    if cfg.print_model_summary:
        print_model_summary(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.min_lr,
    )

    start_epoch = 1
    best_val = float("inf")
    history = {
        "train": [],
        "val": [],
    }

    resume_path = _resolve_resume_path(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        resume_from=resume_from or cfg.resume_from,
    )
    if resume_path is not None:
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {resume_path} at epoch {start_epoch - 1}")

    q_dict = None
    x_dict = None
    if cfg.val_decode_k > 0 and val_loader is not None:
        q_dict, x_dict = model.codec.build_dictionary(
            n=cfg.decode_dict_size,
            device=device,
            dtype=dtype,
        )

    with open(checkpoint_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Learnable module: {model_api['module_name']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {0 if val_loader is None else len(val_loader)}")
    print(f"SR scale: {cfg.sr_scale}")
    if hasattr(model, "up") and hasattr(model.up, "factors"):
        print(f"Transpose upsample factors: {model.up.factors}")

    if start_epoch > cfg.epochs:
        print(
            f"Nothing to train: start_epoch ({start_epoch}) > epochs ({cfg.epochs})."
        )
        return model, history

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            cfg=cfg,
            model_api=model_api,
            epoch=epoch,
        )
        history["train"].append(train_stats)

        if val_loader is not None:
            val_stats = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                dtype=dtype,
                cfg=cfg,
                model_api=model_api,
                epoch=epoch,
                q_dict=q_dict,
                x_dict=x_dict,
            )
            history["val"].append(val_stats)

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                save_checkpoint(
                    path=checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val=best_val,
                    cfg=cfg,
                )

            if epoch % max(cfg.log_every, 1) == 0:
                msg = (
                    f"[{epoch:03d}/{cfg.epochs:03d}] "
                    f"train loss={train_stats['loss']:.6e} "
                    f"train mse={train_stats['mse']:.6e} "
                    f"val loss={val_stats['loss']:.6e} "
                    f"val mse={val_stats['mse']:.6e} "
                    f"val desc_l2={val_stats['desc_l2']:.6e}"
                )
                if "boundary_hr" in train_stats:
                    msg += f" train bdry={train_stats['boundary_hr']:.6e}"
                    msg += f" train aff={train_stats['affinity_hr']:.6e}"
                if "boundary_hr" in val_stats:
                    msg += f" val bdry={val_stats['boundary_hr']:.6e}"
                    msg += f" val aff={val_stats['affinity_hr']:.6e}"
                if "decode_nn" in val_stats:
                    msg += f" val decode_nn={val_stats['decode_nn']:.6e}"
                print(msg)
        else:
            if epoch % max(cfg.log_every, 1) == 0:
                msg = (
                    f"[{epoch:03d}/{cfg.epochs:03d}] "
                    f"train loss={train_stats['loss']:.6e} "
                    f"train mse={train_stats['mse']:.6e}"
                )
                if "boundary_hr" in train_stats:
                    msg += f" train bdry={train_stats['boundary_hr']:.6e}"
                    msg += f" train aff={train_stats['affinity_hr']:.6e}"
                print(msg)

        epoch_ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        should_save_epoch = (epoch % max(cfg.save_every, 1) == 0) or (
            cfg.viz_every > 0 and (epoch % cfg.viz_every == 0)
        )
        if should_save_epoch:
            save_checkpoint(
                path=epoch_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val,
                cfg=cfg,
            )
            _run_epoch_visualization(
                exp_dir=exp_dir,
                cfg=cfg,
                epoch=epoch,
                checkpoint_path=epoch_ckpt_path,
            )

        scheduler.step()

        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(
        path=checkpoint_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        best_val=best_val,
        cfg=cfg,
    )

    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SO(3)/O cubic quotient SR model (4e6e)"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory containing config JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt (or best.pt if last.pt is missing).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional explicit checkpoint path (absolute or relative to checkpoint_dir).",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. '0' or '0,1'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    cfg = load_train_config(config_path=config_path, exp_dir=exp_dir)
    if args.resume_from:
        cfg.resume_from = args.resume_from

    _, history = train_from_dataset(
        cfg=cfg,
        exp_dir=exp_dir,
        resume=args.resume,
        resume_from=args.resume_from,
    )

    if history["train"]:
        print("Last train stats:", history["train"][-1])
    if history["val"]:
        print("Last val stats:", history["val"][-1])


if __name__ == "__main__":
    main()
