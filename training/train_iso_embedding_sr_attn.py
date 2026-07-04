# -*- coding:utf-8 -*-
"""
Train quaternion SR models with local-iso feature-space SR loss.

This trainer is shared across the repo's SR variants and optimizes each
model's `feature_loss_sr(...)` directly. The concrete model class is resolved
from the config's `model.model_module` / `model.model_class` fields or from
the CLI overrides.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.SR_ocrp import resolve_ocrp_upsample_residual_weight
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import get_seed_from_config, set_seed
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quaternion SR model")
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
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1'",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="models.SR_double_conv_SRattn",
        help="Dotted module path containing the model class (default: models.SR_double_conv_SRattn)",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="IsoEmbeddingSRAttn",
        help="Class name to instantiate from model_module (default: IsoEmbeddingSRAttn)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last_checkpoint.pt if present",
    )
    return parser.parse_args()


def _flatten_quat_chw_batch(q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert (B,4,H,W) or (B,H,W,4) quaternion batch to (B,H*W,4)."""
    if q.dim() != 4:
        raise ValueError(f"Expected rank-4 quaternion batch, got {tuple(q.shape)}")
    if q.shape[1] == 4:
        bsz, _, h, w = q.shape
        q_flat = q.permute(0, 2, 3, 1).reshape(bsz, h * w, 4)
        return q_flat, (h, w)
    if q.shape[-1] == 4:
        bsz, h, w, _ = q.shape
        q_flat = q.reshape(bsz, h * w, 4)
        return q_flat, (h, w)
    raise ValueError(f"Expected quaternion axis of size 4 in dim=1 or dim=-1, got {tuple(q.shape)}")


def _to_hwc_quat_single(q: torch.Tensor) -> torch.Tensor:
    """Convert single quaternion image to (H,W,4) from (4,H,W) or (H,W,4)."""
    if q.dim() != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {tuple(q.shape)}")
    if q.shape[0] == 4:
        return q.permute(1, 2, 0)
    if q.shape[-1] == 4:
        return q
    raise ValueError(f"Expected quaternion axis of size 4 in dim=0 or dim=-1, got {tuple(q.shape)}")


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Accept dataloader batches as:
      - (lr, hr)
      - (lr, hr, lr_boundary_map)
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) == 2:
        lr, hr = batch
        return lr, hr, None
    if len(batch) == 3:
        lr, hr, lr_boundary_map = batch
        return lr, hr, lr_boundary_map
    raise ValueError(f"Expected batch length 2 or 3, got {len(batch)}")


def _get_take_first(cfg, split: str):
    """Resolve optional dataset truncation for split."""
    if bool(getattr(cfg, "smoke_test", False)):
        return int(getattr(cfg, "smoke_take_first", 8))
    key = f"{split.lower()}_take_first"
    val = getattr(cfg, key, None)
    return int(val) if val is not None else None


_VIZ_SAMPLE_NAME_RE = re.compile(
    r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
    re.IGNORECASE,
)


def _sample_pair_key_from_path(path_str: str | os.PathLike[str]) -> str | None:
    path = Path(path_str)
    match = _VIZ_SAMPLE_NAME_RE.match(path.name)
    if match is None:
        return None
    return f"{match.group('axis').lower()}_{int(match.group('id'))}"


def _resolve_viz_sample_index(
    data_loader,
    sample_index: int,
    sample_key: str | None = None,
) -> tuple[int, dict[str, object] | None]:
    resolved_index = max(0, int(sample_index))
    dataset = getattr(data_loader, "dataset", None)
    pairs = getattr(dataset, "pairs", None)
    if sample_key is None or pairs is None:
        return resolved_index, None

    sample_key_norm = str(sample_key).strip().lower()
    if not sample_key_norm:
        return resolved_index, None

    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            continue
        lr_path = str(pair[0])
        hr_path = str(pair[1])
        pair_key = _sample_pair_key_from_path(lr_path) or _sample_pair_key_from_path(hr_path)
        if pair_key == sample_key_norm:
            return int(idx), {
                "pair_key": pair_key,
                "lr_path": lr_path,
                "hr_path": hr_path,
            }

    print(
        f"[warning] viz_sample_key={sample_key_norm!r} not found in dataset pairs; "
        f"falling back to viz_sample_index={resolved_index}."
    )
    return resolved_index, None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _sync_module_device_attrs(model: torch.nn.Module, device: torch.device) -> None:
    """Best-effort sync of custom `.device` attrs used by some model components."""
    dev = torch.device(device)
    for module in model.modules():
        if hasattr(module, "device"):
            try:
                setattr(module, "device", dev)
            except Exception:
                # Some modules may expose read-only attributes.
                pass


def _module_param_rows(model: torch.nn.Module) -> list[tuple[str, str, int, int]]:
    model_core = _unwrap_model(model)
    rows: list[tuple[str, str, int, int]] = []
    for module_name, module in model_core.named_modules():
        params = list(module.parameters(recurse=False))
        direct_total = sum(p.numel() for p in params)
        direct_trainable = sum(p.numel() for p in params if p.requires_grad)
        if module_name and direct_total == 0:
            continue
        display_name = "<root>" if not module_name else module_name
        rows.append((display_name, module.__class__.__name__, direct_total, direct_trainable))
    return rows


def _print_model_summary(model: torch.nn.Module, max_rows: int | None = None) -> None:
    model_core = _unwrap_model(model)
    print("Model architecture:")
    print(model_core)

    rows = _module_param_rows(model_core)
    total_params = sum(p.numel() for p in model_core.parameters())
    trainable_params = sum(p.numel() for p in model_core.parameters() if p.requires_grad)

    print("Layer parameter summary (direct params per module):")
    print(f"{'module':<72} {'type':<28} {'params':>12} {'trainable':>12}")
    print("-" * 130)

    rows_to_show = rows
    if max_rows is not None and max_rows > 0:
        rows_to_show = rows[:max_rows]

    for name, cls_name, n_params, n_trainable in rows_to_show:
        print(f"{name:<72} {cls_name:<28} {n_params:>12,} {n_trainable:>12,}")

    if len(rows_to_show) < len(rows):
        hidden = len(rows) - len(rows_to_show)
        print(f"... {hidden} more modules omitted (set model_summary_max_rows to a larger value to show all)")

    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")


def _save_history(history_path: Path, history: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _init_history() -> dict:
    return {
        "train": [],
        "val": [],
        "lr": [],
        "ocrp_upsample_residual_weight": [],
        "train_terms": {},
        "val_terms": {},
    }


def _normalize_history(history: dict | None) -> dict:
    base = _init_history()
    if history is None:
        return base

    base["train"] = list(history.get("train", []))
    base["val"] = list(history.get("val", []))
    base["lr"] = list(history.get("lr", []))
    base["ocrp_upsample_residual_weight"] = list(
        history.get("ocrp_upsample_residual_weight", [])
    )
    base["train_terms"] = {
        str(k): list(v) for k, v in dict(history.get("train_terms", {})).items()
    }
    base["val_terms"] = {
        str(k): list(v) for k, v in dict(history.get("val_terms", {})).items()
    }
    return base


def _apply_ocrp_upsample_residual_schedule(
    model_core: torch.nn.Module,
    cfg,
    *,
    epoch: int,
    total_epochs: int,
) -> float | None:
    if not bool(getattr(cfg, "ocrp_upsample_residual", False)):
        return None
    if not hasattr(model_core, "ocrp"):
        return None

    weight = float(
        resolve_ocrp_upsample_residual_weight(
            cfg,
            epoch=epoch,
            total_epochs=total_epochs,
            for_training=True,
        )
    )
    if hasattr(model_core, "set_ocrp_upsample_residual_weight"):
        model_core.set_ocrp_upsample_residual_weight(weight)
        return weight

    ocrp_module = getattr(model_core, "ocrp", None)
    if hasattr(ocrp_module, "set_upsample_residual_weight"):
        ocrp_module.set_upsample_residual_weight(weight)
        return weight
    if ocrp_module is not None and hasattr(ocrp_module, "upsample_residual_weight"):
        ocrp_module.upsample_residual_weight = weight
        return weight
    return None


def _append_metric_history(metric_history: dict[str, list], metrics: dict[str, float], prior_len: int) -> None:
    keys = set(metric_history.keys()) | set(metrics.keys())
    for key in keys:
        if key not in metric_history:
            metric_history[key] = [None] * prior_len
        value = metrics.get(key, None)
        metric_history[key].append(float(value) if value is not None else None)


def _loss_output_to_tensor_and_info(loss_output) -> tuple[torch.Tensor, dict[str, float]]:
    if (
        isinstance(loss_output, tuple)
        and len(loss_output) == 2
        and isinstance(loss_output[1], dict)
    ):
        loss, raw_info = loss_output
    else:
        loss, raw_info = loss_output, {}

    info: dict[str, float] = {}
    for key, value in raw_info.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            info[str(key)] = float(value.detach().item())
        elif isinstance(value, (int, float)):
            info[str(key)] = float(value)
    return loss, info


def _save_loss_plot(plot_path: Path, history: dict, exp_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = history.get("train", [])
        val   = history.get("val", [])
        lr    = history.get("lr", [])
        train_terms = history.get("train_terms", {})
        val_terms = history.get("val_terms", {})
        epochs = list(range(1, len(train) + 1))
        component_names = sorted(
            key
            for key in (set(train_terms.keys()) | set(val_terms.keys()))
            if key != "loss_total"
        )

        nrows = 3 if component_names else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 8 if component_names else 7), sharex=True)
        if nrows == 2:
            ax1, ax3 = axes
            ax2 = None
        else:
            ax1, ax2, ax3 = axes
        fig.suptitle(exp_name, fontsize=11)

        ax1.plot(epochs, train, label="train", linewidth=1.5)
        ax1.plot(epochs, val,   label="val",   linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Total Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if ax2 is not None:
            color_cycle = plt.rcParams.get("axes.prop_cycle", None)
            colors = color_cycle.by_key().get("color", []) if color_cycle is not None else []
            for idx, name in enumerate(component_names):
                color = colors[idx % len(colors)] if colors else None
                train_vals = train_terms.get(name, [])
                val_vals = val_terms.get(name, [])
                ax2.plot(
                    epochs,
                    train_vals[:len(epochs)],
                    label=f"{name} train",
                    linewidth=1.2,
                    color=color,
                )
                ax2.plot(
                    epochs,
                    val_vals[:len(epochs)],
                    label=f"{name} val",
                    linewidth=1.2,
                    linestyle="--",
                    color=color,
                )
            ax2.set_ylabel("Loss Terms")
            ax2.legend(ncol=2, fontsize=8)
            ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, lr[:len(epochs)], color="tab:orange", linewidth=1.5)
        ax3.set_ylabel("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warning] Could not save loss plot: {exc}")


@torch.no_grad()
def _get_boundary_context_from_lr_map(
    model_core: torch.nn.Module,
    lr_boundary_map_2d: torch.Tensor,
    lr_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """
    Resolve boundary context from an LR boundary map across model API variants.

    Preferred API:
      - _prepare_boundary_context_from_lr_boundary_map(...)

    Backward-compatible API:
      - _prepare_boundary_context(lr_boundary_map=...)
    """
    fn_from_map = getattr(model_core, "_prepare_boundary_context_from_lr_boundary_map", None)
    if callable(fn_from_map):
        return fn_from_map(
            lr_boundary_map=lr_boundary_map_2d,
            lr_shape=lr_shape,
            batch_size=1,
            device=device,
            dtype=dtype,
        )

    fn_prepare = getattr(model_core, "_prepare_boundary_context", None)
    if not callable(fn_prepare):
        return None

    try:
        params = inspect.signature(fn_prepare).parameters
    except (TypeError, ValueError):
        return None

    if "lr_boundary_map" not in params:
        return None

    return fn_prepare(
        lr_boundary_map=lr_boundary_map_2d,
        lr_shape=lr_shape,
        batch_size=1,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def _derive_boundary_debug_maps(
    model_core: torch.nn.Module,
    lr_boundary_map_2d: torch.Tensor,
    lr_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float | None] | None:
    """
    Derive boundary maps for debugging using the model-level boundary pipeline:
      LR boundary -> LR 1px -> HR 1px.
    Returns (lr_boundary_2d, hr_cleaned_boundary_2d, threshold) on CPU.
    """
    ctx = _get_boundary_context_from_lr_map(
        model_core=model_core,
        lr_boundary_map_2d=lr_boundary_map_2d,
        lr_shape=lr_shape,
        device=device,
        dtype=dtype,
    )
    if ctx is not None:
        return (
            ctx["boundary_lr_1px"][0, 0].detach().cpu(),
            ctx["boundary_hr_1px"][0, 0].detach().cpu(),
            None,
        )

    upsampler = getattr(model_core, "upsample_conv", None)
    if (
        upsampler is None
        or not hasattr(upsampler, "_format_lr_boundary_map")
        or not hasattr(upsampler, "_smooth_boundary_to_sdf_like_boundary_prep")
    ):
        return None

    H, W = lr_shape
    boundary_lr = upsampler._format_lr_boundary_map(
        lr_boundary_map=lr_boundary_map_2d,
        batch_size=1,
        lr_shape=(H, W),
        device=device,
        dtype=dtype,
    )

    # Prefer full boundary-prep-aligned 1px pipeline when available.
    if hasattr(upsampler, "_build_hr_1px_boundary_and_maps"):
        boundary_lr_1px, boundary_hr_1px, _, _ = upsampler._build_hr_1px_boundary_and_maps(
            boundary_lr=boundary_lr
        )
        return (
            boundary_lr_1px[0, 0].detach().cpu(),
            boundary_hr_1px[0, 0].detach().cpu(),
            None,
        )

    up_factor = getattr(upsampler, "upsample_factor", (4, 4))
    if isinstance(up_factor, (tuple, list)):
        r_h, r_w = int(up_factor[0]), int(up_factor[1])
    else:
        r_h = r_w = int(up_factor)
    Hr, Wr = H * r_h, W * r_w
    boundary_hr_clean = upsampler._smooth_boundary_to_sdf_like_boundary_prep(
        boundary_lr=boundary_lr,
        hr_shape=(Hr, Wr),
    )
    threshold = getattr(upsampler, "boundary_threshold", None)
    return (
        boundary_lr[0, 0].detach().cpu(),
        boundary_hr_clean[0, 0].detach().cpu(),
        float(threshold) if threshold is not None else None,
    )


@torch.no_grad()
def _derive_boundary_grain_context(
    model_core: torch.nn.Module,
    lr_boundary_map_2d: torch.Tensor,
    lr_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """
    Derive LR/HR grain-id assignments used by boundary-aware upsampling.
    Returns CPU tensors:
      - boundary_lr_1px: (H,W)
      - boundary_hr_1px: (Hr,Wr)
      - lr_labels: (H,W), -1 on LR boundary
      - hr_to_lr_map: (Hr,Wr), -1 on HR boundary/unmapped
    """
    # Preferred path: model-level boundary context pipeline.
    ctx = _get_boundary_context_from_lr_map(
        model_core=model_core,
        lr_boundary_map_2d=lr_boundary_map_2d,
        lr_shape=lr_shape,
        device=device,
        dtype=dtype,
    )
    if ctx is not None:
        need = ("boundary_lr_1px", "boundary_hr_1px", "lr_labels", "hr_to_lr_map")
        if all((k in ctx) and (ctx[k] is not None) for k in need):
            return {
                "boundary_lr_1px": ctx["boundary_lr_1px"][0, 0].detach().cpu(),
                "boundary_hr_1px": ctx["boundary_hr_1px"][0, 0].detach().cpu(),
                "lr_labels": ctx["lr_labels"][0].detach().cpu().to(torch.long),
                "hr_to_lr_map": ctx["hr_to_lr_map"][0].detach().cpu().to(torch.long),
            }

    # Fallback path: upsampler helpers.
    upsampler = getattr(model_core, "upsample_conv", None)
    if (
        upsampler is None
        or not hasattr(upsampler, "_format_lr_boundary_map")
        or not hasattr(upsampler, "_build_hr_1px_boundary_and_maps")
    ):
        return None

    H, W = lr_shape
    boundary_lr = upsampler._format_lr_boundary_map(
        lr_boundary_map=lr_boundary_map_2d,
        batch_size=1,
        lr_shape=(H, W),
        device=device,
        dtype=dtype,
    )
    b_lr_1px, b_hr_1px, hr_to_lr_map, lr_labels = upsampler._build_hr_1px_boundary_and_maps(
        boundary_lr=boundary_lr
    )
    return {
        "boundary_lr_1px": b_lr_1px[0, 0].detach().cpu(),
        "boundary_hr_1px": b_hr_1px[0, 0].detach().cpu(),
        "lr_labels": lr_labels[0].detach().cpu().to(torch.long),
        "hr_to_lr_map": hr_to_lr_map[0].detach().cpu().to(torch.long),
    }


def _save_grain_id_debug_plot(
    out_png: Path,
    lr_labels_2d: torch.Tensor,
    hr_to_lr_map_2d: torch.Tensor,
) -> None:
    """
    Save side-by-side grain-id assignment maps:
      1) LR grain labels
      2) HR grain labels assigned from LR
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:
        print(f"[warning] Could not import matplotlib for grain-id debug plot: {exc}")
        return

    lr_t = lr_labels_2d.detach().cpu().to(torch.long)
    hr_t = hr_to_lr_map_2d.detach().cpu().to(torch.long)
    valid_lr = lr_t[lr_t >= 0]
    valid_hr = hr_t[hr_t >= 0]
    if valid_lr.numel() == 0 and valid_hr.numel() == 0:
        print("[warning] Grain-id debug plot skipped: no valid grain ids found.")
        return

    max_gid = -1
    if valid_lr.numel() > 0:
        max_gid = max(max_gid, int(valid_lr.max().item()))
    if valid_hr.numel() > 0:
        max_gid = max(max_gid, int(valid_hr.max().item()))
    n_labels = max_gid + 1

    # Build a discrete, fixed id->color mapping shared by LR and HR.
    base = np.asarray(plt.get_cmap("tab20").colors, dtype=np.float32)  # (20,3)
    if n_labels <= 20:
        rgb = base[:n_labels]
    else:
        reps = int(np.ceil(n_labels / 20.0))
        rgb = np.tile(base, (reps, 1))[:n_labels]
    cmap = mcolors.ListedColormap(rgb, name="grain_ids_discrete")
    cmap.set_bad(color="black")  # boundary/unmapped (-1)
    bounds = np.arange(-0.5, n_labels + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

    lr = np.ma.masked_where(lr_t.numpy() < 0, lr_t.numpy().astype(np.float32))
    hr = np.ma.masked_where(hr_t.numpy() < 0, hr_t.numpy().astype(np.float32))

    n_lr = int(torch.unique(lr_labels_2d[lr_labels_2d >= 0]).numel())
    n_hr = int(torch.unique(hr_to_lr_map_2d[hr_to_lr_map_2d >= 0]).numel())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180, constrained_layout=True)
    ax0, ax1 = axes
    im0 = ax0.imshow(lr, cmap=cmap, norm=norm, interpolation="nearest")
    ax0.set_title(f"LR grain ids ({lr.shape[0]}x{lr.shape[1]}), grains={n_lr}")
    ax0.axis("off")

    im1 = ax1.imshow(hr, cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_title(f"HR assigned grain ids ({hr.shape[0]}x{hr.shape[1]}), grains={n_hr}")
    ax1.axis("off")

    # Shared, discrete colorbar so the same grain id has the same color in both maps.
    tick_step = 1 if n_labels <= 25 else max(1, int(np.ceil(n_labels / 25)))
    ticks = np.arange(0, n_labels, tick_step, dtype=np.int32)
    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.046, pad=0.02, ticks=ticks)
    cbar.set_label("Grain id")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"Saved grain-id debug plot: {out_png}")


def _build_hr_to_lr_context_masks(
    hr_to_lr_map_2d: torch.Tensor,
    lr_labels_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-HR-pixel LR context masks used by seed attention.

    Returns:
      context_masks: (Hr,Wr,H,W) bool
      parent_y:      (Hr,Wr) long
      parent_x:      (Hr,Wr) long
    Rules:
      - Interior HR pixel (gid>=0): use LR pixels with same gid.
      - HR boundary/unmapped (gid<0): fallback to single parent LR pixel.
    """
    if hr_to_lr_map_2d.ndim != 2 or lr_labels_2d.ndim != 2:
        raise ValueError("Expected rank-2 maps for hr_to_lr_map_2d and lr_labels_2d")

    Hr, Wr = int(hr_to_lr_map_2d.shape[0]), int(hr_to_lr_map_2d.shape[1])
    H, W = int(lr_labels_2d.shape[0]), int(lr_labels_2d.shape[1])

    hr_flat = hr_to_lr_map_2d.reshape(-1)               # (Hr*Wr,)
    lr_flat = lr_labels_2d.reshape(-1)                  # (H*W,)
    context = (hr_flat[:, None] == lr_flat[None, :]) & (hr_flat[:, None] >= 0)

    # Parent mapping for fallback (same floor rule as upsampler).
    y_hr = torch.arange(Hr, dtype=torch.long)
    x_hr = torch.arange(Wr, dtype=torch.long)
    gy, gx = torch.meshgrid(y_hr, x_hr, indexing="ij")
    py = torch.div(gy, max(1, Hr // max(1, H)), rounding_mode="floor").clamp(0, H - 1)
    px = torch.div(gx, max(1, Wr // max(1, W)), rounding_mode="floor").clamp(0, W - 1)
    parent_flat = (py * W + px).reshape(-1)

    boundary_idx = (hr_flat < 0).nonzero(as_tuple=False).squeeze(1)
    if boundary_idx.numel() > 0:
        context[boundary_idx, parent_flat[boundary_idx]] = True

    return context.reshape(Hr, Wr, H, W), py, px


def _save_context_lookup_npz(
    out_npz: Path,
    hr_to_lr_map_2d: torch.Tensor,
    lr_labels_2d: torch.Tensor,
) -> None:
    """
    Save full per-pixel context masks for debugging:
      context_masks[y_hr, x_hr] -> (H,W) LR mask used by that HR pixel.
    """
    context_masks, parent_y, parent_x = _build_hr_to_lr_context_masks(
        hr_to_lr_map_2d=hr_to_lr_map_2d,
        lr_labels_2d=lr_labels_2d,
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        context_masks=context_masks.numpy().astype(np.uint8),
        hr_to_lr_map=hr_to_lr_map_2d.numpy().astype(np.int32),
        lr_labels=lr_labels_2d.numpy().astype(np.int32),
        parent_y=parent_y.numpy().astype(np.int32),
        parent_x=parent_x.numpy().astype(np.int32),
    )
    print(f"Saved full HR->LR context masks: {out_npz}")


def _save_context_probe_plot(
    out_png: Path,
    hr_to_lr_map_2d: torch.Tensor,
    lr_labels_2d: torch.Tensor,
    max_probes: int = 12,
) -> None:
    """
    Save probe visualization of LR context used for selected HR pixels.
    For each probe:
      - left: HR grain-id map with query pixel marker
      - right: LR context mask (unused LR pixels masked out)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] Could not import matplotlib for context probe plot: {exc}")
        return

    Hr, Wr = int(hr_to_lr_map_2d.shape[0]), int(hr_to_lr_map_2d.shape[1])
    H, W = int(lr_labels_2d.shape[0]), int(lr_labels_2d.shape[1])

    context_masks, parent_y, parent_x = _build_hr_to_lr_context_masks(
        hr_to_lr_map_2d=hr_to_lr_map_2d,
        lr_labels_2d=lr_labels_2d,
    )

    # Random probe pixels on HR (re-sampled each call).
    n_total = Hr * Wr
    n_take = max(1, min(int(max_probes), n_total))
    rng = np.random.default_rng()
    picked = rng.choice(n_total, size=n_take, replace=False)
    probes: list[tuple[int, int]] = []
    for idx in picked.tolist():
        y_hr = int(idx // Wr)
        x_hr = int(idx % Wr)
        probes.append((y_hr, x_hr))

    if len(probes) == 0:
        return

    hr_disp = hr_to_lr_map_2d.detach().cpu().numpy().astype(np.float32)
    hr_disp[hr_disp < 0] = np.nan
    lr_base = lr_labels_2d.detach().cpu().numpy().astype(np.float32)
    lr_base[lr_base < 0] = np.nan
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="black")

    fig, axes = plt.subplots(len(probes), 2, figsize=(10, 3 * len(probes)), dpi=170)
    if len(probes) == 1:
        axes = np.array([axes])

    for i, (y_hr, x_hr) in enumerate(probes):
        ax_hr, ax_lr = axes[i]
        gid = int(hr_to_lr_map_2d[y_hr, x_hr].item())

        ax_hr.imshow(hr_disp, cmap=cmap, interpolation="nearest")
        ax_hr.scatter([x_hr], [y_hr], c="red", s=35, marker="x")
        ax_hr.set_title(f"HR query pixel (y={y_hr}, x={x_hr}), gid={gid}")
        ax_hr.axis("off")

        ctx = context_masks[y_hr, x_hr].detach().cpu().numpy().astype(bool)
        used = np.where(ctx, lr_base, np.nan)
        ax_lr.imshow(used, cmap=cmap, interpolation="nearest")
        if gid < 0:
            py = int(parent_y[y_hr, x_hr].item())
            px = int(parent_x[y_hr, x_hr].item())
            ax_lr.scatter([px], [py], c="red", s=30, marker="x")
            ax_lr.set_title(
                f"LR context used (fallback parent y={py}, x={px}); used={int(ctx.sum())}/{H*W}"
            )
        else:
            ax_lr.set_title(f"LR context used (same grain); used={int(ctx.sum())}/{H*W}")
        ax_lr.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"Saved context probe plot: {out_png}")


def _save_boundary_debug_plot(
    out_png: Path,
    boundary_lr_2d: torch.Tensor,
    boundary_hr_clean_2d: torch.Tensor,
    threshold: float | None = None,
) -> None:
    """
    Save side-by-side plot:
      1) LR boundary map from dataloader
      2) Cleaned HR boundary map derived by the model boundary pipeline
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] Could not import matplotlib for boundary debug plot: {exc}")
        return

    lr_np = boundary_lr_2d.numpy()
    hr_np = boundary_hr_clean_2d.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    ax0, ax1 = axes

    im0 = ax0.imshow(lr_np, cmap="gray", vmin=0.0, vmax=1.0)
    ax0.set_title(f"LR boundary map ({lr_np.shape[0]}x{lr_np.shape[1]})")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    im1 = ax1.imshow(hr_np, cmap="magma", vmin=0.0, vmax=1.0)
    if threshold is None:
        ax1.set_title(f"Cleaned HR boundary ({hr_np.shape[0]}x{hr_np.shape[1]})")
    else:
        ax1.set_title(
            f"Cleaned HR boundary ({hr_np.shape[0]}x{hr_np.shape[1]}), thr={threshold:.2f}"
        )
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"Saved boundary debug plot: {out_png}")


def _save_checkpoint(
    ckpt_path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    history: dict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": float(best_val_loss),
            "history": history,
        },
        ckpt_path,
    )


def _load_checkpoint(
    ckpt_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    optimizer_state_loaded = False
    optimizer_state_dict = ckpt.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            optimizer_state_loaded = True
        except ValueError as exc:
            current_opt_state = optimizer.state_dict()
            saved_groups = optimizer_state_dict.get("param_groups", [])
            current_groups = current_opt_state.get("param_groups", [])
            pruned_optimizer_state = None
            if len(saved_groups) == len(current_groups):
                removed_param_ids: set[int] = set()
                pruned_groups = []
                can_prune = True
                for saved_group, current_group in zip(saved_groups, current_groups):
                    saved_params = list(saved_group.get("params", []))
                    current_params = list(current_group.get("params", []))
                    if len(saved_params) < len(current_params):
                        can_prune = False
                        break
                    group_copy = dict(saved_group)
                    group_copy["params"] = saved_params[: len(current_params)]
                    removed_param_ids.update(saved_params[len(current_params) :])
                    pruned_groups.append(group_copy)
                if can_prune and removed_param_ids:
                    pruned_optimizer_state = dict(optimizer_state_dict)
                    pruned_optimizer_state["param_groups"] = pruned_groups
                    pruned_optimizer_state["state"] = {
                        pid: state
                        for pid, state in optimizer_state_dict.get("state", {}).items()
                        if pid not in removed_param_ids
                    }
            if pruned_optimizer_state is not None:
                optimizer.load_state_dict(pruned_optimizer_state)
                optimizer_state_loaded = True
                print(
                    "[warning] Pruned legacy optimizer state for removed disabled-module parameters "
                    f"while loading {ckpt_path.name}."
                )
            else:
                print(
                    f"[warning] Skipping optimizer state restore from {ckpt_path.name}: {exc}"
                )
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        if optimizer_state_dict is None or optimizer_state_loaded:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print(
                f"[warning] Skipping scheduler state restore from {ckpt_path.name} because "
                "optimizer state could not be restored."
            )
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    history = _normalize_history(ckpt.get("history"))
    return start_epoch, best_val_loss, history


def _train_one_epoch(
    model_core: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler,
    memory_debug_every: int = 0,
    cuda_empty_cache_every: int = 0,
    tv_loss_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model_core.train()
    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_steps = 0
    _feature_loss_params = inspect.signature(model_core.feature_loss_sr).parameters
    _supports_tv_loss_weight = "tv_loss_weight" in _feature_loss_params
    _supports_lr_boundary_map = "lr_boundary_map" in _feature_loss_params
    _supports_return_info = "return_info" in _feature_loss_params
    _requires_lr_boundary_map = (
        _supports_lr_boundary_map
        and _feature_loss_params["lr_boundary_map"].default is inspect._empty
    )

    for batch in tqdm(loader, desc="Train", leave=False):
        lr, hr, lr_boundary_map = _unpack_batch(batch)
        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        if lr_boundary_map is not None:
            lr_boundary_map = lr_boundary_map.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        lr_flat, lr_shape = _flatten_quat_chw_batch(lr)
        hr_flat, _ = _flatten_quat_chw_batch(hr)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            loss_kwargs = {
                "lr_shape": lr_shape,
                "normalize_input": True,
            }
            if _supports_lr_boundary_map:
                if lr_boundary_map is None and _requires_lr_boundary_map:
                    raise ValueError(
                        "Model requires lr_boundary_map but dataloader did not provide it. "
                        "Enable return_lr_boundary_map in build_dataloader."
                    )
                if lr_boundary_map is not None:
                    loss_kwargs["lr_boundary_map"] = lr_boundary_map
            if _supports_tv_loss_weight:
                loss_kwargs["tv_loss_weight"] = tv_loss_weight
            if _supports_return_info:
                loss_kwargs["return_info"] = True
            loss_output = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                **loss_kwargs,
            )
            loss, loss_info = _loss_output_to_tensor_and_info(loss_output)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), clip)
            optimizer.step()

        total_loss += float(loss.detach().item())
        for key, value in loss_info.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        n_steps += 1
        if (
            device.type == "cuda"
            and int(cuda_empty_cache_every) > 0
            and (n_steps % int(cuda_empty_cache_every) == 0)
        ):
            torch.cuda.empty_cache()
        if (
            device.type == "cuda"
            and int(memory_debug_every) > 0
            and (n_steps % int(memory_debug_every) == 0)
        ):
            alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
            resv_gb = torch.cuda.memory_reserved(device) / (1024**3)
            max_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            max_resv_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            print(
                f"[cuda-mem] step={n_steps} alloc={alloc_gb:.2f}G reserved={resv_gb:.2f}G "
                f"max_alloc={max_alloc_gb:.2f}G max_reserved={max_resv_gb:.2f}G"
            )

    avg_metrics = {
        key: value / max(1, n_steps) for key, value in metric_sums.items()
    }
    return total_loss / max(1, n_steps), avg_metrics


@torch.no_grad()
def _validate_one_epoch(
    model_core: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tv_loss_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model_core.eval()
    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    n_steps = 0
    _feature_loss_params = inspect.signature(model_core.feature_loss_sr).parameters
    _supports_tv_loss_weight = "tv_loss_weight" in _feature_loss_params
    _supports_lr_boundary_map = "lr_boundary_map" in _feature_loss_params
    _supports_return_info = "return_info" in _feature_loss_params
    _requires_lr_boundary_map = (
        _supports_lr_boundary_map
        and _feature_loss_params["lr_boundary_map"].default is inspect._empty
    )

    for batch in tqdm(loader, desc="Val", leave=False):
        lr, hr, lr_boundary_map = _unpack_batch(batch)
        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        if lr_boundary_map is not None:
            lr_boundary_map = lr_boundary_map.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        lr_flat, lr_shape = _flatten_quat_chw_batch(lr)
        hr_flat, _ = _flatten_quat_chw_batch(hr)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            loss_kwargs = {
                "lr_shape": lr_shape,
                "normalize_input": True,
            }
            if _supports_lr_boundary_map:
                if lr_boundary_map is None and _requires_lr_boundary_map:
                    raise ValueError(
                        "Model requires lr_boundary_map but dataloader did not provide it. "
                        "Enable return_lr_boundary_map in build_dataloader."
                    )
                if lr_boundary_map is not None:
                    loss_kwargs["lr_boundary_map"] = lr_boundary_map
            if _supports_tv_loss_weight:
                loss_kwargs["tv_loss_weight"] = tv_loss_weight
            if _supports_return_info:
                loss_kwargs["return_info"] = True
            loss_output = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                **loss_kwargs,
            )
            loss, loss_info = _loss_output_to_tensor_and_info(loss_output)
        total_loss += float(loss.detach().item())
        for key, value in loss_info.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        n_steps += 1

    avg_metrics = {
        key: value / max(1, n_steps) for key, value in metric_sums.items()
    }
    return total_loss / max(1, n_steps), avg_metrics


def _render_probe_stage_viz(
    model_core: torch.nn.Module,
    sym_class,
    out_dir: Path,
    lr_hwc: torch.Tensor,
    hr_hwc: torch.Tensor,
    sr_flat: torch.Tensor,
    sr_shape: tuple[int, int],
    aux: dict | None,
) -> bool:
    """Render probe-stage galleries when aux contains decoded probe stages."""
    if not isinstance(aux, dict):
        return False

    probe_stages = aux.get("probe_stages")
    if not isinstance(probe_stages, list) or len(probe_stages) == 0:
        return False

    try:
        from utils.stage_probe_utils import (
            decode_probe_stages,
            extract_explicit_scalar_probe_maps,
            render_decoded_probe_gallery,
            render_scalar_probe_gallery,
        )
    except Exception as exc:
        print(f"[warning] Probe-stage utilities unavailable (import failed): {exc}")
        return False

    expected_dim = getattr(model_core, "feature_dim_a1", None)
    expected_dim = int(expected_dim) if expected_dim is not None else None

    decodable_stages: list[dict] = []
    skipped_stages: list[dict[str, object]] = []
    highdim_scalar_maps: list[dict[str, object]] = []

    for stage in probe_stages:
        name = str(stage.get("name", "unknown_stage"))
        feat = stage.get("feat", None)
        hw = tuple(stage.get("shape", ()))
        if not isinstance(feat, torch.Tensor) or len(hw) != 2:
            skipped_stages.append({"name": name, "reason": "missing_or_invalid_feat_or_shape"})
            continue

        if feat.dim() == 3:
            feat_single = feat[0]
        elif feat.dim() == 2:
            feat_single = feat
        else:
            skipped_stages.append({"name": name, "reason": f"unsupported_feat_rank_{int(feat.dim())}"})
            continue

        if feat_single.dim() != 2:
            skipped_stages.append({"name": name, "reason": "expected_flat_feat_rank2"})
            continue

        c_dim = int(feat_single.shape[-1])
        n_dim = int(feat_single.shape[0])
        if expected_dim is not None and c_dim != expected_dim:
            skipped_stages.append(
                {
                    "name": name,
                    "reason": "feature_dim_mismatch",
                    "feature_dim": c_dim,
                    "expected_dim": expected_dim,
                }
            )
            if n_dim == int(hw[0]) * int(hw[1]):
                chan_norm = feat_single.detach().float().norm(dim=-1).reshape(int(hw[0]), int(hw[1])).cpu().numpy()
                highdim_scalar_maps.append(
                    {
                        "name": f"{name}_chan_norm",
                        "array": chan_norm,
                        "cmap": "magma",
                    }
                )
            continue

        decodable_stages.append(stage)

    decoded_stages: list[dict[str, object]] = []
    if decodable_stages:
        try:
            with torch.enable_grad():
                decoded_stages = decode_probe_stages(model_core, decodable_stages, sample_index=0)
        except Exception as exc:
            print(f"[warning] Failed to decode probe stages during viz: {exc}")
            decoded_stages = []

    sr_h, sr_w = int(sr_shape[0]), int(sr_shape[1])
    sr_hwc = sr_flat.reshape(sr_h, sr_w, 4).detach().cpu()

    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_gallery_path = None
    if decoded_stages:
        context_rows = [
            {
                "name": "lr_input",
                "shape": tuple(lr_hwc.shape[:2]),
                "quat_hwc": lr_hwc.detach().cpu(),
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
            *[
                {
                    "name": item["name"],
                    "shape": item["shape"],
                    "quat_hwc": item["quat_hwc"],
                    "hr_target_hwc": hr_hwc.detach().cpu(),
                }
                for item in decoded_stages
            ],
            {
                "name": "sr_output",
                "shape": (sr_h, sr_w),
                "quat_hwc": sr_hwc,
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
            {
                "name": "hr_target",
                "shape": tuple(hr_hwc.shape[:2]),
                "quat_hwc": hr_hwc.detach().cpu(),
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
        ]
        decoded_gallery_path = render_decoded_probe_gallery(
            context_rows,
            sym_class=sym_class,
            out_png=out_dir / "probe_decoded_gallery.png",
        )

    stage_by_name = {str(item["name"]).strip(): item for item in decoded_stages}
    detailed_upsampler_order = (
        "grain_attention_out",
        "upsample_center_hr",
        "upsample_plus_hr",
        "upsample_minus_hr",
        "upsample_shifted_mix_hr",
    )
    detailed_rows = []
    for name in detailed_upsampler_order:
        if name in stage_by_name:
            row = stage_by_name[name]
            detailed_rows.append(
                {
                    "name": row["name"],
                    "shape": row["shape"],
                    "quat_hwc": row["quat_hwc"],
                    "hr_target_hwc": hr_hwc.detach().cpu(),
                }
            )

    upsampler_gallery_path = None
    if detailed_rows:
        upsampler_rows = [
            {
                "name": "lr_input",
                "shape": tuple(lr_hwc.shape[:2]),
                "quat_hwc": lr_hwc.detach().cpu(),
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
            *detailed_rows,
            {
                "name": "sr_output",
                "shape": (sr_h, sr_w),
                "quat_hwc": sr_hwc,
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
            {
                "name": "hr_target",
                "shape": tuple(hr_hwc.shape[:2]),
                "quat_hwc": hr_hwc.detach().cpu(),
                "hr_target_hwc": hr_hwc.detach().cpu(),
            },
        ]
        upsampler_gallery_path = render_decoded_probe_gallery(
            upsampler_rows,
            sym_class=sym_class,
            out_png=out_dir / "probe_upsampler_detail_gallery.png",
        )

    scalar_maps = extract_explicit_scalar_probe_maps(aux, sample_index=0)
    scalar_maps.extend(highdim_scalar_maps)
    scalar_gallery_path = None
    if scalar_maps:
        scalar_gallery_path = render_scalar_probe_gallery(
            scalar_maps,
            out_png=out_dir / "probe_scalar_gallery.png",
        )

    metadata = {
        "probe_stage_names": [str(item["name"]) for item in decoded_stages],
        "skipped_probe_stages": skipped_stages,
        "requested_upsampler_stage_names": list(detailed_upsampler_order),
        "available_upsampler_stage_names": [row["name"] for row in detailed_rows],
        "decoded_gallery": str(decoded_gallery_path) if decoded_gallery_path is not None else None,
        "upsampler_detail_gallery": str(upsampler_gallery_path) if upsampler_gallery_path is not None else None,
        "scalar_gallery": str(scalar_gallery_path) if scalar_gallery_path is not None else None,
    }
    with open(out_dir / "probe_stage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if decoded_gallery_path is not None:
        print(f"Saved probe decoded gallery: {decoded_gallery_path}")
    elif skipped_stages:
        print("[warning] No decodable probe stages for quaternion decode; wrote scalar probe maps and metadata.")
    if upsampler_gallery_path is not None:
        print(f"Saved detailed upsampler gallery: {upsampler_gallery_path}")
    if scalar_gallery_path is not None:
        print(f"Saved probe scalar gallery: {scalar_gallery_path}")
    return True


def _render_sr_hr_lr_ipf(
    model_core: torch.nn.Module,
    data_loader,
    sym_class,
    out_png: Path,
    ref_dir: str = "ALL",
    enable_probe_stage_viz: bool = True,
    sample_index: int = 0,
    sample_key: str | None = None,
    force_cpu: bool = False,
    allow_cpu_fallback: bool = True,
) -> bool:
    """
    Render LR/SR/HR IPF comparison from one selected sample in the loader.

    `sample_index` is interpreted as a global sample index over the loader's
    deterministic iteration order, not a per-batch offset. When `sample_key`
    is provided, it takes precedence and resolves against dataset pair names.
    """
    try:
        from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side
    except Exception as exc:
        print(f"[warning] Visualization unavailable (import failed): {exc}")
        return False

    target_index, resolved_pair_meta = _resolve_viz_sample_index(
        data_loader,
        sample_index=sample_index,
        sample_key=sample_key,
    )
    batch = None
    batch_base_index = 0
    last_nonempty_batch = None
    last_nonempty_base_index = 0
    seen = 0
    for maybe_batch in data_loader:
        lr_tmp, hr_tmp, _ = _unpack_batch(maybe_batch)
        batch_size_tmp = 0 if lr_tmp is None else int(lr_tmp.shape[0])
        if batch_size_tmp <= 0:
            continue
        last_nonempty_batch = maybe_batch
        last_nonempty_base_index = seen
        if target_index < seen + batch_size_tmp:
            batch = maybe_batch
            batch_base_index = seen
            break
        seen += batch_size_tmp

    if batch is None:
        batch = last_nonempty_batch
        batch_base_index = last_nonempty_base_index

    if batch is None:
        print("[warning] Skipping visualization: loader is empty.")
        return False

    lr, hr, lr_boundary_map = _unpack_batch(batch)
    if lr is None or hr is None or lr.shape[0] == 0 or hr.shape[0] == 0:
        print("[warning] Skipping visualization: empty LR/HR batch.")
        return False

    original_device = next(model_core.parameters()).device
    moved_for_viz = False
    try:
        model_was_training = model_core.training
        model_core.eval()

        if force_cpu and original_device.type != "cpu":
            model_core.to("cpu")
            _sync_module_device_attrs(model_core, torch.device("cpu"))
            moved_for_viz = True
        else:
            _sync_module_device_attrs(model_core, next(model_core.parameters()).device)

        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.empty_cache()

        # Decoder backend may run gradient-based optimization, so visualization
        # forward pass cannot be wrapped in inference/no-grad mode.
        with torch.enable_grad():
            device = next(model_core.parameters()).device
            chosen_idx = int(max(0, min(target_index - batch_base_index, int(lr.shape[0]) - 1)))
            resolved_index = int(batch_base_index + chosen_idx)
            if resolved_index != target_index:
                print(
                    f"[warning] viz_sample_index={target_index} out of range for loader; "
                    f"using sample_index={resolved_index}."
                )
            out_png.parent.mkdir(parents=True, exist_ok=True)
            viz_meta: dict[str, object] = {
                "requested_sample_index": int(sample_index),
                "requested_sample_key": None if sample_key is None else str(sample_key),
                "resolved_sample_index": resolved_index,
                "resolved_batch_base_index": int(batch_base_index),
                "resolved_batch_local_index": int(chosen_idx),
                "ref_dir": str(ref_dir),
            }
            if resolved_pair_meta is not None:
                viz_meta.update(resolved_pair_meta)
            elif hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "pairs"):
                pairs = getattr(data_loader.dataset, "pairs", None)
                if isinstance(pairs, (tuple, list)) and 0 <= resolved_index < len(pairs):
                    pair = pairs[resolved_index]
                    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                        lr_path = str(pair[0])
                        hr_path = str(pair[1])
                        viz_meta.update(
                            {
                                "pair_key": _sample_pair_key_from_path(lr_path)
                                or _sample_pair_key_from_path(hr_path),
                                "lr_path": lr_path,
                                "hr_path": hr_path,
                            }
                        )
            with open(out_png.parent / "selected_sample.json", "w") as f:
                json.dump(viz_meta, f, indent=2)
            print(
                "Visualization sample: "
                f"index={viz_meta['resolved_sample_index']}"
                + (
                    f", key={viz_meta['pair_key']}"
                    if "pair_key" in viz_meta and viz_meta["pair_key"] is not None
                    else ""
                )
                + (
                    f", lr={viz_meta['lr_path']}"
                    if "lr_path" in viz_meta
                    else ""
                )
            )

            lr0 = _to_hwc_quat_single(lr[chosen_idx].to(device=device, dtype=torch.float32))
            hr0 = _to_hwc_quat_single(hr[chosen_idx].to(device=device, dtype=torch.float32))
            lr_h, lr_w = int(lr0.shape[0]), int(lr0.shape[1])
            hr_h, hr_w = int(hr0.shape[0]), int(hr0.shape[1])
            lr_boundary0 = None
            if lr_boundary_map is not None:
                lr_boundary0 = lr_boundary_map[chosen_idx].to(device=device, dtype=torch.float32)

            lr_flat = lr0.reshape(-1, 4)
            _forward_sr_params = inspect.signature(model_core.forward_sr).parameters
            _supports_lr_boundary = "lr_boundary_map" in _forward_sr_params
            _requires_lr_boundary = (
                _supports_lr_boundary
                and _forward_sr_params["lr_boundary_map"].default is inspect._empty
            )
            forward_kwargs = {
                "lr_shape": (lr_h, lr_w),
                "normalize_input": True,
            }
            if _supports_lr_boundary:
                if lr_boundary0 is None and _requires_lr_boundary:
                    raise ValueError(
                        "Model forward_sr requires lr_boundary_map but visualization batch did not include it."
                    )
                if lr_boundary0 is not None:
                    forward_kwargs["lr_boundary_map"] = lr_boundary0

            _supports_return_aux = "return_aux" in _forward_sr_params
            _supports_return_probe = "return_probe" in _forward_sr_params
            aux = None
            need_aux = bool(enable_probe_stage_viz)

            if _supports_return_aux and need_aux:
                probe_kwargs = dict(forward_kwargs)
                probe_kwargs["return_aux"] = True
                if _supports_return_probe:
                    probe_kwargs["return_probe"] = bool(enable_probe_stage_viz)
                sr_output = model_core.forward_sr(
                    lr_flat,
                    **probe_kwargs,
                )
                if (
                    isinstance(sr_output, tuple)
                    and len(sr_output) == 2
                    and isinstance(sr_output[1], dict)
                ):
                    q_sr_flat, aux = sr_output
                else:
                    q_sr_flat = sr_output
            else:
                q_sr_flat = model_core.forward_sr(
                    lr_flat,
                    **forward_kwargs,
                )

            if int(q_sr_flat.shape[0]) != hr_h * hr_w:
                raise ValueError(
                    f"SR output size mismatch: got N={int(q_sr_flat.shape[0])}, expected {hr_h * hr_w}"
                )

            lr_np = lr0.detach().cpu().numpy()
            hr_np = hr0.detach().cpu().numpy()
            sr_np = q_sr_flat.reshape(hr_h, hr_w, 4).detach().cpu().numpy()

        render_sr_hr_lr_side_by_side(
            sr_q_arr=sr_np,
            hr_q_arr=hr_np,
            lr_q_arr=lr_np,
            sym_class=sym_class,
            out_png=str(out_png),
            ref_dir=str(ref_dir),
            include_key=True,
            overwrite=True,
            format_input=True,
            dpi=300,
        )
        print(f"Saved LR/SR/HR IPF visualization: {out_png}")

        if enable_probe_stage_viz:
            _render_probe_stage_viz(
                model_core=model_core,
                sym_class=sym_class,
                out_dir=out_png.parent,
                lr_hwc=lr0,
                hr_hwc=hr0,
                sr_flat=q_sr_flat,
                sr_shape=(hr_h, hr_w),
                aux=aux,
            )

        # Boundary debug visualization in eval mode (when boundary map is available).
        if lr_boundary0 is not None:
            grain_ctx = _derive_boundary_grain_context(
                model_core=model_core,
                lr_boundary_map_2d=lr_boundary0,
                lr_shape=(lr_h, lr_w),
                device=device,
                dtype=torch.float32,
            )
            if grain_ctx is not None:
                b_lr_2d = grain_ctx["boundary_lr_1px"]
                b_hr_clean_2d = grain_ctx["boundary_hr_1px"]
                _save_boundary_debug_plot(
                    out_png=out_png.parent / "boundary_maps_lr_vs_cleaned_hr.png",
                    boundary_lr_2d=b_lr_2d,
                    boundary_hr_clean_2d=b_hr_clean_2d,
                    threshold=None,
                )
                _save_grain_id_debug_plot(
                    out_png=out_png.parent / "grain_id_maps_lr_hr.png",
                    lr_labels_2d=grain_ctx["lr_labels"],
                    hr_to_lr_map_2d=grain_ctx["hr_to_lr_map"],
                )
                _save_context_probe_plot(
                    out_png=out_png.parent / "context_probes_hr_to_lr.png",
                    hr_to_lr_map_2d=grain_ctx["hr_to_lr_map"],
                    lr_labels_2d=grain_ctx["lr_labels"],
                    max_probes=12,
                )
                _save_context_lookup_npz(
                    out_npz=out_png.parent / "context_lookup_hr_to_lr.npz",
                    hr_to_lr_map_2d=grain_ctx["hr_to_lr_map"],
                    lr_labels_2d=grain_ctx["lr_labels"],
                )
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if allow_cpu_fallback and (not force_cpu) and ("out of memory" in msg):
            print("[warning] GPU visualization OOM; retrying visualization on CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _render_sr_hr_lr_ipf(
                model_core=model_core,
                data_loader=data_loader,
                sym_class=sym_class,
                out_png=out_png,
                ref_dir=ref_dir,
                enable_probe_stage_viz=enable_probe_stage_viz,
                sample_index=sample_index,
                force_cpu=True,
                allow_cpu_fallback=False,
            )
        print(f"[warning] Failed to render LR/SR/HR IPF visualization: {exc}")
        return False
    finally:
        if moved_for_viz:
            model_core.to(original_device)
            _sync_module_device_attrs(model_core, original_device)
            if original_device.type == "cuda":
                torch.cuda.empty_cache()
        if "model_was_training" in locals() and model_was_training:
            model_core.train()


def _render_final_probe_split_viz(
    model_core: torch.nn.Module,
    loaders: dict[str, object],
    sym_class,
    out_root: Path,
    ref_dir: str = "ALL",
    enable_probe_stage_viz: bool = True,
    sample_index: int = 0,
    sample_key: str | None = None,
    force_cpu: bool = False,
) -> None:
    """Render end-of-training probe-stage visualizations for a selected val/test sample."""
    if not bool(enable_probe_stage_viz):
        return

    for split_key in ("val", "test"):
        data_loader = loaders.get(split_key)
        if data_loader is None:
            continue
        sample_label = (
            str(sample_key).strip().lower()
            if sample_key is not None and str(sample_key).strip()
            else f"{int(sample_index):04d}"
        )
        split_out_dir = out_root / f"{split_key}_sample{sample_label}"
        _render_sr_hr_lr_ipf(
            model_core=model_core,
            data_loader=data_loader,
            sym_class=sym_class,
            out_png=split_out_dir / "lr_sr_hr_ipf.png",
            ref_dir=ref_dir,
            enable_probe_stage_viz=True,
            sample_index=sample_index,
            sample_key=sample_key,
            force_cpu=force_cpu,
        )


def main() -> None: 
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    model_cfg = getattr(cfg, "model", {}) or {}
    model_module = model_cfg.get("model_module", args.model_module) if isinstance(model_cfg, dict) else getattr(model_cfg, "model_module", args.model_module)
    model_class = model_cfg.get("model_class", args.model_class) if isinstance(model_cfg, dict) else getattr(model_cfg, "model_class", args.model_class)
    IsoEmbeddingSRAttn = getattr(importlib.import_module(model_module), model_class)
    print(f"Model: {model_module}.{model_class}")

    _feature_loss_params_cls = inspect.signature(IsoEmbeddingSRAttn.feature_loss_sr).parameters
    _model_supports_lr_boundary = "lr_boundary_map" in _feature_loss_params_cls
    _model_requires_lr_boundary = (
        _model_supports_lr_boundary
        and _feature_loss_params_cls["lr_boundary_map"].default is inspect._empty
    )
    feature_upsampler_type = str(getattr(cfg, "feature_upsampler_type", "shifted_bilinear")).strip().lower()
    use_lr_boundary_map = bool(getattr(cfg, "use_lr_boundary_map", _model_supports_lr_boundary))
    if feature_upsampler_type == "grain_attention":
        use_lr_boundary_map = True
    if _model_requires_lr_boundary:
        use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    print(
        f"LR boundary maps from dataloader: {use_lr_boundary_map} "
        f"(model supports={_model_supports_lr_boundary}, requires={_model_requires_lr_boundary}, "
        f"feature_upsampler_type={feature_upsampler_type})"
    )

    seed = int(get_seed_from_config(cfg))
    set_seed(seed)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")
    allow_tf32 = bool(getattr(cfg, "allow_tf32", True))
    cudnn_benchmark = bool(getattr(cfg, "cudnn_benchmark", False))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        print(
            f"CUDA math settings: allow_tf32={allow_tf32}, cudnn_benchmark={cudnn_benchmark}"
        )
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        free_gb = float(free_bytes) / (1024**3)
        total_gb = float(total_bytes) / (1024**3)
        print(f"CUDA free memory at startup: {free_gb:.2f} / {total_gb:.2f} GB")
        min_free_cuda_gb = float(getattr(cfg, "min_free_cuda_gb", 0.0))
        if min_free_cuda_gb > 0.0 and free_gb < min_free_cuda_gb:
            raise RuntimeError(
                f"Insufficient free CUDA memory at startup: {free_gb:.2f} GB < "
                f"min_free_cuda_gb={min_free_cuda_gb:.2f} GB. "
                "Another GPU process is likely active. "
                "Run `nvidia-smi` and kill the listed PID(s), then retry."
            )
    amp_dtype_str = str(getattr(cfg, "amp_dtype", "bf16")).lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype={amp_dtype_str!r}; use 'bf16' or 'fp16'")
    use_amp = bool(getattr(cfg, "use_amp", True)) and device.type == "cuda"
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if (use_amp and amp_dtype == torch.float16)
        else None
    )
    print(f"AMP enabled: {use_amp} (dtype={amp_dtype_str})")

    loaders = {}
    for split in ("train", "val", "test"):
        try:
            loaders[split] = build_dataloader(
                dataset_root=cfg.dataset_root,
                split=split.capitalize(),
                batch_size=int(cfg.batch_size),
                num_workers=int(cfg.num_workers),
                preload=bool(cfg.preload),
                preload_torch=bool(cfg.preload_torch),
                pin_memory=bool(cfg.pin_memory),
                persistent_workers=bool(getattr(cfg, "persistent_workers", False)),
                prefetch_factor=int(getattr(cfg, "prefetch_factor", 2)),
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                take_first=_get_take_first(cfg, split),
                seed=seed,
                return_lr_boundary_map=use_lr_boundary_map,
                lr_boundary_angle_deg=lr_boundary_angle_deg,
                lr_boundary_mark_both_sides=lr_boundary_mark_both_sides,
            )
        except (RuntimeError, FileNotFoundError) as _e:
            if split == "train":
                raise
            print(f"[warning] {split} split unavailable ({_e}); falling back to train split.")
            loaders[split] = loaders["train"]

    _init_params = set(inspect.signature(IsoEmbeddingSRAttn.__init__).parameters)

    model_kwargs = {
        "crystal": str(getattr(cfg, "crystal", "fcc")),
        "d6_convention": str(getattr(cfg, "d6_convention", "z_axis")),
        "embedding_mode": str(getattr(cfg, "embedding_mode", "tensor_product")),
        "max_harmonic_l": getattr(cfg, "max_harmonic_l", None),
        "embedding_metric_calibration": getattr(cfg, "embedding_metric_calibration", "none"),
        "device": device,
        "upsample_factor": getattr(cfg, "upsample_factor", getattr(cfg, "scale", 4)),
        "feature_upsampler_type": str(getattr(cfg, "feature_upsampler_type", "shifted_bilinear")),
        "upsample_context_kernel_size": int(getattr(cfg, "upsample_context_kernel_size", 3)),
        "upsample_residual": bool(getattr(cfg, "upsample_residual", True)),
        "upsample_transpose_overlap": int(getattr(cfg, "upsample_transpose_overlap", 2)),
        "upsample_boundary_threshold": float(getattr(cfg, "upsample_boundary_threshold", 0.5)),
        "upsample_boundary_smooth_sigma": float(getattr(cfg, "upsample_boundary_smooth_sigma", 2.0)),
        "upsample_boundary_smooth_iters": int(getattr(cfg, "upsample_boundary_smooth_iters", 12)),
        "upsample_boundary_sdf_shift": float(getattr(cfg, "upsample_boundary_sdf_shift", 0.7)),
        "use_boundary_gate": bool(getattr(cfg, "use_boundary_gate", False)),
        "evidence_radius": int(getattr(cfg, "evidence_radius", 1)),
        "sdf_hidden_dim": int(getattr(cfg, "sdf_hidden_dim", 64)),
        "guidance_dim": int(getattr(cfg, "guidance_dim", 16)),
        "stats_code_dim": int(getattr(cfg, "stats_code_dim", 16)),
        "stats_hidden_dim": int(getattr(cfg, "stats_hidden_dim", 32)),
        "extra_stats_dim": int(getattr(cfg, "extra_stats_dim", 0)),
        "num_lr_blocks": int(getattr(cfg, "num_lr_blocks", 1)),
        "num_hr_blocks": int(getattr(cfg, "num_hr_blocks", 1)),
        "use_pre_lr": getattr(cfg, "use_pre_lr", None),
        "use_post_hr": getattr(cfg, "use_post_hr", None),
        "use_refinement": bool(getattr(cfg, "use_refinement", True)),
        "refinement_num_steps": int(getattr(cfg, "refinement_num_steps", 2)),
        "refinement_hidden_dim": int(getattr(cfg, "refinement_hidden_dim", 32)),
        "refinement_kernel_size": int(getattr(cfg, "refinement_kernel_size", 3)),
        "hard_one_sided": bool(getattr(cfg, "hard_one_sided", True)),
        "hard_boundary_band": bool(getattr(cfg, "hard_boundary_band", False)),
        "lambda_feat": float(getattr(cfg, "lambda_feat", 1.0)),
        "lambda_boundary": float(getattr(cfg, "lambda_boundary", 0.5)),
        "lambda_lr_boundary": float(getattr(cfg, "lambda_lr_boundary", 0.10)),
        "lambda_side_correct": float(getattr(cfg, "lambda_side_correct", 0.10)),
        "lambda_side_entropy": float(getattr(cfg, "lambda_side_entropy", 0.002)),
        "boundary_thr_deg": float(getattr(cfg, "boundary_thr_deg", 3.0)),
        "boundary_connectivity": int(getattr(cfg, "boundary_connectivity", 4)),
        "use_focal_boundary": bool(getattr(cfg, "use_focal_boundary", True)),
        "focal_gamma": float(getattr(cfg, "focal_gamma", 2.0)),
        "side_correct_band_kernel": getattr(cfg, "side_correct_band_kernel", (3, 3)),
        "side_correct_rel_gap": float(getattr(cfg, "side_correct_rel_gap", 0.05)),
        "attention_num_heads": int(getattr(cfg, "attention_num_heads", 8)),
        "attention_head_dim": int(getattr(cfg, "attention_head_dim", 32)),
        "attention_num_layers": int(getattr(cfg, "attention_num_blocks", getattr(cfg, "attention_num_layers", 4))),
        "attention_mlp_ratio": float(getattr(cfg, "attention_mlp_ratio", 2.0)),
        "attention_pos_hidden_dim": int(getattr(cfg, "attention_pos_hidden_dim", 64)),
        "attention_dropout": float(getattr(cfg, "attention_dropout", 0.0)),
        "attention_query_chunk_size": int(getattr(cfg, "attention_query_chunk_size", 2048)),
        "attention_lr_patch_size": getattr(cfg, "attention_lr_patch_size", (2, 2)),
        "seed_mode": str(getattr(cfg, "seed_mode", "bilinear")),
        "seed_num_candidates": int(getattr(cfg, "seed_num_candidates", 4)),
        "seed_lr_self_attn_layers": int(getattr(cfg, "seed_lr_self_attn_layers", 1)),
        "seed_lr_self_attn_heads": int(getattr(cfg, "seed_lr_self_attn_heads", 3)),
        "seed_lr_block_size": getattr(cfg, "seed_lr_block_size", (8, 8)),
        "seed_gm_iters": int(getattr(cfg, "seed_gm_iters", 3)),
        "seed_use_hr_median_pool": bool(getattr(cfg, "seed_use_hr_median_pool", False)),
        "seed_hr_median_kernel_size": int(getattr(cfg, "seed_hr_median_kernel_size", 3)),
        "use_lr_conv1": bool(getattr(cfg, "use_lr_conv1", True)),
        "use_lr_conv2": bool(getattr(cfg, "use_lr_conv2", True)),
        "use_lr_conv3": bool(getattr(cfg, "use_lr_conv3", False)),
        "lr_conv1_kernel_size": int(getattr(cfg, "lr_conv1_kernel_size", 3)),
        "lr_conv1_residual_weight": float(getattr(cfg, "lr_conv1_residual_weight", 1.0)),
        "lr_conv2_kernel_size": int(getattr(cfg, "lr_conv2_kernel_size", 9)),
        "lr_conv3_kernel_size": int(getattr(cfg, "lr_conv3_kernel_size", 9)),
        "lr_conv3_dilation": int(getattr(cfg, "lr_conv3_dilation", 1)),
        "hr_conv1_kernel_size": int(getattr(cfg, "hr_conv1_kernel_size", 3)),
        "hr_conv1_residual_weight": float(getattr(cfg, "hr_conv1_residual_weight", 1.0)),
        "use_hr_conv2": bool(getattr(cfg, "use_hr_conv2", False)),
        "hr_conv2_kernel_size": getattr(cfg, "hr_conv2_kernel_size", None),
        "use_residual_hr2": bool(getattr(cfg, "use_residual_hr2", True)),
        "hr_conv2_residual_weight": float(getattr(cfg, "hr_conv2_residual_weight", 1.0)),
        "use_hr_conv3": bool(getattr(cfg, "use_hr_conv3", False)),
        "hr_conv3_kernel_size": getattr(cfg, "hr_conv3_kernel_size", None),
        "use_residual_hr3": bool(getattr(cfg, "use_residual_hr3", True)),
        "hr_conv3_residual_weight": float(getattr(cfg, "hr_conv3_residual_weight", 1.0)),
        "conv_feature_mask_cosine_threshold": float(
            getattr(cfg, "conv_feature_mask_cosine_threshold", 0.98)
        ),
        "conv_feature_mask_l2_threshold": getattr(
            cfg, "conv_feature_mask_l2_threshold", None
        ),
        "conv_feature_mask_soft": bool(getattr(cfg, "conv_feature_mask_soft", False)),
        "conv_feature_mask_temperature": float(
            getattr(cfg, "conv_feature_mask_temperature", 32.0)
        ),
        "hr_conv_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv_feature_mask_cosine_threshold", None
        ),
        "hr_conv_feature_mask_l2_threshold": getattr(
            cfg, "hr_conv_feature_mask_l2_threshold", None
        ),
        "hr_conv_feature_mask_soft": getattr(cfg, "hr_conv_feature_mask_soft", None),
        "hr_conv_feature_mask_temperature": getattr(
            cfg, "hr_conv_feature_mask_temperature", None
        ),
        "hr_conv2_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv2_feature_mask_cosine_threshold", None
        ),
        "hr_conv2_feature_mask_l2_threshold": getattr(
            cfg, "hr_conv2_feature_mask_l2_threshold", None
        ),
        "hr_conv2_feature_mask_soft": getattr(cfg, "hr_conv2_feature_mask_soft", None),
        "hr_conv2_feature_mask_temperature": getattr(
            cfg, "hr_conv2_feature_mask_temperature", None
        ),
        "hr_conv3_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv3_feature_mask_cosine_threshold", None
        ),
        "hr_conv3_feature_mask_l2_threshold": getattr(
            cfg, "hr_conv3_feature_mask_l2_threshold", None
        ),
        "hr_conv3_feature_mask_soft": getattr(cfg, "hr_conv3_feature_mask_soft", None),
        "hr_conv3_feature_mask_temperature": getattr(
            cfg, "hr_conv3_feature_mask_temperature", None
        ),
        "use_residual_lr1": bool(getattr(cfg, "use_residual_lr1", False)),
        "use_residual_lr2": bool(getattr(cfg, "use_residual_lr2", False)),
        "use_residual_lr3": bool(getattr(cfg, "use_residual_lr3", False)),
        "use_residual_hr1": bool(getattr(cfg, "use_residual_hr1", False)),
        "use_attention": bool(getattr(cfg, "use_attention", True)),
        "use_grain_attention": bool(getattr(cfg, "use_grain_attention", True)),
        "grain_attention_boundary_source": str(
            getattr(cfg, "grain_attention_boundary_source", "hr")
        ),
        "enable_lr_grain_attention_layer": bool(
            getattr(cfg, "enable_lr_grain_attention_layer", getattr(cfg, "use_lr_grain_attention", False))
        ),
        "enable_hr_grain_attention_layer": bool(
            getattr(
                cfg,
                "enable_hr_grain_attention_layer",
                bool(getattr(cfg, "use_attention", True)) and bool(getattr(cfg, "use_grain_attention", True)),
            )
        ),
        "enable_hr_block_attention_layer": bool(
            getattr(
                cfg,
                "enable_hr_block_attention_layer",
                bool(getattr(cfg, "use_attention", True)) and (not bool(getattr(cfg, "use_grain_attention", True))),
            )
        ),
        "hr_grain_attention_boundary_source": str(
            getattr(
                cfg,
                "hr_grain_attention_boundary_source",
                getattr(cfg, "grain_attention_boundary_source", "hr"),
            )
        ),
        "grain_attn_boundary_threshold": float(
            getattr(cfg, "grain_attn_boundary_threshold", 0.5)
        ),
        "use_lr_grain_attention": bool(getattr(cfg, "use_lr_grain_attention", False)),
        "num_lr_grain_attn_blocks": int(getattr(cfg, "num_lr_grain_attn_blocks", 1)),
        "lr_grain_attn_num_channels": int(getattr(cfg, "lr_grain_attn_num_channels", 8)),
        "lr_grain_attn_checkpoint": bool(getattr(cfg, "lr_grain_attn_checkpoint", False)),
        "lr_grain_attn_boundary_threshold": float(
            getattr(cfg, "lr_grain_attn_boundary_threshold", 0.5)
        ),
        "num_hr_attn_blocks": int(getattr(cfg, "num_hr_attn_blocks", 1)),
        "hr_attn_num_channels": int(getattr(cfg, "hr_attn_num_channels", 8)),
        "hr_attn_block_size": int(getattr(cfg, "hr_attn_block_size", 16)),
        "hr_attn_tp_out_chunk_size": getattr(cfg, "hr_attn_tp_out_chunk_size", 2048),
        "hr_attn_checkpoint": bool(getattr(cfg, "hr_attn_checkpoint", False)),
        "use_boundary_gate": bool(getattr(cfg, "use_boundary_gate", False)),
        "num_lr_attn_blocks": int(getattr(cfg, "num_lr_attn_blocks", 0)),
        "lr_attn_block_size": int(getattr(cfg, "lr_attn_block_size", 8)),
        "use_hr_conv1": bool(getattr(cfg, "use_hr_conv1", True)),
        "use_masked_spatial_conv": bool(getattr(cfg, "use_masked_spatial_conv", True)),
        "use_masked_upsample": bool(getattr(cfg, "use_masked_upsample", True)),
        "spatial_mask_tau": float(getattr(cfg, "spatial_mask_tau", 0.6)),
        "spatial_mask_min": float(getattr(cfg, "spatial_mask_min", 0.0)),
        "spatial_mask_strength": float(getattr(cfg, "spatial_mask_strength", 1.0)),
        "spatial_hard_threshold": float(getattr(cfg, "spatial_hard_threshold", 0.85)),
        "upsample_mask_tau": getattr(cfg, "upsample_mask_tau", None),
        "upsample_mask_min": getattr(cfg, "upsample_mask_min", None),
        "upsample_mask_strength": getattr(cfg, "upsample_mask_strength", None),
        "upsample_hard_threshold": getattr(cfg, "upsample_hard_threshold", None),
        "mask_eps": float(getattr(cfg, "mask_eps", 1e-6)),
        "hr_attn_mask_tau": float(getattr(cfg, "hr_attn_mask_tau", 0.6)),
        "hr_attn_mask_spatial_sigma": float(getattr(cfg, "hr_attn_mask_spatial_sigma", 0.35)),
        "hr_attn_mask_min": float(getattr(cfg, "hr_attn_mask_min", 0.0)),
        "hr_attn_mask_strength": float(getattr(cfg, "hr_attn_mask_strength", 1.0)),
        "hr_attn_hard_threshold": float(getattr(cfg, "hr_attn_hard_threshold", 0.85)),
        "decoder_cubochoric_resolution": int(getattr(cfg, "decoder_cubochoric_resolution", 1)),
        "decoder_num_starts": int(getattr(cfg, "decoder_num_starts", 2)),
        "decoder_steps": int(getattr(cfg, "decoder_steps", 1)),
        "decoder_lr": float(getattr(cfg, "decoder_lr", 0.05)),
        "decoder_method": str(getattr(cfg, "decoder_method", "cubochoric")),
        "decoder_max_table_rows": getattr(cfg, "decoder_max_table_rows", None),
        "decoder_table_cache_dir": getattr(cfg, "decoder_table_cache_dir", "out/decoder_lookup_tables"),
        "decoder_backend": str(getattr(cfg, "decoder_backend", "optimizing")),
        "feature_irreps": str(getattr(cfg, "feature_irreps", "full")),
        "window_size": int(getattr(cfg, "window_size", 5)),
        "kmax_slots": int(getattr(cfg, "kmax_slots", 10)),
        "cluster_threshold_deg": float(getattr(cfg, "cluster_threshold_deg", 2.0)),
        "cluster_feature_l2_threshold": getattr(cfg, "cluster_feature_l2_threshold", None),
        "cluster_connectivity": int(getattr(cfg, "cluster_connectivity", 8)),
        "num_experts": int(getattr(cfg, "num_experts", 12)),
        "top_k_experts": int(getattr(cfg, "top_k_experts", 2)),
        "phase_dim": int(getattr(cfg, "phase_dim", 32)),
        "ocrp_router_hidden_dim": int(getattr(cfg, "ocrp_router_hidden_dim", 128)),
        "ocrp_router_conv_hidden_dim": int(getattr(cfg, "ocrp_router_conv_hidden_dim", 64)),
        "ocrp_router_slot_mass_power": float(getattr(cfg, "ocrp_router_slot_mass_power", 0.25)),
        "ocrp_router_uniform_slot_mix": float(getattr(cfg, "ocrp_router_uniform_slot_mix", 0.75)),
        "ocrp_router_use_slot_type_meta": bool(getattr(cfg, "ocrp_router_use_slot_type_meta", True)),
        "ocrp_router_geom_logit_bias": float(getattr(cfg, "ocrp_router_geom_logit_bias", 0.0)),
        "ocrp_proposal_hidden_dim": int(getattr(cfg, "ocrp_proposal_hidden_dim", 128)),
        "ocrp_slot_ratio_loss_weight": float(getattr(cfg, "ocrp_slot_ratio_loss_weight", 0.0)),
        "ocrp_router_geom_loss_weight": float(getattr(cfg, "ocrp_router_geom_loss_weight", 0.0)),
        "ocrp_router_geom_boundary_only": bool(getattr(cfg, "ocrp_router_geom_boundary_only", False)),
        "ocrp_slot_ratio_temperature": float(getattr(cfg, "ocrp_slot_ratio_temperature", 1.0)),
        "ocrp_straight_through": bool(getattr(cfg, "ocrp_straight_through", True)),
        "ocrp_mode": str(getattr(cfg, "ocrp_mode", "pixel_patch")),
        "macro_lr_tile_size": int(getattr(cfg, "macro_lr_tile_size", 3)),
        "macro_lr_stride_shape": getattr(cfg, "macro_lr_stride_shape", None),
        "ocrp_token_conditioned_member_bias": getattr(cfg, "ocrp_token_conditioned_member_bias", None),
        "ocrp_upsample_residual": bool(getattr(cfg, "ocrp_upsample_residual", False)),
        "ocrp_upsample_residual_weight": float(getattr(cfg, "ocrp_upsample_residual_weight", 1.0)),
        "ocrp_pool_chunk_size": int(getattr(cfg, "ocrp_pool_chunk_size", 512)),
        "ocrp_router_chunk_size": int(getattr(cfg, "ocrp_router_chunk_size", 512)),
        "ocrp_router_use_mlp_encoder": bool(getattr(cfg, "ocrp_router_use_mlp_encoder", False)),
        "ocrp_router_center_prior_weight": float(getattr(cfg, "ocrp_router_center_prior_weight", 0.0)),
        "ocrp_router_mode": str(getattr(cfg, "ocrp_router_mode", "geometric")),
        "ocrp_router_use_raw_token_ctx": bool(getattr(cfg, "ocrp_router_use_raw_token_ctx", False)),
        "ocrp_pool_center_bias_init": getattr(cfg, "ocrp_pool_center_bias_init", None),
        "ocrp_proposal_query_residual_scale": float(getattr(cfg, "ocrp_proposal_query_residual_scale", 0.5)),
        "ocrp_proposal_chunk_size": int(getattr(cfg, "ocrp_proposal_chunk_size", 128)),
        "ocrp_proposal_token_chunk_size": getattr(cfg, "ocrp_proposal_token_chunk_size", None),
        "rrctp_score_hidden_dim": int(getattr(cfg, "rrctp_score_hidden_dim", 64)),
        "rrctp_router_hidden_dim": int(getattr(cfg, "rrctp_router_hidden_dim", 128)),
        "rrctp_query_hidden_dim": int(getattr(cfg, "rrctp_query_hidden_dim", 64)),
        "rrctp_seed_hidden_dim": int(getattr(cfg, "rrctp_seed_hidden_dim", 128)),
        "rrctp_token_chunk_size": int(getattr(cfg, "rrctp_token_chunk_size", 1024)),
        "decoder_eager_init": bool(getattr(cfg, "decoder_eager_init", False)),
    }

    model_kwargs = {k: v for k, v in model_kwargs.items() if k in _init_params}
    model = IsoEmbeddingSRAttn(**model_kwargs).to(device)

    summary_max_rows_cfg = getattr(cfg, "model_summary_max_rows", None)
    summary_max_rows = int(summary_max_rows_cfg) if summary_max_rows_cfg is not None else None
    print_model_summary = bool(getattr(cfg, "print_model_summary", True))
    if print_model_summary:
        _print_model_summary(model, max_rows=summary_max_rows)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_tb = bool(getattr(getattr(cfg, "logging", {}), "tensorboard", True))
    writer = None
    if use_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            writer = SummaryWriter(log_dir=exp_dir / "runs")
        except Exception as exc:
            print(f"[warning] TensorBoard disabled (could not import SummaryWriter): {exc}")

    checkpoints_dir = Path(cfg.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoints_dir / "history.json"
    best_ckpt = checkpoints_dir / "best_model.pt"
    last_ckpt = checkpoints_dir / "last_checkpoint.pt"
    logging_cfg = getattr(cfg, "logging", {}) or {}
    save_best_only = bool(getattr(logging_cfg, "save_best_only", False))
    save_last_checkpoint = bool(getattr(cfg, "save_last_checkpoint", not save_best_only))
    save_epoch_checkpoints = bool(getattr(cfg, "save_epoch_checkpoints", not save_best_only))
    final_viz = bool(getattr(cfg, "final_viz", True))
    plot_loss_curves = bool(getattr(cfg, "plot_loss_curves", True))

    start_epoch = 0
    best_val_loss = float("inf")
    history = _init_history()

    if args.resume and last_ckpt.exists():
        start_epoch, best_val_loss, history = _load_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    clip = float(getattr(cfg, "clip", 1.0))
    tv_loss_weight = float(getattr(cfg, "tv_loss_weight", 0.0))
    save_every = int(getattr(cfg, "save_every", 1))
    viz_every = int(getattr(cfg, "viz_every", save_every))
    viz_ref_dir = str(getattr(cfg, "viz_ref_dir", "ALL"))
    viz_enable_probe_stage_viz = bool(getattr(cfg, "viz_enable_probe_stage_viz", True))
    viz_sample_index = int(getattr(cfg, "viz_sample_index", 0))
    viz_sample_key_cfg = getattr(cfg, "viz_sample_key", None)
    viz_sample_key = (
        None
        if viz_sample_key_cfg is None or not str(viz_sample_key_cfg).strip()
        else str(viz_sample_key_cfg).strip()
    )
    viz_force_cpu = bool(getattr(cfg, "viz_force_cpu", False))
    memory_debug_every = int(getattr(cfg, "memory_debug_every", 0))
    cuda_empty_cache_every = int(getattr(cfg, "cuda_empty_cache_every", 0))
    epochs = int(cfg.epochs)
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    for epoch in range(start_epoch, epochs):
        model_core = _unwrap_model(model)
        current_ocrp_residual_weight = _apply_ocrp_upsample_residual_schedule(
            model_core,
            cfg,
            epoch=epoch,
            total_epochs=epochs,
        )
        train_loss, train_metrics = _train_one_epoch(
            model_core,
            loaders["train"],
            optimizer,
            device,
            clip=clip,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            memory_debug_every=memory_debug_every,
            cuda_empty_cache_every=cuda_empty_cache_every,
            tv_loss_weight=tv_loss_weight,
        )
        val_loss, val_metrics = _validate_one_epoch(
            model_core,
            loaders["val"],
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            tv_loss_weight=tv_loss_weight,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(float(val_loss))
            else:
                scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["lr"].append(current_lr)
        if current_ocrp_residual_weight is not None:
            history["ocrp_upsample_residual_weight"].append(
                float(current_ocrp_residual_weight)
            )
        _append_metric_history(history["train_terms"], train_metrics, len(history["train"]) - 1)
        _append_metric_history(history["val_terms"], val_metrics, len(history["val"]) - 1)

        if writer is not None:
            writer.add_scalar("Loss/Train", float(train_loss), epoch)
            writer.add_scalar("Loss/Val", float(val_loss), epoch)
            writer.add_scalar("LR", current_lr, epoch)
            if current_ocrp_residual_weight is not None:
                writer.add_scalar(
                    "OCRP/UpsampleResidualWeight",
                    float(current_ocrp_residual_weight),
                    epoch,
                )
            for key, value in train_metrics.items():
                writer.add_scalar(f"LossTerms/Train/{key}", float(value), epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"LossTerms/Val/{key}", float(value), epoch)

        metric_bits = []
        for label, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in ("loss_feat", "loss_boundary", "loss_side_correct", "loss_side_entropy"):
                if key in metrics:
                    metric_bits.append(f"{label}_{key.replace('loss_', '')}={metrics[key]:.3e}")
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train={train_loss:.6e} val={val_loss:.6e} lr={current_lr:.2e}"
            + (
                f" | ocrp_residual={current_ocrp_residual_weight:.3f}"
                if current_ocrp_residual_weight is not None
                else ""
            )
            + (f" | {' '.join(metric_bits)}" if metric_bits else "")
        )

        if save_last_checkpoint:
            _save_checkpoint(
                last_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            _save_checkpoint(
                best_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )

        if save_epoch_checkpoints and save_every > 0 and ((epoch + 1) % save_every == 0):
            epoch_ckpt = checkpoints_dir / f"epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(
                epoch_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )

        if viz_every > 0 and ((epoch + 1) % viz_every == 0):
            viz_dir = exp_dir / "visualizations" / f"epoch_{epoch + 1:04d}"
            _render_sr_hr_lr_ipf(
                model_core=model_core,
                data_loader=loaders["val"],
                sym_class=sym_class,
                out_png=viz_dir / "lr_sr_hr_ipf.png",
                ref_dir=viz_ref_dir,
                enable_probe_stage_viz=viz_enable_probe_stage_viz,
                sample_index=viz_sample_index,
                sample_key=viz_sample_key,
                force_cpu=viz_force_cpu,
            )

        _save_history(history_path, history)
        if plot_loss_curves:
            _save_loss_plot(
                exp_dir / "visualizations" / "loss_curves.png",
                history,
                exp_name=exp_dir.name,
            )

    if final_viz:
        final_viz_dir = exp_dir / "visualizations" / "final"
        _render_sr_hr_lr_ipf(
            model_core=_unwrap_model(model),
            data_loader=loaders["val"],
            sym_class=sym_class,
            out_png=final_viz_dir / "lr_sr_hr_ipf.png",
            ref_dir=viz_ref_dir,
            enable_probe_stage_viz=viz_enable_probe_stage_viz,
            sample_index=viz_sample_index,
            sample_key=viz_sample_key,
            force_cpu=viz_force_cpu,
        )
        _render_final_probe_split_viz(
            model_core=_unwrap_model(model),
            loaders=loaders,
            sym_class=sym_class,
            out_root=final_viz_dir,
            ref_dir=viz_ref_dir,
            enable_probe_stage_viz=viz_enable_probe_stage_viz,
            sample_index=viz_sample_index,
            sample_key=viz_sample_key,
            force_cpu=viz_force_cpu,
        )

    if writer is not None:
        writer.close()

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    main()
