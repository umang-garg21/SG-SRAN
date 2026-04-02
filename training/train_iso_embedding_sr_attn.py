# -*- coding:utf-8 -*-
"""
Train IsoEmbeddingSRAttn with local-iso feature-space SR loss.

This trainer is specialized for models.SR_double_conv_SRattn.IsoEmbeddingSRAttn
and optimizes model.feature_loss_sr(...) directly.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import get_seed_from_config, set_seed
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IsoEmbeddingSRAttn")
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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _save_history(history_path: Path, history: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _save_loss_plot(plot_path: Path, history: dict, exp_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = history.get("train", [])
        val   = history.get("val", [])
        lr    = history.get("lr", [])
        epochs = list(range(1, len(train) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        fig.suptitle(exp_name, fontsize=11)

        ax1.plot(epochs, train, label="train", linewidth=1.5)
        ax1.plot(epochs, val,   label="val",   linewidth=1.5, linestyle="--")
        ax1.set_ylabel("MSE Loss (feature space)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, lr[:len(epochs)], color="tab:orange", linewidth=1.5)
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warning] Could not save loss plot: {exc}")


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
    if hasattr(model_core, "_prepare_boundary_context"):
        ctx = model_core._prepare_boundary_context(
            lr_boundary_map=lr_boundary_map_2d,
            lr_shape=lr_shape,
            batch_size=1,
            device=device,
            dtype=dtype,
        )
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
    if hasattr(model_core, "_prepare_boundary_context"):
        ctx = model_core._prepare_boundary_context(
            lr_boundary_map=lr_boundary_map_2d,
            lr_shape=lr_shape,
            batch_size=1,
            device=device,
            dtype=dtype,
        )
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
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    history = ckpt.get("history", {"train": [], "val": [], "lr": []})
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
) -> float:
    model_core.train()
    total_loss = 0.0
    n_steps = 0
    _feature_loss_params = inspect.signature(model_core.feature_loss_sr).parameters
    _supports_tv_loss_weight = "tv_loss_weight" in _feature_loss_params
    _supports_lr_boundary_map = "lr_boundary_map" in _feature_loss_params
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
            loss = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                **loss_kwargs,
            )
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

    return total_loss / max(1, n_steps)


@torch.no_grad()
def _validate_one_epoch(
    model_core: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tv_loss_weight: float = 0.0,
) -> float:
    model_core.eval()
    total_loss = 0.0
    n_steps = 0
    _feature_loss_params = inspect.signature(model_core.feature_loss_sr).parameters
    _supports_tv_loss_weight = "tv_loss_weight" in _feature_loss_params
    _supports_lr_boundary_map = "lr_boundary_map" in _feature_loss_params
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
            loss = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                **loss_kwargs,
            )
        total_loss += float(loss.detach().item())
        n_steps += 1

    return total_loss / max(1, n_steps)


def _render_sr_hr_lr_ipf(
    model_core: torch.nn.Module,
    data_loader,
    sym_class,
    out_png: Path,
    ref_dir: str = "ALL",
) -> bool:
    """
    Render LR/SR/HR IPF comparison from the first sample in data_loader.
    """
    try:
        from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side
    except Exception as exc:
        print(f"[warning] Visualization unavailable (import failed): {exc}")
        return False

    batch = next(iter(data_loader), None)
    if batch is None:
        print("[warning] Skipping visualization: loader is empty.")
        return False

    lr, hr, lr_boundary_map = _unpack_batch(batch)
    if lr is None or hr is None or lr.shape[0] == 0 or hr.shape[0] == 0:
        print("[warning] Skipping visualization: empty LR/HR batch.")
        return False

    try:
        model_was_training = model_core.training
        model_core.eval()

        device = next(model_core.parameters()).device
        lr0 = _to_hwc_quat_single(lr[0].to(device=device, dtype=torch.float32))
        hr0 = _to_hwc_quat_single(hr[0].to(device=device, dtype=torch.float32))
        lr_h, lr_w = int(lr0.shape[0]), int(lr0.shape[1])
        hr_h, hr_w = int(hr0.shape[0]), int(hr0.shape[1])
        lr_boundary0 = None
        if lr_boundary_map is not None:
            lr_boundary0 = lr_boundary_map[0].to(device=device, dtype=torch.float32)

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

        out_png.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"[warning] Failed to render LR/SR/HR IPF visualization: {exc}")
        return False
    finally:
        if "model_was_training" in locals() and model_was_training:
            model_core.train()


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
    use_lr_boundary_map = bool(getattr(cfg, "use_lr_boundary_map", _model_supports_lr_boundary))
    if _model_requires_lr_boundary:
        use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    print(
        f"LR boundary maps from dataloader: {use_lr_boundary_map} "
        f"(model supports={_model_supports_lr_boundary}, requires={_model_requires_lr_boundary})"
    )

    seed = int(get_seed_from_config(cfg))
    set_seed(seed)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")
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
        "device": device,
        "upsample_factor": getattr(cfg, "upsample_factor", getattr(cfg, "scale", 4)),
        "upsample_context_kernel_size": int(getattr(cfg, "upsample_context_kernel_size", 3)),
        "upsample_residual": bool(getattr(cfg, "upsample_residual", True)),
        "upsample_transpose_overlap": int(getattr(cfg, "upsample_transpose_overlap", 2)),
        "upsample_boundary_threshold": float(getattr(cfg, "upsample_boundary_threshold", 0.5)),
        "upsample_boundary_smooth_sigma": float(getattr(cfg, "upsample_boundary_smooth_sigma", 2.0)),
        "upsample_boundary_smooth_iters": int(getattr(cfg, "upsample_boundary_smooth_iters", 12)),
        "upsample_boundary_sdf_shift": float(getattr(cfg, "upsample_boundary_sdf_shift", 0.7)),
        "evidence_radius": int(getattr(cfg, "evidence_radius", 1)),
        "sdf_hidden_dim": int(getattr(cfg, "sdf_hidden_dim", 64)),
        "guidance_dim": int(getattr(cfg, "guidance_dim", 16)),
        "stats_code_dim": int(getattr(cfg, "stats_code_dim", 16)),
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
        "use_lr_conv1": bool(getattr(cfg, "use_lr_conv1", True)),
        "use_lr_conv2": bool(getattr(cfg, "use_lr_conv2", True)),
        "use_lr_conv3": bool(getattr(cfg, "use_lr_conv3", False)),
        "lr_conv1_kernel_size": int(getattr(cfg, "lr_conv1_kernel_size", 3)),
        "lr_conv2_kernel_size": int(getattr(cfg, "lr_conv2_kernel_size", 9)),
        "lr_conv3_kernel_size": int(getattr(cfg, "lr_conv3_kernel_size", 9)),
        "lr_conv3_dilation": int(getattr(cfg, "lr_conv3_dilation", 1)),
        "hr_conv1_kernel_size": int(getattr(cfg, "hr_conv1_kernel_size", 3)),
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
    }

    model_kwargs = {k: v for k, v in model_kwargs.items() if k in _init_params}
    model = IsoEmbeddingSRAttn(**model_kwargs).to(device)

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

    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train": [], "val": [], "lr": []}

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
    memory_debug_every = int(getattr(cfg, "memory_debug_every", 0))
    cuda_empty_cache_every = int(getattr(cfg, "cuda_empty_cache_every", 0))
    epochs = int(cfg.epochs)
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    for epoch in range(start_epoch, epochs):
        model_core = _unwrap_model(model)
        train_loss = _train_one_epoch(
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
        val_loss = _validate_one_epoch(
            model_core,
            loaders["val"],
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            tv_loss_weight=tv_loss_weight,
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["lr"].append(current_lr)

        if writer is not None:
            writer.add_scalar("Loss/Train", float(train_loss), epoch)
            writer.add_scalar("Loss/Val", float(val_loss), epoch)
            writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train={train_loss:.6e} val={val_loss:.6e} lr={current_lr:.2e}"
        )

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

        if save_every > 0 and ((epoch + 1) % save_every == 0):
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
            )

        _save_history(history_path, history)
        _save_loss_plot(
            exp_dir / "visualizations" / "loss_curves.png",
            history,
            exp_name=exp_dir.name,
        )

    final_viz_dir = exp_dir / "visualizations" / "final"
    _render_sr_hr_lr_ipf(
        model_core=_unwrap_model(model),
        data_loader=loaders["test"] if "test" in loaders else loaders["val"],
        sym_class=sym_class,
        out_png=final_viz_dir / "lr_sr_hr_ipf.png",
        ref_dir=viz_ref_dir,
    )

    if writer is not None:
        writer.close()

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    main()
