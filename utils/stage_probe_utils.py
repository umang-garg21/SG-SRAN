"""Utilities for decoding and visualizing intermediate irrep probe stages."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import numpy as np
import torch

from visualization.ipf_render import render_ipf_rgb


def pick_most_free_cuda_gpu() -> int | None:
    """Return the physical CUDA GPU index with the most free memory."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0 if torch.cuda.is_available() else None

    best_idx: int | None = None
    best_free_mib = -1
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpu_idx = int(parts[0])
            free_mib = int(parts[1])
        except ValueError:
            continue
        if free_mib > best_free_mib:
            best_idx = gpu_idx
            best_free_mib = free_mib
    return best_idx


def sanitize_probe_tensor(x: torch.Tensor) -> torch.Tensor:
    """Clamp NaN/Inf-heavy tensors before handing them to visualization code."""
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


def _left_mult_matrix_wxyz_batch(q_syms: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first symmetry quaternions to left-multiplication matrices."""
    w, x, y, z = q_syms.unbind(dim=-1)
    r0 = torch.stack([w, -x, -y, -z], dim=-1)
    r1 = torch.stack([x, w, -z, y], dim=-1)
    r2 = torch.stack([y, z, w, -x], dim=-1)
    r3 = torch.stack([z, -y, x, w], dim=-1)
    return torch.stack([r0, r1, r2, r3], dim=1)


def quat_ang_err_deg(
    q_pred_hwc: torch.Tensor,
    q_tgt_hwc: torch.Tensor,
    sym=None,
) -> torch.Tensor:
    """True symmetry-aware pairwise misorientation in degrees."""
    if sym is None:
        raise ValueError("sym must be provided for symmetry-aware pairwise misorientation.")

    q1 = q_pred_hwc
    q2 = q_tgt_hwc
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sym_ops_quat = torch.as_tensor(np.asarray(sym.data, dtype=np.float32))
    sym_ops = _left_mult_matrix_wxyz_batch(sym_ops_quat)
    q2_variants = torch.einsum("gij,hwj->hwgi", sym_ops, q2)
    dots = (q1.unsqueeze(-2) * q2_variants).sum(dim=-1).abs().clamp(0.0, 1.0)
    best = dots.max(dim=-1).values
    return torch.rad2deg(2.0 * torch.acos(best))


def resize_quat_target(target_hwc: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a quaternion image for same-resolution error comparisons in galleries."""
    h, w = int(target_hwc.shape[0]), int(target_hwc.shape[1])
    if (h, w) == tuple(size_hw):
        return target_hwc
    q = target_hwc.permute(2, 0, 1).unsqueeze(0)
    q = torch.nn.functional.interpolate(q, size=size_hw, mode="bilinear", align_corners=False)
    q = q.squeeze(0).permute(1, 2, 0)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _project_lr_array_to_hr_sparse(
    arr: np.ndarray,
    lr_hw: tuple[int, int],
    hr_hw: tuple[int, int],
    fill_value: float | int = 0,
) -> np.ndarray:
    """Embed an LR 2D/3D array on an HR canvas; unsampled pixels use fill_value."""
    h_lr, w_lr = int(lr_hw[0]), int(lr_hw[1])
    h_hr, w_hr = int(hr_hw[0]), int(hr_hw[1])
    if arr.shape[:2] != (h_lr, w_lr):
        raise ValueError(
            f"Array shape {tuple(arr.shape[:2])} does not match lr_hw={(h_lr, w_lr)}"
        )
    if h_lr <= 0 or w_lr <= 0 or h_hr <= 0 or w_hr <= 0:
        raise ValueError(f"Invalid shapes lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}")

    scale_y = h_hr // h_lr
    scale_x = w_hr // w_lr
    if h_lr * scale_y != h_hr or w_lr * scale_x != w_hr:
        raise ValueError(
            "HR shape must be an integer multiple of LR shape for sparse projection: "
            f"lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}"
        )

    out_shape = (h_hr, w_hr) + tuple(arr.shape[2:])
    out = np.full(out_shape, fill_value, dtype=arr.dtype)
    y_idx = np.arange(h_lr, dtype=np.int64) * scale_y + (scale_y // 2)
    x_idx = np.arange(w_lr, dtype=np.int64) * scale_x + (scale_x // 2)
    y_idx = np.clip(y_idx, 0, h_hr - 1)
    x_idx = np.clip(x_idx, 0, w_hr - 1)
    out[np.ix_(y_idx, x_idx)] = arr
    return out


def decode_probe_stages(
    model_obj,
    probe_stages: list[dict[str, Any]],
    sample_index: int = 0,
) -> list[dict[str, Any]]:
    """Decode a list of flattened A1 feature stages back to quaternion images."""
    decoded: list[dict[str, Any]] = []
    for stage in probe_stages:
        feat = stage["feat"]
        hw = tuple(stage["shape"])
        if isinstance(feat, torch.Tensor) and feat.dim() == 3:
            feat_single = feat[int(sample_index)]
        elif isinstance(feat, torch.Tensor) and feat.dim() == 2:
            feat_single = feat
        else:
            raise ValueError(f"Unsupported probe feature shape for stage {stage['name']!r}: {type(feat)} {getattr(feat, 'shape', None)}")

        feat_single = sanitize_probe_tensor(feat_single)
        with torch.enable_grad():
            q_flat = model_obj.decode(feat_single)
        decoded.append(
            {
                "name": str(stage["name"]),
                "shape": hw,
                "quat_hwc": q_flat.reshape(hw[0], hw[1], 4).detach().cpu(),
            }
        )
    return decoded


def _pick_aux_tensor(x: Any, sample_index: int = 0) -> torch.Tensor | None:
    if not isinstance(x, torch.Tensor):
        return None
    if x.dim() == 4:
        return x[int(sample_index)].detach().cpu()
    if x.dim() == 3:
        if x.shape[1] > 8 and x.shape[2] > 8 and x.shape[0] == 1:
            return x[0].detach().cpu()
        if x.shape[1] > 8 and x.shape[2] > 8 and x.shape[0] > 4:
            return x[int(sample_index)].detach().cpu()
        return x.detach().cpu()
    if x.dim() == 2:
        return x.detach().cpu()
    return None


def extract_scalar_probe_maps(
    aux: dict[str, Any],
    sample_index: int = 0,
) -> list[dict[str, Any]]:
    """Collect scalar diagnostics from the modified one-sided model aux dict."""

    maps: list[dict[str, Any]] = []

    evidence = aux.get("tensor")
    ev = _pick_aux_tensor(evidence, sample_index=sample_index)
    if ev is not None:
        if ev.dim() == 3:
            ev_mean = ev.mean(dim=0)
        else:
            ev_mean = ev
        maps.append({"name": "evidence_mean", "array": ev_mean.numpy(), "cmap": "inferno"})

    for key, cmap in (
        ("boundary_lr_1px", "gray"),
        ("boundary_hr_1px", "gray"),
        ("center_valid_hr", "gray"),
    ):
        arr = _pick_aux_tensor(aux.get(key), sample_index=sample_index)
        if arr is not None:
            maps.append({"name": key, "array": arr.squeeze().numpy(), "cmap": cmap, "vmin": 0.0, "vmax": 1.0})

    owner = _pick_aux_tensor(aux.get("hr_to_lr_owner"), sample_index=sample_index)
    if owner is not None:
        maps.append({"name": "hr_to_lr_owner", "array": owner.numpy(), "cmap": "nipy_spectral"})

    boundary_prob = _pick_aux_tensor(aux.get("boundary_prob_hr"), sample_index=sample_index)
    if boundary_prob is not None:
        maps.append({"name": "boundary_prob_hr", "array": boundary_prob.squeeze().numpy(), "cmap": "magma", "vmin": 0.0, "vmax": 1.0})

    boundary_logits_hr_refined = _pick_aux_tensor(aux.get("boundary_logits_hr_refined"), sample_index=sample_index)
    if boundary_logits_hr_refined is not None:
        maps.append(
            {
                "name": "boundary_prob_hr_refined",
                "array": torch.sigmoid(boundary_logits_hr_refined).squeeze().numpy(),
                "cmap": "magma",
                "vmin": 0.0,
                "vmax": 1.0,
            }
        )

    for key, cmap in (
        ("sdf_hr", "viridis"),
        ("boundary_band_hr", "inferno"),
    ):
        arr = _pick_aux_tensor(aux.get(key), sample_index=sample_index)
        if arr is not None:
            maps.append({"name": key, "array": arr.squeeze().numpy(), "cmap": cmap})

    side_probs = _pick_aux_tensor(aux.get("side_probs_hr"), sample_index=sample_index)
    if side_probs is not None and side_probs.dim() == 3 and side_probs.shape[0] == 2:
        maps.append({"name": "side_prob_plus", "array": side_probs[0].numpy(), "cmap": "cividis", "vmin": 0.0, "vmax": 1.0})
        maps.append({"name": "side_prob_minus", "array": side_probs[1].numpy(), "cmap": "cividis", "vmin": 0.0, "vmax": 1.0})

    return maps


def select_upsampler_stage_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the decoded stage that best represents the direct pre-mixing upsample output."""
    if not rows:
        raise ValueError("rows is empty")

    prioritized_exact = (
        "upsample_center_hr",
        "upsample_direct_hr",
        "upsample_base_hr",
        "upsample_shifted_mix_hr",
        "upsample_mixed_hr",
        "upsample_out_hr",
        "upsample_hr",
    )
    rows_by_name = {str(row["name"]).strip().lower(): row for row in rows}
    for key in prioritized_exact:
        if key in rows_by_name:
            return rows_by_name[key]

    upsample_rows = [row for row in rows if str(row["name"]).strip().lower().startswith("upsample_")]
    non_branch_rows = [
        row
        for row in upsample_rows
        if not any(tok in str(row["name"]).strip().lower() for tok in ("center", "plus", "minus"))
    ]
    if non_branch_rows:
        return non_branch_rows[-1]

    for row in rows:
        name = str(row["name"]).strip().lower()
        if "grain_attention_out" in name:
            return row

    if upsample_rows:
        return upsample_rows[-1]

    raise ValueError("Could not identify a decoded upsampler stage row.")


def _label_boundary_mask(labels_2d: np.ndarray) -> np.ndarray:
    mask = np.zeros(labels_2d.shape, dtype=bool)
    valid = labels_2d >= 0
    if labels_2d.ndim != 2:
        raise ValueError(f"Expected label image with shape (H,W), got {labels_2d.shape}")

    diff_y = labels_2d[1:, :] != labels_2d[:-1, :]
    valid_y = valid[1:, :] & valid[:-1, :]
    edge_y = diff_y & valid_y
    mask[1:, :] |= edge_y
    mask[:-1, :] |= edge_y

    diff_x = labels_2d[:, 1:] != labels_2d[:, :-1]
    valid_x = valid[:, 1:] & valid[:, :-1]
    edge_x = diff_x & valid_x
    mask[:, 1:] |= edge_x
    mask[:, :-1] |= edge_x
    return mask


def _label_centers(labels_2d: np.ndarray) -> list[tuple[int, float, float]]:
    if labels_2d.ndim != 2:
        raise ValueError(f"Expected label image with shape (H,W), got {labels_2d.shape}")
    centers: list[tuple[int, float, float]] = []
    ids = np.unique(labels_2d[labels_2d >= 0])
    for gid in ids.tolist():
        ys, xs = np.nonzero(labels_2d == int(gid))
        if ys.size == 0:
            continue
        centers.append((int(gid), float(xs.mean()), float(ys.mean())))
    return centers


def _annotate_grain_ids(ax, labels_2d: np.ndarray, color: str = "white") -> None:
    for gid, x_ctr, y_ctr in _label_centers(labels_2d):
        ax.text(
            x_ctr,
            y_ctr,
            str(gid),
            fontsize=8.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        )


def render_upsampler_boundary_overlay(
    lr_stage_row: dict[str, Any],
    lr_labels_dense: torch.Tensor,
    upsampler_stage_row: dict[str, Any],
    sdf_hr: torch.Tensor,
    hr_to_lr_owner: torch.Tensor,
    sym_class,
    out_png: str | Path,
    ref_dir: str = "X",
    pixels_per_image_pixel: int = 1,
) -> Path:
    """Render LR/HR grain correspondence beside the direct upsampled output with learned SDF overlay."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lr_quat_hwc = lr_stage_row.get("quat_hwc")
    hr_quat_hwc = upsampler_stage_row.get("quat_hwc")
    if not isinstance(lr_quat_hwc, torch.Tensor) or lr_quat_hwc.dim() != 3 or int(lr_quat_hwc.shape[-1]) != 4:
        raise ValueError("lr_stage_row['quat_hwc'] must be a quaternion image tensor of shape (H,W,4).")
    if not isinstance(hr_quat_hwc, torch.Tensor) or hr_quat_hwc.dim() != 3 or int(hr_quat_hwc.shape[-1]) != 4:
        raise ValueError("upsampler_stage_row['quat_hwc'] must be a quaternion image tensor of shape (H,W,4).")

    if not isinstance(lr_labels_dense, torch.Tensor):
        raise ValueError("lr_labels_dense must be a torch.Tensor.")
    if lr_labels_dense.dim() == 3 and int(lr_labels_dense.shape[0]) == 1:
        lr_labels_dense = lr_labels_dense[0]
    if lr_labels_dense.dim() != 2:
        raise ValueError(f"Expected lr_labels_dense with shape (H,W), got {tuple(lr_labels_dense.shape)}")

    if not isinstance(sdf_hr, torch.Tensor):
        raise ValueError("sdf_hr must be a torch.Tensor.")
    if sdf_hr.dim() == 3 and int(sdf_hr.shape[0]) == 1:
        sdf_hr = sdf_hr[0]
    if sdf_hr.dim() != 2:
        raise ValueError(f"Expected sdf_hr with shape (H,W), got {tuple(sdf_hr.shape)}")

    if not isinstance(hr_to_lr_owner, torch.Tensor):
        raise ValueError("hr_to_lr_owner must be a torch.Tensor.")
    if hr_to_lr_owner.dim() == 3 and int(hr_to_lr_owner.shape[0]) == 1:
        hr_to_lr_owner = hr_to_lr_owner[0]
    if hr_to_lr_owner.dim() != 2:
        raise ValueError(f"Expected hr_to_lr_owner with shape (H,W), got {tuple(hr_to_lr_owner.shape)}")

    lr_quat_hwc = lr_quat_hwc.detach().cpu()
    hr_quat_hwc = hr_quat_hwc.detach().cpu()
    lr_labels_dense = lr_labels_dense.detach().cpu().to(dtype=torch.long)
    sdf_hr = sdf_hr.detach().cpu().to(dtype=torch.float32)
    hr_to_lr_owner = hr_to_lr_owner.detach().cpu().to(dtype=torch.long)

    lr_hw = tuple(int(v) for v in lr_quat_hwc.shape[:2])
    hr_hw = tuple(int(v) for v in hr_quat_hwc.shape[:2])
    if tuple(int(v) for v in lr_labels_dense.shape) != lr_hw:
        raise ValueError(
            f"lr_labels_dense shape {tuple(lr_labels_dense.shape)} does not match LR stage shape {lr_hw}"
        )
    if tuple(int(v) for v in sdf_hr.shape) != hr_hw:
        raise ValueError(f"sdf_hr shape {tuple(sdf_hr.shape)} does not match HR stage shape {hr_hw}")
    if tuple(int(v) for v in hr_to_lr_owner.shape) != hr_hw:
        raise ValueError(
            f"hr_to_lr_owner shape {tuple(hr_to_lr_owner.shape)} does not match HR stage shape {hr_hw}"
        )

    lr_rgb_native = render_ipf_rgb(lr_quat_hwc.numpy().astype(np.float32), sym_class, ref_dir=ref_dir)
    hr_rgb = render_ipf_rgb(hr_quat_hwc.numpy().astype(np.float32), sym_class, ref_dir=ref_dir)
    lr_rgb = _project_lr_array_to_hr_sparse(lr_rgb_native, lr_hw, hr_hw, fill_value=0)

    lr_labels_np = lr_labels_dense.numpy()
    lr_labels_hr_np = _project_lr_array_to_hr_sparse(lr_labels_np, lr_hw, hr_hw, fill_value=-1)
    lr_boundary_np = _label_boundary_mask(lr_labels_hr_np)
    valid_lr = lr_labels_hr_np >= 0
    max_label = int(lr_labels_hr_np[valid_lr].max()) if bool(valid_lr.any()) else 0
    label_cmap = plt.get_cmap("tab20", max(1, max_label + 1))
    label_overlay = np.ma.masked_where(~valid_lr, lr_labels_hr_np)
    label_alpha = np.where(valid_lr, 0.28, 0.0)

    sdf_np = np.clip(sdf_hr.numpy(), 0.0, 1.0)
    sdf_alpha = 0.62 * sdf_np
    hr_owner_np = hr_to_lr_owner.numpy()
    hr_owner_boundary_np = _label_boundary_mask(hr_owner_np)

    stage_name = str(upsampler_stage_row.get("name", "upsampler_stage"))
    scale = max(1, int(pixels_per_image_pixel))
    panel_h, panel_w = hr_hw
    panel_h_px = panel_h * scale
    panel_w_px = panel_w * scale
    dpi_out = 180
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2.0 * panel_w_px / float(dpi_out), panel_h_px / float(dpi_out)),
        dpi=dpi_out,
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    axes[0].imshow(lr_rgb, interpolation="nearest", resample=False)
    axes[0].imshow(label_overlay, cmap=label_cmap, alpha=label_alpha, interpolation="nearest", resample=False)
    axes[0].contour(lr_boundary_np.astype(np.float32), levels=[0.5], colors=["#ffffff"], linewidths=0.9)
    _annotate_grain_ids(axes[0], lr_labels_hr_np, color="white")
    axes[0].axis("off")
    axes[0].text(
        0.01,
        0.99,
        f"LR->HR sparse + lr_labels_dense (IPF-{ref_dir})",
        transform=axes[0].transAxes,
        fontsize=9,
        color="white",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )

    axes[1].imshow(hr_rgb, interpolation="nearest", resample=False)
    sdf_artist = axes[1].imshow(sdf_np, cmap="magma", alpha=sdf_alpha, vmin=0.0, vmax=1.0, interpolation="nearest", resample=False)
    axes[1].contour(sdf_np, levels=[0.35, 0.55, 0.75], colors=["#ffe600"], linewidths=0.8, alpha=0.9)
    axes[1].contour(hr_owner_boundary_np.astype(np.float32), levels=[0.5], colors=["#00e5ff"], linewidths=0.8, alpha=0.8)
    _annotate_grain_ids(axes[1], hr_owner_np, color="white")
    axes[1].axis("off")
    axes[1].text(
        0.01,
        0.99,
        f"{stage_name}: sdf_hr + hr_to_lr_owner (IPF-{ref_dir})",
        transform=axes[1].transAxes,
        fontsize=9,
        color="white",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )

    fig.savefig(out_path, dpi=dpi_out)
    plt.close(fig)
    return out_path


def render_sdf_comparison(
    oldprep_sdf_hr: torch.Tensor,
    learned_sdf_hr: torch.Tensor,
    out_png: str | Path,
) -> Path:
    """Render a direct comparison between old prep SDF and learned sdf_hr."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(oldprep_sdf_hr, torch.Tensor) or not isinstance(learned_sdf_hr, torch.Tensor):
        raise ValueError("oldprep_sdf_hr and learned_sdf_hr must be torch.Tensor instances.")

    oldprep = oldprep_sdf_hr.detach().cpu().to(dtype=torch.float32)
    learned = learned_sdf_hr.detach().cpu().to(dtype=torch.float32)
    if oldprep.dim() == 3 and int(oldprep.shape[0]) == 1:
        oldprep = oldprep[0]
    if learned.dim() == 3 and int(learned.shape[0]) == 1:
        learned = learned[0]
    if oldprep.dim() != 2 or learned.dim() != 2:
        raise ValueError(
            f"Expected rank-2 SDF maps, got {tuple(oldprep.shape)} and {tuple(learned.shape)}"
        )
    if tuple(int(v) for v in oldprep.shape) != tuple(int(v) for v in learned.shape):
        raise ValueError(
            f"SDF shapes must match, got {tuple(oldprep.shape)} and {tuple(learned.shape)}"
        )

    oldprep_np = oldprep.numpy()
    learned_np = learned.numpy()
    diff_np = learned_np - oldprep_np
    abs_diff = np.abs(diff_np)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9), constrained_layout=True)
    fig.suptitle("Old Prep SDF Vs Learned sdf_hr", fontsize=14, y=1.02)

    im0 = axes[0].imshow(oldprep_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Old Prep SDF", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(learned_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Learned sdf_hr", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    vmax = float(max(1e-6, abs_diff.max()))
    im2 = axes[2].imshow(diff_np, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Difference: learned - oldprep", fontsize=11)
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.02,
        f"mean |diff| = {abs_diff.mean():.4f}\nmax |diff| = {abs_diff.max():.4f}",
        transform=axes[2].transAxes,
        fontsize=9,
        color="white",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def sample_attention_probe_pixels(
    boundary_mask_hr: torch.Tensor,
    num_total: int = 10,
    num_boundary: int = 3,
    seed: int = 0,
) -> list[dict[str, int | bool]]:
    """Sample HR probe pixels with a guaranteed boundary subset when available."""
    mask = boundary_mask_hr
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if mask.dim() != 2:
        raise ValueError(f"Expected boundary mask with shape (H,W), got {tuple(mask.shape)}")

    H, W = int(mask.shape[0]), int(mask.shape[1])
    rng = np.random.default_rng(int(seed))

    boundary_coords = np.argwhere(mask.detach().cpu().numpy() > 0.5)
    non_boundary_coords = np.argwhere(mask.detach().cpu().numpy() <= 0.5)

    n_boundary = min(int(num_boundary), len(boundary_coords), int(num_total))
    chosen: list[dict[str, int | bool]] = []
    used: set[tuple[int, int]] = set()

    if n_boundary > 0:
        sel = rng.choice(len(boundary_coords), size=n_boundary, replace=False)
        for idx in np.atleast_1d(sel):
            y, x = map(int, boundary_coords[int(idx)])
            chosen.append({"y_hr": y, "x_hr": x, "is_boundary": True})
            used.add((y, x))

    remaining = max(0, int(num_total) - len(chosen))
    available_non_boundary = [tuple(map(int, xy)) for xy in non_boundary_coords if tuple(map(int, xy)) not in used]
    if remaining > 0 and available_non_boundary:
        sel_idx = rng.choice(len(available_non_boundary), size=min(remaining, len(available_non_boundary)), replace=False)
        for idx in np.atleast_1d(sel_idx):
            y, x = available_non_boundary[int(idx)]
            chosen.append({"y_hr": y, "x_hr": x, "is_boundary": False})
            used.add((y, x))

    remaining = max(0, int(num_total) - len(chosen))
    if remaining > 0:
        all_coords = [(y, x) for y in range(H) for x in range(W) if (y, x) not in used]
        if all_coords:
            sel_idx = rng.choice(len(all_coords), size=min(remaining, len(all_coords)), replace=False)
            for idx in np.atleast_1d(sel_idx):
                y, x = all_coords[int(idx)]
                chosen.append({"y_hr": y, "x_hr": x, "is_boundary": bool(mask[y, x] > 0.5)})
                used.add((y, x))

    return chosen


def compute_attention_probe_traces(
    model_obj,
    aux: dict[str, Any],
    probe_points: list[dict[str, int | bool]],
    sample_index: int = 0,
) -> list[dict[str, Any]]:
    """Compute LR whole-grain attention heatmaps for selected HR probe pixels."""
    helper = getattr(model_obj, "grain_attention_helper", None)
    if helper is None:
        raise ValueError("Model does not have an active grain_attention_helper.")

    feat_lr = aux.get("feat_lr_a1_post_lr")
    hr_to_lr_owner = aux.get("hr_to_lr_owner")
    lr_labels = aux.get("lr_labels")
    if feat_lr is None or hr_to_lr_owner is None or lr_labels is None:
        raise ValueError("Attention probe requires feat_lr_a1_post_lr, hr_to_lr_owner, and lr_labels in aux.")

    if feat_lr.dim() == 3:
        feat_lr = feat_lr[int(sample_index) : int(sample_index) + 1]
    elif feat_lr.dim() == 2:
        feat_lr = feat_lr.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected feat_lr_a1_post_lr shape: {tuple(feat_lr.shape)}")

    if lr_labels.dim() == 3:
        lr_labels = lr_labels[int(sample_index) : int(sample_index) + 1]
    elif lr_labels.dim() == 2:
        lr_labels = lr_labels.unsqueeze(0)

    if hr_to_lr_owner.dim() == 3:
        hr_to_lr_owner = hr_to_lr_owner[int(sample_index) : int(sample_index) + 1]
    elif hr_to_lr_owner.dim() == 2:
        hr_to_lr_owner = hr_to_lr_owner.unsqueeze(0)

    H_lr, W_lr = int(lr_labels.shape[-2]), int(lr_labels.shape[-1])
    feat_lr_img = feat_lr.reshape(1, H_lr, W_lr, feat_lr.shape[-1]).permute(0, 3, 1, 2).contiguous()
    pixel_coords = [(int(item["y_hr"]), int(item["x_hr"])) for item in probe_points]
    trace_raw = helper.trace_hr_attention_pixels(
        feat_lr_img=feat_lr_img,
        hr_to_lr_map=hr_to_lr_owner,
        lr_labels=lr_labels,
        pixel_coords=pixel_coords,
    )
    traces: list[dict[str, Any]] = []
    for item, trace in zip(probe_points, trace_raw):
        merged = dict(trace)
        merged["is_boundary"] = bool(item["is_boundary"])
        traces.append(merged)
    return traces


def _bright_black_blue_red_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bright_black_blue_red",
        [
            (0.0, "#000000"),
            (0.18, "#001a88"),
            (0.45, "#1268ff"),
            (1.0, "#ff2a00"),
        ],
    )


def _masked_map(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(arr, mask=mask <= 0.0)


def _cmap_with_black(cmap_name: str):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")
    return cmap


class InteractiveAttentionProbeFigure:
    """Interactive one-point grain-attention probe driven by mouse clicks."""

    def __init__(
        self,
        model_obj,
        aux: dict[str, Any],
        sr_quat_hwc: torch.Tensor,
        sym_class,
        sample_index: int = 0,
        out_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.model_obj = model_obj
        self.aux = aux
        self.sym_class = sym_class
        self.sample_index = int(sample_index)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.sr_rgb = render_ipf_rgb(sr_quat_hwc.detach().cpu().numpy().astype(np.float32), sym_class, ref_dir="X")
        self.boundary_mask_hr = aux.get("boundary_hr_1px")
        if not isinstance(self.boundary_mask_hr, torch.Tensor):
            raise ValueError("Interactive attention probe requires aux['boundary_hr_1px'].")
        if self.boundary_mask_hr.dim() == 4:
            self.boundary_mask_hr = self.boundary_mask_hr[self.sample_index, 0].detach().cpu()
        elif self.boundary_mask_hr.dim() == 3:
            self.boundary_mask_hr = self.boundary_mask_hr[self.sample_index].detach().cpu()
        else:
            self.boundary_mask_hr = self.boundary_mask_hr.detach().cpu()

        self.H, self.W = int(self.sr_rgb.shape[0]), int(self.sr_rgb.shape[1])
        self.rng = np.random.default_rng(int(seed))
        boundary_np = self.boundary_mask_hr.numpy() > 0.5
        self.boundary_coords = np.argwhere(boundary_np)
        self.nonboundary_coords = np.argwhere(~boundary_np)
        self._current_point: tuple[int, int] | None = None
        self._current_trace: dict[str, Any] | None = None

        self.fig = plt.figure(figsize=(16.5, 9.5), constrained_layout=True)
        self.axes = np.asarray(self.fig.subplots(2, 3, squeeze=False))
        self._sr_ax = self.axes[0, 0]
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._key_cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _sample_point(self, boundary: bool) -> tuple[int, int]:
        coords = self.boundary_coords if boundary else self.nonboundary_coords
        if coords.size == 0:
            coords = self.nonboundary_coords if boundary else self.boundary_coords
        if coords.size == 0:
            return self.H // 2, self.W // 2
        idx = int(self.rng.integers(0, len(coords)))
        y, x = map(int, coords[idx])
        return y, x

    def _trace_point(self, y_hr: int, x_hr: int) -> dict[str, Any]:
        y = int(np.clip(y_hr, 0, self.H - 1))
        x = int(np.clip(x_hr, 0, self.W - 1))
        is_boundary = bool(self.boundary_mask_hr[y, x] > 0.5)
        trace = compute_attention_probe_traces(
            model_obj=self.model_obj,
            aux=self.aux,
            probe_points=[{"y_hr": y, "x_hr": x, "is_boundary": is_boundary}],
            sample_index=self.sample_index,
        )[0]
        self._current_point = (y, x)
        self._current_trace = trace
        return trace

    def _draw(self, trace: dict[str, Any]) -> None:
        self.fig.clf()
        self.axes = np.asarray(self.fig.subplots(2, 3, squeeze=False))
        ax0, ax1, ax2, ax3, ax4, ax5 = self.axes.reshape(-1)
        self._sr_ax = ax0

        y_hr = int(trace["y_hr"])
        x_hr = int(trace["x_hr"])
        parent_y = int(trace["parent_y_lr"])
        parent_x = int(trace["parent_x_lr"])
        query_y = int(trace["query_y_lr"])
        query_x = int(trace["query_x_lr"])
        grain_mask = trace["grain_mask_lr"].detach().cpu().numpy()
        heatmap = trace["heatmap_lr"].detach().cpu().numpy()
        sim_map = trace["invariant_similarity_lr"].detach().cpu().numpy()
        bias_map = trace["distance_bias_lr"].detach().cpu().numpy()
        impact_map = trace["context_impact_lr"].detach().cpu().numpy()
        context_shift = float(trace.get("context_shift_norm", 0.0))

        self.fig.suptitle(
            "Interactive Grain-Attention Probe\n"
            "Click the SR panel to recompute live. Keys: b=random boundary, n=random non-boundary, s=save snapshot, q=close.",
            fontsize=14,
            y=1.01,
        )

        ax0.imshow(self.sr_rgb)
        ax0.scatter([x_hr], [y_hr], s=38, c="white", marker="o", edgecolors="black", linewidths=0.8)
        ax0.set_title(f"SR IPF-X | HR({y_hr},{x_hr})", fontsize=11)
        ax0.axis("off")

        ax1.imshow(grain_mask, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.scatter([parent_x], [parent_y], s=32, c="red", marker="x", linewidths=1.1)
        ax1.scatter([query_x], [query_y], s=30, c="yellow", marker="o", edgecolors="black", linewidths=0.6)
        ax1.set_title(
            "LR owner grain"
            + (" | query replaced" if bool(trace["query_replaced"]) else " | parent=query"),
            fontsize=11,
        )
        ax1.axis("off")

        sim_vmin = float(np.min(sim_map))
        sim_vmax = float(np.max(sim_map))
        if abs(sim_vmax - sim_vmin) < 1e-8:
            sim_vmax = sim_vmin + 1e-8
        sim_handle = ax2.imshow(
            _masked_map(sim_map, grain_mask),
            cmap=_cmap_with_black("coolwarm"),
            vmin=sim_vmin,
            vmax=sim_vmax,
        )
        ax2.set_title("LR invariant similarity", fontsize=11)
        ax2.axis("off")
        cbar = self.fig.colorbar(sim_handle, ax=ax2, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        bias_vmin = float(np.min(bias_map))
        bias_vmax = float(np.max(bias_map))
        if abs(bias_vmax - bias_vmin) < 1e-8:
            bias_vmax = bias_vmin + 1e-8
        bias_handle = ax3.imshow(
            _masked_map(bias_map, grain_mask),
            cmap=_cmap_with_black("viridis"),
            vmin=bias_vmin,
            vmax=bias_vmax,
        )
        ax3.set_title("LR distance bias", fontsize=11)
        ax3.axis("off")
        cbar = self.fig.colorbar(bias_handle, ax=ax3, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        bright_black_blue_red = _bright_black_blue_red_cmap()

        attn_handle = ax4.imshow(
            heatmap,
            cmap=bright_black_blue_red,
            vmin=0.0,
            vmax=max(float(np.max(heatmap)), 1e-8),
        )
        ax4.set_title("LR attention weight", fontsize=11)
        ax4.axis("off")
        cbar = self.fig.colorbar(attn_handle, ax=ax4, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        impact_handle = ax5.imshow(
            impact_map,
            cmap=bright_black_blue_red,
            vmin=0.0,
            vmax=max(float(np.max(impact_map)), 1e-8),
        )
        ax5.set_title("LR context impact", fontsize=11)
        ax5.axis("off")
        cbar = self.fig.colorbar(impact_handle, ax=ax5, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        status = (
            f"boundary={bool(trace['is_boundary'])}   "
            f"gid={int(trace['gid'])}   "
            f"query_replaced={bool(trace['query_replaced'])}   "
            f"|ctx-q|={context_shift:.4f}   "
            f"attn_max={float(np.max(heatmap)):.4f}   "
            f"impact_max={float(np.max(impact_map)):.4f}"
        )
        self.fig.text(0.5, 0.012, status, ha="center", va="bottom", fontsize=10)
        self.fig.canvas.draw_idle()

    def _save_snapshot(self) -> None:
        if self.out_dir is None or self._current_point is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        y, x = self._current_point
        out_path = self.out_dir / f"interactive_attention_probe_y{y:03d}_x{x:03d}.png"
        self.fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved interactive snapshot: {out_path}")

    def _on_click(self, event) -> None:
        if event.inaxes != self._sr_ax or event.xdata is None or event.ydata is None:
            return
        trace = self._trace_point(int(round(event.ydata)), int(round(event.xdata)))
        self._draw(trace)

    def _on_key(self, event) -> None:
        if event.key == "b":
            trace = self._trace_point(*self._sample_point(boundary=True))
            self._draw(trace)
        elif event.key == "n":
            trace = self._trace_point(*self._sample_point(boundary=False))
            self._draw(trace)
        elif event.key == "s":
            self._save_snapshot()
        elif event.key == "q":
            plt.close(self.fig)

    def show(self, initial_point: tuple[int, int] | None = None, block: bool = True) -> None:
        if initial_point is None:
            initial_point = self._sample_point(boundary=True)
        trace = self._trace_point(*initial_point)
        self._draw(trace)
        plt.show(block=block)


def render_attention_probe_gallery(
    probe_traces: list[dict[str, Any]],
    sr_quat_hwc: torch.Tensor,
    sym_class,
    out_png: str | Path,
) -> Path:
    """Render selected HR probe pixels and the actual LR-grain attention ingredients."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not probe_traces:
        raise ValueError("probe_traces is empty")

    sr_rgb = render_ipf_rgb(sr_quat_hwc.detach().cpu().numpy().astype(np.float32), sym_class, ref_dir="X")

    def _global_limits(key: str) -> tuple[float, float]:
        values = [trace[key].detach().cpu().numpy() for trace in probe_traces]
        mins = [float(np.min(v)) for v in values]
        maxs = [float(np.max(v)) for v in values]
        vmin = min(mins) if mins else 0.0
        vmax = max(maxs) if maxs else 1.0
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8
        return vmin, vmax

    sim_vmin, sim_vmax = _global_limits("invariant_similarity_lr")
    bias_vmin, bias_vmax = _global_limits("distance_bias_lr")
    bright_black_blue_red = _bright_black_blue_red_cmap()

    n_rows = len(probe_traces)
    fig, axes = plt.subplots(
        n_rows,
        6,
        figsize=(18.2, max(2.65 * n_rows, 4.0)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        "Grain-Attention Probe\nShowing grain support, score ingredients, final weights, and LR context impact",
        fontsize=14,
        y=1.01,
    )
    for title, ax in zip(
        (
            "Column 1: SR IPF-X with HR Probe Pixel",
            "Column 2: LR Owner Grain / Parent / Query",
            "Column 3: LR Invariant Similarity",
            "Column 4: LR Distance Bias",
            "Column 5: LR Attention Weight",
            "Column 6: LR Context Impact",
        ),
        axes[0],
    ):
        ax.set_title(title, fontsize=11)

    sim_handle = None
    bias_handle = None
    for ridx, trace in enumerate(probe_traces):
        y_hr = int(trace["y_hr"])
        x_hr = int(trace["x_hr"])
        parent_y = int(trace["parent_y_lr"])
        parent_x = int(trace["parent_x_lr"])
        query_y = int(trace["query_y_lr"])
        query_x = int(trace["query_x_lr"])
        grain_mask = trace["grain_mask_lr"].detach().cpu().numpy()
        heatmap = trace["heatmap_lr"].detach().cpu().numpy()
        sim_map = trace["invariant_similarity_lr"].detach().cpu().numpy()
        bias_map = trace["distance_bias_lr"].detach().cpu().numpy()
        impact_map = trace["context_impact_lr"].detach().cpu().numpy()
        context_shift = float(trace.get("context_shift_norm", 0.0))
        row_name = (
            f"Row {ridx + 1}: HR({y_hr},{x_hr}) "
            f"{'boundary' if trace['is_boundary'] else 'non-boundary'} "
            f"gid={int(trace['gid'])}  |ctx-q|={context_shift:.3f}"
        )

        ax0 = axes[ridx, 0]
        ax0.imshow(sr_rgb)
        ax0.scatter([x_hr], [y_hr], s=28, c="white", marker="o", edgecolors="black", linewidths=0.6)
        ax0.axis("off")
        ax0.set_ylabel(row_name, fontsize=9, rotation=0, labelpad=72, va="center")

        ax1 = axes[ridx, 1]
        ax1.imshow(grain_mask, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.scatter([parent_x], [parent_y], s=26, c="red", marker="x", linewidths=1.0)
        ax1.scatter([query_x], [query_y], s=24, c="yellow", marker="o", edgecolors="black", linewidths=0.5)
        ax1.axis("off")
        ax1.set_title(
            "parent=query" if not bool(trace["query_replaced"]) else "query replaced",
            fontsize=9,
        )

        ax2 = axes[ridx, 2]
        sim_handle = ax2.imshow(
            _masked_map(sim_map, grain_mask),
            cmap=_cmap_with_black("coolwarm"),
            vmin=sim_vmin,
            vmax=sim_vmax,
        )
        ax2.axis("off")

        ax3 = axes[ridx, 3]
        bias_handle = ax3.imshow(
            _masked_map(bias_map, grain_mask),
            cmap=_cmap_with_black("viridis"),
            vmin=bias_vmin,
            vmax=bias_vmax,
        )
        ax3.axis("off")

        ax4 = axes[ridx, 4]
        row_attn_vmax = max(float(np.max(heatmap)), 1e-8)
        attn_handle = ax4.imshow(
            heatmap,
            cmap=bright_black_blue_red,
            vmin=0.0,
            vmax=row_attn_vmax,
        )
        ax4.axis("off")
        cbar = fig.colorbar(attn_handle, ax=ax4, fraction=0.046, pad=0.02)
        cbar.set_label("attn", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax5 = axes[ridx, 5]
        row_impact_vmax = max(float(np.max(impact_map)), 1e-8)
        impact_handle = ax5.imshow(
            impact_map,
            cmap=bright_black_blue_red,
            vmin=0.0,
            vmax=row_impact_vmax,
        )
        ax5.axis("off")
        cbar = fig.colorbar(impact_handle, ax=ax5, fraction=0.046, pad=0.02)
        cbar.set_label("impact", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if sim_handle is not None:
        cbar = fig.colorbar(sim_handle, ax=axes[:, 2], fraction=0.018, pad=0.01)
        cbar.set_label("sim", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    if bias_handle is not None:
        cbar = fig.colorbar(bias_handle, ax=axes[:, 3], fraction=0.018, pad=0.01)
        cbar.set_label("bias", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_decoded_probe_gallery(
    rows: list[dict[str, Any]],
    sym_class,
    out_png: str | Path,
    pixels_per_image_pixel: int = 1,
    left_label_gutter_px: int = 260,
) -> Path:
    """Render a clearly labeled stage-by-stage decoded gallery."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)

    prepared_rows: list[dict[str, Any]] = []
    target_hw: tuple[int, int] | None = None
    for row in rows:
        if str(row.get("name", "")).strip().lower() == "hr_target":
            target_hw = tuple(int(v) for v in row["shape"])
            break
    if target_hw is None:
        target_hw = (
            max(int(row["shape"][0]) for row in rows),
            max(int(row["shape"][1]) for row in rows),
        )

    scale = max(1, int(pixels_per_image_pixel))
    panel_h, panel_w = int(target_hw[0]), int(target_hw[1])
    panel_h_px = panel_h * scale
    panel_w_px = panel_w * scale
    label_gutter_px = max(0, int(left_label_gutter_px))
    dpi_out = 180
    fig_px_w = label_gutter_px + 4 * panel_w_px
    fig_px_h = n_rows * panel_h_px
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(fig_px_w / float(dpi_out), fig_px_h / float(dpi_out)),
        dpi=dpi_out,
        squeeze=False,
    )
    left_frac = float(label_gutter_px) / float(fig_px_w) if fig_px_w > 0 else 0.0
    fig.subplots_adjust(left=left_frac, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    error_min = None
    error_max = None
    for row in rows:
        quat_hwc = row["quat_hwc"]
        row_hw = tuple(int(v) for v in row["shape"])
        rgb_x, rgb_y, rgb_z = render_ipf_rgb(quat_hwc.numpy().astype(np.float32), sym_class, ref_dir="ALL")
        if str(row["name"]) == "hr_target":
            err = torch.zeros(tuple(row["shape"]), dtype=torch.float32)
        else:
            err = quat_ang_err_deg(
                quat_hwc,
                resize_quat_target(row["hr_target_hwc"], row["shape"]),
                sym=sym_class,
            )

        is_lr_like_row = ("lr" in str(row["name"]).strip().lower()) and (row_hw != target_hw)
        if is_lr_like_row:
            rgb_x = _project_lr_array_to_hr_sparse(rgb_x, row_hw, target_hw, fill_value=0)
            rgb_y = _project_lr_array_to_hr_sparse(rgb_y, row_hw, target_hw, fill_value=0)
            rgb_z = _project_lr_array_to_hr_sparse(rgb_z, row_hw, target_hw, fill_value=0)
            err_np = _project_lr_array_to_hr_sparse(
                err.numpy().astype(np.float32),
                row_hw,
                target_hw,
                fill_value=np.nan,
            )
            err = torch.from_numpy(err_np)

        err_finite = err[torch.isfinite(err)]
        if err_finite.numel() > 0:
            row_min = float(err_finite.min())
            row_max = float(err_finite.max())
        else:
            row_min = 0.0
            row_max = 0.0
        error_min = row_min if error_min is None else min(error_min, row_min)
        error_max = row_max if error_max is None else max(error_max, row_max)
        prepared_rows.append(
            {
                **row,
                "rgb_x": rgb_x,
                "rgb_y": rgb_y,
                "rgb_z": rgb_z,
                "err": err,
            }
        )
    error_vmin = 0.0 if error_min is None else min(0.0, float(error_min))
    error_vmax = max(float(error_max) if error_max is not None else 0.0, 1e-6)

    for ridx, row in enumerate(prepared_rows):
        panels = (
            (row["rgb_x"], None, None, None),
            (row["rgb_y"], None, None, None),
            (row["rgb_z"], None, None, None),
            (row["err"].numpy(), "inferno", 0.0, error_vmax),
        )
        for cidx, (panel, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[ridx, cidx]
            if cmap is None:
                ax.imshow(panel, interpolation="nearest", resample=False)
            else:
                panel_arr = np.asarray(panel)
                if np.isnan(panel_arr).any():
                    err_cmap = plt.get_cmap(cmap).copy()
                    err_cmap.set_bad(color="black")
                    err_im = ax.imshow(
                        np.ma.masked_invalid(panel_arr),
                        cmap=err_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                        resample=False,
                    )
                else:
                    err_im = ax.imshow(
                        panel_arr,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                        resample=False,
                    )
            ax.axis("off")
            if ridx == 0:
                col_name = (
                    "IPF-X"
                    if cidx == 0
                    else "IPF-Y"
                    if cidx == 1
                    else "IPF-Z"
                    if cidx == 2
                    else "Misorientation (deg)"
                )
                ax.text(
                    0.01,
                    0.99,
                    col_name,
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    ha="left",
                    va="top",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
                )
        h, w = row["shape"]
        row_label = f"Row {ridx + 1}: {row['name']} ({h}x{w})"
        if label_gutter_px > 0:
            y_center = 1.0 - ((float(ridx) + 0.5) / float(n_rows))
            fig.text(
                0.01,
                y_center,
                row_label,
                fontsize=10,
                color="black",
                ha="left",
                va="center",
            )
        else:
            axes[ridx, 0].text(
                0.01,
                0.01,
                row_label,
                transform=axes[ridx, 0].transAxes,
                fontsize=8,
                color="white",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
            )

    fig.savefig(out_path, dpi=dpi_out)
    plt.close(fig)
    return out_path


def render_scalar_probe_gallery(
    scalar_maps: list[dict[str, Any]],
    out_png: str | Path,
    ncols: int = 4,
) -> Path:
    """Render scalar probe maps with explicit panel naming."""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not scalar_maps:
        raise ValueError("scalar_maps is empty")

    n_maps = len(scalar_maps)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_maps / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        "Scalar Probe Diagnostics\nPanels are read left-to-right, top-to-bottom",
        fontsize=14,
        y=1.02,
    )
    flat_axes = axes.reshape(-1)
    for ax in flat_axes[n_maps:]:
        ax.axis("off")

    for idx, (ax, item) in enumerate(zip(flat_axes, scalar_maps), start=1):
        if item["name"] == "hr_to_lr_owner":
            arr = np.asarray(item["array"], dtype=np.int64)
            unique_vals = np.unique(arr)
            if unique_vals.size == 0:
                raise ValueError("hr_to_lr_owner map is empty")

            vmin = int(unique_vals.min())
            vmax = int(unique_vals.max())
            num_bins = max(1, vmax - vmin + 1)
            base = plt.get_cmap("nipy_spectral")
            colors = base(np.linspace(0.0, 1.0, num_bins))
            cmap = ListedColormap(colors)
            bounds = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
            norm = BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        else:
            im = ax.imshow(
                item["array"],
                cmap=item.get("cmap", "viridis"),
                vmin=item.get("vmin", None),
                vmax=item.get("vmax", None),
            )
        ax.set_title(f"Panel {idx}: {item['name']}", fontsize=10)
        ax.axis("off")
        if item["name"] == "hr_to_lr_owner":
            ticks = unique_vals.astype(int).tolist()
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.02,
                ticks=ticks,
                boundaries=bounds,
                spacing="proportional",
            )
            cbar.ax.set_yticklabels([str(v) for v in ticks])
            cbar.ax.tick_params(labelsize=6)

            font_size = float(np.clip(120.0 / max(arr.shape), 4.5, 8.0))
            for grain_id in ticks:
                coords = np.argwhere(arr == grain_id)
                if coords.size == 0:
                    continue
                centroid = coords.mean(axis=0)
                d2 = ((coords - centroid) ** 2).sum(axis=1)
                y, x = coords[int(d2.argmin())]
                rgba = cmap(norm(grain_id))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.58 else "white"
                stroke_color = "white" if text_color == "black" else "black"
                txt = ax.text(
                    float(x),
                    float(y),
                    str(grain_id),
                    color=text_color,
                    fontsize=font_size,
                    ha="center",
                    va="center",
                )
                txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground=stroke_color)])
        else:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
