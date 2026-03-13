#!/usr/bin/env python3
"""Plot spatial Step-B quantities (boundary/affinity/error maps) for SO3/O SR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from so3_o_cubic_sr import (
    affinity_to_boundary,
    descriptor_to_local_affinity,
    normalize_quaternion,
    standardize_quaternion_sign,
)
from training.data_loading import build_dataloader


def _extract_lr_hr_from_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        lr = batch.get("lr_q", batch.get("lr"))
        hr = batch.get("hr_q", batch.get("hr"))
        if lr is None or hr is None:
            raise KeyError(f"Unsupported dict batch keys: {list(batch.keys())}")
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


def _resolve_ckpt_path(exp_dir: Path, ckpt: str) -> Path:
    p = Path(ckpt)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    candidates = [
        exp_dir / "checkpoints_so3o_sr_4e6e_full" / ckpt,
        exp_dir / "checkpoints_so3o_sr_4e6e_learnable" / ckpt,
        exp_dir / "checkpoints_so3o_sr_4e6e" / ckpt,
        exp_dir / "checkpoints" / ckpt,
        exp_dir / ckpt,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{ckpt}'. Tried: {', '.join(str(c) for c in candidates)}"
    )


def _build_model_from_cfg(cfg: dict[str, Any], dtype: torch.dtype, device: torch.device):
    upsample_mode = str(cfg.get("upsample_mode", "transpose")).lower()

    if upsample_mode == "masked_learnable":
        learnable_version = str(
            cfg.get("learnable_version", cfg.get("learnable_variant", "v1"))
        ).lower()
        if learnable_version in ("v2", "nextgen", "learnable_v2"):
            from so3_o_cubic_sr_4e6e_learnable_v2 import SO3OQuotientSRNet4e6e as LearnableSR
        else:
            from so3_o_cubic_sr_4e6e_learnable import SO3OQuotientSRNet4e6e as LearnableSR

        model = LearnableSR(
            hidden_mul4=int(cfg.get("hidden_mul4", 16)),
            hidden_mul6=int(cfg.get("hidden_mul6", 8)),
            num_blocks_lr=int(cfg.get("num_blocks_lr", cfg.get("num_blocks_pre", 3))),
            num_blocks_hr=int(cfg.get("num_blocks_hr", cfg.get("num_blocks_post", 3))),
            sr_scale=int(cfg.get("sr_scale", 4)),
            passive_input=bool(cfg.get("passive_input", True)),
            dtype=dtype,
            affinity_kernel_size=int(cfg.get("affinity_kernel_size", 3)),
            use_boundary_prior=bool(cfg.get("use_boundary_prior", True)),
            upsample_mode="masked_learnable",
        )
    else:
        from so3_o_cubic_sr import SO3OQuotientSRNet4e6e as DefaultSR

        model = DefaultSR(
            hidden_mul4=int(cfg.get("hidden_mul4", 16)),
            hidden_mul6=int(cfg.get("hidden_mul6", 8)),
            num_blocks_pre=int(cfg.get("num_blocks_pre", 3)),
            num_blocks_post=int(cfg.get("num_blocks_post", 3)),
            upsample_mode=str(cfg.get("upsample_mode", "transpose")),
            sr_scale=int(cfg.get("sr_scale", 4)),
            refine_per_stage=int(cfg.get("refine_per_stage", 1)),
            passive_input=bool(cfg.get("passive_input", True)),
            affinity_kernel_size=int(cfg.get("affinity_kernel_size", 3)),
            affinity_tau=float(cfg.get("affinity_tau", 0.6)),
            affinity_hidden=int(cfg.get("affinity_hidden", 32)),
            masked_interp_sigma=float(cfg.get("masked_interp_sigma", 0.75)),
            hr_aux_enabled=bool(cfg.get("hr_aux_enabled", True)),
            hr_aux_coupling_init=float(cfg.get("hr_aux_coupling_init", 1.0)),
            dtype=dtype,
        )

    return model.to(device=device, dtype=dtype)


def _load_config_fallback(exp_dir: Path, cfg_from_ckpt: dict[str, Any] | None) -> dict[str, Any]:
    if cfg_from_ckpt:
        return dict(cfg_from_ckpt)
    for cfg_name in ("config.json", "config_smoke.json"):
        p = exp_dir / cfg_name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config found in checkpoint and no config.json/config_smoke.json in exp_dir."
    )


def _imshow(ax: plt.Axes, arr: np.ndarray, title: str, vmin: float | None, vmax: float | None) -> None:
    im = ax.imshow(arr, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_spatial_panel(out_png: Path, maps: dict[str, np.ndarray], sample_id: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=130)
    axes = axes.ravel()

    _imshow(axes[0], maps["boundary_pred"], "Boundary Pred", 0.0, 1.0)
    _imshow(axes[1], maps["boundary_gt"], "Boundary GT", 0.0, 1.0)
    _imshow(axes[2], maps["boundary_abs_err"], "|Boundary Error|", 0.0, 1.0)
    _imshow(axes[3], maps["affinity_pred_mean"], "Affinity Pred (mean)", 0.0, 1.0)
    _imshow(axes[4], maps["affinity_gt_mean"], "Affinity GT (mean)", 0.0, 1.0)
    _imshow(axes[5], maps["affinity_abs_err_mean"], "|Affinity Error| (mean)", 0.0, 1.0)

    desc_vmax = float(np.percentile(maps["descriptor_l2_err"], 99.0))
    if desc_vmax <= 0.0:
        desc_vmax = 1.0
    shell_vmax = float(np.percentile(maps["shell_abs_err"], 99.0))
    if shell_vmax <= 0.0:
        shell_vmax = 1.0

    _imshow(axes[6], maps["descriptor_l2_err"], "Descriptor L2 Error", 0.0, desc_vmax)
    _imshow(axes[7], maps["shell_abs_err"], "Shell Error", 0.0, shell_vmax)

    fig.suptitle(f"Step-B Spatial Quantities - Sample {sample_id:03d}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Step-B spatial quantities (boundary/affinity/error) for SO3/O SR checkpoints."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory.")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename/path.")
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        choices=["Train", "Val", "Test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples to render.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Default: exp_dir/visualizations_spatial/epoch_XXXX_<split>.",
    )
    parser.add_argument(
        "--affinity_target_tau",
        type=float,
        default=-1.0,
        help="Override GT affinity tau. Default uses config affinity_target_tau if >0 else affinity_tau.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save raw numpy arrays for each sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    ckpt_path = _resolve_ckpt_path(exp_dir, args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_obj["model_state_dict"] if "model_state_dict" in ckpt_obj else ckpt_obj
    cfg = _load_config_fallback(exp_dir, ckpt_obj.get("config", None) if isinstance(ckpt_obj, dict) else None)

    dataset_root = str(cfg.get("dataset_root", ""))
    if not dataset_root:
        raise ValueError("Missing dataset_root in checkpoint config.")

    dtype_name = str(cfg.get("dtype", "float32")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32

    model = _build_model_from_cfg(cfg=cfg, dtype=dtype, device=device)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[warn] Missing checkpoint keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[warn] Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")
    model.eval()

    epoch = int(ckpt_obj.get("epoch", 0)) if isinstance(ckpt_obj, dict) else 0
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = exp_dir / "visualizations_spatial" / f"epoch_{epoch:04d}_{args.split.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(
        dataset_root=dataset_root,
        split=args.split,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        preload=bool(cfg.get("preload", False)),
        preload_torch=bool(cfg.get("preload_torch", False)),
        take_first=max(1, int(args.max_samples)),
        seed=int(cfg.get("seed", 42)),
    )

    affinity_tau = float(cfg.get("affinity_tau", 0.6))
    affinity_target_tau_cfg = float(cfg.get("affinity_target_tau", 0.0))
    if args.affinity_target_tau > 0:
        affinity_target_tau = float(args.affinity_target_tau)
    elif affinity_target_tau_cfg > 0:
        affinity_target_tau = affinity_target_tau_cfg
    else:
        affinity_target_tau = affinity_tau

    kernel_size = int(cfg.get("affinity_kernel_size", 3))
    center_idx = (kernel_size * kernel_size) // 2
    target_radius = float(model.codec.descriptor_radius)

    print(f"Device: {device}")
    print(f"Saving maps to: {out_dir}")

    saved = 0
    with torch.no_grad():
        for batch in loader:
            lr_q, hr_q = _extract_lr_hr_from_batch(batch)
            lr_q = _normalize_map_channel_first(
                _to_channel_first_quat_map(lr_q).to(device=device, dtype=dtype, non_blocking=True)
            )
            hr_q = _normalize_map_channel_first(
                _to_channel_first_quat_map(hr_q).to(device=device, dtype=dtype, non_blocking=True)
            )

            if str(cfg.get("upsample_mode", "transpose")).lower() == "masked_learnable":
                pred = model(lr_q)
                x_pred = pred["descriptor_hr"]
                aux = {
                    "affinity_hr": pred.get("affinity_hr", None),
                    "boundary_hr": pred.get("boundary_hr", None),
                }
            else:
                out = model.forward_descriptor(lr_q, return_aux=True)
                if isinstance(out, tuple):
                    x_pred, aux = out
                else:
                    x_pred = out
                    aux = {}

            x_gt = model.codec.encode_map(hr_q)
            aff_gt = descriptor_to_local_affinity(
                x_desc=x_gt,
                kernel_size=kernel_size,
                tau=affinity_target_tau,
                eps=1e-6,
            )
            boundary_gt = affinity_to_boundary(aff_gt, center_idx=center_idx)

            if "affinity_hr" in aux and aux["affinity_hr"] is not None:
                aff_pred = aux["affinity_hr"]
            else:
                aff_pred = descriptor_to_local_affinity(
                    x_desc=x_pred,
                    kernel_size=kernel_size,
                    tau=affinity_tau,
                    eps=1e-6,
                )

            if "boundary_hr" in aux and aux["boundary_hr"] is not None:
                boundary_pred = aux["boundary_hr"]
            else:
                boundary_pred = affinity_to_boundary(aff_pred, center_idx=center_idx)

            non_center_idx = [i for i in range(aff_pred.shape[1]) if i != center_idx]
            if non_center_idx:
                aff_pred_mean = aff_pred[:, non_center_idx].mean(dim=1)
                aff_gt_mean = aff_gt[:, non_center_idx].mean(dim=1)
            else:
                aff_pred_mean = aff_pred.mean(dim=1)
                aff_gt_mean = aff_gt.mean(dim=1)

            desc_l2 = torch.norm(x_pred - x_gt, dim=1)
            shell_err = torch.abs(torch.norm(x_pred, dim=1) - target_radius)

            bsz = x_pred.shape[0]
            for b in range(bsz):
                if saved >= int(args.max_samples):
                    break

                maps_np = {
                    "boundary_pred": boundary_pred[b, 0].detach().cpu().numpy(),
                    "boundary_gt": boundary_gt[b, 0].detach().cpu().numpy(),
                    "boundary_abs_err": torch.abs(boundary_pred[b, 0] - boundary_gt[b, 0]).detach().cpu().numpy(),
                    "affinity_pred_mean": aff_pred_mean[b].detach().cpu().numpy(),
                    "affinity_gt_mean": aff_gt_mean[b].detach().cpu().numpy(),
                    "affinity_abs_err_mean": torch.abs(aff_pred_mean[b] - aff_gt_mean[b]).detach().cpu().numpy(),
                    "descriptor_l2_err": desc_l2[b].detach().cpu().numpy(),
                    "shell_abs_err": shell_err[b].detach().cpu().numpy(),
                }

                out_png = out_dir / f"spatial_stepb_{saved:03d}.png"
                _plot_spatial_panel(out_png=out_png, maps=maps_np, sample_id=saved)

                if args.save_npz:
                    npz_path = out_dir / f"spatial_stepb_{saved:03d}.npz"
                    np.savez_compressed(npz_path, **maps_np)

                saved += 1
                print(f"Saved {out_png.name}")

            if saved >= int(args.max_samples):
                break

    print(f"Done. Saved {saved} sample(s).")


if __name__ == "__main__":
    main()
