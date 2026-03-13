#!/usr/bin/env python3
"""Visualize stage-wise outputs (descriptor/boundary/affinity) for SO3/O SR models."""

from __future__ import annotations

import argparse
import json
import math
import re
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
from utils.quat_ops import format_quaternions
from utils.symmetry_utils import resolve_symmetry
from visualization.ipf_render import render_ipf_rgb


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
        exp_dir / "checkpoints_so3o_sr_4e6e_e3nn_consistent_lite" / ckpt,
        exp_dir / "checkpoints_so3o_sr_4e6e_e3nn_consistent" / ckpt,
        exp_dir / "checkpoints_so3o_sr_4e6e_learnable_v2" / ckpt,
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


def _load_config_fallback(exp_dir: Path, cfg_from_ckpt: dict[str, Any] | None) -> dict[str, Any]:
    if cfg_from_ckpt:
        return dict(cfg_from_ckpt)

    for cfg_name in ("config.json", "config_smoke.json", "config_v1_full.json", "config_v2_full.json"):
        p = exp_dir / cfg_name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        "No config found in checkpoint and no known config file in exp_dir."
    )


def _build_model_from_cfg(cfg: dict[str, Any], dtype: torch.dtype, device: torch.device):
    model_variant = str(cfg.get("model_variant", "")).strip().lower()
    if model_variant in ("full", "lite", "light", "lightweight") or "lam_cross_aff" in cfg:
        from so3_o_cubic_sr_4e6e_e3nn_consistent import (
            SO3OQuotientSRNet4e6eE3NN,
            SO3OQuotientSRNet4e6eE3NNLite,
        )

        model_cls = SO3OQuotientSRNet4e6eE3NNLite if model_variant in ("lite", "light", "lightweight") else SO3OQuotientSRNet4e6eE3NN
        model = model_cls(
            hidden_mul4=int(cfg.get("hidden_mul4", 16)),
            hidden_mul6=int(cfg.get("hidden_mul6", 8)),
            num_blocks_lr=int(cfg.get("num_blocks_lr", 2)),
            num_blocks_hr=int(cfg.get("num_blocks_hr", 1)),
            sr_scale=int(cfg.get("sr_scale", 4)),
            passive_input=bool(cfg.get("passive_input", True)),
            affinity_kernel_size=int(cfg.get("affinity_kernel_size", 3)),
            affinity_tau=float(cfg.get("affinity_tau", 0.35)),
            affinity_hidden=int(cfg.get("affinity_hidden", 32)),
            hard_mask_mode=str(cfg.get("hard_mask_mode", "hard")),
            hard_mask_threshold=float(cfg.get("hard_mask_threshold", 0.5)),
            hard_mask_temperature=float(cfg.get("hard_mask_temperature", 0.05)),
            stage_refine_blocks=int(cfg.get("stage_refine_blocks", 1)),
            dtype=dtype,
        )
        return model.to(device=device, dtype=dtype)

    upsample_mode = str(cfg.get("upsample_mode", "transpose")).lower()
    if upsample_mode == "masked_learnable":
        learnable_version = str(cfg.get("learnable_version", cfg.get("learnable_variant", "v1"))).lower()
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
        return model.to(device=device, dtype=dtype)

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


def _load_symmetry_name(dataset_root: str, cfg: dict[str, Any]) -> str:
    info_path = Path(dataset_root) / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        sym = info.get("symmetry", None)
        if sym:
            return str(sym)
    return str(cfg.get("symmetry_group", cfg.get("symmetry", "O")))


def _build_dictionary_flexible(model, n: int, device: torch.device, dtype: torch.dtype):
    try:
        return model.codec.build_dictionary(n=n, device=device, dtype=dtype)
    except TypeError:
        return model.codec.build_dictionary(
            n=n,
            device=device,
            dtype=dtype,
            sampling="random",
            fz_resolution=3,
            fz_method="cubochoric",
            fz_point_group="O",
        )


def _stage_ids(pred: dict[str, torch.Tensor]) -> list[int]:
    out = set()
    rx = re.compile(r"^stage(\d+)_")
    for k in pred.keys():
        m = rx.match(k)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _select_stage_descriptor(pred: dict[str, torch.Tensor], stage_idx: int) -> torch.Tensor | None:
    keys = [
        f"stage{stage_idx}_provisional_descriptor_hr",
        f"stage{stage_idx}_skip_hr",
    ]
    for k in keys:
        if k in pred:
            return pred[k]
    return None


def _decode_descriptor_to_rgb(
    x_desc_chw: torch.Tensor,
    model,
    q_dict: torch.Tensor,
    x_dict: torch.Tensor,
    sym_class,
    ref_dir: str,
    chunk: int,
    topk: int,
) -> np.ndarray:
    x_last = x_desc_chw.permute(1, 2, 0).contiguous()
    dec = model.codec.decode_by_dictionary(
        x_last,
        q_dict=q_dict,
        x_dict=x_dict,
        chunk=int(chunk),
        topk=int(topk),
    )
    q = standardize_quaternion_sign(normalize_quaternion(dec.quaternions)).detach().cpu().numpy()
    q = format_quaternions(
        q,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=sym_class,
        to_quat_first=False,
    )
    rgb = render_ipf_rgb(q, sym_class, ref_dir=ref_dir)
    if isinstance(rgb, list):
        return rgb[2]  # fallback to Z if list returned unexpectedly
    return rgb


def _affinity_mean(aff: torch.Tensor) -> torch.Tensor:
    k2 = int(aff.shape[0])
    c = k2 // 2
    idx = [i for i in range(k2) if i != c]
    if not idx:
        return aff.mean(dim=0)
    return aff[idx].mean(dim=0)


def _plot_stage_panel(
    out_png: Path,
    desc_imgs: list[np.ndarray],
    desc_labels: list[str],
    boundary_maps: list[np.ndarray],
    boundary_labels: list[str],
    affinity_maps: list[np.ndarray],
    affinity_labels: list[str],
    sample_id: int,
) -> None:
    ncols = max(len(desc_imgs), len(boundary_maps), len(affinity_maps))
    fig, axes = plt.subplots(3, ncols, figsize=(3.8 * ncols, 10.0), dpi=140)
    if ncols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for j in range(ncols):
        ax = axes[0, j]
        if j < len(desc_imgs):
            ax.imshow(desc_imgs[j])
            ax.set_title(desc_labels[j], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(ncols):
        ax = axes[1, j]
        if j < len(boundary_maps):
            ax.imshow(boundary_maps[j], cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(boundary_labels[j], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(ncols):
        ax = axes[2, j]
        if j < len(affinity_maps):
            ax.imshow(affinity_maps[j], cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(affinity_labels[j], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Stage Outputs - Sample {sample_id:03d}", fontsize=14)
    axes[0, 0].set_ylabel("Decoded IPF", fontsize=11)
    axes[1, 0].set_ylabel("Boundary", fontsize=11)
    axes[2, 0].set_ylabel("Affinity Mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-stage outputs for SO3/O SR models."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory.")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename/path.")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"], help="Dataset split.")
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples to render.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--decode_dict_size", type=int, default=12000, help="Descriptor decode dictionary size.")
    parser.add_argument("--decode_chunk", type=int, default=2048, help="Decode chunk size.")
    parser.add_argument("--decode_topk", type=int, default=1, help="Decode top-k shortlist.")
    parser.add_argument("--ref_dir", type=str, default="Z", choices=["X", "Y", "Z"], help="IPF reference direction.")
    parser.add_argument(
        "--affinity_target_tau",
        type=float,
        default=-1.0,
        help="Override GT affinity tau. Default uses config affinity_target_tau if >0 else affinity_tau.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: exp_dir/visualizations_stages/epoch_XXXX_split).",
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

    symmetry_name = _load_symmetry_name(dataset_root, cfg)
    sym_class = resolve_symmetry(symmetry_name)

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
        out_dir = exp_dir / "visualizations_stages" / f"epoch_{epoch:04d}_{args.split.lower()}"
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

    q_dict, x_dict = _build_dictionary_flexible(
        model=model,
        n=int(args.decode_dict_size),
        device=device,
        dtype=dtype,
    )

    affinity_tau = float(cfg.get("affinity_tau", 0.6))
    affinity_target_tau_cfg = float(cfg.get("affinity_target_tau", 0.0))
    if args.affinity_target_tau > 0:
        affinity_target_tau = float(args.affinity_target_tau)
    elif affinity_target_tau_cfg > 0:
        affinity_target_tau = affinity_target_tau_cfg
    else:
        affinity_target_tau = affinity_tau

    print(f"Using symmetry group: {symmetry_name}")
    print(f"Saving stage visualizations to: {out_dir}")

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

            pred = model(lr_q)
            x_gt = model.codec.encode_map(hr_q)

            k = int(math.isqrt(int(pred["affinity_hr"].shape[1])))
            aff_gt = descriptor_to_local_affinity(
                x_desc=x_gt,
                kernel_size=k,
                tau=affinity_target_tau,
                eps=1e-6,
            )
            center_idx = (k * k) // 2
            boundary_gt = affinity_to_boundary(aff_gt, center_idx=center_idx)

            stage_ids = _stage_ids(pred)

            bsz = lr_q.shape[0]
            for b in range(bsz):
                if saved >= int(args.max_samples):
                    break

                desc_entries: list[tuple[str, torch.Tensor]] = [("LR", pred["descriptor_lr"][b])]
                for sid in stage_ids:
                    desc_map = _select_stage_descriptor(pred, sid)
                    if desc_map is not None:
                        desc_entries.append((f"S{sid}", desc_map[b]))
                desc_entries.append(("Final", pred["descriptor_hr"][b]))
                desc_entries.append(("GT", x_gt[b]))

                boundary_entries: list[tuple[str, torch.Tensor]] = [("LR", pred["boundary_lr"][b, 0])]
                for sid in stage_ids:
                    k_b = f"stage{sid}_boundary_hr"
                    if k_b in pred:
                        boundary_entries.append((f"S{sid}", pred[k_b][b, 0]))
                boundary_entries.append(("Final", pred["boundary_hr"][b, 0]))
                boundary_entries.append(("GT", boundary_gt[b, 0]))

                affinity_entries: list[tuple[str, torch.Tensor]] = [("LR", _affinity_mean(pred["affinity_lr"][b]))]
                for sid in stage_ids:
                    k_a = f"stage{sid}_affinity_hr"
                    if k_a in pred:
                        affinity_entries.append((f"S{sid}", _affinity_mean(pred[k_a][b])))
                affinity_entries.append(("Final", _affinity_mean(pred["affinity_hr"][b])))
                affinity_entries.append(("GT", _affinity_mean(aff_gt[b])))

                desc_imgs: list[np.ndarray] = []
                desc_labels: list[str] = []
                for label, x_desc_chw in desc_entries:
                    rgb = _decode_descriptor_to_rgb(
                        x_desc_chw=x_desc_chw,
                        model=model,
                        q_dict=q_dict,
                        x_dict=x_dict,
                        sym_class=sym_class,
                        ref_dir=args.ref_dir,
                        chunk=int(args.decode_chunk),
                        topk=int(args.decode_topk),
                    )
                    desc_imgs.append(rgb)
                    desc_labels.append(label)

                boundary_maps = [x.detach().cpu().numpy() for _, x in boundary_entries]
                boundary_labels = [k for k, _ in boundary_entries]
                affinity_maps = [x.detach().cpu().numpy() for _, x in affinity_entries]
                affinity_labels = [k for k, _ in affinity_entries]

                out_png = out_dir / f"stage_outputs_{saved:03d}.png"
                _plot_stage_panel(
                    out_png=out_png,
                    desc_imgs=desc_imgs,
                    desc_labels=desc_labels,
                    boundary_maps=boundary_maps,
                    boundary_labels=boundary_labels,
                    affinity_maps=affinity_maps,
                    affinity_labels=affinity_labels,
                    sample_id=saved,
                )

                saved += 1
                print(f"Saved {out_png.name}")

            if saved >= int(args.max_samples):
                break

    print(f"Done. Saved {saved} stage panel(s).")


if __name__ == "__main__":
    main()
