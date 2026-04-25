#!/usr/bin/env python3
"""Decode and visualize OCRP macro stages for a single dataset sample."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.infer_iso_embedding_sr_attn import (  # noqa: E402
    _flatten_quat_chw,
    _resolve_checkpoint,
    _to_hwc_quat_single,
    _unpack_batch,
)
from models.SR_ocrp import ClusterSlotBuilder, IsoEmbeddingSROCRP  # noqa: E402
from training.config_utils import load_and_prepare_config  # noqa: E402
from training.data_loading import build_dataloader  # noqa: E402
from utils.stage_probe_utils import (  # noqa: E402
    decode_probe_stages,
    pick_most_free_cuda_gpu,
    quat_ang_err_deg,
    render_decoded_probe_gallery,
    render_scalar_probe_gallery,
    resize_quat_target,
)
from utils.symmetry_utils import resolve_symmetry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OCRP macro stages for a single sample.")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path relative to exp_dir or absolute path. Defaults to logs/run_config.json when present, else config_new.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside checkpoints/ or an absolute path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Val",
        choices=["Train", "Val", "Test"],
        help="Dataset split to probe.",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index inside the selected split.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <exp_dir>/analysis/ocrp_macro_stage_probe/<split>_sampleXXXX.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value. By default the most-free GPU is selected.",
    )
    return parser.parse_args()


def _resolve_config_path(exp_dir: Path, config_arg: str | None) -> Path:
    if config_arg is None:
        candidates = (
            exp_dir / "logs" / "run_config.json",
            exp_dir / "config_new.json",
            exp_dir / "config.json",
        )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find a config in {exp_dir}")

    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path
    return exp_dir / config_arg


def _default_out_dir(exp_dir: Path, split: str, sample_idx: int) -> Path:
    return exp_dir / "analysis" / "ocrp_macro_stage_probe" / f"{str(split).lower()}_sample{int(sample_idx):04d}"


def _load_ocrp_model_from_checkpoint(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
) -> IsoEmbeddingSROCRP:
    init_params = set(inspect.signature(IsoEmbeddingSROCRP.__init__).parameters)
    model_kwargs = {
        "crystal": str(getattr(cfg, "crystal", "fcc")),
        "d6_convention": str(getattr(cfg, "d6_convention", "z_axis")),
        "device": device,
        "feature_irreps": str(getattr(cfg, "feature_irreps", "full")),
        "use_lr_conv1": bool(getattr(cfg, "use_lr_conv1", True)),
        "lr_conv1_kernel_size": int(getattr(cfg, "lr_conv1_kernel_size", 5)),
        "use_residual_lr1": bool(getattr(cfg, "use_residual_lr1", True)),
        "conv_feature_mask_cosine_threshold": float(
            getattr(cfg, "conv_feature_mask_cosine_threshold", 0.98)
        ),
        "conv_feature_mask_soft": bool(getattr(cfg, "conv_feature_mask_soft", False)),
        "conv_feature_mask_temperature": float(
            getattr(cfg, "conv_feature_mask_temperature", 32.0)
        ),
        "upsample_factor": getattr(cfg, "upsample_factor", getattr(cfg, "scale", 4)),
        "window_size": int(getattr(cfg, "window_size", 5)),
        "kmax_slots": int(getattr(cfg, "kmax_slots", 4)),
        "cluster_threshold_deg": float(getattr(cfg, "cluster_threshold_deg", 2.0)),
        "cluster_connectivity": int(getattr(cfg, "cluster_connectivity", 8)),
        "phase_dim": int(getattr(cfg, "phase_dim", 32)),
        "ocrp_router_hidden_dim": int(getattr(cfg, "ocrp_router_hidden_dim", 128)),
        "ocrp_router_conv_hidden_dim": int(getattr(cfg, "ocrp_router_conv_hidden_dim", 64)),
        "ocrp_proposal_hidden_dim": int(getattr(cfg, "ocrp_proposal_hidden_dim", 128)),
        "ocrp_straight_through": bool(getattr(cfg, "ocrp_straight_through", True)),
        "ocrp_mode": str(getattr(cfg, "ocrp_mode", "pixel_patch")),
        "macro_lr_tile_size": int(getattr(cfg, "macro_lr_tile_size", 3)),
        "ocrp_token_conditioned_member_bias": getattr(
            cfg, "ocrp_token_conditioned_member_bias", None
        ),
        "use_hr_conv1": bool(getattr(cfg, "use_hr_conv1", True)),
        "hr_conv1_kernel_size": int(getattr(cfg, "hr_conv1_kernel_size", 7)),
        "use_residual_hr1": bool(getattr(cfg, "use_residual_hr1", True)),
        "decoder_cubochoric_resolution": int(getattr(cfg, "decoder_cubochoric_resolution", 1)),
        "decoder_num_starts": int(getattr(cfg, "decoder_num_starts", 2)),
        "decoder_steps": int(getattr(cfg, "decoder_steps", 1)),
        "decoder_lr": float(getattr(cfg, "decoder_lr", 0.05)),
        "decoder_method": str(getattr(cfg, "decoder_method", "cubochoric")),
        "decoder_max_table_rows": getattr(cfg, "decoder_max_table_rows", None),
        "decoder_table_cache_dir": getattr(
            cfg, "decoder_table_cache_dir", "out/decoder_lookup_tables"
        ),
        "decoder_eager_init": bool(getattr(cfg, "decoder_eager_init", False)),
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in init_params}
    model = IsoEmbeddingSROCRP(**model_kwargs).to(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _as_unbatched(x: Any) -> Any:
    if not isinstance(x, torch.Tensor):
        return x
    if x.dim() > 0 and x.shape[0] == 1:
        return x[0]
    return x


def _build_probe_stage(name: str, feat: torch.Tensor, shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "name": str(name),
        "feat": torch.nan_to_num(feat.detach(), nan=0.0, posinf=1e4, neginf=-1e4),
        "shape": (int(shape[0]), int(shape[1])),
    }


def _assemble_macro_feature_map(
    model_obj: IsoEmbeddingSROCRP,
    patch_tensor: torch.Tensor,
    grid_shape: tuple[int, int],
    hr_shape: tuple[int, int],
) -> torch.Tensor:
    squeeze = patch_tensor.dim() == 3
    patch_batched = patch_tensor.unsqueeze(0) if squeeze else patch_tensor
    feat_hr = model_obj.ocrp._assemble_macro_patch_tokens(
        patch_batched,
        grid_shape=grid_shape,
        hr_crop_shape=hr_shape,
    )
    return feat_hr.squeeze(0) if squeeze else feat_hr


def _assemble_macro_scalar_image(
    model_obj: IsoEmbeddingSROCRP,
    token_values: torch.Tensor,
    grid_shape: tuple[int, int],
    hr_shape: tuple[int, int],
) -> np.ndarray:
    squeeze = token_values.dim() == 2
    token_batched = token_values.unsqueeze(0) if squeeze else token_values
    bsz, nwin, patch_tokens = token_batched.shape
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    patch_shape = getattr(model_obj.ocrp, "hr_patch_shape", None)
    if patch_shape is None:
        patch_size = int(model_obj.ocrp.hr_patch_size)
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    if nwin != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} windows, got {nwin}")
    if patch_tokens != patch_h * patch_w:
        raise ValueError(f"Expected {patch_h * patch_w} patch tokens, got {patch_tokens}")

    img = (
        token_batched.view(bsz, grid_h, grid_w, patch_h, patch_w)
        .permute(0, 1, 3, 2, 4)
        .reshape(bsz, grid_h * patch_h, grid_w * patch_w)
    )
    crop_h, crop_w = int(hr_shape[0]), int(hr_shape[1])
    img = img[:, :crop_h, :crop_w]
    arr = img[0].detach().cpu().numpy() if squeeze else img.detach().cpu().numpy()
    return arr


def _stage_error_stats(
    quat_hwc: torch.Tensor,
    target_hwc: torch.Tensor,
    sym_class,
) -> dict[str, float]:
    err = quat_ang_err_deg(
        quat_hwc,
        resize_quat_target(target_hwc, tuple(int(v) for v in quat_hwc.shape[:2])),
        sym=sym_class,
    )
    finite = err[torch.isfinite(err)]
    if finite.numel() == 0:
        return {
            "mean_deg": float("nan"),
            "median_deg": float("nan"),
            "p90_deg": float("nan"),
            "p95_deg": float("nan"),
            "max_deg": float("nan"),
        }
    return {
        "mean_deg": float(finite.mean().item()),
        "median_deg": float(finite.median().item()),
        "p90_deg": float(torch.quantile(finite, 0.90).item()),
        "p95_deg": float(torch.quantile(finite, 0.95).item()),
        "max_deg": float(finite.max().item()),
    }


def _stage_note(name: str) -> str:
    if name == "lr_input":
        return "Input LR quaternion field."
    if name == "encode_lr":
        return "Encoder output decoded back on the LR grid."
    if name == "lr_conv1_pre_ocrp":
        return "LR spatial context just before OCRP support-bank construction."
    if name.startswith("support_slot") and name.endswith("_medoid_ctx"):
        return "Representative medoid feature chosen for that slot on each support tile."
    if name.startswith("support_slot") and name.endswith("_pooled_mean"):
        return "Within-slot pooled summary, averaged over HR patch tokens."
    if name.startswith("slot") and name.endswith("_proposal_hr"):
        return "Decoded HR patch proposal emitted by this slot before routing selection."
    if name == "selected_patch_out_hr":
        return "Hard-routed OCRP output before the HR post-conv."
    if name == "hr_conv1_post_ocrp":
        return "HR post-conv refinement applied after OCRP assembly."
    if name == "sr_output":
        return "Final decoded SR prediction."
    if name == "hr_target":
        return "Ground-truth HR quaternion field."
    return ""


def _write_stage_metrics_csv(
    rows: list[dict[str, Any]],
    target_hwc: torch.Tensor,
    sym_class,
    out_csv: Path,
) -> list[dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    for row in rows:
        quat_hwc = row["quat_hwc"]
        stats = (
            {
                "mean_deg": 0.0,
                "median_deg": 0.0,
                "p90_deg": 0.0,
                "p95_deg": 0.0,
                "max_deg": 0.0,
            }
            if row["name"] == "hr_target"
            else _stage_error_stats(quat_hwc, target_hwc, sym_class)
        )
        metrics_rows.append(
            {
                "stage": row["name"],
                "height": int(row["shape"][0]),
                "width": int(row["shape"][1]),
                "mean_deg": stats["mean_deg"],
                "median_deg": stats["median_deg"],
                "p90_deg": stats["p90_deg"],
                "p95_deg": stats["p95_deg"],
                "max_deg": stats["max_deg"],
                "note": _stage_note(str(row["name"])),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "height", "width", "mean_deg", "median_deg", "p90_deg", "p95_deg", "max_deg", "note"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    return metrics_rows


def _write_summary_md(
    *,
    out_path: Path,
    exp_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    sample_idx: int,
    device: torch.device,
    support_grid_shape: tuple[int, int],
    hr_shape: tuple[int, int],
    active_slots: list[int],
    slot_coverage: list[float],
    slot_usage: list[float],
    gallery_paths: dict[str, Path | None],
    metrics_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# OCRP Macro Stage Probe")
    lines.append("")
    lines.append(f"- Experiment: `{exp_dir}`")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Split / sample: `{split}` / `{sample_idx}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Support grid: `{support_grid_shape[0]}x{support_grid_shape[1]}`")
    lines.append(f"- HR shape: `{hr_shape[0]}x{hr_shape[1]}`")
    lines.append(f"- Active slots: `{active_slots}`")
    lines.append("")
    lines.append("## Slot Usage")
    lines.append("")
    lines.append("| slot | support coverage | HR owner usage |")
    lines.append("| --- | ---: | ---: |")
    for slot_idx, coverage, usage in zip(active_slots, slot_coverage, slot_usage):
        lines.append(f"| {slot_idx} | {coverage:.4f} | {usage:.4f} |")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for key, path in gallery_paths.items():
        if path is not None:
            lines.append(f"- {key}: `{path}`")
    lines.append("")
    lines.append("## Stage Metrics")
    lines.append("")
    lines.append("| stage | shape | mean deg | p95 deg | max deg | note |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in metrics_rows:
        shape_txt = f"{int(row['height'])}x{int(row['width'])}"
        lines.append(
            f"| {row['stage']} | {shape_txt} | {row['mean_deg']:.3f} | {row['p95_deg']:.3f} | {row['max_deg']:.3f} | {row['note']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decode_stage_collection(
    model: IsoEmbeddingSROCRP,
    stages: list[dict[str, Any]],
    hr_target_hwc: torch.Tensor,
) -> list[dict[str, Any]]:
    decoded = decode_probe_stages(model, stages, sample_index=0)
    return [
        {
            "name": item["name"],
            "shape": item["shape"],
            "quat_hwc": item["quat_hwc"],
            "hr_target_hwc": hr_target_hwc,
        }
        for item in decoded
    ]


def main() -> None:
    args = parse_args()
    selected_gpu: int | None = None
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        print(f"Using requested GPU(s): {args.gpu_ids}")
    elif torch.cuda.is_available():
        selected_gpu = pick_most_free_cuda_gpu()
        if selected_gpu is not None:
            print(f"Auto-selected most-free GPU: {selected_gpu}")

    exp_dir = Path(args.exp_dir).resolve()
    config_path = _resolve_config_path(exp_dir, args.config)
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else _default_out_dir(exp_dir, args.split, int(args.sample_idx))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = exp_dir / "logs" / f"ocrp_macro_stage_probe_{str(args.split).lower()}_run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{selected_gpu}" if selected_gpu is not None else "cuda")
    else:
        device = torch.device("cpu")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=str(args.split).capitalize(),
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=int(args.sample_idx) + 1,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=False,
    )

    model = _load_ocrp_model_from_checkpoint(cfg, checkpoint_path, device=device)
    if not isinstance(model, IsoEmbeddingSROCRP):
        raise TypeError(f"Expected IsoEmbeddingSROCRP, got {type(model).__name__}")
    if str(model.ocrp.ocrp_mode) != "macro_tile":
        raise ValueError(f"Expected OCRP macro_tile mode, got {model.ocrp.ocrp_mode!r}")

    selected_batch = None
    for idx, batch in enumerate(loader):
        if idx == int(args.sample_idx):
            selected_batch = batch
            break
    if selected_batch is None:
        raise IndexError(f"sample_idx={args.sample_idx} is out of range for split={args.split!r}")

    lr_batch, hr_batch, _ = _unpack_batch(selected_batch)
    lr = lr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
    hr = hr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)

    lr_flat, lr_shape = _flatten_quat_chw(lr)
    lr_hwc = _to_hwc_quat_single(lr).detach().cpu()
    hr_hwc = _to_hwc_quat_single(hr).detach().cpu()

    with torch.enable_grad():
        feat_lr = model.encode(lr_flat)
        feat_hr, hr_shape, aux = model._forward_sr_features(
            lr_quats=lr_flat,
            feat_lr=feat_lr,
            lr_shape=lr_shape,
            return_aux=True,
        )
        sr_flat = model.decode(feat_hr)

    aux = {key: _as_unbatched(val) for key, val in aux.items()}

    support_grid_shape = tuple(int(v) for v in aux["support_grid_shape"])
    hr_shape = tuple(int(v) for v in hr_shape)

    slot_valid = aux["slot_valid"].to(dtype=torch.float32)
    owner_idx = aux["owner_idx"].to(dtype=torch.long)
    active_mask = slot_valid.mean(dim=0) > 0.01
    slot_usage_all = [(owner_idx == slot_idx).to(dtype=torch.float32).mean().item() for slot_idx in range(slot_valid.shape[1])]
    for slot_idx, usage in enumerate(slot_usage_all):
        if usage > 0.01:
            active_mask[slot_idx] = True
    active_slots = [idx for idx, flag in enumerate(active_mask.tolist()) if bool(flag)]
    if not active_slots:
        active_slots = [0]

    main_probe_stages: list[dict[str, Any]] = [
        _build_probe_stage("encode_lr", feat_lr, lr_shape),
    ]
    feat_lr_pre_ocrp = aux.get("feat_lr_pre_ocrp")
    if isinstance(feat_lr_pre_ocrp, torch.Tensor):
        main_probe_stages.append(_build_probe_stage("lr_conv1_pre_ocrp", feat_lr_pre_ocrp, lr_shape))

    feat_hr_raw_ocrp = aux.get("feat_hr_raw_ocrp")
    if isinstance(feat_hr_raw_ocrp, torch.Tensor):
        main_probe_stages.append(_build_probe_stage("selected_patch_out_hr", feat_hr_raw_ocrp, hr_shape))

    feat_hr_post = aux.get("feat_hr_post_hr_conv")
    if isinstance(feat_hr_post, torch.Tensor):
        main_probe_stages.append(_build_probe_stage("hr_conv1_post_ocrp", feat_hr_post, hr_shape))

    support_probe_stages: list[dict[str, Any]] = []
    slot_ctx = aux.get("slot_ctx")
    slot_pooled_ctx = aux.get("slot_pooled_ctx")
    if isinstance(slot_ctx, torch.Tensor):
        for slot_idx in active_slots:
            support_probe_stages.append(
                _build_probe_stage(
                    f"support_slot{slot_idx}_medoid_ctx",
                    slot_ctx[:, slot_idx, :],
                    support_grid_shape,
                )
            )
    if isinstance(slot_pooled_ctx, torch.Tensor):
        for slot_idx in active_slots:
            pooled_mean = slot_pooled_ctx[:, slot_idx, :, :].mean(dim=1)
            support_probe_stages.append(
                _build_probe_stage(
                    f"support_slot{slot_idx}_pooled_mean",
                    pooled_mean,
                    support_grid_shape,
                )
            )

    proposal_probe_stages: list[dict[str, Any]] = []
    patch_prop = aux.get("patch_prop")
    if isinstance(patch_prop, torch.Tensor):
        for slot_idx in active_slots:
            proposal_feat_hr = _assemble_macro_feature_map(
                model,
                patch_prop[:, slot_idx, :, :],
                grid_shape=support_grid_shape,
                hr_shape=hr_shape,
            )
            proposal_probe_stages.append(
                _build_probe_stage(
                    f"slot{slot_idx}_proposal_hr",
                    proposal_feat_hr,
                    hr_shape,
                )
            )

    main_decoded_rows = _decode_stage_collection(model, main_probe_stages, hr_hwc)
    support_decoded_rows = _decode_stage_collection(model, support_probe_stages, hr_hwc)
    proposal_decoded_rows = _decode_stage_collection(model, proposal_probe_stages, hr_hwc)

    sr_hwc = sr_flat.reshape(hr_shape[0], hr_shape[1], 4).detach().cpu()

    lr_row = {
        "name": "lr_input",
        "shape": tuple(int(v) for v in lr_hwc.shape[:2]),
        "quat_hwc": lr_hwc,
        "hr_target_hwc": hr_hwc,
    }
    sr_row = {
        "name": "sr_output",
        "shape": tuple(int(v) for v in sr_hwc.shape[:2]),
        "quat_hwc": sr_hwc,
        "hr_target_hwc": hr_hwc,
    }
    hr_row = {
        "name": "hr_target",
        "shape": tuple(int(v) for v in hr_hwc.shape[:2]),
        "quat_hwc": hr_hwc,
        "hr_target_hwc": hr_hwc,
    }

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    main_gallery_rows = [lr_row, *main_decoded_rows, sr_row, hr_row]
    support_gallery_rows = [lr_row, *support_decoded_rows, hr_row]
    proposal_gallery_rows = [lr_row, *proposal_decoded_rows, *main_decoded_rows[-2:], sr_row, hr_row]

    main_gallery_path = render_decoded_probe_gallery(
        main_gallery_rows,
        sym_class=sym_class,
        out_png=out_dir / "decoded_main_gallery.png",
    )
    support_gallery_path = render_decoded_probe_gallery(
        support_gallery_rows,
        sym_class=sym_class,
        out_png=out_dir / "decoded_support_context_gallery.png",
    )
    proposal_gallery_path = render_decoded_probe_gallery(
        proposal_gallery_rows,
        sym_class=sym_class,
        out_png=out_dir / "decoded_slot_proposal_gallery.png",
    )

    scalar_maps: list[dict[str, Any]] = []
    cluster_ids = aux.get("cluster_ids")
    if isinstance(cluster_ids, torch.Tensor):
        cluster_counts = torch.tensor(
            [int(torch.unique(cluster_ids[win_idx]).numel()) for win_idx in range(cluster_ids.shape[0])],
            dtype=torch.float32,
        ).view(*support_grid_shape)
        scalar_maps.append({"name": "support_cluster_count", "array": cluster_counts.cpu().numpy(), "cmap": "viridis"})

    slot_meta = aux.get("slot_meta")
    if isinstance(slot_meta, torch.Tensor):
        for slot_idx in active_slots:
            mass_map = slot_meta[:, slot_idx, ClusterSlotBuilder.META_MASS].view(*support_grid_shape)
            valid_map = slot_valid[:, slot_idx].view(*support_grid_shape)
            scalar_maps.append(
                {
                    "name": f"support_slot{slot_idx}_mass",
                    "array": mass_map.detach().cpu().numpy(),
                    "cmap": "magma",
                    "vmin": 0.0,
                    "vmax": 1.0,
                }
            )
            scalar_maps.append(
                {
                    "name": f"support_slot{slot_idx}_valid",
                    "array": valid_map.detach().cpu().numpy(),
                    "cmap": "gray",
                    "vmin": 0.0,
                    "vmax": 1.0,
                }
            )

    router_logits = aux.get("router_logits")
    if isinstance(router_logits, torch.Tensor):
        owner_prob = torch.softmax(router_logits, dim=-1)
        owner_conf = owner_prob.max(dim=-1).values
        topk = torch.topk(owner_prob, k=min(2, owner_prob.shape[-1]), dim=-1).values
        owner_margin = (
            topk[..., 0] - topk[..., 1]
            if topk.shape[-1] == 2
            else topk[..., 0]
        )
        owner_idx_hr = _assemble_macro_scalar_image(
            model,
            owner_idx.to(dtype=torch.float32),
            grid_shape=support_grid_shape,
            hr_shape=hr_shape,
        )
        owner_conf_hr = _assemble_macro_scalar_image(
            model,
            owner_conf,
            grid_shape=support_grid_shape,
            hr_shape=hr_shape,
        )
        owner_margin_hr = _assemble_macro_scalar_image(
            model,
            owner_margin,
            grid_shape=support_grid_shape,
            hr_shape=hr_shape,
        )
        scalar_maps.extend(
            [
                {
                    "name": "owner_idx_hr",
                    "array": owner_idx_hr,
                    "cmap": "tab10",
                    "vmin": 0.0,
                    "vmax": float(max(active_slots) if active_slots else 0),
                },
                {
                    "name": "owner_confidence_hr",
                    "array": owner_conf_hr,
                    "cmap": "viridis",
                    "vmin": 0.0,
                    "vmax": 1.0,
                },
                {
                    "name": "owner_margin_hr",
                    "array": owner_margin_hr,
                    "cmap": "cividis",
                    "vmin": 0.0,
                    "vmax": 1.0,
                },
            ]
        )

    scalar_gallery_path = render_scalar_probe_gallery(
        scalar_maps,
        out_png=out_dir / "scalar_routing_gallery.png",
    )

    unique_metric_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row in [lr_row, *main_decoded_rows, *support_decoded_rows, *proposal_decoded_rows, sr_row, hr_row]:
        if row["name"] in seen_names:
            continue
        seen_names.add(row["name"])
        unique_metric_rows.append(row)
    metrics_rows = _write_stage_metrics_csv(
        unique_metric_rows,
        target_hwc=hr_hwc,
        sym_class=sym_class,
        out_csv=out_dir / "stage_metrics.csv",
    )

    bundle = {
        "split": str(args.split),
        "sample_idx": int(args.sample_idx),
        "support_grid_shape": support_grid_shape,
        "hr_shape": hr_shape,
        "active_slots": active_slots,
        "main_probe_stage_names": [item["name"] for item in main_probe_stages],
        "support_probe_stage_names": [item["name"] for item in support_probe_stages],
        "proposal_probe_stage_names": [item["name"] for item in proposal_probe_stages],
        "scalar_probe_names": [item["name"] for item in scalar_maps],
        "lr_quat_hwc": lr_hwc,
        "sr_quat_hwc": sr_hwc,
        "hr_quat_hwc": hr_hwc,
    }
    torch.save(bundle, out_dir / "probe_bundle.pt")

    slot_coverages = [float(slot_valid[:, slot_idx].mean().item()) for slot_idx in active_slots]
    slot_usages = [float(slot_usage_all[slot_idx]) for slot_idx in active_slots]
    gallery_paths = {
        "decoded_main_gallery": main_gallery_path,
        "decoded_support_context_gallery": support_gallery_path,
        "decoded_slot_proposal_gallery": proposal_gallery_path,
        "scalar_routing_gallery": scalar_gallery_path,
        "stage_metrics_csv": out_dir / "stage_metrics.csv",
        "probe_bundle": out_dir / "probe_bundle.pt",
    }
    _write_summary_md(
        out_path=out_dir / "summary.md",
        exp_dir=exp_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        split=str(args.split),
        sample_idx=int(args.sample_idx),
        device=device,
        support_grid_shape=support_grid_shape,
        hr_shape=hr_shape,
        active_slots=active_slots,
        slot_coverage=slot_coverages,
        slot_usage=slot_usages,
        gallery_paths=gallery_paths,
        metrics_rows=metrics_rows,
    )

    metadata = {
        "exp_dir": str(exp_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": str(args.split),
        "sample_idx": int(args.sample_idx),
        "device": str(device),
        "support_grid_shape": list(support_grid_shape),
        "hr_shape": list(hr_shape),
        "active_slots": active_slots,
        "slot_coverages": slot_coverages,
        "slot_usages": slot_usages,
        "main_probe_stage_names": [item["name"] for item in main_probe_stages],
        "support_probe_stage_names": [item["name"] for item in support_probe_stages],
        "proposal_probe_stage_names": [item["name"] for item in proposal_probe_stages],
        "scalar_probe_names": [item["name"] for item in scalar_maps],
        "decoded_main_gallery": str(main_gallery_path),
        "decoded_support_context_gallery": str(support_gallery_path),
        "decoded_slot_proposal_gallery": str(proposal_gallery_path),
        "scalar_routing_gallery": str(scalar_gallery_path),
        "stage_metrics_csv": str(out_dir / "stage_metrics.csv"),
        "summary_md": str(out_dir / "summary.md"),
        "probe_bundle": str(out_dir / "probe_bundle.pt"),
    }
    with open(out_dir / "probe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved main decoded gallery: {main_gallery_path}")
    print(f"Saved support-context gallery: {support_gallery_path}")
    print(f"Saved slot-proposal gallery: {proposal_gallery_path}")
    print(f"Saved scalar routing gallery: {scalar_gallery_path}")
    print(f"Saved stage metrics: {out_dir / 'stage_metrics.csv'}")
    print(f"Saved summary: {out_dir / 'summary.md'}")
    print(f"Saved bundle: {out_dir / 'probe_bundle.pt'}")


if __name__ == "__main__":
    main()
