# -*- coding:utf-8 -*-
"""Run inference for IsoEmbeddingSRAttn checkpoints."""

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
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for IsoEmbeddingSRAttn")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config file name inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename in checkpoints_dir, or absolute path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        choices=["Train", "Val", "Test"],
        help="Dataset split to run inference on.",
    )
    parser.add_argument(
        "--take_first",
        type=int,
        default=None,
        help="Optional sample cap for quick runs.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on number of dataloader batches.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <exp_dir>/inference/<split_lower>).",
    )
    parser.add_argument(
        "--viz_ref_dir",
        type=str,
        default="ALL",
        help="IPF reference direction: X, Y, Z, or ALL.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0'.",
    )
    return parser.parse_args()


def _flatten_quat_chw(q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if q.dim() != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {tuple(q.shape)}")
    if q.shape[0] == 4:
        _, h, w = q.shape
        q_flat = q.permute(1, 2, 0).reshape(h * w, 4)
        return q_flat, (h, w)
    if q.shape[-1] == 4:
        h, w, _ = q.shape
        q_flat = q.reshape(h * w, 4)
        return q_flat, (h, w)
    raise ValueError(f"Expected quaternion axis of size 4 in dim=0 or dim=-1, got {tuple(q.shape)}")


def _to_hwc_quat_single(q: torch.Tensor) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {tuple(q.shape)}")
    if q.shape[0] == 4:
        return q.permute(1, 2, 0)
    if q.shape[-1] == 4:
        return q
    raise ValueError(f"Expected quaternion axis of size 4 in dim=0 or dim=-1, got {tuple(q.shape)}")


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) == 2:
        lr, hr = batch
        return lr, hr, None
    if len(batch) == 3:
        lr, hr, lr_boundary_map = batch
        return lr, hr, lr_boundary_map
    raise ValueError(f"Expected batch length 2 or 3, got {len(batch)}")


def _resolve_checkpoint(cfg, exp_dir: Path, ckpt_arg: str) -> Path:
    ckpt_candidate = Path(ckpt_arg)
    if ckpt_candidate.is_absolute():
        return ckpt_candidate
    checkpoints_dir = Path(getattr(cfg, "checkpoints_dir", exp_dir / "checkpoints"))
    return checkpoints_dir / ckpt_arg


def _resolve_model_class(cfg):
    model_cfg = getattr(cfg, "model", {}) or {}
    if isinstance(model_cfg, dict):
        model_module = model_cfg.get(
            "model_module",
            getattr(cfg, "model_module", "models.SR_double_conv_SRattn_a1"),
        )
        model_class = model_cfg.get(
            "model_class",
            getattr(cfg, "model_class", "IsoEmbeddingSRAttn"),
        )
    else:
        model_module = getattr(
            model_cfg,
            "model_module",
            getattr(cfg, "model_module", "models.SR_double_conv_SRattn_a1"),
        )
        model_class = getattr(
            model_cfg,
            "model_class",
            getattr(cfg, "model_class", "IsoEmbeddingSRAttn"),
        )
    return getattr(importlib.import_module(model_module), model_class), model_module, model_class


def _load_model_from_checkpoint(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_cls, model_module, model_class = _resolve_model_class(cfg)
    print(f"Model: {model_module}.{model_class}")

    init_params = set(inspect.signature(model_cls.__init__).parameters)
    model_kwargs = {
        "crystal": str(getattr(cfg, "crystal", "fcc")),
        "d6_convention": str(getattr(cfg, "d6_convention", "z_axis")),
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
        "conv_feature_mask_soft": bool(getattr(cfg, "conv_feature_mask_soft", False)),
        "conv_feature_mask_temperature": float(
            getattr(cfg, "conv_feature_mask_temperature", 32.0)
        ),
        "hr_conv_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv_feature_mask_cosine_threshold", None
        ),
        "hr_conv_feature_mask_soft": getattr(cfg, "hr_conv_feature_mask_soft", None),
        "hr_conv_feature_mask_temperature": getattr(
            cfg, "hr_conv_feature_mask_temperature", None
        ),
        "hr_conv2_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv2_feature_mask_cosine_threshold", None
        ),
        "hr_conv2_feature_mask_soft": getattr(cfg, "hr_conv2_feature_mask_soft", None),
        "hr_conv2_feature_mask_temperature": getattr(
            cfg, "hr_conv2_feature_mask_temperature", None
        ),
        "hr_conv3_feature_mask_cosine_threshold": getattr(
            cfg, "hr_conv3_feature_mask_cosine_threshold", None
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
        "macro_lr_stride_shape": getattr(cfg, "macro_lr_stride_shape", None),
        "ocrp_token_conditioned_member_bias": getattr(cfg, "ocrp_token_conditioned_member_bias", None),
        "ocrp_pool_chunk_size": int(getattr(cfg, "ocrp_pool_chunk_size", 512)),
        "ocrp_router_chunk_size": int(getattr(cfg, "ocrp_router_chunk_size", 512)),
        "ocrp_proposal_chunk_size": int(getattr(cfg, "ocrp_proposal_chunk_size", 128)),
        "ocrp_proposal_token_chunk_size": getattr(cfg, "ocrp_proposal_token_chunk_size", None),
        "decoder_eager_init": bool(getattr(cfg, "decoder_eager_init", False)),
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in init_params}
    model = model_cls(**model_kwargs).to(device)

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


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    run_config_path = exp_dir / "logs" / "inference_run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split = str(args.split).capitalize()
    model_cls, _, _ = _resolve_model_class(cfg)
    forward_sr_params_cls = inspect.signature(model_cls.forward_sr).parameters
    model_supports_lr_boundary = "lr_boundary_map" in forward_sr_params_cls
    model_requires_lr_boundary = (
        model_supports_lr_boundary
        and forward_sr_params_cls["lr_boundary_map"].default is inspect._empty
    )
    feature_upsampler_type = str(getattr(cfg, "feature_upsampler_type", "shifted_bilinear")).strip().lower()
    use_lr_boundary_map = bool(getattr(cfg, "use_lr_boundary_map", model_supports_lr_boundary))
    if feature_upsampler_type == "grain_attention":
        use_lr_boundary_map = True
    if model_requires_lr_boundary:
        use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    print(
        f"LR boundary maps from dataloader: {use_lr_boundary_map} "
        f"(model supports={model_supports_lr_boundary}, requires={model_requires_lr_boundary}, "
        f"feature_upsampler_type={feature_upsampler_type})"
    )

    if args.take_first is not None:
        take_first = int(args.take_first)
    else:
        split_key = f"{split.lower()}_take_first"
        take_first_cfg = getattr(cfg, split_key, None)
        take_first = int(take_first_cfg) if take_first_cfg is not None else None

    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=split,
        batch_size=int(getattr(cfg, "batch_size", 1)),
        num_workers=int(getattr(cfg, "num_workers", 0)),
        preload=bool(getattr(cfg, "preload", False)),
        preload_torch=bool(getattr(cfg, "preload_torch", False)),
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        shuffle=False,
        take_first=take_first,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=use_lr_boundary_map,
        lr_boundary_angle_deg=lr_boundary_angle_deg,
        lr_boundary_mark_both_sides=lr_boundary_mark_both_sides,
    )

    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else (exp_dir / "inference" / split.lower())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_dir = out_dir / "sr_quaternions"
    ipf_dir = out_dir / "ipf"
    sr_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    records = []
    forward_sr_params = inspect.signature(model.forward_sr).parameters
    forward_supports_lr_boundary = "lr_boundary_map" in forward_sr_params
    forward_requires_lr_boundary = (
        forward_supports_lr_boundary
        and forward_sr_params["lr_boundary_map"].default is inspect._empty
    )

    total_written = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(loader, desc=f"Infer-{split}", leave=False)):
            if args.max_batches is not None and bidx >= int(args.max_batches):
                break

            lr_batch, hr_batch, lr_boundary_batch = _unpack_batch(batch)
            lr_batch = lr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            hr_batch = hr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            if lr_boundary_batch is not None:
                lr_boundary_batch = lr_boundary_batch.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            bsz = int(lr_batch.shape[0])

            for j in range(bsz):
                lr = lr_batch[j]
                hr = hr_batch[j]
                lr_boundary = lr_boundary_batch[j] if lr_boundary_batch is not None else None

                lr_flat, lr_shape = _flatten_quat_chw(lr)
                hr_hwc = _to_hwc_quat_single(hr)
                hr_h, hr_w = int(hr_hwc.shape[0]), int(hr_hwc.shape[1])

                # Optimizing decoder performs an internal gradient-based solve.
                # Keep no_grad for the outer loop, but enable grad for decode.
                with torch.enable_grad():
                    forward_kwargs = {
                        "lr_shape": lr_shape,
                        "normalize_input": True,
                    }
                    if forward_supports_lr_boundary:
                        if lr_boundary is None and forward_requires_lr_boundary:
                            raise ValueError(
                                "Model forward_sr requires lr_boundary_map but dataloader did not provide it."
                            )
                        if lr_boundary is not None:
                            forward_kwargs["lr_boundary_map"] = lr_boundary
                    sr_flat = model.forward_sr(lr_flat, **forward_kwargs)
                if int(sr_flat.shape[0]) != int(hr_h * hr_w):
                    raise ValueError(
                        f"SR size mismatch: got N={int(sr_flat.shape[0])}, expected {int(hr_h * hr_w)}"
                    )

                sr_np = sr_flat.reshape(hr_h, hr_w, 4).detach().cpu().numpy().astype(np.float32)
                hr_np = hr_hwc.detach().cpu().numpy().astype(np.float32)
                lr_np = _to_hwc_quat_single(lr).detach().cpu().numpy().astype(np.float32)

                sid = total_written
                sr_path = sr_dir / f"sample_{sid:06d}_sr.npy"
                lr_path = sr_dir / f"sample_{sid:06d}_lr.npy"
                hr_path = sr_dir / f"sample_{sid:06d}_hr.npy"
                np.save(sr_path, sr_np)
                np.save(lr_path, lr_np)
                np.save(hr_path, hr_np)

                ipf_path = ipf_dir / f"sample_{sid:06d}_lr_sr_hr_ipf.png"
                render_sr_hr_lr_side_by_side(
                    sr_q_arr=sr_np,
                    hr_q_arr=hr_np,
                    lr_q_arr=lr_np,
                    sym_class=sym_class,
                    out_png=str(ipf_path),
                    ref_dir=str(args.viz_ref_dir),
                    include_key=True,
                    overwrite=True,
                    format_input=True,
                    dpi=300,
                )

                records.append(
                    {
                        "sample_id": sid,
                        "batch_index": bidx,
                        "in_batch_index": j,
                        "lr_shape": [int(lr_shape[0]), int(lr_shape[1])],
                        "hr_shape": [int(hr_h), int(hr_w)],
                        "sr_npy": str(sr_path),
                        "lr_npy": str(lr_path),
                        "hr_npy": str(hr_path),
                        "ipf_png": str(ipf_path),
                    }
                )
                total_written += 1

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "exp_dir": str(exp_dir),
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "split": split,
                "num_samples": total_written,
                "records": records,
            },
            f,
            indent=2,
        )

    print("Inference complete.")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples written: {total_written}")
    print(f"Output dir: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
