# -*- coding:utf-8 -*-
"""Dedicated inference entrypoint for boundary-aware IsoEmbeddingSRAttn."""

from __future__ import annotations

import argparse
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

from inference.infer_iso_embedding_sr_attn import (  # Reuse tested generic helpers.
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
    _resolve_model_class,
    _to_hwc_quat_single,
    _unpack_batch,
)
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side


EXPECTED_MODEL_MODULE = "models.SR_double_conv_SRattn_a1_Boundary_aware"
EXPECTED_MODEL_CLASS = "IsoEmbeddingSRAttn"
DEFAULT_EXP_REL = Path("experiments/IN718/iso_embedding_sr_attn_boundary_aware_01")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_exp_dir = str((repo_root / DEFAULT_EXP_REL).resolve())

    parser = argparse.ArgumentParser(
        description="Inference for boundary-aware IsoEmbeddingSRAttn checkpoints"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=default_exp_dir,
        help=f"Experiment directory (default: {default_exp_dir}).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_new.json",
        help="Config file name inside exp_dir (default: config_new.json).",
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
        help="Output directory (default: <exp_dir>/inference/<split_lower>_boundary_aware).",
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
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0' or '6,7'.",
    )
    parser.add_argument(
        "--allow_other_model",
        action="store_true",
        help=(
            "Allow inference even if config model is not "
            f"{EXPECTED_MODEL_MODULE}.{EXPECTED_MODEL_CLASS}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    config_path = exp_dir / args.config
    run_config_path = exp_dir / "logs" / "inference_boundary_aware_run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cls, model_module, model_class = _resolve_model_class(cfg)
    print(f"Configured model: {model_module}.{model_class}")
    if (
        not args.allow_other_model
        and (model_module != EXPECTED_MODEL_MODULE or model_class != EXPECTED_MODEL_CLASS)
    ):
        raise ValueError(
            "This dedicated boundary-aware inference expects "
            f"{EXPECTED_MODEL_MODULE}.{EXPECTED_MODEL_CLASS}, got "
            f"{model_module}.{model_class}. "
            "Use --allow_other_model to bypass."
        )

    split = str(args.split).capitalize()
    forward_sr_params_cls = inspect.signature(model_cls.forward_sr).parameters
    model_supports_lr_boundary = "lr_boundary_map" in forward_sr_params_cls
    model_requires_lr_boundary = (
        model_supports_lr_boundary
        and forward_sr_params_cls["lr_boundary_map"].default is inspect._empty
    )
    if not model_supports_lr_boundary:
        raise ValueError(
            "Configured model forward_sr does not support lr_boundary_map. "
            "This script is for boundary-aware inference."
        )

    # Boundary-aware inference should always use LR boundary maps.
    use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    print(
        "LR boundary maps from dataloader: True "
        f"(model supports={model_supports_lr_boundary}, requires={model_requires_lr_boundary})"
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
        else (exp_dir / "inference" / f"{split.lower()}_boundary_aware")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_dir = out_dir / "sr_quaternions"
    ipf_dir = out_dir / "ipf"
    sr_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    records = []

    total_written = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(loader, desc=f"Infer-BA-{split}", leave=False)):
            if args.max_batches is not None and bidx >= int(args.max_batches):
                break

            lr_batch, hr_batch, lr_boundary_batch = _unpack_batch(batch)
            lr_batch = lr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            hr_batch = hr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            if lr_boundary_batch is None:
                raise ValueError(
                    "Boundary-aware inference requires lr_boundary_map, but dataloader returned None."
                )
            lr_boundary_batch = lr_boundary_batch.to(
                device=device, dtype=torch.float32, non_blocking=True
            )

            bsz = int(lr_batch.shape[0])
            for j in range(bsz):
                lr = lr_batch[j]
                hr = hr_batch[j]
                lr_boundary = lr_boundary_batch[j]

                lr_flat, lr_shape = _flatten_quat_chw(lr)
                hr_hwc = _to_hwc_quat_single(hr)
                hr_h, hr_w = int(hr_hwc.shape[0]), int(hr_hwc.shape[1])

                # Optimizing decoder performs an internal gradient-based solve.
                with torch.enable_grad():
                    sr_flat = model.forward_sr(
                        lr_flat,
                        lr_shape=lr_shape,
                        normalize_input=True,
                        lr_boundary_map=lr_boundary,
                    )
                if int(sr_flat.shape[0]) != int(hr_h * hr_w):
                    raise ValueError(
                        f"SR size mismatch: got N={int(sr_flat.shape[0])}, expected {int(hr_h * hr_w)}"
                    )

                sr_np = sr_flat.reshape(hr_h, hr_w, 4).detach().cpu().numpy().astype(np.float32)
                hr_np = hr_hwc.detach().cpu().numpy().astype(np.float32)
                lr_np = _to_hwc_quat_single(lr).detach().cpu().numpy().astype(np.float32)
                lr_boundary_np = lr_boundary.detach().cpu().numpy().astype(np.float32)

                sid = total_written
                sr_path = sr_dir / f"sample_{sid:06d}_sr.npy"
                lr_path = sr_dir / f"sample_{sid:06d}_lr.npy"
                hr_path = sr_dir / f"sample_{sid:06d}_hr.npy"
                bmap_path = sr_dir / f"sample_{sid:06d}_lr_boundary.npy"
                np.save(sr_path, sr_np)
                np.save(lr_path, lr_np)
                np.save(hr_path, hr_np)
                np.save(bmap_path, lr_boundary_np)

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
                        "lr_boundary_npy": str(bmap_path),
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
                "model_module": model_module,
                "model_class": model_class,
                "boundary_aware_inference": True,
                "num_samples": total_written,
                "records": records,
            },
            f,
            indent=2,
        )

    print("Boundary-aware inference complete.")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples written: {total_written}")
    print(f"Output dir: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

