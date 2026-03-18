# -*- coding:utf-8 -*-
"""Run inference for IsoEmbeddingSRAttn checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.SR_double_conv_SRattn import IsoEmbeddingSRAttn
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


def _resolve_checkpoint(cfg, exp_dir: Path, ckpt_arg: str) -> Path:
    ckpt_candidate = Path(ckpt_arg)
    if ckpt_candidate.is_absolute():
        return ckpt_candidate
    checkpoints_dir = Path(getattr(cfg, "checkpoints_dir", exp_dir / "checkpoints"))
    return checkpoints_dir / ckpt_arg


def _load_model_from_checkpoint(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
) -> IsoEmbeddingSRAttn:
    model = IsoEmbeddingSRAttn(
        crystal=str(getattr(cfg, "crystal", "fcc")),
        d6_convention=str(getattr(cfg, "d6_convention", "z_axis")),
        device=device,
        upsample_factor=int(getattr(cfg, "scale", 4)),
        upsample_residual=bool(getattr(cfg, "upsample_residual", True)),
        num_hr_attn_blocks=int(getattr(cfg, "num_hr_attn_blocks", 1)),
        hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
        hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
        hr_attn_tp_out_chunk_size=getattr(cfg, "hr_attn_tp_out_chunk_size", 2048),
        hr_attn_checkpoint=bool(getattr(cfg, "hr_attn_checkpoint", False)),
        decoder_cubochoric_resolution=int(getattr(cfg, "decoder_cubochoric_resolution", 1)),
        decoder_num_starts=int(getattr(cfg, "decoder_num_starts", 2)),
        decoder_steps=int(getattr(cfg, "decoder_steps", 1)),
        decoder_lr=float(getattr(cfg, "decoder_lr", 0.05)),
        decoder_method=str(getattr(cfg, "decoder_method", "cubochoric")),
        decoder_max_table_rows=getattr(cfg, "decoder_max_table_rows", None),
        decoder_table_cache_dir=getattr(
            cfg, "decoder_table_cache_dir", "out/decoder_lookup_tables"
        ),
        decoder_backend=str(getattr(cfg, "decoder_backend", "optimizing")),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
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

    total_written = 0
    with torch.no_grad():
        for bidx, (lr_batch, hr_batch) in enumerate(tqdm(loader, desc=f"Infer-{split}", leave=False)):
            if args.max_batches is not None and bidx >= int(args.max_batches):
                break

            lr_batch = lr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            hr_batch = hr_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            bsz = int(lr_batch.shape[0])

            for j in range(bsz):
                lr = lr_batch[j]
                hr = hr_batch[j]

                lr_flat, lr_shape = _flatten_quat_chw(lr)
                hr_hwc = _to_hwc_quat_single(hr)
                hr_h, hr_w = int(hr_hwc.shape[0]), int(hr_hwc.shape[1])

                # Optimizing decoder performs an internal gradient-based solve.
                # Keep no_grad for the outer loop, but enable grad for decode.
                with torch.enable_grad():
                    sr_flat = model.forward_sr(lr_flat, lr_shape=lr_shape, normalize_input=True)
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
