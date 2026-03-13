#!/usr/bin/env python3
"""Render SR/HR/LR IPF comparison images for SO3/O cubic SR checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from so3_o_cubic_sr import normalize_quaternion, standardize_quaternion_sign
from training.data_loading import build_dataloader
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import (
    render_input_output_side_by_side,
    render_sr_hr_lr_side_by_side,
)


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

    for cfg_name in ("config.json", "config_smoke.json"):
        p = exp_dir / cfg_name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "No config found in checkpoint and no config.json/config_smoke.json in exp_dir."
    )


def _load_symmetry_name(dataset_root: str, cfg: dict[str, Any]) -> str:
    info_path = Path(dataset_root) / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        sym = info.get("symmetry", None)
        if sym:
            return str(sym)
    return str(cfg.get("symmetry_group", cfg.get("symmetry", "O")))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SO3/O cubic SR model outputs as SR/HR/LR comparison PNGs."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best.pt",
        help="Checkpoint file (absolute path or relative to checkpoints dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Val",
        choices=["Train", "Val", "Test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument("--max_samples", type=int, default=8, help="Number of samples to render.")
    parser.add_argument(
        "--decode_dict_size",
        type=int,
        default=20000,
        help="Quaternion dictionary size for NN descriptor decode.",
    )
    parser.add_argument(
        "--decode_chunk",
        type=int,
        default=2048,
        help="Chunk size for descriptor NN decode.",
    )
    parser.add_argument(
        "--decode_topk",
        type=int,
        default=1,
        help="Top-k shortlist size for decoder retrieval.",
    )
    parser.add_argument(
        "--decode_refine_steps",
        type=int,
        default=0,
        help="Local refinement steps after dictionary init.",
    )
    parser.add_argument(
        "--decode_refine_lr",
        type=float,
        default=1e-2,
        help="Refinement step size for quaternion optimization.",
    )
    parser.add_argument(
        "--dict_sampling",
        type=str,
        default="random",
        choices=["random", "fz"],
        help="Dictionary sampling mode.",
    )
    parser.add_argument(
        "--dict_fz_resolution",
        type=int,
        default=3,
        help="FZ sampling resolution for dict_sampling=fz.",
    )
    parser.add_argument(
        "--dict_fz_method",
        type=str,
        default="cubochoric",
        help="FZ sampling method for dict_sampling=fz.",
    )
    parser.add_argument(
        "--dict_fz_point_group",
        type=str,
        default="O",
        help="Point group for FZ sampling (supports aliases via resolve_symmetry).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Default: exp_dir/visualizations/epoch_XXXX",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default="ALL",
        choices=["X", "Y", "Z", "ALL"],
        help="IPF reference direction.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Visualization batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (default=0 for compatibility).",
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
        out_dir = exp_dir / "visualizations" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symmetry_name = _load_symmetry_name(dataset_root, cfg)
    sym_class = resolve_symmetry(symmetry_name)
    print(f"Using symmetry group: {symmetry_name}")
    print(f"Saving visualizations to: {out_dir}")

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

    q_dict, x_dict = model.codec.build_dictionary(
        n=int(args.decode_dict_size),
        device=device,
        dtype=dtype,
        sampling=str(args.dict_sampling),
        fz_resolution=int(args.dict_fz_resolution),
        fz_method=str(args.dict_fz_method),
        fz_point_group=str(args.dict_fz_point_group),
    )

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

            dec = model.forward_quaternion_nn(
                q_lr=lr_q,
                q_dict=q_dict,
                x_dict=x_dict,
                chunk=int(args.decode_chunk),
                topk=int(args.decode_topk),
                refine_steps=int(args.decode_refine_steps),
                refine_lr=float(args.decode_refine_lr),
            )

            sr_q = standardize_quaternion_sign(normalize_quaternion(dec.quaternions))
            # sr_q: [B, sH, sW, 4]
            bsz = sr_q.shape[0]

            for b in range(bsz):
                if saved >= int(args.max_samples):
                    break

                sr_np = sr_q[b].detach().cpu().numpy()
                hr_np = hr_q[b].permute(1, 2, 0).detach().cpu().numpy()
                lr_np = lr_q[b].permute(1, 2, 0).detach().cpu().numpy()

                out_png = out_dir / f"sr_hr_lr_comparison_{saved:03d}.png"
                render_sr_hr_lr_side_by_side(
                    sr_q_arr=sr_np,
                    hr_q_arr=hr_np,
                    lr_q_arr=lr_np,
                    sym_class=sym_class,
                    out_png=str(out_png),
                    ref_dir=args.ref_dir,
                    include_key=True,
                    overwrite=True,
                )

                if lr_np.shape == sr_np.shape:
                    io_png = out_dir / f"input_output_comparison_{saved:03d}.png"
                    render_input_output_side_by_side(
                        input_q_arr=lr_np,
                        output_q_arr=sr_np,
                        sym_class=sym_class,
                        out_png=str(io_png),
                        ref_dir=args.ref_dir,
                        include_key=True,
                        overwrite=True,
                    )

                saved += 1
                print(f"Rendered sample {saved} -> {out_png.name}")

            if saved >= int(args.max_samples):
                break

    print(f"Done. Saved {saved} comparison image(s).")


if __name__ == "__main__":
    main()
