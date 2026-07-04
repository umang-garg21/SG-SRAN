"""Run inference for Jangid/UCSB QEDSR-style baseline checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.data_loading import build_dataloader  # noqa: E402
from training.train_jangid_baseline import (  # noqa: E402
    build_model,
    conjugate_scalar_first_chw,
    forward_baseline_active_from_passive,
    to_chw,
    to_hwc_numpy,
)
from utils import reduce_to_fz_min_angle  # noqa: E402
from utils.symmetry_utils import (  # noqa: E402
    proper_symmetry_quaternions,
    resolve_symmetry,
)
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Jangid/UCSB QEDSR-style baseline")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument("--config", default="config.json", type=str, help="Config filename inside exp_dir.")
    parser.add_argument(
        "--checkpoint",
        default="best_model.pt",
        type=str,
        help="Checkpoint filename in checkpoints/, or absolute path.",
    )
    parser.add_argument("--split", default="Test", choices=["Train", "Val", "Test"], help="Dataset split.")
    parser.add_argument("--take_first", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional dataloader batch cap.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--viz_ref_dir", type=str, default="ALL", help="IPF reference direction.")
    parser.add_argument(
        "--max_visualizations",
        type=int,
        default=5,
        help="Maximum number of IPF comparison figures to render; all arrays are still exported.",
    )
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--no_reduce_fz",
        action="store_true",
        help="Do not reduce exported SR quaternions to the fundamental zone.",
    )
    return parser.parse_args()


def _resolve_checkpoint(exp_dir: Path, checkpoint_arg: str) -> Path:
    candidate = Path(checkpoint_arg)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return exp_dir / "checkpoints" / checkpoint_arg


def _make_loader(cfg: dict, split: str, take_first: int | None):
    if take_first is None:
        cfg_take_first = cfg.get(f"{split.lower()}_take_first", cfg.get("take_first"))
        take_first = int(cfg_take_first) if cfg_take_first is not None else None
    return build_dataloader(
        cfg["dataset_root"],
        split=split,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        seed=int(cfg.get("seed", 42)),
        preload=bool(cfg.get("preload", False)),
        preload_torch=bool(cfg.get("preload_torch", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        take_first=take_first,
    )


def _load_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    checkpoint_path = _resolve_checkpoint(exp_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    with open(Path(cfg["dataset_root"]) / "dataset_info.json", "r") as f:
        dataset_info = json.load(f)
    symmetry_str = dataset_info.get("symmetry", cfg.get("symmetry", cfg.get("symmetry_group", "Oh")))
    sym_class = resolve_symmetry(symmetry_str)
    proper_symmetry = sym_class.proper_subgroup
    proper_rotation_count = int(proper_symmetry_quaternions(sym_class).shape[0])

    loader = _make_loader(cfg, args.split, args.take_first)
    model = _load_model(cfg, checkpoint_path, device)

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "inference" / args.split.lower()
    sr_dir = out_dir / "sr_quaternions"
    ipf_dir = out_dir / "ipf"
    sr_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_written = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(loader, desc=f"Infer-{args.split}", leave=False)):
            if args.max_batches is not None and bidx >= int(args.max_batches):
                break
            lr_batch = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            hr_batch = batch[1].to(device=device, dtype=torch.float32, non_blocking=True)
            bsz = int(lr_batch.shape[0])

            for j in range(bsz):
                lr_passive_chw = to_chw(lr_batch[j : j + 1])
                hr_passive_chw = to_chw(hr_batch[j : j + 1])
                sr_active_chw = forward_baseline_active_from_passive(model, lr_passive_chw)
                sr_passive_chw = conjugate_scalar_first_chw(sr_active_chw)

                sr_np = to_hwc_numpy(sr_passive_chw[0]).astype(np.float32, copy=False)
                hr_np = to_hwc_numpy(hr_passive_chw[0]).astype(np.float32, copy=False)
                lr_np = to_hwc_numpy(lr_passive_chw[0]).astype(np.float32, copy=False)

                if not args.no_reduce_fz:
                    sr_np = reduce_to_fz_min_angle(
                        sr_np,
                        sym=proper_symmetry,
                        normalize=True,
                        hemisphere=True,
                        return_op_map=False,
                    ).astype(np.float32, copy=False)

                sid = total_written
                sr_path = sr_dir / f"sample_{sid:06d}_sr.npy"
                lr_path = sr_dir / f"sample_{sid:06d}_lr.npy"
                hr_path = sr_dir / f"sample_{sid:06d}_hr.npy"
                np.save(sr_path, sr_np)
                np.save(lr_path, lr_np)
                np.save(hr_path, hr_np)

                ipf_path = None
                if sid < int(args.max_visualizations):
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
                        "lr_shape": [int(lr_np.shape[0]), int(lr_np.shape[1])],
                        "hr_shape": [int(hr_np.shape[0]), int(hr_np.shape[1])],
                        "sr_npy": str(sr_path),
                        "lr_npy": str(lr_path),
                        "hr_npy": str(hr_path),
                        "ipf_png": str(ipf_path) if ipf_path is not None else None,
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
                "split": args.split,
                "num_samples": total_written,
                "symmetry_group": symmetry_str,
                "resolved_point_group": getattr(sym_class, "name", str(sym_class)),
                "fz_rotation_group": getattr(
                    proper_symmetry, "name", str(proper_symmetry)
                ),
                "proper_rotation_count": proper_rotation_count,
                "records": records,
            },
            f,
            indent=2,
        )

    print("Inference complete.")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples written: {total_written}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
