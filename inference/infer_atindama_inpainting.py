"""Inference for the adapted Atindama partial-convolution inpainting baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.train_atindama_inpainting import (  # noqa: E402
    AUTHORS_DIR,
    PeriodicEulerInpaintingDataset,
    load_authors_model,
    normalized_zxz_to_passive_quaternion,
)
from utils.symmetry_utils import (  # noqa: E402
    proper_symmetry_quaternions,
    resolve_symmetry,
)
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with the Atindama inpainting baseline")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--split", default="Test", choices=["Train", "Val", "Test"])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--take_first", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_visualizations", type=int, default=5)
    return parser.parse_args()


def _resolve_checkpoint(exp_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_absolute() or path.exists():
        return path
    return exp_dir / "checkpoints" / checkpoint


def _quat_angle_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction / np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(prediction * target, axis=-1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _symmetry_angle_degrees(
    prediction: np.ndarray, target: np.ndarray, symmetry
) -> np.ndarray:
    """Minimum quaternion angle over proper crystal-symmetry equivalents."""
    prediction = prediction / np.maximum(
        np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-12
    )
    target = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    pred_flat = prediction.reshape(-1, 4)
    target_flat = target.reshape(-1, 4)

    operators = proper_symmetry_quaternions(symmetry)
    operators = operators.copy()
    operators[:, 1:] *= -1.0

    best_dot = np.zeros(pred_flat.shape[0], dtype=np.float32)
    pw, px, py, pz = np.moveaxis(pred_flat, -1, 0)
    for operator in operators:
        ow, ox, oy, oz = operator
        candidate = np.stack(
            [
                ow * pw - ox * px - oy * py - oz * pz,
                ow * px + ox * pw + oy * pz - oz * py,
                ow * py - ox * pz + oy * pw + oz * px,
                ow * pz + ox * py - oy * px + oz * pw,
            ],
            axis=-1,
        )
        best_dot = np.maximum(
            best_dot, np.abs(np.sum(candidate * target_flat, axis=-1))
        )
    angle = np.degrees(2.0 * np.arccos(best_dot.clip(0.0, 1.0)))
    return angle.reshape(prediction.shape[:-1])


def _minimum_angle_symmetry_representative(
    quaternion: np.ndarray, symmetry
) -> np.ndarray:
    """Finite minimum-angle representative under proper crystal symmetry."""
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    operators = proper_symmetry_quaternions(symmetry)
    operators = operators.copy()
    operators[:, 1:] *= -1.0

    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    best = q.copy()
    best_score = np.abs(qw)
    for operator in operators:
        ow, ox, oy, oz = operator
        candidate = np.stack(
            [
                ow * qw - ox * qx - oy * qy - oz * qz,
                ow * qx + ox * qw + oy * qz - oz * qy,
                ow * qy - ox * qz + oy * qw + oz * qx,
                ow * qz + ox * qy - oy * qx + oz * qw,
            ],
            axis=-1,
        )
        score = np.abs(candidate[..., 0])
        update = score > best_score
        best[update] = candidate[update]
        best_score[update] = score[update]
    best[best[..., 0] < 0.0] *= -1.0
    best /= np.maximum(np.linalg.norm(best, axis=-1, keepdims=True), 1e-12)
    return best.astype(np.float32, copy=False)


def sanitize_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    known_mask: torch.Tensor,
    scale,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize Euler channels and replace non-finite pixels without HR leakage."""
    if isinstance(scale, (list, tuple)):
        scale_y, scale_x = int(scale[0]), int(scale[1])
    else:
        scale_y = scale_x = int(scale)

    sampled = target[:, :, ::scale_y, ::scale_x]
    fallback = sampled.repeat_interleave(scale_y, dim=2).repeat_interleave(
        scale_x, dim=3
    )
    fallback = fallback[:, :, : target.shape[2], : target.shape[3]]

    valid_pixel = torch.isfinite(prediction).all(dim=1, keepdim=True)
    prediction = torch.where(valid_pixel, prediction, fallback)
    prediction = prediction.clone()
    # Euler rotations remain equivalent under these periods.  Do not clamp Phi:
    # finite values outside the canonical interval are still valid rotations.
    prediction[:, 0] = torch.remainder(prediction[:, 0], 1.0)
    prediction[:, 1] = torch.remainder(prediction[:, 1], 2.0)
    prediction[:, 2] = torch.remainder(prediction[:, 2], 1.0)
    composite = target * known_mask + prediction * (1.0 - known_mask)
    invalid_counts = (~valid_pixel[:, 0]).sum(dim=(1, 2))
    return composite, invalid_counts


def _authors_refinement_compatibility(mask: np.ndarray, patch_size: int = 3) -> dict:
    """Check the authors' requirement for a fully known exemplar patch."""
    from scipy import ndimage

    known = mask[0] > 0.5
    candidates = ndimage.binary_erosion(
        known, np.ones((patch_size, patch_size), dtype=bool)
    )
    count = int(candidates.sum())
    return {
        "patch_size": int(patch_size),
        "fully_known_source_patch_centers": count,
        "compatible": count > 0,
        "reason": (
            "compatible"
            if count > 0
            else "The published Criminisi stage requires fully known source patches; "
            "periodic SR masks contain none at patch_size=3."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / args.config, "r") as handle:
        cfg = json.load(handle)
    checkpoint_path = _resolve_checkpoint(exp_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PeriodicEulerInpaintingDataset(
        dataset_root=cfg["dataset_root"],
        split=args.split,
        scale=cfg["scale"],
        cache_dir=exp_dir / "cache" / "euler_zxz" if cfg.get("cache_euler", True) else None,
        take_first=args.take_first,
        return_quaternions=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("inference_num_workers", 2)),
        pin_memory=True,
    )

    model = load_authors_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    with open(Path(cfg["dataset_root"]) / "dataset_info.json", "r") as handle:
        dataset_info = json.load(handle)
    symmetry_str = dataset_info.get(
        "symmetry", cfg.get("symmetry_group", "Oh")
    )
    symmetry = resolve_symmetry(symmetry_str)
    proper_rotation_count = int(proper_symmetry_quaternions(symmetry).shape[0])

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "inference" / args.split.lower()
    arrays_dir = out_dir / "sr_quaternions"
    ipf_dir = out_dir / "ipf"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)

    records = []
    all_unknown_angles = []
    all_unknown_symmetry_angles = []
    total_invalid_predictions = 0
    sample_id = 0
    compatibility = None
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Infer-{args.split}", leave=False)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            target, known_mask, lr_quat, hr_quat = batch
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            known_mask_device = known_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction, _ = model(target * known_mask_device, known_mask_device)
            composite, invalid_counts = sanitize_prediction(
                prediction, target, known_mask_device, cfg["scale"]
            )

            for item_index in range(target.shape[0]):
                euler_hwc = composite[item_index].permute(1, 2, 0).cpu().numpy()
                raw_sr_quat = normalized_zxz_to_passive_quaternion(euler_hwc)
                sr_quat = _minimum_angle_symmetry_representative(
                    raw_sr_quat, symmetry
                )
                lr_np = lr_quat[item_index].numpy().astype(np.float32, copy=False)
                hr_np = hr_quat[item_index].numpy().astype(np.float32, copy=False)
                mask_np = known_mask[item_index].numpy()
                unknown = mask_np[0] < 0.5
                unknown_angles = _quat_angle_degrees(raw_sr_quat, hr_np)[unknown]
                unknown_symmetry_angles = _symmetry_angle_degrees(
                    raw_sr_quat, hr_np, symmetry
                )[unknown]
                all_unknown_angles.append(unknown_angles)
                all_unknown_symmetry_angles.append(unknown_symmetry_angles)
                invalid_count = int(invalid_counts[item_index].item())
                total_invalid_predictions += invalid_count

                sr_path = arrays_dir / f"sample_{sample_id:06d}_sr.npy"
                lr_path = arrays_dir / f"sample_{sample_id:06d}_lr.npy"
                hr_path = arrays_dir / f"sample_{sample_id:06d}_hr.npy"
                np.save(sr_path, sr_quat)
                np.save(lr_path, lr_np)
                np.save(hr_path, hr_np)

                ipf_path = None
                if sample_id < args.max_visualizations:
                    ipf_path = ipf_dir / f"sample_{sample_id:06d}_lr_sr_hr_ipf.png"
                    render_sr_hr_lr_side_by_side(
                        sr_q_arr=sr_quat,
                        hr_q_arr=hr_np,
                        lr_q_arr=lr_np,
                        sym_class=symmetry,
                        out_png=str(ipf_path),
                        ref_dir="ALL",
                        include_key=True,
                        overwrite=True,
                        format_input=False,
                        dpi=300,
                    )

                if compatibility is None:
                    compatibility = _authors_refinement_compatibility(mask_np, patch_size=3)
                records.append(
                    {
                        "sample_id": sample_id,
                        "mean_unknown_error_deg_no_symmetry": float(unknown_angles.mean()),
                        "median_unknown_error_deg_no_symmetry": float(np.median(unknown_angles)),
                        "mean_unknown_error_deg_crystal_symmetry": float(
                            unknown_symmetry_angles.mean()
                        ),
                        "invalid_predicted_pixels_replaced": invalid_count,
                        "sr_npy": str(sr_path),
                        "lr_npy": str(lr_path),
                        "hr_npy": str(hr_path),
                        "ipf_png": str(ipf_path) if ipf_path else None,
                    }
                )
                sample_id += 1

    angles = np.concatenate(all_unknown_angles) if all_unknown_angles else np.empty(0)
    symmetry_angles = (
        np.concatenate(all_unknown_symmetry_angles)
        if all_unknown_symmetry_angles
        else np.empty(0)
    )
    summary = {
        "exp_dir": str(exp_dir),
        "config": str(exp_dir / args.config),
        "checkpoint": str(checkpoint_path),
        "authors_source": str(AUTHORS_DIR),
        "split": args.split,
        "num_samples": sample_id,
        "symmetry_group": symmetry_str,
        "resolved_point_group": getattr(symmetry, "name", str(symmetry)),
        "fz_rotation_group": getattr(
            symmetry.proper_subgroup, "name", str(symmetry.proper_subgroup)
        ),
        "proper_rotation_count": proper_rotation_count,
        "metrics_no_crystal_symmetry": {
            "mean_unknown_error_deg": float(angles.mean()) if angles.size else None,
            "median_unknown_error_deg": float(np.median(angles)) if angles.size else None,
            "p95_unknown_error_deg": float(np.percentile(angles, 95)) if angles.size else None,
        },
        "metrics_with_crystal_symmetry": {
            "mean_unknown_error_deg": (
                float(symmetry_angles.mean()) if symmetry_angles.size else None
            ),
            "median_unknown_error_deg": (
                float(np.median(symmetry_angles)) if symmetry_angles.size else None
            ),
            "p95_unknown_error_deg": (
                float(np.percentile(symmetry_angles, 95))
                if symmetry_angles.size
                else None
            ),
        },
        "invalid_predicted_pixels_replaced_with_nearest_observation": (
            total_invalid_predictions
        ),
        "published_criminisi_refinement": compatibility,
        "records": records,
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("Inference complete.")
    print(f"Device: {device}")
    print(f"Samples: {sample_id}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
