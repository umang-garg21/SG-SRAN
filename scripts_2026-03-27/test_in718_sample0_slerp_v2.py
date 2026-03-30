#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from orix.quaternion import symmetry as SYM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boundary_aware_slerp import make_fcc_symmetry_4x4, qnorm, seam_crossing_heatmap
from boundary_aware_slerp_v2 import (
    SymBilinearSlerpUpsampleV2,
    compute_thin_gb_mask,
    run_boundary_smoothed_slerp,
)
from training.quaternion_dataset import QuaternionDataset
from visualization.ipf_render import render_ipf_image, render_ipf_rgb


def _load_sym_ops_from_disk_or_fallback(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    npy = "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy"
    if os.path.exists(npy):
        arr = np.load(npy)
        return torch.tensor(arr, dtype=dtype, device=device)
    return make_fcc_symmetry_4x4(device=device, dtype=dtype)


def _try_segment_labels(q_lr: torch.Tensor, sym_ops: torch.Tensor) -> torch.Tensor:
    """
    Attempt to segment LR grains using optional local utilities.
    Falls back to single-grain labels when unavailable.
    """
    H, W = q_lr.shape[-2], q_lr.shape[-1]
    try:
        from segment_grains import cleanup_small_grains_cuda, segment_grains_graph

        labels_np, nG = segment_grains_graph(q_lr.cpu(), sym_ops.cpu(), thr_deg=3.0)
        labels = torch.from_numpy(labels_np).long().to(q_lr.device)
        labels = cleanup_small_grains_cuda(
            labels,
            q_lr,
            sym_ops,
            min_pixels=3,
            max_iter=1,
        )
        print(f"[segmentation] segmented grains: {int(labels.max().item()) + 1} (initial graph: {nG})")
        return labels
    except Exception as exc:
        print(f"[segmentation] unavailable ({exc}); using fallback single-grain labels.")
        return torch.zeros((H, W), dtype=torch.long, device=q_lr.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SymBilinearSlerpUpsampleV2 on IN718 sample.")
    parser.add_argument("--sample-index", type=int, default=0, help="Test split sample index.")
    parser.add_argument("--scale", type=int, default=4, help="Upsampling factor.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/data/warren/materials/EBSD/IN718_FZ_2D_SR_x4",
        help="Dataset root containing dataset_info.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ipf_sym_aware_slerp_v2_sample0",
        help="Output directory.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable label-aware mode and run pure symmetry-aware bilinear SLERP.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ds = QuaternionDataset(dataset_root=args.dataset_dir, split="Test")
    if args.sample_index < 0 or args.sample_index >= len(ds):
        raise IndexError(f"sample-index {args.sample_index} out of range [0, {len(ds)-1}]")

    q_lr = ds[args.sample_index][0].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,4,H,W)
    q_lr = qnorm(q_lr)

    sym_ops = _load_sym_ops_from_disk_or_fallback(device=q_lr.device, dtype=q_lr.dtype)
    sym_class = SYM.O

    upsampler = SymBilinearSlerpUpsampleV2(
        scale_factor=args.scale,
        seam_threshold_deg=5.0,
        device=str(q_lr.device),
        dtype=q_lr.dtype,
    ).to(q_lr.device)

    if args.no_labels:
        with torch.inference_mode():
            q_hr = upsampler(q_lr)
        labels_hr_smooth = None
    else:
        labels_lr = _try_segment_labels(q_lr, sym_ops)
        q_hr, labels_hr_smooth = run_boundary_smoothed_slerp(
            q_lr=q_lr,
            labels_lr=labels_lr,
            scale=args.scale,
            upsampler=upsampler,
            smooth_iterations=40,
            smooth_lam=0.15,
            use_sdf=True,
        )

    q_hr = qnorm(q_hr)

    q_lr_np = q_lr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    q_hr_np = q_hr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)

    render_ipf_image(
        q_lr_np,
        sym_class,
        out_png=os.path.join(args.out_dir, "ipf_lr.png"),
        ref_dir="Z",
        include_key=True,
        overwrite=True,
    )
    render_ipf_image(
        q_hr_np,
        sym_class,
        out_png=os.path.join(args.out_dir, f"ipf_slerp_v2_x{args.scale}.png"),
        ref_dir="Z",
        include_key=True,
        overwrite=True,
    )

    heat = seam_crossing_heatmap(q_hr, sym_ops, threshold_deg=5.0)
    heat_np = heat.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(heat_np, cmap="hot")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "seam_heatmap.png"), dpi=200)
    plt.close()

    if labels_hr_smooth is not None:
        labels_hr_np = labels_hr_smooth.detach().cpu().numpy()
        gb_mask = compute_thin_gb_mask(labels_hr_np, connectivity=4)
        ipf_hr_rgb = render_ipf_rgb(q_hr_np, sym_class, ref_dir="Z")
        overlay = ipf_hr_rgb.copy().astype(np.float32)
        overlay[gb_mask] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        overlay = np.clip(overlay, 0.0, 1.0)

        plt.figure(figsize=(6, 6))
        plt.imshow((overlay * 255.0).astype(np.uint8))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "ipf_slerp_v2_gb_overlay.png"), dpi=200)
        plt.close()

    print(f"[done] wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
