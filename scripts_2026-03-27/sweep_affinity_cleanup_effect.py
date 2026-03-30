#!/usr/bin/env python
"""Sweep statistical-cleanup knobs and visualize feature-space delta maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.model_stage_walkthrough import load_lr_input, load_model_from_experiment


def _parse_list_float(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_list_int(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_dir", required=True, type=str)
    p.add_argument("--config", type=str, default="config_train.json")
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--split", type=str, default="Test")
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--connect_thresholds", type=str, default="0.90,0.94,0.97")
    p.add_argument("--reassign_margins", type=str, default="-0.20,-0.05,0.12")
    p.add_argument("--snap_alphas", type=str, default="0.9,1.0")
    p.add_argument("--feature_consistency_thresholds", type=str, default="0.85,0.95,0.99")
    p.add_argument("--max_iters", type=str, default="1,2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else exp_dir / "visualizations" / "cleanup_effect_postpatch"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, _ = load_model_from_experiment(
        exp_dir=exp_dir,
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    model.eval()
    if not hasattr(model, "statistical_cleanup") or model.statistical_cleanup is None:
        raise RuntimeError("Model has no statistical_cleanup module enabled.")

    arr, sample_label = load_lr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
    )
    h_lr, w_lr, _ = arr.shape
    lr = torch.from_numpy(arr.reshape(h_lr * w_lr, 4)).to(model.device, dtype=torch.float32)

    with torch.no_grad():
        feat_lr = model.encode_a1(lr)
        saved_cleanup = model.statistical_cleanup
        model.statistical_cleanup = None
        feat_ref, hr_shape = model._forward_sr_features(feat_lr, (h_lr, w_lr))
        model.statistical_cleanup = saved_cleanup

    if feat_ref.dim() == 2:
        feat_ref_b = feat_ref.unsqueeze(0)
    else:
        feat_ref_b = feat_ref
    h_hr, w_hr = hr_shape

    rows: list[dict[str, float | int | str]] = []
    ct_list = _parse_list_float(args.connect_thresholds)
    mg_list = _parse_list_float(args.reassign_margins)
    sa_list = _parse_list_float(args.snap_alphas)
    fc_list = _parse_list_float(args.feature_consistency_thresholds)
    it_list = _parse_list_int(args.max_iters)

    cleanup = model.statistical_cleanup
    for ct in ct_list:
        for mg in mg_list:
            for sa in sa_list:
                for fc in fc_list:
                    for it in it_list:
                        cleanup.connect_threshold = float(ct)
                        cleanup.reassign_margin = float(mg)
                        cleanup.snap_alpha = float(sa)
                        cleanup.feature_consistency_threshold = float(fc)
                        cleanup.max_iters = int(it)

                        with torch.no_grad():
                            feat_cur, _ = model._forward_sr_features(feat_lr, (h_lr, w_lr))
                        if feat_cur.dim() == 2:
                            feat_cur_b = feat_cur.unsqueeze(0)
                        else:
                            feat_cur_b = feat_cur

                        d = (feat_cur_b - feat_ref_b).abs()
                        d_map = d.mean(dim=-1).reshape(h_hr, w_hr).detach().cpu()
                        row = {
                            "connect_threshold": float(ct),
                            "reassign_margin": float(mg),
                            "snap_alpha": float(sa),
                            "feature_consistency_threshold": float(fc),
                            "max_iters": int(it),
                            "mean_abs_delta": float(d.mean().item()),
                            "max_abs_delta": float(d.max().item()),
                            "frac_delta_gt_1e-6": float((d > 1e-6).float().mean().item()),
                        }
                        rows.append(row)

                        stem = (
                            f"ct_{ct:.2f}_mg_{mg:+.2f}_sa_{sa:.2f}_"
                            f"fc_{fc:.2f}_it_{it}"
                        ).replace("+", "p").replace("-", "m")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(d_map.numpy(), cmap="magma")
                        ax.set_title(f"{stem} | mean={row['mean_abs_delta']:.2e}")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                        fig.tight_layout()
                        fig.savefig(out_dir / f"{stem}_delta.png", dpi=180)
                        plt.close(fig)

    rows_sorted = sorted(rows, key=lambda r: float(r["mean_abs_delta"]), reverse=True)
    report = {
        "sample": sample_label,
        "num_settings": len(rows_sorted),
        "top10": rows_sorted[:10],
        "all": rows_sorted,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote cleanup sweep to: {out_dir}")
    print(f"Top setting: {rows_sorted[0] if rows_sorted else 'none'}")


if __name__ == "__main__":
    main()
