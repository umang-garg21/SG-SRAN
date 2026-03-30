# -*- coding:utf-8 -*-
"""Plot feature-space loss versus quaternion-space loss across checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature loss and quaternion geodesic loss across checkpoints."
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Val",
        choices=["Train", "Val", "Test"],
        help="Dataset split for evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="all",
        help=(
            "Checkpoint selector: 'all', a filename in checkpoints/, "
            "a glob pattern (e.g. 'epoch_*.pt'), or an absolute path."
        ),
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Optional cap on number of checkpoints after sorting.",
    )
    parser.add_argument(
        "--take_first",
        type=int,
        default=None,
        help="Optional cap on dataset samples.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on number of dataloader batches.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch-size override for evaluation.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="feature_vs_quaternion_loss",
        help="Output file stem under <exp_dir>/diagnostics.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0'.",
    )
    return parser.parse_args()


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def _flatten_quaternion_batch(q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if q.dim() != 4:
        raise ValueError(f"Expected rank-4 quaternion batch, got {tuple(q.shape)}")
    if q.shape[1] == 4:
        bsz, _, h, w = q.shape
        return q.permute(0, 2, 3, 1).reshape(bsz, h * w, 4), (h, w)
    if q.shape[-1] == 4:
        bsz, h, w, _ = q.shape
        return q.reshape(bsz, h * w, 4), (h, w)
    raise ValueError(
        f"Expected quaternion axis of size 4 in dim=1 or dim=-1, got {tuple(q.shape)}"
    )


def _quaternion_geodesic_rad(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    q1 = _normalize_quaternions(q_pred)
    q2 = _normalize_quaternions(q_true)
    dots = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0 - eps)
    return 2.0 * torch.acos(dots)


def _resolve_take_first(cfg: Any, split: str, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    key = f"{split.lower()}_take_first"
    val = getattr(cfg, key, None)
    return int(val) if val is not None else None


def _checkpoint_epoch_key(path: Path) -> tuple[int, str]:
    """
    Sort key:
      1) by inferred epoch id when available
      2) stable by name
    """
    m = re.search(r"epoch[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), path.name
    if path.name == "best_model.pt":
        return 10**9, path.name
    return 10**8, path.name


def _resolve_checkpoints(exp_dir: Path, selector: str) -> list[Path]:
    checkpoints_dir = exp_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    sel = str(selector).strip()
    if sel.lower() == "all":
        pts = sorted(checkpoints_dir.glob("*.pt"), key=_checkpoint_epoch_key)
        if len(pts) == 0:
            raise FileNotFoundError(f"No .pt checkpoints found in: {checkpoints_dir}")
        return pts

    p = Path(sel)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return [p]

    if any(ch in sel for ch in ["*", "?", "[", "]"]):
        pts = sorted(checkpoints_dir.glob(sel), key=_checkpoint_epoch_key)
        if len(pts) == 0:
            raise FileNotFoundError(
                f"No checkpoints matched pattern '{sel}' in {checkpoints_dir}"
            )
        return pts

    candidate = checkpoints_dir / sel
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return [candidate]


def _extract_epoch_from_blob_or_name(ckpt_blob: dict[str, Any], ckpt_path: Path) -> int | None:
    if isinstance(ckpt_blob, dict) and "epoch" in ckpt_blob:
        try:
            return int(ckpt_blob["epoch"])
        except Exception:
            pass
    m = re.search(r"epoch[_-]?(\d+)", ckpt_path.stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


@torch.no_grad()
def _feature_loss_batch(
    model: torch.nn.Module,
    lr_flat: torch.Tensor,
    hr_flat: torch.Tensor,
    lr_shape: tuple[int, int],
) -> float:
    loss = model.feature_loss_sr(
        lr_flat,
        hr_flat,
        lr_shape=lr_shape,
        normalize_input=True,
    )
    return float(loss.detach().item())


def _quat_loss_batch(
    model: torch.nn.Module,
    lr_flat: torch.Tensor,
    hr_flat: torch.Tensor,
    lr_shape: tuple[int, int],
) -> tuple[float, float]:
    # Decoder backend performs a local optimization step, so grad context is required.
    with torch.enable_grad():
        q_pred = model.forward_sr(
            lr_flat,
            lr_shape=lr_shape,
            normalize_input=True,
        )
    q_pred = q_pred.detach()
    q_true = _normalize_quaternions(hr_flat.detach())
    geod = _quaternion_geodesic_rad(q_pred, q_true)
    geod_mean_rad = float(geod.mean().item())
    geod_mean_deg = float(geod_mean_rad * (180.0 / np.pi))
    return geod_mean_rad, geod_mean_deg


def _evaluate_checkpoint(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()

    feat_sum = 0.0
    quat_rad_sum = 0.0
    quat_deg_sum = 0.0
    n_batches = 0

    for bidx, (lr, hr) in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches is not None and bidx >= int(max_batches):
            break

        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        lr_flat, lr_shape = _flatten_quaternion_batch(lr)
        hr_flat, _ = _flatten_quaternion_batch(hr)

        feat_loss = _feature_loss_batch(model, lr_flat, hr_flat, lr_shape)
        quat_rad, quat_deg = _quat_loss_batch(model, lr_flat, hr_flat, lr_shape)

        feat_sum += feat_loss
        quat_rad_sum += quat_rad
        quat_deg_sum += quat_deg
        n_batches += 1

    if n_batches == 0:
        return {
            "feature_loss": float("nan"),
            "quat_geodesic_rad": float("nan"),
            "quat_geodesic_deg": float("nan"),
            "num_batches": 0.0,
        }

    return {
        "feature_loss": feat_sum / n_batches,
        "quat_geodesic_rad": quat_rad_sum / n_batches,
        "quat_geodesic_deg": quat_deg_sum / n_batches,
        "num_batches": float(n_batches),
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "epoch",
        "feature_loss",
        "quat_geodesic_rad",
        "quat_geodesic_deg",
        "num_batches",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _plot_results(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    feat = np.array([float(r["feature_loss"]) for r in rows], dtype=np.float64)
    quat = np.array([float(r["quat_geodesic_deg"]) for r in rows], dtype=np.float64)
    epochs = np.array(
        [np.nan if r["epoch"] is None else float(r["epoch"]) for r in rows], dtype=np.float64
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Left: scatter feature-vs-quat
    ax0 = axes[0]
    ax0.scatter(feat, quat, s=42, c="#2563eb", alpha=0.9, edgecolors="white", linewidths=0.7)
    for r in rows:
        label = f"e{r['epoch']}" if r["epoch"] is not None else Path(r["checkpoint"]).stem[:12]
        ax0.annotate(
            label,
            (float(r["feature_loss"]), float(r["quat_geodesic_deg"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#334155",
        )
    ax0.set_xlabel("Feature Loss (MSE)")
    ax0.set_ylabel("Quaternion Geodesic Loss (deg)")
    ax0.set_title("Feature Loss vs Quaternion Loss")
    ax0.grid(True, alpha=0.25)

    # Right: trend over checkpoints/epochs
    ax1 = axes[1]
    x = np.arange(len(rows), dtype=np.float64)
    use_epoch = np.all(np.isfinite(epochs))
    xvals = epochs if use_epoch else x
    x_label = "Epoch" if use_epoch else "Checkpoint index"
    ax1.plot(xvals, feat, marker="o", color="#2563eb", label="Feature Loss (MSE)")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Feature Loss (MSE)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, alpha=0.25)

    ax1b = ax1.twinx()
    ax1b.plot(xvals, quat, marker="s", color="#dc2626", label="Quat Geodesic (deg)")
    ax1b.set_ylabel("Quaternion Geodesic Loss (deg)", color="#dc2626")
    ax1b.tick_params(axis="y", labelcolor="#dc2626")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.set_title("Checkpoint Trend")

    fig.suptitle(f"Feature vs Quaternion Loss ({split} split)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    config_path = exp_dir / args.config

    # Defer project imports so `--help` is lightweight and robust.
    from training.config_utils import load_and_prepare_config
    from training.data_loading import build_dataloader
    from utils.runtime_helpers import (
        assert_expected_model_import,
        build_iso_embedding_sr_attn_from_config,
    )

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_and_prepare_config(config_path, save_path=None)

    # Build loader once.
    split = str(args.split).capitalize()
    take_first = _resolve_take_first(cfg, split, args.take_first)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(getattr(cfg, "batch_size", 1))
    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=split,
        batch_size=batch_size,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        preload=bool(getattr(cfg, "preload", False)),
        preload_torch=bool(getattr(cfg, "preload_torch", False)),
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        shuffle=False,
        take_first=take_first,
        seed=int(getattr(cfg, "seed", 42)),
    )

    model = build_iso_embedding_sr_attn_from_config(cfg, device=device)
    assert_expected_model_import(type(model))

    ckpt_paths = _resolve_checkpoints(exp_dir, args.checkpoint)
    if args.max_checkpoints is not None:
        ckpt_paths = ckpt_paths[: int(args.max_checkpoints)]

    print(f"Evaluating {len(ckpt_paths)} checkpoint(s) on split={split}")
    rows: list[dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        print(f"\n[checkpoint] {ckpt_path}")
        blob = torch.load(ckpt_path, map_location=device)
        state = blob.get("model_state_dict", blob)
        model.load_state_dict(state, strict=True)
        metrics = _evaluate_checkpoint(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )
        epoch = _extract_epoch_from_blob_or_name(blob if isinstance(blob, dict) else {}, ckpt_path)

        row = {
            "checkpoint": str(ckpt_path),
            "epoch": epoch,
            **metrics,
        }
        rows.append(row)
        print(
            f"feature_loss={metrics['feature_loss']:.6e}  "
            f"quat_deg={metrics['quat_geodesic_deg']:.6f}  "
            f"batches={int(metrics['num_batches'])}"
        )

    out_dir = exp_dir / "diagnostics"
    out_prefix = str(args.out_prefix)
    csv_path = out_dir / f"{out_prefix}.csv"
    json_path = out_dir / f"{out_prefix}.json"
    png_path = out_dir / f"{out_prefix}.png"

    _save_csv(csv_path, rows)
    _save_json(
        json_path,
        {
            "exp_dir": str(exp_dir),
            "config": str(config_path),
            "split": split,
            "num_checkpoints": len(rows),
            "rows": rows,
        },
    )
    _plot_results(png_path, rows, split=split)

    print("\nSaved:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  PNG : {png_path}")


if __name__ == "__main__":
    main()
