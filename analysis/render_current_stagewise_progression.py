#!/usr/bin/env python3
"""Recompute the OCRP stagewise decoded-error panel from the corrected checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.infer_iso_embedding_sr_attn import (  # noqa: E402
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
    _to_hwc_quat_single,
    _unpack_batch,
)
from training.config_utils import load_and_prepare_config  # noqa: E402
from training.data_loading import build_dataloader  # noqa: E402
from utils.stage_probe_utils import quat_ang_err_deg  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402


PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
EVAL_DIR = PAPER_DIR / "evals"

DEFAULT_EXP = (
    ROOT
    / "experiments/IN718/direct_reynolds_isometric_seed_runs/"
    / "ocrp_direct_reynolds_isometric_l4_s42_fresh_allepochs_20260707_2205"
)

STAGE_SPECS = [
    ("encode_lr", "Encode LR", "feat_lr", "lr"),
    ("context_refine", "Context\naggregation", "feat_lr_pre_ocrp", "lr"),
    ("routed_patch", "Routed patch\nsynthesis", "feat_hr_raw_ocrp", "hr"),
    ("hr_refine_1", "HR refine 1", "feat_hr_post_hr_conv1", "hr"),
    ("hr_refine_2", "HR refine 2", "feat_hr_post_hr_conv2", "hr"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--config", default="config_new.json")
    parser.add_argument("--checkpoint", default="epoch_0012.pt")
    parser.add_argument("--split", default="Test", choices=["Train", "Val", "Test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stop", type=int, default=None)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--decode-chunk", type=int, default=8192)
    parser.add_argument("--out-stem", default="main_stagewise_progression")
    parser.add_argument("--out-tag", default="current_direct_reynolds_stagewise_notebook_ckpt_20260708")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7dbe2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.facecolor": "#fcfcfd",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def resolve_config_path(exp_dir: Path, config_arg: str) -> Path:
    path = Path(config_arg)
    return path if path.is_absolute() else exp_dir / path


def upsample_feature_tokens(
    feat: torch.Tensor,
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> torch.Tensor:
    if tuple(from_shape) == tuple(to_shape):
        return feat
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    batch, n_tokens, channels = feat.shape
    h_src, w_src = int(from_shape[0]), int(from_shape[1])
    h_dst, w_dst = int(to_shape[0]), int(to_shape[1])
    if n_tokens != h_src * w_src:
        raise ValueError(f"Feature token count {n_tokens} does not match shape {from_shape}")
    image = feat.reshape(batch, h_src, w_src, channels).permute(0, 3, 1, 2).contiguous()
    up = F.interpolate(image, size=(h_dst, w_dst), mode="nearest")
    return up.permute(0, 2, 3, 1).reshape(batch, h_dst * w_dst, channels)


def decode_feature_tokens(model: Any, feat: torch.Tensor, chunk: int) -> torch.Tensor:
    if feat.dim() == 3:
        if feat.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for stage decode, got {tuple(feat.shape)}")
        feat = feat[0]
    feat = torch.nan_to_num(feat.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    decoded: list[torch.Tensor] = []
    for start in range(0, feat.shape[0], int(chunk)):
        part = feat[start : start + int(chunk)]
        with torch.enable_grad():
            decoded.append(model.decode(part))
    return torch.cat(decoded, dim=0)


def finite_stats(err: torch.Tensor) -> tuple[float, float, float, float]:
    finite = err[torch.isfinite(err)]
    if finite.numel() == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (
        float(finite.mean().item()),
        float(torch.median(finite).item()),
        float(torch.quantile(finite, 0.90).item()),
        float(torch.quantile(finite, 0.95).item()),
    )


def collect_stage_rows(model: Any, aux: dict[str, Any], feat_lr: torch.Tensor) -> list[tuple[str, str, torch.Tensor, tuple[int, int]]]:
    rows: list[tuple[str, str, torch.Tensor, tuple[int, int]]] = []
    lr_shape = tuple(int(v) for v in aux["_lr_shape"])
    hr_shape = tuple(int(v) for v in aux["_hr_shape"])
    for key, label, tensor_name, shape_kind in STAGE_SPECS:
        if tensor_name == "feat_lr":
            tensor = feat_lr
        else:
            tensor = aux.get(tensor_name)
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = lr_shape if shape_kind == "lr" else hr_shape
        rows.append((key, label, tensor, shape))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_stagewise(summary_rows: list[dict[str, Any]], out_stem: str) -> None:
    setup_style()
    labels = [str(row["label"]) for row in summary_rows]
    means = np.asarray([float(row["mean_deg"]) for row in summary_rows], dtype=np.float64)
    stds = np.asarray([float(row["std_patch_mean_deg"]) for row in summary_rows], dtype=np.float64)
    xs = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.7, 3.75))
    if len(labels) >= 3:
        ax.axvspan(1.5, 2.5, color="#f6d365", alpha=0.22, lw=0)
    ax.errorbar(
        xs,
        means,
        yerr=stds,
        color="#16213e",
        marker="o",
        lw=3.0,
        ms=6.2,
        capsize=4,
        zorder=5,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean decoded $d_{\mathrm{Stab}}$ ($^\circ$)")
    ax.set_xlabel("OCRP pipeline stage")
    if len(means) >= 3:
        drop = means[1] - means[2]
        ax.annotate(
            rf"$- {drop:.2f}^\circ$",
            xy=(1.5, (means[1] + means[2]) / 2.0),
            xytext=(2.35, max(means) - 0.18 * (max(means) - min(means) + 1e-6)),
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#3a3f58"},
            ha="left",
            va="center",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#ccd3df", "alpha": 0.96},
        )
        ax.text(
            2.0,
            ax.get_ylim()[1] - 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "Routed patch step",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#7a5b00",
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    exp_dir = args.exp_dir.resolve()
    config_path = resolve_config_path(exp_dir, args.config)
    out_tag = str(args.out_tag)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading config: {config_path}", flush=True)
    cfg = load_and_prepare_config(config_path, EVAL_DIR / f"{out_tag}_resolved_config.json")

    print("selecting device", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}", flush=True)
    print(f"resolving checkpoint: {args.checkpoint}", flush=True)
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    print(f"loading checkpoint: {checkpoint_path}", flush=True)
    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    print("model loaded", flush=True)
    if not hasattr(model, "_forward_sr_features"):
        raise TypeError(f"Expected OCRP model with _forward_sr_features, got {type(model).__name__}")
    model.eval()

    print(f"building dataloader for split={args.split}", flush=True)
    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=str(args.split).capitalize(),
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=args.max_samples,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=False,
    )
    print(f"dataloader length: {len(loader)}", flush=True)
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    stage_patch_values: dict[str, list[float]] = {}
    stage_rows: list[dict[str, Any]] = []
    n_seen = 0

    sample_start = max(0, int(args.sample_start))
    sample_stop = len(loader) if args.sample_stop is None else min(len(loader), int(args.sample_stop))
    if sample_stop <= sample_start:
        raise ValueError(f"Empty sample range: start={sample_start}, stop={sample_stop}, len={len(loader)}")
    target_count = sample_stop - sample_start

    for batch_idx, batch in enumerate(loader):
        if batch_idx < sample_start:
            continue
        if batch_idx >= sample_stop:
            break
        lr_batch, hr_batch, _ = _unpack_batch(batch)
        lr = lr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
        lr_flat, lr_shape = _flatten_quat_chw(lr)
        hr_hwc = _to_hwc_quat_single(hr).detach().cpu()

        with torch.no_grad():
            feat_lr = model.encode(lr_flat)
            feat_hr, hr_shape, aux = model._forward_sr_features(
                lr_quats=lr_flat,
                feat_lr=feat_lr,
                lr_shape=lr_shape,
                return_aux=True,
            )
        aux = {key: (val[0] if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == 1 else val) for key, val in aux.items()}
        aux["_lr_shape"] = tuple(int(v) for v in lr_shape)
        aux["_hr_shape"] = tuple(int(v) for v in hr_shape)

        rows = collect_stage_rows(model, aux, feat_lr)
        for key, label, tensor, shape in rows:
            feat_stage = tensor.unsqueeze(0) if tensor.dim() == 2 else tensor
            feat_stage = upsample_feature_tokens(feat_stage, shape, tuple(int(v) for v in hr_shape))
            q_stage = decode_feature_tokens(model, feat_stage, chunk=int(args.decode_chunk))
            q_stage_hwc = q_stage.reshape(int(hr_shape[0]), int(hr_shape[1]), 4).detach().cpu()
            err = quat_ang_err_deg(q_stage_hwc, hr_hwc, sym=sym_class)
            mean_deg, median_deg, p90_deg, p95_deg = finite_stats(err)
            stage_patch_values.setdefault(key, []).append(mean_deg)
            stage_rows.append(
                {
                    "sample_index": batch_idx,
                    "stage": key,
                    "label": label.replace("\n", " "),
                    "mean_deg": mean_deg,
                    "median_deg": median_deg,
                    "p90_deg": p90_deg,
                    "p95_deg": p95_deg,
                }
            )
        n_seen += 1
        print(f"stagewise decoded {n_seen}/{target_count} samples (global index {batch_idx})", flush=True)

    summary_rows: list[dict[str, Any]] = []
    labels_by_key = {key: label for key, label, _, _ in STAGE_SPECS}
    for key, _, _, _ in STAGE_SPECS:
        vals = np.asarray(stage_patch_values.get(key, []), dtype=np.float64)
        if vals.size == 0:
            continue
        summary_rows.append(
            {
                "stage": key,
                "label": labels_by_key[key],
                "mean_deg": float(np.nanmean(vals)),
                "std_patch_mean_deg": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
                "n_samples": int(vals.size),
            }
        )

    payload = {
        "provenance": {
            "exp_dir": str(exp_dir),
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "split": str(args.split),
            "max_samples": args.max_samples,
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "device": str(device),
            "embedding_mode": getattr(cfg, "embedding_mode", None),
            "max_harmonic_l": getattr(cfg, "max_harmonic_l", None),
            "embedding_metric_calibration": getattr(cfg, "embedding_metric_calibration", None),
        },
        "summary": summary_rows,
    }
    (EVAL_DIR / f"{out_tag}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(EVAL_DIR / f"{out_tag}_per_sample.csv", stage_rows)
    plot_stagewise(summary_rows, args.out_stem)
    print(f"wrote {FIG_DIR / (args.out_stem + '.pdf')}")
    print(f"wrote {EVAL_DIR / (out_tag + '.json')}")


if __name__ == "__main__":
    main()
