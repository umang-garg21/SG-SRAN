#!/usr/bin/env python3
"""Render Fig. 8 with stagewise errors and a decoded OCRP sample walkthrough."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.metric_panel_hardened import (  # noqa: E402
    boundary_mask_fast,
    configure_symmetry,
    conjugated_ops,
)
from inference.infer_iso_embedding_sr_attn import (  # noqa: E402
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
    _to_hwc_quat_single,
    _unpack_batch,
)
from training.config_utils import load_and_prepare_config  # noqa: E402
from training.data_loading import build_dataloader  # noqa: E402
from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.ipf_render import render_ipf_rgb  # noqa: E402


PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
EVAL_DIR = PAPER_DIR / "evals"
OUT_PDF = FIG_DIR / "main_stagewise_progression.pdf"
OUT_PNG = FIG_DIR / "main_stagewise_progression.png"
PANEL_CACHE = EVAL_DIR / "fig8_walkthrough_panels_full_sample0.npz"
PANEL_B_EMBED_SCALE = 4
PNG_DPI = 1000
NMI_FONT = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
PANEL_A_Y_MAX_DEG = 6.0
PANEL_A_Y_TICK_DEG = 1.0

IN718_EXP = (
    ROOT
    / "experiments/IN718/direct_reynolds_isometric_seed_runs/"
    / "ocrp_direct_reynolds_isometric_l4_s42_fresh_allepochs_20260707_2205"
)

STAGEWISE_INPUTS = [
    {
        "key": "IN718",
        "title": "IN718 (FCC)",
        "path": EVAL_DIR / "current_direct_reynolds_stagewise_notebook_ckpt_20260708.json",
    },
    {
        "key": "Ti_Al_1pct",
        "title": "Ti-6Al-4V (HCP)",
        "path": EVAL_DIR / "current_direct_reynolds_stagewise_ti_20260707.json",
    },
]

LABELS_BY_STAGE = {
    "encode_lr": "Encoded LR\nfeatures",
    "context_refine": "Context\naggregation",
    "routed_patch": "Routed patch\nsynthesis",
    "hr_refine_1": "HR refine 1",
    "hr_refine_2": "HR refine 2",
}

GRID_ROWS = [
    ("lr_sparse", "LR samples"),
    ("encoded_lr", "Encoded LR"),
    ("context_refine", "Context"),
    ("routed_patch", "Routed patch"),
    ("final_ocrp", "Final OCRP"),
    ("hr_target", "HR target"),
]
GRID_COLS = ["IPF-X", "IPF-Y", "IPF-Z", "Boundary"]
DISPLAY_STRIDE = 1


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": NMI_FONT,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7dbe2",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.70,
            "axes.facecolor": "#fcfcfd",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_stagewise_payload(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["material_key"] = spec["key"]
    payload["material_title"] = spec["title"]
    payload["source_json"] = str(path.relative_to(ROOT))
    return payload


def plot_stagewise_panel(ax: plt.Axes, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = ["#16213e", "#b24725", "#24705a"]
    markers = ["o", "s", "^"]
    all_y: list[float] = []
    all_err: list[float] = []
    shares: list[dict[str, Any]] = []

    for idx, payload in enumerate(payloads):
        rows = payload["summary"]
        labels = [LABELS_BY_STAGE.get(str(row["stage"]), str(row["label"])) for row in rows]
        means = np.asarray([float(row["mean_deg"]) for row in rows], dtype=np.float64)
        stds = np.asarray([float(row["std_patch_mean_deg"]) for row in rows], dtype=np.float64)
        xs = np.arange(len(labels))
        stage_indices = {str(row["stage"]): i for i, row in enumerate(rows)}
        encode_idx = stage_indices.get("encode_lr", 0)
        context_idx = stage_indices.get("context_refine", 1)
        routed_idx = stage_indices.get("routed_patch", 2)
        final_idx = stage_indices.get("hr_refine_2", len(rows) - 1)
        offset = (idx - (len(payloads) - 1) / 2.0) * 0.07
        total_drop = means[encode_idx] - means[final_idx]
        upsampler_drop = means[context_idx] - means[routed_idx]
        share_pct = 100.0 * upsampler_drop / total_drop if total_drop > 0 else float("nan")
        label_fraction = 0.30 if payload["material_key"] == "IN718" else 0.55
        shares.append(
            {
                "text": f"{share_pct:.1f}%",
                "color": colors[idx % len(colors)],
                "y": float(means[routed_idx] + label_fraction * upsampler_drop),
                "material_key": payload["material_key"],
                "material_title": payload["material_title"],
                "upsampler_share_of_total_drop_percent": float(share_pct),
            }
        )
        all_y.extend(means.tolist())
        all_err.extend(stds.tolist())
        ax.errorbar(
            xs + offset,
            means,
            yerr=stds,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            lw=2.3,
            ms=5.4,
            capsize=3.2,
            zorder=5,
            label=str(payload["material_title"]),
        )

    labels = [
        LABELS_BY_STAGE.get(str(row["stage"]), str(row["label"]))
        for row in payloads[0]["summary"]
    ]
    xs = np.arange(len(labels))
    if len(labels) >= 3:
        ax.axvspan(1.5, 2.5, color="#f6d365", alpha=0.22, lw=0)
        for ann in shares:
            ax.text(
                2.03,
                float(ann["y"]),
                str(ann["text"]),
                color=str(ann["color"]),
                ha="center",
                va="center",
                fontsize=8.7,
                fontweight="bold",
            )

    ymin = 0.0
    ymax = PANEL_A_Y_MAX_DEG
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("OCRP pipeline stage")
    ax.set_ylabel(r"Mean decoded $d_{\mathrm{Stab}}$ ($^\circ$)")
    ax.yaxis.set_major_locator(MultipleLocator(PANEL_A_Y_TICK_DEG))
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, edgecolor="#ccd3df", fontsize=8.2)
    ax.text(-0.095, 1.05, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")
    return shares


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


def decode_feature_tokens(model: Any, feat: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    if feat.dim() == 3:
        if feat.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for stage decode, got {tuple(feat.shape)}")
        feat = feat[0]
    feat = torch.nan_to_num(feat.detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    decoded: list[torch.Tensor] = []
    for start in range(0, feat.shape[0], int(chunk)):
        part = feat[start : start + int(chunk)]
        with torch.enable_grad():
            decoded.append(model.decode(part).detach())
    return torch.cat(decoded, dim=0)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-8)
    q = np.where(q[..., :1] < 0.0, -q, q)
    return q.astype(np.float32, copy=False)


def sparse_project(arr: np.ndarray, lr_hw: tuple[int, int], hr_hw: tuple[int, int]) -> np.ndarray:
    h_lr, w_lr = int(lr_hw[0]), int(lr_hw[1])
    h_hr, w_hr = int(hr_hw[0]), int(hr_hw[1])
    out_shape = (h_hr, w_hr) + tuple(arr.shape[2:])
    out = np.zeros(out_shape, dtype=arr.dtype)
    sy = h_hr // h_lr
    sx = w_hr // w_lr
    y_idx = np.arange(h_lr, dtype=np.int64) * sy + sy // 2
    x_idx = np.arange(w_lr, dtype=np.int64) * sx + sx // 2
    out[np.ix_(y_idx, x_idx)] = arr
    return out


def decimate_hr_for_display(q: np.ndarray, stride: int) -> np.ndarray:
    if int(stride) <= 1:
        return q
    return q[:: int(stride), :: int(stride)]


def render_ipf_triplet(q: np.ndarray, sym: Any) -> list[np.ndarray]:
    q_fmt = format_quaternions(
        normalize_quat(q),
        normalize=True,
        hemisphere=True,
        reduce_fz=False,
        sym=sym,
        to_quat_first=False,
    )
    return [np.asarray(rgb, dtype=np.float32) for rgb in render_ipf_rgb(q_fmt, sym, ref_dir="ALL")]


def stage_to_panels(
    q: np.ndarray,
    *,
    sym: Any,
    ops: np.ndarray,
    sparse: bool,
    lr_hw: tuple[int, int],
    hr_hw: tuple[int, int],
    display_stride: int = 1,
) -> list[np.ndarray]:
    stride = max(1, int(display_stride))
    if sparse:
        ipfs_lr = render_ipf_triplet(q, sym)
        display_hr_hw = (int(hr_hw[0]) // stride, int(hr_hw[1]) // stride)
        ipfs = [sparse_project(rgb, lr_hw, display_hr_hw) for rgb in ipfs_lr]
        boundary = boundary_mask_fast(normalize_quat(q), ops, threshold_deg=5.0).astype(np.float32)
        boundary_hr = sparse_project(boundary[..., None], lr_hw, display_hr_hw)[..., 0]
    else:
        q_display = decimate_hr_for_display(q, stride)
        ipfs = render_ipf_triplet(q_display, sym)
        boundary_hr = boundary_mask_fast(normalize_quat(q_display), ops, threshold_deg=5.0).astype(np.float32)
    boundary_rgb = np.zeros(boundary_hr.shape + (3,), dtype=np.float32)
    boundary_rgb[boundary_hr > 0.5] = 1.0
    return [*ipfs, boundary_rgb]


def collect_walkthrough(sample_index: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exp_dir = IN718_EXP.resolve()
    print("  loading OCRP config/checkpoint", flush=True)
    cfg = load_and_prepare_config(exp_dir / "config_new.json", EVAL_DIR / "fig8_walkthrough_resolved_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, "epoch_0012.pt")
    checkpoint_mtime_ns = int(checkpoint_path.stat().st_mtime_ns)
    if sample_index == 0 and PANEL_CACHE.exists():
        cached = np.load(PANEL_CACHE, allow_pickle=False)
        cached_meta = json.loads(str(cached["meta"].item()))
        if (
            cached_meta.get("checkpoint") == str(checkpoint_path.relative_to(ROOT))
            and int(cached_meta.get("checkpoint_mtime_ns", -1)) == checkpoint_mtime_ns
        ):
            rows = []
            for key, label in GRID_ROWS:
                rows.append(
                    {
                        "key": key,
                        "label": label,
                        "panels": [np.asarray(cached[f"{key}__{idx}"], dtype=np.float32) for idx in range(len(GRID_COLS))],
                        "sparse": key in {"lr_sparse", "encoded_lr", "context_refine"},
                    }
                )
            cached_meta["loaded_from_panel_cache"] = str(PANEL_CACHE.relative_to(ROOT))
            print("  loaded cached full-sample walkthrough panels", flush=True)
            return rows, cached_meta

    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    model.eval()
    print("  loading test sample 0", flush=True)
    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split="Test",
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=sample_index + 1,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=False,
    )

    batch = None
    for batch_idx, item in enumerate(loader):
        if batch_idx == sample_index:
            batch = item
            break
    if batch is None:
        raise ValueError(f"Could not load Test sample {sample_index}")

    lr_batch, hr_batch, _ = _unpack_batch(batch)
    lr = _to_hwc_quat_single(lr_batch[0]).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    hr = _to_hwc_quat_single(hr_batch[0]).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    lr_flat, lr_shape = _flatten_quat_chw(lr)
    hr_hwc = hr.detach().cpu().numpy().astype(np.float32)

    print("  forwarding OCRP stages without attention dumps", flush=True)
    with torch.no_grad():
        feat_lr = model.encode(lr_flat)
        feat_pre = feat_lr
        if getattr(model, "use_lr_conv1", False):
            feat_pre = model.conv_lr1(feat_pre, lr_shape)
        feat_hr_raw_ocrp, hr_shape = model.ocrp(
            lr_quats=lr_flat,
            feat_lr=feat_pre,
            cluster_feat_lr=feat_lr,
            lr_shape=lr_shape,
            return_aux=False,
        )
        feat_hr_after_conv1 = feat_hr_raw_ocrp
        if getattr(model, "use_hr_conv1", False):
            feat_hr_after_conv1 = model.conv_hr1(feat_hr_after_conv1, hr_shape)
        feat_hr_after_conv2 = feat_hr_after_conv1
        if getattr(model, "use_hr_conv2", False):
            feat_hr_after_conv2 = model.conv_hr2(feat_hr_after_conv2, hr_shape)

    lr_hw = tuple(int(v) for v in lr_shape)
    hr_hw = tuple(int(v) for v in hr_shape)
    sym = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    ops = conjugated_ops(configure_symmetry(getattr(cfg, "symmetry_group", "Oh")))

    def decode_lr_stage(name: str, feat: torch.Tensor) -> np.ndarray:
        print(f"  decoding LR stage: {name}", flush=True)
        q = decode_feature_tokens(model, feat.unsqueeze(0) if feat.dim() == 2 else feat)
        return q.reshape(lr_hw[0], lr_hw[1], 4).detach().cpu().numpy().astype(np.float32)

    def decode_hr_stage(name: str, feat: torch.Tensor) -> np.ndarray:
        print(f"  decoding HR stage: {name}", flush=True)
        feat_b = feat.unsqueeze(0) if feat.dim() == 2 else feat
        q = decode_feature_tokens(model, feat_b)
        return q.reshape(hr_hw[0], hr_hw[1], 4).detach().cpu().numpy().astype(np.float32)

    lr_hwc = _to_hwc_quat_single(lr).detach().cpu().numpy().astype(np.float32)
    stages = [
        {"key": "lr_sparse", "label": "LR samples", "quat": lr_hwc, "sparse": True},
        {"key": "encoded_lr", "label": "Encoded LR", "quat": decode_lr_stage("Encoded LR", feat_lr), "sparse": True},
        {
            "key": "context_refine",
            "label": "Context",
            "quat": decode_lr_stage("Context", feat_pre),
            "sparse": True,
        },
        {
            "key": "routed_patch",
            "label": "Routed patch",
            "quat": decode_hr_stage("Routed patch", feat_hr_raw_ocrp),
            "sparse": False,
        },
        {
            "key": "final_ocrp",
            "label": "Final OCRP",
            "quat": decode_hr_stage("Final OCRP", feat_hr_after_conv2),
            "sparse": False,
        },
        {"key": "hr_target", "label": "HR target", "quat": hr_hwc, "sparse": False},
    ]

    panel_rows: list[dict[str, Any]] = []
    for stage in stages:
        print(f"  rendering walkthrough row: {stage['label']}", flush=True)
        panels = stage_to_panels(
            stage["quat"],
            sym=sym,
            ops=ops,
            sparse=bool(stage["sparse"]),
            lr_hw=lr_hw,
            hr_hw=hr_hw,
            display_stride=DISPLAY_STRIDE,
        )
        panel_rows.append({**stage, "panels": panels})
    print("  walkthrough rows complete", flush=True)

    provenance = {
        "sample_index": int(sample_index),
        "exp_dir": str(exp_dir.relative_to(ROOT)),
        "config": str((exp_dir / "config_new.json").relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_mtime_ns": checkpoint_mtime_ns,
        "split": "Test",
        "lr_shape": list(lr_hw),
        "hr_shape": list(hr_hw),
        "full_sample": True,
        "trace_method": "Manual full-field stage trace avoids materializing OCRP attention-alpha auxiliary tensors.",
        "device": str(device),
    }
    if sample_index == 0:
        cache_payload: dict[str, Any] = {"meta": np.array(json.dumps(provenance))}
        for row in panel_rows:
            for idx, panel in enumerate(row["panels"]):
                cache_payload[f"{row['key']}__{idx}"] = np.asarray(panel, dtype=np.float32)
        np.savez_compressed(PANEL_CACHE, **cache_payload)
        print(f"  cached full-sample walkthrough panels to {PANEL_CACHE}", flush=True)
    return panel_rows, provenance


def plot_walkthrough_panel(parent_spec: Any, fig: plt.Figure, rows: list[dict[str, Any]]) -> None:
    bbox = parent_spec.get_position(fig)
    nrows = len(rows)
    ncols = len(GRID_COLS)
    fig_w, fig_h = fig.get_size_inches()
    gap_in = 0.010
    gap_x = gap_in / fig_w
    gap_y = gap_in / fig_h
    title_band = 0.020
    usable_h = max(0.1, bbox.height - title_band)
    cell_side_in = min(
        (bbox.width * fig_w - (ncols - 1) * gap_in) / ncols,
        (usable_h * fig_h - (nrows - 1) * gap_in) / nrows,
    )
    cell_w = cell_side_in / fig_w
    cell_h = cell_side_in / fig_h
    grid_w = ncols * cell_w + (ncols - 1) * gap_x
    grid_h = nrows * cell_h + (nrows - 1) * gap_y
    grid_left = bbox.x0 + 0.5 * (bbox.width - grid_w)
    grid_bottom = bbox.y0 + 0.5 * (usable_h - grid_h)

    for r_idx, row in enumerate(rows):
        for c_idx, col in enumerate(GRID_COLS):
            left = grid_left + c_idx * (cell_w + gap_x)
            bottom = grid_bottom + (nrows - 1 - r_idx) * (cell_h + gap_y)
            ax = fig.add_axes([left, bottom, cell_w, cell_h])
            panel = np.asarray(row["panels"][c_idx], dtype=np.float32)
            if PANEL_B_EMBED_SCALE > 1:
                panel = np.repeat(
                    np.repeat(panel, PANEL_B_EMBED_SCALE, axis=0),
                    PANEL_B_EMBED_SCALE,
                    axis=1,
                )
            ax.imshow(panel, interpolation="nearest", resample=False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r_idx == 0:
                ax.set_title(col, fontsize=8.2, pad=1.3)
    fig.text(bbox.x0 - 0.012, bbox.y1 + 0.005, "b", fontsize=13, fontweight="bold")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("loading stagewise summaries", flush=True)
    payloads = [load_stagewise_payload(spec) for spec in STAGEWISE_INPUTS]
    print("collecting walkthrough sample", flush=True)
    walkthrough_rows, walkthrough_provenance = collect_walkthrough(sample_index=0)
    print("rendering figure", flush=True)

    fig = plt.figure(figsize=(9.2, 6.0))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.86, 1.14],
        wspace=0.050,
        left=0.075,
        right=0.992,
        top=0.940,
        bottom=0.075,
    )
    bbox_a = outer[0, 0].get_position(fig)
    ax_a_h = min(bbox_a.height, 0.58)
    ax_a = fig.add_axes(
        [
            bbox_a.x0,
            bbox_a.y0 + 0.5 * (bbox_a.height - ax_a_h),
            bbox_a.width,
            ax_a_h,
        ]
    )
    shares = plot_stagewise_panel(ax_a, payloads)
    plot_walkthrough_panel(outer[0, 1], fig, walkthrough_rows)

    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    out_json = EVAL_DIR / "fig8_stagewise_interpretability_provenance_notebook_ckpt_20260708.json"
    out_json.write_text(
        json.dumps(
            {
                "figure": str(OUT_PDF.relative_to(PAPER_DIR)),
                "panel_a_sources": [
                    {
                        "material_key": payload["material_key"],
                        "material_title": payload["material_title"],
                        "source_json": payload["source_json"],
                    }
                    for payload in payloads
                ],
                "panel_a_upsampler_shares": shares,
                "panel_b": walkthrough_provenance,
                "panel_b_rows": [row["key"] for row in walkthrough_rows],
                "panel_b_columns": GRID_COLS,
                "panel_b_display_stride": DISPLAY_STRIDE,
                "panel_b_embed_scale": PANEL_B_EMBED_SCALE,
                "png_dpi": PNG_DPI,
                "panel_b_ipf_rendering": "Quaternion maps are normalized and hemisphere-aligned before IPF coloring; the IPF color key handles symmetry coloring for the display panel.",
                "lr_display": "LR and LR-token decoded stages are rendered on the HR canvas only at sampled LR sites; unsampled HR pixels are black.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
