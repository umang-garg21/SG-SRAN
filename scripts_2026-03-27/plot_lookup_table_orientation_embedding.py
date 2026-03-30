#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Export decoder lookup-table quaternions and plot an MTEX-style orientation embedding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_REF_DIRS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export all decoder lookup-table quaternions and create an MTEX-style "
            "orientation-embedding figure (IPF-colored feature embedding)."
        )
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint filename in exp_dir/checkpoints, or absolute path. "
            "Not required for lookup-table extraction."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="table",
        choices=["table", "a1", "full"],
        help=(
            "Features used for embedding: 'table' uses decoder table features directly; "
            "'a1'/'full' recompute with encoder on lookup quaternions."
        ),
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="Z",
        choices=["X", "Y", "Z"],
        help="IPF reference direction.",
    )
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=30000,
        help="Maximum points to render in scatter panels (all points are still exported).",
    )
    parser.add_argument(
        "--table_encode_chunk_size",
        type=int,
        default=8192,
        help="Chunk size when recomputing features with encoder (a1/full modes).",
    )
    parser.add_argument(
        "--respect_max_table_rows",
        action="store_true",
        help=(
            "Respect cfg.decoder_max_table_rows. By default this script forces full "
            "table generation (decoder_max_table_rows=None)."
        ),
    )
    parser.add_argument(
        "--decoder_cubochoric_resolution",
        type=int,
        default=None,
        help="Optional override for decoder lookup table cubochoric resolution.",
    )
    parser.add_argument(
        "--decoder_table_cache_dir",
        type=str,
        default=None,
        help="Optional override for decoder table cache directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for plotting subsampling.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="lookup_table_orientation_embedding",
        help="Output file stem under <exp_dir>/diagnostics.",
    )
    return parser.parse_args()


def _resolve_checkpoint(exp_dir: Path, checkpoint: str | None) -> Path | None:
    if checkpoint is None:
        return None
    p = Path(checkpoint)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    cp = (exp_dir / "checkpoints" / checkpoint).resolve()
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    return cp


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based 2D PCA."""
    x_center = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_center, full_matrices=False)
    coords = u[:, :2] * s[:2]
    components = vt[:2]
    explained = (s[:2] ** 2) / np.maximum((s**2).sum(), 1e-12)
    return coords.astype(np.float32), components.astype(np.float32), explained.astype(np.float32)


@torch.no_grad()
def _encode_features(
    model,
    quats: torch.Tensor,
    mode: str,
    *,
    chunk_size: int,
) -> torch.Tensor:
    mode_l = str(mode).lower()
    if mode_l == "table":
        return model.decoder.table_feat

    out_chunks: list[torch.Tensor] = []
    n = int(quats.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        q_chunk = quats[start:end]
        if mode_l == "a1":
            out_chunks.append(model.encoder.forward_a1(q_chunk).to(torch.float32))
        elif mode_l == "full":
            out_chunks.append(model.encoder.forward_full(q_chunk).to(torch.float32))
        else:
            raise ValueError(f"Unsupported feature mode: {mode}")
    return torch.cat(out_chunks, dim=0)


def _build_figure(
    *,
    quats_np: np.ndarray,
    colors_np: np.ndarray,
    pca_xy: np.ndarray,
    feature_norm: np.ndarray,
    explained: np.ndarray,
    sym_obj,
    ref_dir: str,
    plot_max_points: int | None,
    seed: int,
    title: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    # Lazy imports for help robustness.
    from orix.quaternion import Orientation

    n_total = int(quats_np.shape[0])
    if plot_max_points is None or int(plot_max_points) <= 0 or n_total <= int(plot_max_points):
        idx = np.arange(n_total, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n_total, size=int(plot_max_points), replace=False))

    q_plot = quats_np[idx]
    c_plot = colors_np[idx]
    xy_plot = pca_xy[idx]
    fn_plot = feature_norm[idx]

    ori_plot = Orientation(q_plot)
    ori_plot.symmetry = sym_obj

    fig = plt.figure(figsize=(15.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # Panel 1: IPF key + lookup orientations.
    ax0 = fig.add_subplot(gs[0, 0], projection="ipf", symmetry=sym_obj.laue)
    ax0.plot_ipf_color_key()
    ax0.scatter(ori_plot, c=c_plot, s=8, alpha=0.80)
    ax0.set_title(f"Lookup orientations in FZ (IPF-{ref_dir})")

    # Panel 2: Feature embedding colored by IPF.
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(xy_plot[:, 0], xy_plot[:, 1], c=c_plot, s=7, alpha=0.75, linewidths=0)
    ax1.set_xlabel(f"PC1 ({100.0 * float(explained[0]):.1f}% var)")
    ax1.set_ylabel(f"PC2 ({100.0 * float(explained[1]):.1f}% var)")
    ax1.set_title("Lookup feature embedding colored by IPF")
    ax1.grid(True, alpha=0.25)

    # Panel 3: Same embedding colored by feature norm.
    ax2 = fig.add_subplot(gs[0, 2])
    sc = ax2.scatter(
        xy_plot[:, 0],
        xy_plot[:, 1],
        c=fn_plot,
        cmap="viridis",
        s=7,
        alpha=0.80,
        linewidths=0,
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Embedding colored by lookup feature norm")
    ax2.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("||feature||", rotation=90)

    fig.suptitle(title, fontsize=12)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Defer heavier project and optional plotting imports.
    from training.config_utils import load_and_prepare_config
    from utils.runtime_helpers import (
        assert_expected_model_import,
        build_iso_embedding_sr_attn_from_config,
        load_checkpoint_state_compat,
    )
    from orix import plot as orix_plot
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    exp_dir = Path(args.exp_dir).resolve()
    config_path = exp_dir / str(args.config)
    out_dir = exp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_and_prepare_config(config_path, save_path=None)

    # Ensure the script uses the full lookup table unless explicitly asked otherwise.
    if not bool(args.respect_max_table_rows):
        cfg.decoder_max_table_rows = None

    if args.decoder_cubochoric_resolution is not None:
        cfg.decoder_cubochoric_resolution = int(args.decoder_cubochoric_resolution)

    if args.decoder_table_cache_dir is not None:
        cfg.decoder_table_cache_dir = str(args.decoder_table_cache_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_iso_embedding_sr_attn_from_config(cfg, device=device).eval()
    assert_expected_model_import(type(model))

    ckpt_path = _resolve_checkpoint(exp_dir, args.checkpoint)
    if ckpt_path is not None:
        blob = torch.load(ckpt_path, map_location=device)
        state = blob.get("model_state_dict", blob)
        load_checkpoint_state_compat(model, state, context=f"checkpoint {ckpt_path}")

    dec = model.decoder
    if not hasattr(dec, "table_quats") or not hasattr(dec, "table_feat"):
        raise RuntimeError("Model decoder does not expose table_quats/table_feat buffers.")

    quats_t = dec.table_quats.detach().to(device=model.device, dtype=torch.float32)
    feat_t = _encode_features(
        model,
        quats_t,
        mode=str(args.feature_mode),
        chunk_size=int(args.table_encode_chunk_size),
    ).detach().to(torch.float32)

    quats_np = quats_t.detach().cpu().numpy().astype(np.float32, copy=False)
    feat_np = feat_t.detach().cpu().numpy().astype(np.float32, copy=False)

    # IPF colors.
    group_name = str(getattr(model.encoder, "group_name", "")).upper()
    if group_name == "O":
        sym_obj = symmetry.Oh
        crystal_name = "fcc"
    elif group_name == "D6":
        sym_obj = symmetry.D6h
        crystal_name = "hcp"
    else:
        raise ValueError(f"Unsupported encoder group_name: {group_name}")

    ori = Orientation(quats_np)
    ori.symmetry = sym_obj

    ckey = orix_plot.IPFColorKeyTSL(sym_obj.laue)
    ref_dir = str(args.reference_dir).upper()
    ckey.direction = Vector3d(_REF_DIRS[ref_dir])
    colors_np = np.asarray(ckey.orientation2color(ori), dtype=np.float32)
    colors_np = np.clip(colors_np, 0.0, 1.0)

    pca_xy, pca_components, pca_explained = _pca_2d(feat_np)
    feat_norm = np.linalg.norm(feat_np, axis=1).astype(np.float32, copy=False)

    stem = str(args.out_prefix)
    out_quats = out_dir / f"{stem}_quats.npy"
    out_feat = out_dir / f"{stem}_feat.npy"
    out_npz = out_dir / f"{stem}_data.npz"
    out_meta = out_dir / f"{stem}_meta.json"
    out_png = out_dir / f"{stem}.png"
    out_svg = out_dir / f"{stem}.svg"

    np.save(out_quats, quats_np)
    np.save(out_feat, feat_np)
    np.savez_compressed(
        out_npz,
        quats_passive_wxyz=quats_np,
        feature=feat_np,
        ipf_color_rgb=colors_np,
        pca_xy=pca_xy,
        pca_components=pca_components,
        pca_explained=pca_explained,
        feature_norm=feat_norm,
    )

    title = (
        f"Lookup orientation embedding | crystal={crystal_name} | group={group_name} | "
        f"feature_mode={str(args.feature_mode).lower()} | N={int(quats_np.shape[0])}"
    )
    _build_figure(
        quats_np=quats_np,
        colors_np=colors_np,
        pca_xy=pca_xy,
        feature_norm=feat_norm,
        explained=pca_explained,
        sym_obj=sym_obj,
        ref_dir=ref_dir,
        plot_max_points=args.plot_max_points,
        seed=int(args.seed),
        title=title,
        out_png=out_png,
        out_svg=out_svg,
    )

    meta: dict[str, Any] = {
        "exp_dir": str(exp_dir),
        "config": str(config_path),
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "device": str(device),
        "crystal": crystal_name,
        "group_name": group_name,
        "d6_convention": str(getattr(cfg, "d6_convention", "z_axis")),
        "feature_mode": str(args.feature_mode).lower(),
        "reference_dir": ref_dir,
        "num_lookup_quaternions": int(quats_np.shape[0]),
        "feature_dim": int(feat_np.shape[1]),
        "decoder_cubochoric_resolution": int(getattr(cfg, "decoder_cubochoric_resolution", -1)),
        "decoder_method": str(getattr(cfg, "decoder_method", "cubochoric")),
        "decoder_max_table_rows": getattr(cfg, "decoder_max_table_rows", None),
        "plot_max_points": int(args.plot_max_points),
        "pca_explained": [float(v) for v in pca_explained.tolist()],
        "outputs": {
            "quats_npy": str(out_quats),
            "feat_npy": str(out_feat),
            "data_npz": str(out_npz),
            "meta_json": str(out_meta),
            "figure_png": str(out_png),
            "figure_svg": str(out_svg),
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved lookup-table exports and figure:")
    print(f"  quats : {out_quats}")
    print(f"  feat  : {out_feat}")
    print(f"  npz   : {out_npz}")
    print(f"  meta  : {out_meta}")
    print(f"  png   : {out_png}")
    print(f"  svg   : {out_svg}")


if __name__ == "__main__":
    main()
