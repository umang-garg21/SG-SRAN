#!/usr/bin/env python3
"""Plot quaternion <-> irrep-feature mapping for cubic symmetry.

Modes
-----
1) Data mode (existing workflow):
   - provide --input and --features
2) Random SO(3) mode:
   - provide --random-so3 N
   - features are encoded on the fly with the e3nn invariant encoder

Outputs
-------
- overview mapping figure (stereo <-> feature space <-> key [+ spatial panels when HxW])
- 3D discretization surface in feature PCA space
- additional diagnostics: hexbin, stereo density, feature marginals,
  bin-centroid correspondence, optional spatial bin-id map
- optional symmetry-invariance diagnostics for random mode
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
import numpy as np
import orix.plot  # noqa: F401 - registers matplotlib projections
from orix.projections.stereographic import _vector2xy
from orix.quaternion import Orientation
from orix.vector import Vector3d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.symmetry_utils import resolve_symmetry


DIR_VECTORS = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}


def _quat_axis_to_last(arr: np.ndarray) -> np.ndarray:
    axes = [i for i, s in enumerate(arr.shape) if s == 4]
    if len(axes) != 1:
        raise ValueError(
            f"Expected exactly one quaternion axis of size 4, got shape {arr.shape}"
        )
    q_axis = axes[0]
    return arr if q_axis == arr.ndim - 1 else np.moveaxis(arr, q_axis, -1)


def _load_array(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), "npy:data"
    if suffix == ".npz":
        z = np.load(path)
        if key is not None:
            if key not in z:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(z.keys())}")
            return z[key], f"npz:{key}"
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"No arrays in {path}")
        return z[keys[0]], f"npz:{keys[0]}"
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _normalize_quats(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _build_fcc_syms_inv_wxyz_np() -> np.ndarray:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [0, -inv_sqrt_2, -inv_sqrt_2, 0],
            [0, -inv_sqrt_2, 0, -inv_sqrt_2],
            [0, 0, -inv_sqrt_2, -inv_sqrt_2],
            [0, -inv_sqrt_2, inv_sqrt_2, 0],
            [0, 0, -inv_sqrt_2, inv_sqrt_2],
            [0, -inv_sqrt_2, 0, inv_sqrt_2],
            [half, -half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, half],
        ],
        dtype=np.float32,
    )


def _cubic_misorientation_deg_pairs(
    q1: np.ndarray,
    q2: np.ndarray,
    sym_inv: np.ndarray,
) -> np.ndarray:
    q1n = _normalize_quats(q1)
    q2n = _normalize_quats(q2)

    best = np.full((q1n.shape[0],), np.inf, dtype=np.float32)
    for s in sym_inv:
        s_batch = np.broadcast_to(s[None, :], q2n.shape)
        q2s = _quat_mul_np(s_batch, q2n)
        dot = np.abs(np.sum(q1n * q2s, axis=-1))
        dot = np.clip(dot, -1.0, 1.0)
        ang = 2.0 * np.arccos(dot) * (180.0 / np.pi)
        best = np.minimum(best, ang.astype(np.float32, copy=False))
    return best


def _build_feature_layout_from_meta(
    feat_meta: dict[str, Any] | None,
    feature_dim: int,
) -> list[tuple[int, int, int, int, bool]]:
    if not feat_meta:
        return []
    if "Ls" not in feat_meta or "basis_ranks" not in feat_meta:
        return []

    ls = [int(v) for v in feat_meta["Ls"]]
    basis_ranks = {str(k): int(v) for k, v in dict(feat_meta["basis_ranks"]).items()}
    stack_re_im = bool(feat_meta.get("stack_re_im", True))

    layout: list[tuple[int, int, int, int, bool]] = []
    off = 0
    for l in ls:
        rank = int(basis_ranks.get(str(l), 0))
        if rank <= 0:
            return []
        block = (2 * l + 1) * rank
        width = 2 * block if stack_re_im else block
        layout.append((l, off, off + width, rank, stack_re_im))
        off += width

    if off != int(feature_dim):
        return []
    return layout


def _parse_angle_list(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("jacobian-angles-deg cannot be empty")
    arr = np.asarray(vals, dtype=np.float32)
    if np.any(arr <= 0):
        raise ValueError("jacobian-angles-deg must be positive")
    return np.sort(arr)


def _sample_random_so3_quats(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q = _normalize_quats(q)
    return q


def _fit_pca2(x: np.ndarray, fit_n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    if n > fit_n:
        idx = rng.choice(n, size=fit_n, replace=False)
        xf = x[idx]
    else:
        xf = x

    xf64 = xf.astype(np.float64, copy=False)
    mean = xf64.mean(axis=0)
    xc = xf64 - mean
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    w = evecs[:, :2]
    denom = max(float(np.sum(evals)), 1e-12)
    var_ratio = (evals[:2] / denom).astype(np.float64, copy=False)
    return mean.astype(np.float32), w.astype(np.float32), var_ratio


def _project_pca2(x: np.ndarray, mean: np.ndarray, w: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = np.empty((x.shape[0], 2), dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        out[s:e] = (x[s:e] - mean) @ w
    return out


def _digitize_2d(
    xy: np.ndarray,
    bins: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(x_min, x_max, bins + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, bins + 1, dtype=np.float32)
    ix = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, bins - 1)

    counts = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(counts, (iy, ix), 1)
    return ix.astype(np.int32), iy.astype(np.int32), counts, x_edges, y_edges


def _bin_color_lut(bins: int) -> np.ndarray:
    lut = np.zeros((bins, bins, 3), dtype=np.float32)
    for iy in range(bins):
        for ix in range(bins):
            hue = float(ix + 0.5) / float(bins)
            sat = 0.35 + 0.65 * (float(iy + 0.5) / float(bins))
            val = 0.96
            lut[iy, ix] = hsv_to_rgb([hue, sat, val])
    return lut


def _parse_ls(ls_text: str) -> tuple[int, ...]:
    ls = tuple(int(x.strip()) for x in ls_text.split(",") if x.strip())
    if not ls:
        raise ValueError("encode-ls cannot be empty")
    if any(l < 0 for l in ls):
        raise ValueError(f"Invalid ls: {ls}")
    return ls


def _load_encoder_module():
    mod_path = Path(__file__).resolve().parent / "encode_bunge_irreps_invariants.py"
    mod_name = "encode_bunge_irreps_invariants"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load encoder module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    # Needed for Python 3.12 dataclass internals during dynamic import.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_encoder_and_syms(args: argparse.Namespace):
    import torch

    mod = _load_encoder_module()
    ls = _parse_ls(args.encode_ls)
    cfg = mod.InvariantEncoderConfig(
        Ls=ls,
        stack_re_im=not bool(args.encode_no_stack_re_im),
        rel_tol=float(args.encode_rel_tol),
        abs_tol=float(args.encode_abs_tol),
        eig_tol=float(args.encode_eig_tol),
        canonicalize_basis=True,
        normalize_output=not bool(args.encode_no_normalize_output),
    )

    dev = str(args.encode_device).strip().lower()
    if dev in {"", "auto"}:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    sym_np = mod.build_fcc_syms_inv_wxyz_np()
    sym_q = torch.as_tensor(sym_np, dtype=torch.float32, device=device)

    model = mod.BestInvariantFeatureModel(
        sym_quats_wxyz_bunge=sym_q,
        encoder_cfg=cfg,
        use_head=False,
    ).to(device)
    model.eval()

    basis_ranks = {
        str(l): int(getattr(model.encoder, f"U_{l}").shape[-1])
        for l in cfg.Ls
    }

    return mod, model, sym_q, device, cfg, basis_ranks


def _encode_quats_in_batches(
    q_flat: np.ndarray,
    model,
    device,
    batch_size: int,
) -> np.ndarray:
    import torch

    q_t = torch.as_tensor(q_flat, dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for s in range(0, int(q_t.shape[0]), int(batch_size)):
            e = min(s + int(batch_size), int(q_t.shape[0]))
            z = model(q_t[s:e]).detach().cpu().numpy()
            out.append(z)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _run_symmetry_invariance_check(
    q_flat: np.ndarray,
    mod,
    model,
    sym_q,
    device,
    verify_samples: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    n = q_flat.shape[0]
    m = min(int(verify_samples), n)
    rng = np.random.default_rng(seed)
    if n > m:
        idx = rng.choice(n, size=m, replace=False)
    else:
        idx = np.arange(n)

    q_sub_np = q_flat[idx]
    q_sub = torch.as_tensor(q_sub_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        base = model(q_sub)
        l2_all = []
        linf_all = []
        per_op_l2_mean = []
        for j in range(int(sym_q.shape[0])):
            s = sym_q[j].unsqueeze(0).expand_as(q_sub)
            qg = mod.quat_mul_torch(s, q_sub)
            zg = model(qg)
            d = zg - base
            l2 = torch.linalg.norm(d, dim=-1)
            linf = d.abs().max(dim=-1).values
            l2_all.append(l2.detach().cpu().numpy())
            linf_all.append(linf.detach().cpu().numpy())
            per_op_l2_mean.append(float(l2.mean().item()))

        q_neg = -q_sub
        d_sign = model(q_neg) - base
        sign_l2 = torch.linalg.norm(d_sign, dim=-1).detach().cpu().numpy()
        sign_linf = d_sign.abs().max(dim=-1).values.detach().cpu().numpy()

    l2_all_np = np.concatenate(l2_all, axis=0)
    linf_all_np = np.concatenate(linf_all, axis=0)
    return {
        "sample_count": int(m),
        "sym_l2_all": l2_all_np,
        "sym_linf_all": linf_all_np,
        "per_op_l2_mean": np.asarray(per_op_l2_mean, dtype=np.float64),
        "sign_l2": sign_l2,
        "sign_linf": sign_linf,
        "sym_l2_max": float(np.max(l2_all_np)),
        "sym_l2_mean": float(np.mean(l2_all_np)),
        "sym_linf_max": float(np.max(linf_all_np)),
        "sym_linf_mean": float(np.mean(linf_all_np)),
        "sign_l2_max": float(np.max(sign_l2)),
        "sign_l2_mean": float(np.mean(sign_l2)),
        "sign_linf_max": float(np.max(sign_linf)),
        "sign_linf_mean": float(np.mean(sign_linf)),
        "subset_quats": q_sub_np,
    }


def _plot_invariance_diagnostics(inv: dict[str, Any], out_dir: Path) -> list[str]:
    saved: list[str] = []

    eps = 1e-20
    fig1, ax1 = plt.subplots(figsize=(8.2, 4.4), dpi=170)
    ax1.hist(np.log10(inv["sym_l2_all"] + eps), bins=100, alpha=0.85, label="symmetry orbit")
    ax1.hist(np.log10(inv["sign_l2"] + eps), bins=100, alpha=0.6, label="q vs -q")
    ax1.set_xlabel("log10(L2 feature error + 1e-20)")
    ax1.set_ylabel("count")
    ax1.set_title("Invariance error distribution")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    p1 = out_dir / "symmetry_invariance_hist.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)
    saved.append(str(p1))

    fig2, ax2 = plt.subplots(figsize=(9.2, 4.0), dpi=170)
    op = np.arange(inv["per_op_l2_mean"].shape[0])
    ax2.bar(op, inv["per_op_l2_mean"], width=0.85)
    ax2.set_xlabel("cubic symmetry op index")
    ax2.set_ylabel("mean L2 error")
    ax2.set_title("Per-operation invariance error")
    ax2.grid(alpha=0.25, axis="y")
    p2 = out_dir / "symmetry_invariance_per_op.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(str(p2))

    return saved


def _plot_orbit_examples(
    inv: dict[str, Any],
    mod,
    sym_q,
    sym,
    v_ref,
    out_dir: Path,
    n_orbits: int,
) -> str:
    import torch

    q_seed = inv["subset_quats"]
    n = min(int(n_orbits), int(q_seed.shape[0]))
    if n <= 0:
        return ""

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n, endpoint=False))
    fig = plt.figure(figsize=(7.0, 6.0), dpi=170)
    ax = fig.add_subplot(111, projection="stereographic")

    sym_cpu = sym_q.detach().cpu()
    for i in range(n):
        q0 = torch.as_tensor(q_seed[i : i + 1], dtype=torch.float32)
        q_rep = q0.repeat(int(sym_cpu.shape[0]), 1)
        q_orbit = mod.quat_mul_torch(sym_cpu, q_rep).numpy()
        ori = Orientation(q_orbit)
        v = ori * v_ref
        x, y = _vector2xy(v, pole=-1)
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        Axes.scatter(ax, x, y, s=12, color=colors[i], alpha=0.75, edgecolors="none")

        ori0 = Orientation(q0.numpy())
        v0 = ori0 * v_ref
        x0, y0 = _vector2xy(v0, pole=-1)
        x0 = float(np.asarray(x0).reshape(-1)[0])
        y0 = float(np.asarray(y0).reshape(-1)[0])
        Axes.scatter(
            ax,
            [x0],
            [y0],
            s=75,
            color=colors[i],
            marker="*",
            edgecolors="black",
            linewidths=0.4,
        )

    ax.set_title(f"Random quaternion orbits under cubic symmetry ({n} examples)")
    ax.set_labels("RD", "TD", None)
    ax.show_hemisphere_label()
    p = out_dir / "symmetry_orbit_examples.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _plot_1_misorientation_vs_feature_distance(
    q_flat: np.ndarray,
    x_flat: np.ndarray,
    sym_inv_np: np.ndarray,
    out_dir: Path,
    pair_count: int,
    seed: int,
) -> tuple[str, dict[str, float]]:
    n = int(q_flat.shape[0])
    m = min(int(pair_count), max(n - 1, 1))
    rng = np.random.default_rng(seed)

    i = rng.integers(0, n, size=m, endpoint=False)
    j = rng.integers(0, n, size=m, endpoint=False)
    same = i == j
    j[same] = (j[same] + 1) % n

    q1 = q_flat[i]
    q2 = q_flat[j]
    mis_deg = _cubic_misorientation_deg_pairs(q1, q2, sym_inv_np)
    fd = np.linalg.norm(x_flat[i] - x_flat[j], axis=-1).astype(np.float32, copy=False)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=170)
    hb = ax.hexbin(
        mis_deg,
        fd,
        gridsize=85,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    ax.set_xlabel("Cubic misorientation (deg)")
    ax.set_ylabel("Feature distance ||f(q1)-f(q2)||")
    ax.set_title("1) Misorientation vs feature distance")
    ax.grid(alpha=0.2)
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10(count)")

    # Median trend.
    bins = np.linspace(0.0, max(1.0, float(np.max(mis_deg))), 32)
    mids = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(mids, np.nan, dtype=np.float32)
    for k in range(len(mids)):
        mask = (mis_deg >= bins[k]) & (mis_deg < bins[k + 1])
        if np.any(mask):
            med[k] = float(np.median(fd[mask]))
    valid = np.isfinite(med)
    if np.any(valid):
        ax.plot(mids[valid], med[valid], color="crimson", linewidth=2.0, label="median")
        ax.legend(loc="upper left")

    p = out_dir / "1_misorientation_vs_feature_distance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(mis_deg, fd)[0, 1] if m > 3 else np.nan
    stats = {
        "pair_count": float(m),
        "mis_deg_mean": float(np.mean(mis_deg)),
        "mis_deg_max": float(np.max(mis_deg)),
        "feat_dist_mean": float(np.mean(fd)),
        "feat_dist_max": float(np.max(fd)),
        "pearson_corr": float(corr) if np.isfinite(corr) else float("nan"),
    }
    return str(p), stats


def _plot_2_orbit_collapse_residual(
    q_flat: np.ndarray,
    mod,
    model,
    sym_q,
    device,
    out_dir: Path,
    sample_count: int,
    seed: int,
) -> tuple[str, dict[str, float]]:
    import torch

    n = int(q_flat.shape[0])
    m = min(int(sample_count), n)
    rng = np.random.default_rng(seed)
    if n > m:
        idx = rng.choice(n, size=m, replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)

    q_np = q_flat[idx]
    q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        base = model(q)
        op_err = []
        for j in range(int(sym_q.shape[0])):
            s = sym_q[j].unsqueeze(0).expand_as(q)
            qg = mod.quat_mul_torch(s, q)
            zg = model(qg)
            d = torch.linalg.norm(zg - base, dim=-1)
            op_err.append(d.detach().cpu().numpy())

    E = np.stack(op_err, axis=1)  # (m, G)
    mean_err = np.mean(E, axis=1)
    max_err = np.max(E, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=170)
    eps = 1e-20
    axes[0].hist(np.log10(mean_err + eps), bins=100, alpha=0.85, color="#1f77b4")
    axes[0].set_title("Per-orbit mean error")
    axes[0].set_xlabel("log10(mean ||Δf|| + 1e-20)")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2)

    axes[1].hist(np.log10(max_err + eps), bins=100, alpha=0.85, color="#d62728")
    axes[1].set_title("Per-orbit max error")
    axes[1].set_xlabel("log10(max ||Δf|| + 1e-20)")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)

    fig.suptitle("2) Orbit-collapse residual")
    p = out_dir / "2_orbit_collapse_residual.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "sample_count": float(m),
        "mean_err_mean": float(np.mean(mean_err)),
        "mean_err_max": float(np.max(mean_err)),
        "max_err_mean": float(np.mean(max_err)),
        "max_err_max": float(np.max(max_err)),
    }
    return str(p), stats


def _plot_3_knn_retrieval_map(
    q_flat: np.ndarray,
    x_flat: np.ndarray,
    sym: Any,
    v_ref: Any,
    sym_inv_np: np.ndarray,
    out_dir: Path,
    pool_size: int,
    query_count: int,
    k: int,
    seed: int,
) -> tuple[str, dict[str, float]]:
    n = int(q_flat.shape[0])
    rng = np.random.default_rng(seed)
    p = min(int(pool_size), n)
    if n > p:
        pool_idx = rng.choice(n, size=p, replace=False)
    else:
        pool_idx = np.arange(n, dtype=np.int64)

    xp = x_flat[pool_idx]
    qp = q_flat[pool_idx]
    qn = min(int(query_count), p)
    q_idx_local = rng.choice(p, size=qn, replace=False)

    ori_pool = Orientation(qp, symmetry=sym)
    v_pool = ori_pool * v_ref
    v_key_pool = v_pool.in_fundamental_sector(sym)
    sx, sy = _vector2xy(v_key_pool, pole=-1)
    sx = np.asarray(sx).reshape(-1)
    sy = np.asarray(sy).reshape(-1)
    valid = np.isfinite(sx) & np.isfinite(sy)

    fig, axes = plt.subplots(qn, 2, figsize=(12.0, 4.2 * qn), dpi=170)
    if qn == 1:
        axes = np.asarray([axes])

    mean_mis = []
    for r, qi in enumerate(q_idx_local):
        xq = xp[qi]
        dq = np.linalg.norm(xp - xq[None, :], axis=-1)
        order = np.argsort(dq)
        order = order[order != qi][: int(k)]
        local_all = np.arange(p, dtype=np.int64)

        mis = _cubic_misorientation_deg_pairs(
            np.broadcast_to(qp[qi][None, :], (order.shape[0], 4)),
            qp[order],
            sym_inv_np,
        )
        mean_mis.append(float(np.mean(mis)))

        axf = axes[r, 0]
        axf.scatter(xp[:, 0], xp[:, 1], s=2, color="#cccccc", alpha=0.35, edgecolors="none")
        axf.scatter(xp[order, 0], xp[order, 1], s=20, color="#1f77b4", alpha=0.9, edgecolors="none")
        axf.scatter([xq[0]], [xq[1]], s=90, color="crimson", marker="*", edgecolors="black", linewidths=0.4)
        axf.set_title(f"Query {r+1}: feature kNN (k={k})")
        axf.set_xlabel("feature dim 0")
        axf.set_ylabel("feature dim 1")
        axf.grid(alpha=0.2)

        axs = axes[r, 1]
        axs.scatter(sx[valid], sy[valid], s=2, color="#cccccc", alpha=0.35, edgecolors="none")
        vord = valid[order]
        if np.any(vord):
            good = order[vord]
            axs.scatter(sx[good], sy[good], s=20, color="#1f77b4", alpha=0.9, edgecolors="none")
        if valid[qi]:
            axs.scatter([sx[qi]], [sy[qi]], s=90, color="crimson", marker="*", edgecolors="black", linewidths=0.4)
        axs.set_title(f"Query {r+1}: stereo (mean mis={np.mean(mis):.2f} deg)")
        axs.set_xlabel("stereo x")
        axs.set_ylabel("stereo y")
        axs.set_aspect("equal", adjustable="box")
        axs.grid(alpha=0.2)

    fig.suptitle("3) kNN retrieval map (feature-space neighbors in stereo space)")
    p_out = out_dir / "3_knn_retrieval_map.png"
    fig.savefig(p_out, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "pool_size": float(p),
        "query_count": float(qn),
        "k": float(k),
        "mean_misorientation_deg": float(np.mean(mean_mis)) if mean_mis else float("nan"),
    }
    return str(p_out), stats


def _plot_4_per_l_energy_decomposition(
    feats: np.ndarray,
    layout: list[tuple[int, int, int, int, bool]],
    out_dir: Path,
) -> tuple[list[str], dict[str, float]]:
    if not layout:
        return [], {}

    saved: list[str] = []
    stats: dict[str, float] = {}

    f2 = feats.reshape(-1, feats.shape[-1]) if feats.ndim > 2 else feats
    e_blocks = []
    ls = []
    for (l, s, e, rank, _stack) in layout:
        eb = np.linalg.norm(f2[:, s:e], axis=-1).astype(np.float32, copy=False)
        e_blocks.append(eb)
        ls.append(l)
        stats[f"l{l}_energy_mean"] = float(np.mean(eb))
    E = np.stack(e_blocks, axis=1)  # (N, L)
    Er = E / np.maximum(np.sum(E, axis=1, keepdims=True), 1e-12)

    if feats.ndim == 3:
        h, w = feats.shape[:2]
        E_img = E.reshape(h, w, -1)
        Er_img = Er.reshape(h, w, -1)

        n_l = len(ls)
        fig1, ax1 = plt.subplots(1, n_l, figsize=(4.2 * n_l, 4.2), dpi=170, constrained_layout=True)
        if n_l == 1:
            ax1 = [ax1]
        for i, l in enumerate(ls):
            im = ax1[i].imshow(E_img[..., i], cmap="viridis")
            ax1[i].set_title(f"l={l} ||block||")
            ax1[i].axis("off")
            fig1.colorbar(im, ax=ax1[i], fraction=0.046, pad=0.04)
        p1 = out_dir / "4_per_l_energy_absolute.png"
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        saved.append(str(p1))

        fig2, ax2 = plt.subplots(1, n_l, figsize=(4.2 * n_l, 4.2), dpi=170, constrained_layout=True)
        if n_l == 1:
            ax2 = [ax2]
        for i, l in enumerate(ls):
            im = ax2[i].imshow(Er_img[..., i], cmap="magma", vmin=0.0, vmax=1.0)
            ax2[i].set_title(f"l={l} ratio")
            ax2[i].axis("off")
            fig2.colorbar(im, ax=ax2[i], fraction=0.046, pad=0.04)
        p2 = out_dir / "4_per_l_energy_ratio.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        saved.append(str(p2))
    else:
        fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=170)
        x = np.arange(len(ls))
        means = np.mean(Er, axis=0)
        ax.bar(x, means, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"l={l}" for l in ls])
        ax.set_ylabel("mean ratio")
        ax.set_title("4) Per-l energy decomposition (mean ratio)")
        ax.grid(alpha=0.2, axis="y")
        p = out_dir / "4_per_l_energy_ratio_bar.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved, stats


def _plot_5_jacobian_sensitivity(
    q_flat: np.ndarray,
    mod,
    model,
    device,
    out_dir: Path,
    sample_count: int,
    angles_deg: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[str, dict[str, float]]:
    import torch

    n = int(q_flat.shape[0])
    m = min(int(sample_count), n)
    rng = np.random.default_rng(seed)
    if n > m:
        idx = rng.choice(n, size=m, replace=False)
    else:
        idx = np.arange(n, dtype=np.int64)

    q0_np = q_flat[idx]
    axes = rng.normal(size=(m, 3)).astype(np.float32)
    axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), 1e-12)

    q0 = torch.as_tensor(q0_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        f0 = model(q0).detach()

    med = []
    p10 = []
    p90 = []
    mean = []

    for a_deg in angles_deg:
        a = float(a_deg) * (np.pi / 180.0)
        half = 0.5 * a
        w = np.cos(half).astype(np.float32)
        s = np.sin(half).astype(np.float32)
        dq_np = np.empty((m, 4), dtype=np.float32)
        dq_np[:, 0] = w
        dq_np[:, 1:] = axes * s
        dq = torch.as_tensor(dq_np, dtype=torch.float32, device=device)

        # Left perturbation in Bunge/passive convention.
        qp = mod.quat_mul_torch(dq, q0)
        qp = qp / qp.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        vals = []
        with torch.no_grad():
            for s0 in range(0, m, int(batch_size)):
                e0 = min(s0 + int(batch_size), m)
                fp = model(qp[s0:e0])
                d = torch.linalg.norm(fp - f0[s0:e0], dim=-1)
                vals.append(d.detach().cpu().numpy())
        v = np.concatenate(vals, axis=0)
        mean.append(float(np.mean(v)))
        med.append(float(np.median(v)))
        p10.append(float(np.percentile(v, 10)))
        p90.append(float(np.percentile(v, 90)))

    mean = np.asarray(mean, dtype=np.float64)
    med = np.asarray(med, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    p90 = np.asarray(p90, dtype=np.float64)
    slope = mean / np.maximum(angles_deg.astype(np.float64), 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=170)
    ax1.plot(angles_deg, mean, "-o", label="mean")
    ax1.plot(angles_deg, med, "-s", label="median")
    ax1.fill_between(angles_deg, p10, p90, alpha=0.2, label="p10-p90")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("perturbation angle (deg)")
    ax1.set_ylabel("||Δfeature||")
    ax1.set_title("5) Jacobian sensitivity vs angle")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(loc="best")

    ax2.plot(angles_deg, slope, "-o", color="#7f3c8d")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("perturbation angle (deg)")
    ax2.set_ylabel("mean ||Δf|| / Δθ(deg)")
    ax2.set_title("Approximate local sensitivity")
    ax2.grid(alpha=0.25, which="both")

    p = out_dir / "5_jacobian_sensitivity_vs_angle.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "sample_count": float(m),
        "angle_min_deg": float(np.min(angles_deg)),
        "angle_max_deg": float(np.max(angles_deg)),
        "mean_slope": float(np.mean(slope)),
        "max_slope": float(np.max(slope)),
    }
    return str(p), stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Data mode
    p.add_argument("--input", type=Path, default=None, help="Quaternion input .npy/.npz")
    p.add_argument("--input-key", type=str, default=None, help="Array key if --input is .npz")
    p.add_argument("--features", type=Path, default=None, help="Feature .npz with key `features`")
    p.add_argument("--features-key", type=str, default="features", help="Feature key in --features")

    # Random mode
    p.add_argument(
        "--random-so3",
        type=int,
        default=0,
        help="If >0, sample this many random unit quaternions and encode on-the-fly.",
    )

    # Quaternion conventions
    p.add_argument(
        "--scalar-last",
        action="store_true",
        help="Interpret input as [x,y,z,w] and convert to [w,x,y,z]",
    )
    p.add_argument(
        "--quaternion-convention",
        type=str,
        default="bunge",
        choices=["bunge", "passive", "active"],
        help="If active, conjugate to Bunge/passive before plotting.",
    )

    # Geometry + mapping
    p.add_argument("--symmetry", type=str, default="Oh", help="Symmetry name (default: Oh)")
    p.add_argument("--ref-dir", type=str, default="Z", choices=["X", "Y", "Z"])
    p.add_argument("--bins", type=int, default=28, help="2D bin count per PCA axis")
    p.add_argument("--fit-points", type=int, default=50000, help="Max points used to fit PCA")
    p.add_argument("--max-points", type=int, default=35000, help="Max points shown in scatters")
    p.add_argument("--seed", type=int, default=0)

    # On-the-fly encoding options
    p.add_argument("--encode-ls", type=str, default="4,6,8,10,12")
    p.add_argument("--encode-device", type=str, default="auto")
    p.add_argument("--encode-batch-size", type=int, default=8192)
    p.add_argument("--encode-rel-tol", type=float, default=1e-8)
    p.add_argument("--encode-abs-tol", type=float, default=1e-6)
    p.add_argument("--encode-eig-tol", type=float, default=1e-5)
    p.add_argument("--encode-no-stack-re-im", action="store_true")
    p.add_argument("--encode-no-normalize-output", action="store_true")

    # Extra diagnostics
    p.add_argument(
        "--verify-symmetry",
        action="store_true",
        help="Numerically verify cubic invariance (requires on-the-fly encoder).",
    )
    p.add_argument("--verify-samples", type=int, default=4096)
    p.add_argument("--orbit-examples", type=int, default=6)
    p.add_argument(
        "--run-top5",
        action="store_true",
        help="Generate requested plots #1-#5 (misorientation, orbit collapse, kNN, per-l energy, Jacobian).",
    )
    p.add_argument("--pair-count", type=int, default=50000)
    p.add_argument("--orbit-samples", type=int, default=2048)
    p.add_argument("--retrieval-pool", type=int, default=12000)
    p.add_argument("--retrieval-queries", type=int, default=6)
    p.add_argument("--retrieval-k", type=int, default=8)
    p.add_argument("--jacobian-samples", type=int, default=1024)
    p.add_argument(
        "--jacobian-angles-deg",
        type=str,
        default="0.1,0.2,0.5,1,2,5,10",
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/quat_irrep_mapping"),
        help="Output directory for figures and metadata",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    random_mode = int(args.random_so3) > 0
    if not random_mode and args.input is None:
        raise ValueError("Provide --input/--features or use --random-so3 N")

    # Load/build quaternions.
    if random_mode:
        q_last = _sample_random_so3_quats(int(args.random_so3), seed=int(args.seed))
        q_src = f"random_so3:{int(args.random_so3)}"
    else:
        q_raw, q_src = _load_array(args.input, args.input_key)
        q_last = _quat_axis_to_last(np.asarray(q_raw, dtype=np.float32))
        if bool(args.scalar_last):
            q_last = q_last[..., [3, 0, 1, 2]]
        q_last = _normalize_quats(q_last)

    conv = str(args.quaternion_convention).lower()
    if conv == "active":
        q_last = _quat_conjugate(q_last)

    spatial_shape = q_last.shape[:-1]
    n_total = int(np.prod(spatial_shape))
    q_flat = q_last.reshape(n_total, 4)

    # Load or encode features.
    encoder_bundle = None
    basis_ranks: dict[str, int] | None = None
    feat_meta: dict[str, Any] | None = None

    if args.features is not None:
        fz = np.load(args.features)
        if args.features_key not in fz:
            raise KeyError(f"Feature key '{args.features_key}' not found. Keys: {list(fz.keys())}")
        feats = np.asarray(fz[args.features_key], dtype=np.float32)
        if feats.shape[:-1] != q_last.shape[:-1]:
            raise ValueError(
                f"Quaternion shape {q_last.shape[:-1]} != feature shape {feats.shape[:-1]}"
            )
        x_flat = feats.reshape(n_total, feats.shape[-1])
        feat_src = f"file:{args.features}"
        if "meta_json" in fz:
            try:
                feat_meta = json.loads(str(fz["meta_json"]))
            except Exception:
                feat_meta = None
        if feat_meta is not None and "basis_ranks" in feat_meta:
            try:
                basis_ranks = {str(k): int(v) for k, v in dict(feat_meta["basis_ranks"]).items()}
            except Exception:
                basis_ranks = basis_ranks
    else:
        mod, model, sym_q, dev, cfg, basis_ranks = _build_encoder_and_syms(args)
        x_flat = _encode_quats_in_batches(
            q_flat=q_flat,
            model=model,
            device=dev,
            batch_size=int(args.encode_batch_size),
        )
        feats = x_flat.reshape(*spatial_shape, x_flat.shape[-1])
        feat_src = f"encoded_on_the_fly:Ls={list(cfg.Ls)}"
        encoder_bundle = (mod, model, sym_q, dev)
        feat_meta = {
            "Ls": list(cfg.Ls),
            "basis_ranks": basis_ranks,
            "stack_re_im": (not bool(args.encode_no_stack_re_im)),
        }

    # If invariance requested and encoder not built yet, build it.
    if bool(args.verify_symmetry) and encoder_bundle is None:
        mod, model, sym_q, dev, cfg, basis_ranks2 = _build_encoder_and_syms(args)
        encoder_bundle = (mod, model, sym_q, dev)
        if basis_ranks is None:
            basis_ranks = basis_ranks2

    # Top-5 requested diagnostics (#2 and #5) require encoder access.
    if bool(args.run_top5) and encoder_bundle is None:
        mod, model, sym_q, dev, cfg, basis_ranks2 = _build_encoder_and_syms(args)
        encoder_bundle = (mod, model, sym_q, dev)
        if basis_ranks is None:
            basis_ranks = basis_ranks2

    # PCA projection for mapping plots.
    mean, w, var_ratio = _fit_pca2(x_flat, fit_n=int(args.fit_points), seed=int(args.seed))
    xy_all = _project_pca2(x_flat, mean, w)

    x_lo = float(np.percentile(xy_all[:, 0], 1.0))
    x_hi = float(np.percentile(xy_all[:, 0], 99.0))
    y_lo = float(np.percentile(xy_all[:, 1], 1.0))
    y_hi = float(np.percentile(xy_all[:, 1], 99.0))
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    ix_all, iy_all, counts, x_edges, y_edges = _digitize_2d(
        xy=xy_all,
        bins=int(args.bins),
        x_min=x_lo,
        x_max=x_hi,
        y_min=y_lo,
        y_max=y_hi,
    )
    lut = _bin_color_lut(int(args.bins))
    rgb_all = lut[iy_all, ix_all]

    rng = np.random.default_rng(int(args.seed))
    if n_total > int(args.max_points):
        sidx = rng.choice(n_total, size=int(args.max_points), replace=False)
    else:
        sidx = np.arange(n_total, dtype=np.int64)

    q_s = q_flat[sidx]
    xy_s = xy_all[sidx]
    rgb_s = rgb_all[sidx]
    ix_s = ix_all[sidx]
    iy_s = iy_all[sidx]

    sym = resolve_symmetry(args.symmetry)
    v_ref = DIR_VECTORS[str(args.ref_dir).upper()]

    ori_s = Orientation(q_s, symmetry=sym)
    v_s = ori_s * v_ref
    v_key_s = v_s.in_fundamental_sector(sym)
    x_st, y_st = _vector2xy(v_key_s, pole=-1)
    x_st = np.asarray(x_st).reshape(-1)
    y_st = np.asarray(y_st).reshape(-1)
    valid_st = np.isfinite(x_st) & np.isfinite(y_st)

    is_spatial_2d = len(spatial_shape) == 2
    if is_spatial_2d:
        h, w_img = int(spatial_shape[0]), int(spatial_shape[1])
        spatial_rgb = rgb_all.reshape(h, w_img, 3)
        bin_id = (iy_all * int(args.bins) + ix_all).reshape(h, w_img)

        ori_img = Orientation(q_last)
        ori_img.symmetry = sym
        ckey = orix.plot.IPFColorKeyTSL(sym.laue)
        ckey.direction = v_ref
        ipf_rgb = ckey.orientation2color(ori_img)
    else:
        spatial_rgb = None
        ipf_rgb = None
        bin_id = None

    saved_plots: list[str] = []

    # Overview figure.
    if is_spatial_2d:
        fig = plt.figure(figsize=(16, 9), dpi=160)
        gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)
    else:
        fig = plt.figure(figsize=(15, 4.8), dpi=160)
        gs = fig.add_gridspec(1, 3, wspace=0.25)

    ax_st = fig.add_subplot(gs[0, 0], projection="stereographic")
    ax_st.scatter(
        x_st[valid_st],
        y_st[valid_st],
        c=rgb_s[valid_st],
        s=7,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )
    ax_st.set_title(f"Stereographic in cubic FS (IPF-{args.ref_dir.upper()})")
    ax_st.set_labels("RD", "TD", None)
    ax_st.show_hemisphere_label()

    ax_fs = fig.add_subplot(gs[0, 1])
    ax_fs.scatter(
        xy_s[:, 0],
        xy_s[:, 1],
        c=rgb_s,
        s=7,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )
    ax_fs.set_xlabel("PC1")
    ax_fs.set_ylabel("PC2")
    ax_fs.set_title(
        f"Irrep feature space (PCA-2D)\nvar={var_ratio[0]:.3f}, {var_ratio[1]:.3f}"
    )
    ax_fs.grid(alpha=0.2)

    ax_key = fig.add_subplot(gs[0, 2], projection="ipf", symmetry=sym.laue)
    ax_key.plot_ipf_color_key(show_title=False)
    ax_key.set_title(f"IPF key ({sym.name}, ref {args.ref_dir.upper()})")

    if is_spatial_2d:
        ax_hist = fig.add_subplot(gs[1, 0])
        hm = ax_hist.imshow(
            np.log1p(counts),
            origin="lower",
            cmap="magma",
            aspect="auto",
            extent=[x_lo, x_hi, y_lo, y_hi],
        )
        ax_hist.set_xlabel("PC1")
        ax_hist.set_ylabel("PC2")
        ax_hist.set_title("Discretized feature bins (log count)")
        fig.colorbar(hm, ax=ax_hist, fraction=0.046, pad=0.04)

        ax_sp = fig.add_subplot(gs[1, 1])
        ax_sp.imshow(spatial_rgb)
        ax_sp.set_title("Spatial map colored by feature bin")
        ax_sp.axis("off")

        ax_ipf = fig.add_subplot(gs[1, 2])
        ax_ipf.imshow(ipf_rgb)
        ax_ipf.set_title(f"Spatial IPF-{args.ref_dir.upper()} map")
        ax_ipf.axis("off")

    fig.suptitle("Quaternion <-> Irrep Feature Mapping", fontsize=14)
    overview_path = out_dir / "quat_irrep_mapping_overview.png"
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(str(overview_path))

    # 3D discretization surface.
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="xy")
    zz = counts.astype(np.float32)

    fig2 = plt.figure(figsize=(10, 7), dpi=170)
    ax3 = fig2.add_subplot(111, projection="3d")
    ax3.plot_surface(
        xx,
        yy,
        zz,
        facecolors=lut,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax3.set_xlabel("PC1 bin center")
    ax3.set_ylabel("PC2 bin center")
    ax3.set_zlabel("count")
    ax3.set_title("Irrep discretization surface (bin counts)")
    ax3.view_init(elev=28, azim=38)
    surface_path = out_dir / "quat_irrep_feature_surface.png"
    fig2.savefig(surface_path, bbox_inches="tight")
    plt.close(fig2)
    saved_plots.append(str(surface_path))

    # Additional plot 1: feature-space hexbin.
    fig3, axh = plt.subplots(figsize=(7.2, 6.0), dpi=170)
    hb = axh.hexbin(
        xy_all[:, 0],
        xy_all[:, 1],
        gridsize=int(args.bins),
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axh.set_xlabel("PC1")
    axh.set_ylabel("PC2")
    axh.set_title("Feature-space density (hexbin, log)")
    fig3.colorbar(hb, ax=axh, fraction=0.046, pad=0.04, label="log10(count)")
    p_hex = out_dir / "quat_irrep_feature_hexbin.png"
    fig3.savefig(p_hex, bbox_inches="tight")
    plt.close(fig3)
    saved_plots.append(str(p_hex))

    # Additional plot 2: stereographic density.
    fig4, axs = plt.subplots(figsize=(7.2, 6.0), dpi=170)
    if np.any(valid_st):
        h2, x2e, y2e = np.histogram2d(x_st[valid_st], y_st[valid_st], bins=120)
        im2 = axs.imshow(
            np.log1p(h2.T),
            origin="lower",
            cmap="magma",
            extent=[x2e[0], x2e[-1], y2e[0], y2e[-1]],
            aspect="equal",
        )
        fig4.colorbar(im2, ax=axs, fraction=0.046, pad=0.04, label="log(1+count)")
    axs.set_xlabel("stereo x")
    axs.set_ylabel("stereo y")
    axs.set_title("Stereographic density in cubic fundamental sector")
    p_stden = out_dir / "quat_irrep_stereo_density.png"
    fig4.savefig(p_stden, bbox_inches="tight")
    plt.close(fig4)
    saved_plots.append(str(p_stden))

    # Additional plot 3: feature component marginals.
    comp_show = min(8, x_flat.shape[1])
    ncols = 4
    nrows = int(math.ceil(comp_show / ncols))
    fig5, axes5 = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), dpi=170)
    if nrows == 1 and ncols == 1:
        axes_flat = [axes5]
    elif nrows == 1:
        axes_flat = list(axes5)
    elif ncols == 1:
        axes_flat = list(axes5)
    else:
        axes_flat = [a for row in axes5 for a in row]

    x_sub = x_flat[sidx]
    for i, ax in enumerate(axes_flat):
        if i >= comp_show:
            ax.axis("off")
            continue
        ax.hist(x_sub[:, i], bins=120, color="#2f5d95", alpha=0.85)
        ax.set_title(f"feature[{i}]")
        ax.grid(alpha=0.2)
    fig5.suptitle("Feature marginals (sampled points)")
    p_marg = out_dir / "quat_irrep_feature_marginals.png"
    fig5.savefig(p_marg, bbox_inches="tight")
    plt.close(fig5)
    saved_plots.append(str(p_marg))

    # Additional plot 4: bin-centroid correspondence between feature and stereo spaces.
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=170)
    bid_s = iy_s * int(args.bins) + ix_s
    uniq = np.unique(bid_s)
    min_bin_count = 20
    for b in uniq:
        mask = bid_s == b
        if int(np.sum(mask)) < min_bin_count:
            continue
        iy = int(b // int(args.bins))
        ix = int(b % int(args.bins))
        c = lut[iy, ix]

        cf = np.mean(xy_s[mask], axis=0)
        ax6a.scatter(cf[0], cf[1], s=20, color=c, edgecolors="none")

        mask_st = mask & valid_st
        if np.any(mask_st):
            csx = float(np.mean(x_st[mask_st]))
            csy = float(np.mean(y_st[mask_st]))
            ax6b.scatter(csx, csy, s=20, color=c, edgecolors="none")

    ax6a.set_title("Bin centroids in feature PCA space")
    ax6a.set_xlabel("PC1")
    ax6a.set_ylabel("PC2")
    ax6a.grid(alpha=0.2)

    ax6b.set_title("Same-bin centroids in stereographic space")
    ax6b.set_xlabel("stereo x")
    ax6b.set_ylabel("stereo y")
    ax6b.set_aspect("equal", adjustable="box")
    ax6b.grid(alpha=0.2)

    p_cent = out_dir / "quat_irrep_bin_centroid_correspondence.png"
    fig6.savefig(p_cent, bbox_inches="tight")
    plt.close(fig6)
    saved_plots.append(str(p_cent))

    # Optional spatial bin-id map.
    if is_spatial_2d and bin_id is not None:
        fig7, ax7 = plt.subplots(figsize=(8.6, 6.0), dpi=170)
        im7 = ax7.imshow(bin_id, cmap="nipy_spectral", interpolation="nearest")
        ax7.set_title("Discrete feature-bin ID map (spatial)")
        ax7.axis("off")
        fig7.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label="bin id")
        p_binid = out_dir / "quat_irrep_spatial_bin_id_map.png"
        fig7.savefig(p_binid, bbox_inches="tight")
        plt.close(fig7)
        saved_plots.append(str(p_binid))

    top5_stats: dict[str, Any] = {}
    if bool(args.run_top5):
        if encoder_bundle is not None:
            mod, model, sym_q, dev = encoder_bundle
            sym_inv_np = sym_q.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            mod = model = dev = None
            sym_inv_np = _build_fcc_syms_inv_wxyz_np()

        p1, s1 = _plot_1_misorientation_vs_feature_distance(
            q_flat=q_flat,
            x_flat=x_flat,
            sym_inv_np=sym_inv_np,
            out_dir=out_dir,
            pair_count=int(args.pair_count),
            seed=int(args.seed) + 11,
        )
        saved_plots.append(p1)
        top5_stats["1_misorientation_vs_feature_distance"] = s1

        if encoder_bundle is not None:
            p2, s2 = _plot_2_orbit_collapse_residual(
                q_flat=q_flat,
                mod=mod,
                model=model,
                sym_q=sym_q,
                device=dev,
                out_dir=out_dir,
                sample_count=int(args.orbit_samples),
                seed=int(args.seed) + 22,
            )
            saved_plots.append(p2)
            top5_stats["2_orbit_collapse_residual"] = s2

        p3, s3 = _plot_3_knn_retrieval_map(
            q_flat=q_flat,
            x_flat=x_flat,
            sym=sym,
            v_ref=v_ref,
            sym_inv_np=sym_inv_np,
            out_dir=out_dir,
            pool_size=int(args.retrieval_pool),
            query_count=int(args.retrieval_queries),
            k=int(args.retrieval_k),
            seed=int(args.seed) + 33,
        )
        saved_plots.append(p3)
        top5_stats["3_knn_retrieval_map"] = s3

        layout = _build_feature_layout_from_meta(feat_meta, feature_dim=int(x_flat.shape[-1]))
        p4s, s4 = _plot_4_per_l_energy_decomposition(
            feats=feats,
            layout=layout,
            out_dir=out_dir,
        )
        saved_plots.extend(p4s)
        if s4:
            top5_stats["4_per_l_energy_decomposition"] = s4

        if encoder_bundle is not None:
            jac_angles = _parse_angle_list(str(args.jacobian_angles_deg))
            p5, s5 = _plot_5_jacobian_sensitivity(
                q_flat=q_flat,
                mod=mod,
                model=model,
                device=dev,
                out_dir=out_dir,
                sample_count=int(args.jacobian_samples),
                angles_deg=jac_angles,
                batch_size=int(args.encode_batch_size),
                seed=int(args.seed) + 44,
            )
            saved_plots.append(p5)
            top5_stats["5_jacobian_sensitivity"] = s5

    invariance_stats = None
    if bool(args.verify_symmetry):
        if encoder_bundle is None:
            raise RuntimeError("Internal error: encoder bundle missing for verify-symmetry")
        mod, model, sym_q, dev = encoder_bundle
        inv = _run_symmetry_invariance_check(
            q_flat=q_flat,
            mod=mod,
            model=model,
            sym_q=sym_q,
            device=dev,
            verify_samples=int(args.verify_samples),
            seed=int(args.seed),
        )
        invariance_stats = {
            k: v
            for k, v in inv.items()
            if k
            in {
                "sample_count",
                "sym_l2_max",
                "sym_l2_mean",
                "sym_linf_max",
                "sym_linf_mean",
                "sign_l2_max",
                "sign_l2_mean",
                "sign_linf_max",
                "sign_linf_mean",
            }
        }
        saved_plots.extend(_plot_invariance_diagnostics(inv, out_dir))
        p_orbit = _plot_orbit_examples(
            inv=inv,
            mod=mod,
            sym_q=sym_q,
            sym=sym,
            v_ref=v_ref,
            out_dir=out_dir,
            n_orbits=int(args.orbit_examples),
        )
        if p_orbit:
            saved_plots.append(p_orbit)

    meta: dict[str, Any] = {
        "mode": "random_so3" if random_mode else "data",
        "input_path": str(args.input) if args.input is not None else None,
        "input_source": q_src,
        "features_path": str(args.features) if args.features is not None else None,
        "features_source": feat_src,
        "features_key": str(args.features_key),
        "quaternion_shape": list(q_last.shape),
        "feature_shape": list(feats.shape),
        "n_total": int(n_total),
        "n_scatter": int(sidx.shape[0]),
        "symmetry": str(sym.name),
        "ref_dir": str(args.ref_dir).upper(),
        "bins": int(args.bins),
        "pca_var_ratio": [float(var_ratio[0]), float(var_ratio[1])],
        "scalar_order_input": "xyzw" if bool(args.scalar_last) else "wxyz",
        "quaternion_convention_after_conversion": "bunge_passive_wxyz",
        "basis_ranks": basis_ranks,
        "run_top5": bool(args.run_top5),
        "top5_stats": top5_stats,
        "verify_symmetry": bool(args.verify_symmetry),
        "invariance_stats": invariance_stats,
        "saved_plots": saved_plots,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for pth in saved_plots:
        print(f"saved: {pth}")
    print(f"saved: {meta_path}")
    if invariance_stats is not None:
        print("invariance:")
        for k, v in invariance_stats.items():
            if k == "sample_count":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {float(v):.6e}")


if __name__ == "__main__":
    main()
