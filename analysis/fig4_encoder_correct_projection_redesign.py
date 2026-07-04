"""Regenerate Fig. 4-style encoder diagnostics with the corrected projection set.

This module is meant to be called from
``analysis/Fig4_encoder_correct_projection_redesign.ipynb``.

It does not overwrite the paper figure.  It writes redesigned candidate assets to
``analysis/out/fig4_encoder_correct_projection_redesign`` by default.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reynolds_qsr_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "out" / "fig4_encoder_correct_projection_redesign"


@dataclass(frozen=True)
class EmbeddingSpec:
    label: str
    group_name: str
    d6_convention: str
    embedding_mode: str
    max_harmonic_l: int | None
    role: str


CORRECT_FCC = EmbeddingSpec(
    label="FCC / O direct Reynolds, l≤4",
    group_name="O",
    d6_convention="z_axis",
    embedding_mode="direct_reynolds",
    max_harmonic_l=4,
    role="correct",
)
CORRECT_HCP = EmbeddingSpec(
    label="HCP / D6 direct Reynolds, l≤6",
    group_name="D6",
    d6_convention="z_axis",
    embedding_mode="direct_reynolds",
    max_harmonic_l=6,
    role="correct",
)
LEGACY_HCP = EmbeddingSpec(
    label="HCP / D6 tensor-product legacy",
    group_name="D6",
    d6_convention="z_axis",
    embedding_mode="tensor_product",
    max_harmonic_l=None,
    role="legacy",
)


def _import_embedding_code():
    """Import after callers have chosen the Python environment."""

    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from models.local_iso_embedding import (  # noqa: WPS433
        LocalIsoCrystalEmbedding,
        build_fcc_syms_mtex,
        build_hcp_syms_mtex,
    )

    return LocalIsoCrystalEmbedding, build_fcc_syms_mtex, build_hcp_syms_mtex


def axis_angle_to_matrix(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Active rotation matrices for a batch of unit axes and one angle."""

    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    one_minus_c = 1.0 - c

    R = torch.empty((axis.shape[0], 3, 3), dtype=axis.dtype, device=axis.device)
    R[:, 0, 0] = c + x * x * one_minus_c
    R[:, 0, 1] = x * y * one_minus_c - z * s
    R[:, 0, 2] = x * z * one_minus_c + y * s
    R[:, 1, 0] = y * x * one_minus_c + z * s
    R[:, 1, 1] = c + y * y * one_minus_c
    R[:, 1, 2] = y * z * one_minus_c - x * s
    R[:, 2, 0] = z * x * one_minus_c - y * s
    R[:, 2, 1] = z * y * one_minus_c + x * s
    R[:, 2, 2] = c + z * z * one_minus_c
    return R


def axis_angle_to_matrix_variable(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Active rotation matrices for a batch of unit axes and per-row angles."""

    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    angle_rad = angle_rad.reshape(-1).to(dtype=axis.dtype, device=axis.device)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    one_minus_c = 1.0 - c

    R = torch.empty((axis.shape[0], 3, 3), dtype=axis.dtype, device=axis.device)
    R[:, 0, 0] = c + x * x * one_minus_c
    R[:, 0, 1] = x * y * one_minus_c - z * s
    R[:, 0, 2] = x * z * one_minus_c + y * s
    R[:, 1, 0] = y * x * one_minus_c + z * s
    R[:, 1, 1] = c + y * y * one_minus_c
    R[:, 1, 2] = y * z * one_minus_c - x * s
    R[:, 2, 0] = z * x * one_minus_c - y * s
    R[:, 2, 1] = z * y * one_minus_c + x * s
    R[:, 2, 2] = c + z * z * one_minus_c
    return R


def quaternion_to_matrix_active(q: torch.Tensor) -> torch.Tensor:
    """Scalar-first active quaternion to rotation matrix."""

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0
    R = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - two * (y * y + z * z)
    R[..., 0, 1] = two * (x * y - z * w)
    R[..., 0, 2] = two * (x * z + y * w)
    R[..., 1, 0] = two * (x * y + z * w)
    R[..., 1, 1] = 1 - two * (x * x + z * z)
    R[..., 1, 2] = two * (y * z - x * w)
    R[..., 2, 0] = two * (x * z - y * w)
    R[..., 2, 1] = two * (y * z + x * w)
    R[..., 2, 2] = 1 - two * (x * x + y * y)
    return R


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternion product q1 ⊗ q2."""

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def random_quaternions(n: int, *, seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Uniform random unit quaternions in scalar-first convention."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    q = torch.randn((int(n), 4), generator=generator, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    q = torch.where(q[:, :1] < 0.0, -q, q)
    return q


def fibonacci_sphere_directions(
    n: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Deterministic Fibonacci directions on S^2."""

    n = int(n)
    i = torch.arange(n, dtype=dtype)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * (i + float(seed))
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def fibonacci_quaternions(
    n: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Low-discrepancy Hopf/Fibonacci unit quaternions in scalar-first convention."""

    n = int(n)
    i = torch.arange(n, dtype=dtype)
    # Hopf coordinates on S^3.  u1 is stratified, while u2/u3 use irrational
    # rotations to avoid the clumping that pure random sampling can create in
    # the Fig. 4 diagnostics.
    u1 = (i + 0.5) / float(n)
    phi1 = (math.sqrt(5.0) - 1.0) / 2.0
    phi2 = math.sqrt(3.0) - 1.0
    u2 = torch.frac((i + 0.5 + 0.173 * float(seed)) * phi1)
    u3 = torch.frac((i + 0.5 + 0.419 * float(seed)) * phi2)
    a = torch.sqrt((1.0 - u1).clamp_min(0.0))
    b = torch.sqrt(u1.clamp_min(0.0))
    t2 = 2.0 * math.pi * u2
    t3 = 2.0 * math.pi * u3
    q = torch.stack(
        [
            a * torch.cos(t2),
            a * torch.sin(t2),
            b * torch.cos(t3),
            b * torch.sin(t3),
        ],
        dim=-1,
    )
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return torch.where(q[:, :1] < 0.0, -q, q)


def sphere_grid(n_theta: int, n_phi: int, *, dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return theta, phi, and unit axes on a polar/azimuthal grid."""

    theta = torch.linspace(0.0, math.pi, int(n_theta), dtype=dtype)
    phi = torch.linspace(0.0, 2.0 * math.pi, int(n_phi), dtype=dtype)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    axes = torch.stack(
        [
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ],
        dim=-1,
    )
    return theta_grid, phi_grid, axes


def build_embedding(spec: EmbeddingSpec, *, dtype: torch.dtype = torch.float64):
    LocalIsoCrystalEmbedding, _, _ = _import_embedding_code()
    return LocalIsoCrystalEmbedding(
        spec.group_name,
        d6_convention=spec.d6_convention,
        embedding_mode=spec.embedding_mode,
        max_harmonic_l=spec.max_harmonic_l,
        dtype=dtype,
        device="cpu",
    ).eval()


def eval_embedding_batched(embedding, R: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Evaluate embedding.forward_irreps on rotation matrices without large peak memory."""

    chunks = []
    with torch.no_grad():
        for start in range(0, int(R.shape[0]), int(chunk_size)):
            chunks.append(embedding.forward_irreps(R[start : start + chunk_size]).detach().cpu())
    return torch.cat(chunks, dim=0)


def finite_difference_jacobian(embedding, *, eps: float = 1.0e-4) -> torch.Tensor:
    """Numerical Jacobian dE_I / d(axis-angle) at identity."""

    eye = torch.eye(3, dtype=torch.float64)
    cols = []
    for axis in eye:
        Rp = axis_angle_to_matrix(axis[None, :], eps)
        Rm = axis_angle_to_matrix(axis[None, :], -eps)
        yp = eval_embedding_batched(embedding, Rp, chunk_size=1).reshape(-1)
        ym = eval_embedding_batched(embedding, Rm, chunk_size=1).reshape(-1)
        cols.append((yp - ym) / (2.0 * eps))
    return torch.stack(cols, dim=1)


def calibration_report(embedding) -> Dict[str, object]:
    """Return local metric diagnostics and a single scalar normalization."""

    J = finite_difference_jacobian(embedding)
    G = J.T @ J
    evals = torch.linalg.eigvalsh(G).clamp_min(0.0)
    scale = float(torch.sqrt(evals.mean()).item())
    anisotropy = float((evals.max() / evals.min().clamp_min(1.0e-30)).item())
    return {
        "J": J,
        "G": G,
        "eigvals": evals,
        "scale": scale if scale > 0.0 else 1.0,
        "anisotropy": anisotropy,
    }


def field_for_axes(
    embedding,
    axes: torch.Tensor,
    *,
    angle_deg: float = 60.0,
    scale: float = 1.0,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute ||E(R(n, angle)) - E(I)|| / scale over flattened axes."""

    flat_axes = axes.reshape(-1, 3).to(dtype=torch.float64)
    R = axis_angle_to_matrix(flat_axes, math.radians(float(angle_deg)))
    y = eval_embedding_batched(embedding, R, chunk_size=chunk_size)
    y0 = eval_embedding_batched(embedding, torch.eye(3, dtype=torch.float64)[None, :, :], chunk_size=1)
    return ((y - y0).norm(dim=-1) / float(scale)).reshape(axes.shape[:-1])


def symmetry_aware_misorientation_deg(q1: torch.Tensor, q2: torch.Tensor, syms: torch.Tensor) -> torch.Tensor:
    """min_S 2 acos(|<q1, q2⊗S>|) in degrees."""

    q2s = quat_mul(q2[:, None, :], syms[None, :, :])
    dots = (q1[:, None, :] * q2s).sum(dim=-1).abs().clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dots.max(dim=1).values)
    return angles * (180.0 / math.pi)


def pair_distance_curve(
    embedding,
    *,
    group_name: str,
    scale: float,
    n_pairs: int,
    seed: int,
    chunk_size: int = 8192,
) -> Dict[str, np.ndarray]:
    """Low-discrepancy pair embedding distance versus symmetry-aware misorientation."""

    _, build_fcc_syms_mtex, build_hcp_syms_mtex = _import_embedding_code()
    q1 = fibonacci_quaternions(n_pairs, seed=seed)
    q_delta = fibonacci_quaternions(n_pairs, seed=seed + 1009)
    q2 = quat_mul(q1, q_delta)
    syms = build_fcc_syms_mtex(dtype=torch.float64) if group_name == "O" else build_hcp_syms_mtex(dtype=torch.float64)
    mis_deg = symmetry_aware_misorientation_deg(q1, q2, syms)
    y1 = eval_embedding_batched(embedding, quaternion_to_matrix_active(q1), chunk_size=chunk_size)
    y2 = eval_embedding_batched(embedding, quaternion_to_matrix_active(q2), chunk_size=chunk_size)
    dist = (y1 - y2).norm(dim=-1) / float(scale)
    return {"mis_deg": mis_deg.numpy(), "dist": dist.numpy()}


def local_small_angle_curve(
    embedding,
    *,
    scale: float,
    n_samples: int,
    seed: int,
    max_deg: float = 5.0,
    chunk_size: int = 8192,
) -> Dict[str, np.ndarray]:
    """Dedicated 0--max_deg local curve for the inset in panel e."""

    axes = fibonacci_sphere_directions(n_samples, seed=seed, dtype=torch.float64)
    angles_deg = (torch.arange(int(n_samples), dtype=torch.float64) + 0.5) * (float(max_deg) / float(n_samples))
    R = axis_angle_to_matrix_variable(axes, torch.deg2rad(angles_deg))
    y = eval_embedding_batched(embedding, R, chunk_size=chunk_size)
    y0 = eval_embedding_batched(embedding, torch.eye(3, dtype=torch.float64)[None, :, :], chunk_size=1)
    dist = (y - y0).norm(dim=-1) / float(scale)
    return {"mis_deg": angles_deg.numpy(), "dist": dist.numpy()}


def binned_median_iqr(x: np.ndarray, y: np.ndarray, *, bins: np.ndarray) -> Dict[str, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(centers, np.nan, dtype=float)
    q25 = np.full_like(centers, np.nan, dtype=float)
    q75 = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    for idx in range(len(centers)):
        mask = (x >= bins[idx]) & (x < bins[idx + 1])
        counts[idx] = int(mask.sum())
        if counts[idx] > 0:
            med[idx] = float(np.median(y[mask]))
            q25[idx] = float(np.quantile(y[mask], 0.25))
            q75[idx] = float(np.quantile(y[mask], 0.75))
    return {"centers": centers, "median": med, "q25": q25, "q75": q75, "counts": counts}


def symmetry_spread(
    embedding,
    *,
    group_name: str,
    n_samples: int,
    seed: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Relative spread of E(q⊗S) over proper crystal symmetries S."""

    _, build_fcc_syms_mtex, build_hcp_syms_mtex = _import_embedding_code()
    dtype = getattr(embedding, "dtype", torch.float64)
    q = fibonacci_quaternions(n_samples, seed=seed, dtype=dtype)
    syms = build_fcc_syms_mtex(dtype=dtype) if group_name == "O" else build_hcp_syms_mtex(dtype=dtype)
    q_orbit = quat_mul(q[:, None, :], syms[None, :, :]).reshape(-1, 4)
    y = eval_embedding_batched(embedding, quaternion_to_matrix_active(q_orbit), chunk_size=chunk_size)
    y = y.reshape(n_samples, syms.shape[0], -1)
    center = y.mean(dim=1, keepdim=True)
    spread = (y - center).norm(dim=-1).max(dim=1).values
    denom = center.norm(dim=-1).squeeze(1).clamp_min(1.0e-12)
    return (spread / denom).numpy()


PAPER_CMAP = "plasma"
PANEL_COLOR = "#1f2933"
FCC_COLOR = "#3f6f9f"
HCP_COLOR = "#b96f5d"
LIMIT_COLOR = "#7a4e3a"


def _panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.04) -> None:
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=PANEL_COLOR,
        va="top",
        ha="left",
    )


def _surface_panel(
    ax,
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    *,
    label: str | None = None,
    title: str | None = None,
    cmap_name: str = PAPER_CMAP,
    zoom: float = 1.0,
) -> ScalarMappable:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    cmap = plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
    colors = cmap(norm(values))
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=float(zoom))
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="semibold")
    if label is not None:
        ax.text2D(
            0.5,
            -0.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color=PANEL_COLOR,
        )
    return ScalarMappable(norm=norm, cmap=cmap)


def _heatmap_panel(
    ax,
    values: np.ndarray,
    *,
    tag: str,
    cmap_name: str = PAPER_CMAP,
) -> ScalarMappable:
    im = ax.imshow(
        values,
        origin="upper",
        aspect="auto",
        extent=(0, 360, 180, 0),
        cmap=cmap_name,
        interpolation="bicubic",
    )
    ax.set_xlabel(r"azimuth $\varphi$ (deg)", color=PANEL_COLOR)
    ax.set_ylabel(r"polar $\theta$ (deg)", color=PANEL_COLOR)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.tick_params(colors="#52616b")
    ax.text(
        0.985,
        0.93,
        tag,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#334155", edgecolor="none", alpha=0.84),
    )
    return im


def plot_corrected_figure(
    *,
    out_dir: Path,
    theta: torch.Tensor,
    phi: torch.Tensor,
    fields: Dict[str, np.ndarray],
    pair_curves: Dict[str, Dict[str, np.ndarray]],
    local_curves: Dict[str, Dict[str, np.ndarray]],
    spreads: Dict[str, np.ndarray],
    summaries: Dict[str, Dict[str, object]],
) -> Tuple[Path, Path]:
    """Create paper-ready Fig. 4 using the corrected projection sets."""

    out_dir.mkdir(parents=True, exist_ok=True)
    theta_np = theta.numpy()
    phi_np = phi.numpy()
    plt.rcParams.update(
        {
            "axes.edgecolor": "#52616b",
            "axes.labelcolor": PANEL_COLOR,
            "xtick.color": "#52616b",
            "ytick.color": "#52616b",
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(12.0, 13.3), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 0.90, 1.00], hspace=0.18, wspace=0.16)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    _panel_letter(ax_a, "a", x=-0.03, y=1.02)
    _panel_letter(ax_b, "b", x=-0.03, y=1.02)
    sm_a = _surface_panel(
        ax_a,
        theta_np,
        phi_np,
        fields["fcc_correct"],
        label=r"FCC / $O$",
        zoom=1.38,
    )
    sm_b = _surface_panel(
        ax_b,
        theta_np,
        phi_np,
        fields["hcp_correct"],
        label=r"HCP / $D_6$",
        zoom=1.38,
    )
    fig.colorbar(sm_a, ax=ax_a, shrink=0.62, fraction=0.045, pad=0.02, label=r"$s(\mathbf{n})$")
    fig.colorbar(sm_b, ax=ax_b, shrink=0.62, fraction=0.045, pad=0.02, label=r"$s(\mathbf{n})$")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    _panel_letter(ax_c, "c")
    _panel_letter(ax_d, "d")
    im_c = _heatmap_panel(ax_c, fields["fcc_correct"], tag="FCC")
    im_d = _heatmap_panel(ax_d, fields["hcp_correct"], tag="HCP")
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02, label=r"$s(\mathbf{n})$")
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.02, label=r"$s(\mathbf{n})$")

    ax_e = fig.add_subplot(gs[2, 0])
    _panel_letter(ax_e, "e")
    max_x = max(float(np.nanmax(data["mis_deg"])) for data in pair_curves.values())
    main_bins = np.linspace(0.0, max_x, 70)
    inset_bins = np.linspace(0.0, 5.0, 16)
    stats_by_key = {}
    for key, color, label in [
        ("fcc_correct", FCC_COLOR, r"FCC ($O$, $|G|$=24)"),
        ("hcp_correct", HCP_COLOR, r"HCP ($D_6$, $|G|$=12)"),
    ]:
        data = pair_curves[key]
        stats = binned_median_iqr(data["mis_deg"], data["dist"], bins=main_bins)
        stats_by_key[key] = {
            "main": stats,
            "inset": binned_median_iqr(
                local_curves[key]["mis_deg"],
                local_curves[key]["dist"],
                bins=inset_bins,
            ),
        }
        valid = np.isfinite(stats["median"])
        ax_e.plot(stats["centers"][valid], stats["median"][valid], color=color, lw=2.0, label=label)
        ax_e.fill_between(
            stats["centers"][valid],
            stats["q25"][valid],
            stats["q75"][valid],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    small = np.linspace(0.0, 5.0, 100)
    ax_e.axvline(11.0, color=LIMIT_COLOR, ls="--", lw=1.3)
    ax_e.text(
        11.4,
        0.04,
        r"$\sim 11^\circ$ first-order limit",
        rotation=90,
        color=LIMIT_COLOR,
        fontsize=8,
        va="bottom",
    )
    ax_e.set_xlim(0.0, max_x)
    ax_e.set_xlabel(r"misorientation $\Delta\omega$ (deg)")
    ax_e.set_ylabel(r"embedding distance $\|\Delta f\|$")
    ax_e.legend(frameon=False, fontsize=9, loc="upper left")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    axins = ax_e.inset_axes([0.54, 0.12, 0.38, 0.30])
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(0.94)
    inset_ymax = 0.0
    axins.plot(small, np.deg2rad(small), color="#555555", ls=":", lw=1.0, alpha=0.9, zorder=1)
    for key, color, style, width in [
        ("fcc_correct", FCC_COLOR, "-", 1.7),
        ("hcp_correct", HCP_COLOR, (0, (3, 2)), 1.6),
    ]:
        stats = stats_by_key[key]["inset"]
        valid = np.isfinite(stats["median"])
        axins.fill_between(
            stats["centers"][valid],
            stats["q25"][valid],
            stats["q75"][valid],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        axins.plot(
            stats["centers"][valid],
            stats["median"][valid],
            color=color,
            lw=width,
            ls=style,
            zorder=3,
        )
        if np.any(valid):
            inset_ymax = max(inset_ymax, float(np.nanmax(stats["q75"][valid])))
    axins.set_xlim(0, 5)
    axins.set_ylim(0, max(inset_ymax, float(np.deg2rad(5.0))) * 1.12)
    axins.set_title(r"$0$--$5^\circ$ (linear)", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    ax_f = fig.add_subplot(gs[2, 1])
    _panel_letter(ax_f, "f")
    fcc_log = np.log10(np.clip(spreads["fcc_correct"], 1.0e-18, None))
    hcp_log = np.log10(np.clip(spreads["hcp_correct"], 1.0e-18, None))
    both = np.concatenate([fcc_log, hcp_log])
    lo = float(np.nanmin(both))
    hi = float(np.nanmax(both))
    if hi - lo < 0.12:
        mid = 0.5 * (hi + lo)
        lo, hi = mid - 0.12, mid + 0.12
    else:
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    bins = np.linspace(lo, hi, 56)
    ax_f.hist(fcc_log, bins=bins, color=FCC_COLOR, alpha=0.48, label="FCC (24 copies)")
    ax_f.hist(hcp_log, bins=bins, color=HCP_COLOR, alpha=0.52, label="HCP (12 copies)")
    ax_f.axvline(float(np.median(fcc_log)), color=FCC_COLOR, ls="--", lw=1.6)
    ax_f.axvline(float(np.median(hcp_log)), color=HCP_COLOR, ls="--", lw=1.6)
    ax_f.text(
        0.03,
        0.93,
        "right-symmetry copies\nruntime precision",
        transform=ax_f.transAxes,
        color="#666666",
        fontsize=9,
    )
    ax_f.set_xlabel(r"$\log_{10}$ relative spread over symmetry copies")
    ax_f.set_ylabel("orientations")
    ax_f.legend(frameon=False, fontsize=9, loc="upper right")
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    png = out_dir / "fig4_encoder_corrected_projection_redesign.png"
    pdf = out_dir / "fig4_encoder_corrected_projection_redesign.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_hcp_old_vs_corrected(
    *,
    out_dir: Path,
    theta: torch.Tensor,
    phi: torch.Tensor,
    old_field: np.ndarray,
    new_field: np.ndarray,
) -> Tuple[Path, Path]:
    """Direct diagnostic for the old Fig. 4b HCP panel."""

    out_dir.mkdir(parents=True, exist_ok=True)
    theta_np = theta.numpy()
    phi_np = phi.numpy()
    old_z = (old_field - np.mean(old_field)) / np.std(old_field)
    new_z = (new_field - np.mean(new_field)) / np.std(new_field)
    diff = new_z - old_z

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_old = fig.add_subplot(gs[0, 0], projection="3d")
    ax_new = fig.add_subplot(gs[0, 1], projection="3d")
    ax_diff = fig.add_subplot(gs[0, 2])
    _surface_panel(ax_old, theta_np, phi_np, old_z, title="legacy HCP tensor route\n2×2e+1×4e+1×6e")
    _surface_panel(ax_new, theta_np, phi_np, new_z, title="corrected HCP direct projector\n1×2e+1×4e+2×6e")
    div = TwoSlopeNorm(vcenter=0.0, vmin=float(np.nanmin(diff)), vmax=float(np.nanmax(diff)))
    im_diff = ax_diff.imshow(diff, origin="upper", aspect="auto", extent=(0, 360, 180, 0), cmap="coolwarm", norm=div)
    ax_diff.set_title("corrected − legacy\nz-scored field")
    ax_diff.set_xlabel(r"$\varphi$ (deg)")
    ax_diff.set_ylabel(r"$\theta$ (deg)")
    fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.02)

    for ax, arr, title in [
        (fig.add_subplot(gs[1, 0]), old_z, "legacy unrolled"),
        (fig.add_subplot(gs[1, 1]), new_z, "corrected unrolled"),
    ]:
        im = ax.imshow(arr, origin="upper", aspect="auto", extent=(0, 360, 180, 0), cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(r"$\varphi$ (deg)")
        ax.set_ylabel(r"$\theta$ (deg)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    corr = float(np.corrcoef(old_z.reshape(-1), new_z.reshape(-1))[0, 1])
    mae = float(np.mean(np.abs(diff)))
    p95 = float(np.quantile(np.abs(diff), 0.95))
    max_abs = float(np.max(np.abs(diff)))
    ax_text.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Why this panel changes:",
                "",
                "Legacy tensor route duplicated the l=2 block",
                "and kept only one l=6 direction.",
                "",
                "Correct direct Reynolds route keeps the",
                "projector rank itself: r2=1, r4=1, r6=2.",
                "",
                f"normalized-field corr: {corr:.4f}",
                f"mean |Δ z-score|: {mae:.4f}",
                f"p95 |Δ z-score|: {p95:.4f}",
                f"max |Δ z-score|: {max_abs:.4f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
    )
    fig.suptitle("Fig. 4b redesign audit: HCP old tensor route vs corrected projector basis", fontsize=14, fontweight="bold")

    png = out_dir / "fig4b_hcp_old_vs_corrected_projection.png"
    pdf = out_dir / "fig4b_hcp_old_vs_corrected_projection.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_all(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    n_theta: int = 181,
    n_phi: int = 361,
    n_pairs: int = 24000,
    n_spread: int = 24000,
    angle_deg: float = 60.0,
    chunk_size: int = 8192,
) -> Dict[str, object]:
    """Run the complete redesign calculation and save figures/data.

    The defaults are intentionally moderate so the notebook can be rerun on CPU.
    Increase ``n_theta``, ``n_phi`` and ``n_pairs`` for camera-ready regeneration.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "fcc_correct": CORRECT_FCC,
        "hcp_correct": CORRECT_HCP,
        "hcp_legacy": LEGACY_HCP,
    }
    embeddings = {key: build_embedding(spec) for key, spec in specs.items()}
    summaries: Dict[str, Dict[str, object]] = {}
    calibrations = {}
    for key, emb in embeddings.items():
        cal = calibration_report(emb)
        calibrations[key] = cal
        summaries[key] = {
            "label": specs[key].label,
            "role": specs[key].role,
            "embedding_mode": specs[key].embedding_mode,
            "group_name": specs[key].group_name,
            "max_harmonic_l": specs[key].max_harmonic_l,
            "irreps_full": str(emb.irreps_full),
            "irreps_a1": str(emb.irreps_a1),
            "dim_full": int(emb.irreps_full.dim),
            "dim_a1": int(emb.irreps_a1.dim),
            "local_metric_scale": float(cal["scale"]),
            "local_metric_eigvals": [float(v) for v in cal["eigvals"].detach().cpu().tolist()],
            "local_metric_anisotropy": float(cal["anisotropy"]),
        }

    theta, phi, axes = sphere_grid(n_theta, n_phi)
    fields = {
        key: field_for_axes(
            emb,
            axes,
            angle_deg=angle_deg,
            scale=calibrations[key]["scale"],
            chunk_size=chunk_size,
        ).numpy()
        for key, emb in embeddings.items()
    }

    pair_curves = {
        "fcc_correct": pair_distance_curve(
            embeddings["fcc_correct"],
            group_name="O",
            scale=calibrations["fcc_correct"]["scale"],
            n_pairs=n_pairs,
            seed=100,
            chunk_size=chunk_size,
        ),
        "hcp_correct": pair_distance_curve(
            embeddings["hcp_correct"],
            group_name="D6",
            scale=calibrations["hcp_correct"]["scale"],
            n_pairs=n_pairs,
            seed=200,
            chunk_size=chunk_size,
        ),
    }
    local_curves = {
        "fcc_correct": local_small_angle_curve(
            embeddings["fcc_correct"],
            scale=calibrations["fcc_correct"]["scale"],
            n_samples=max(2048, n_pairs // 8),
            seed=500,
            max_deg=5.0,
            chunk_size=chunk_size,
        ),
        "hcp_correct": local_small_angle_curve(
            embeddings["hcp_correct"],
            scale=calibrations["hcp_correct"]["scale"],
            n_samples=max(2048, n_pairs // 8),
            seed=600,
            max_deg=5.0,
            chunk_size=chunk_size,
        ),
    }
    # Panel f intentionally uses the runtime precision of the encoder path.  In
    # float64 the right-orbit residual is almost algebraically constant, so the
    # histogram collapses into two needle-like bars.  Float32 exposes the real
    # single-precision numerical floor seen by the network implementation.
    spread_embeddings = {
        "fcc_correct": build_embedding(CORRECT_FCC, dtype=torch.float32),
        "hcp_correct": build_embedding(CORRECT_HCP, dtype=torch.float32),
    }
    spreads = {
        "fcc_correct": symmetry_spread(
            spread_embeddings["fcc_correct"],
            group_name="O",
            n_samples=n_spread,
            seed=300,
            chunk_size=chunk_size,
        ),
        "hcp_correct": symmetry_spread(
            spread_embeddings["hcp_correct"],
            group_name="D6",
            n_samples=n_spread,
            seed=400,
            chunk_size=chunk_size,
        ),
    }
    summaries["symmetry_spread_runtime_precision"] = {
        "dtype": "float32",
        "n_samples": int(n_spread),
        "fcc_log10_relative_spread_median": float(
            np.median(np.log10(np.clip(spreads["fcc_correct"], 1.0e-18, None)))
        ),
        "hcp_log10_relative_spread_median": float(
            np.median(np.log10(np.clip(spreads["hcp_correct"], 1.0e-18, None)))
        ),
    }

    hcp_old_z = (fields["hcp_legacy"] - fields["hcp_legacy"].mean()) / fields["hcp_legacy"].std()
    hcp_new_z = (fields["hcp_correct"] - fields["hcp_correct"].mean()) / fields["hcp_correct"].std()
    summaries["hcp_legacy_vs_corrected"] = {
        "normalized_field_corr": float(np.corrcoef(hcp_old_z.reshape(-1), hcp_new_z.reshape(-1))[0, 1]),
        "mean_abs_zscore_diff": float(np.mean(np.abs(hcp_new_z - hcp_old_z))),
        "p95_abs_zscore_diff": float(np.quantile(np.abs(hcp_new_z - hcp_old_z), 0.95)),
        "max_abs_zscore_diff": float(np.max(np.abs(hcp_new_z - hcp_old_z))),
    }

    fig4_png, fig4_pdf = plot_corrected_figure(
        out_dir=out_dir,
        theta=theta,
        phi=phi,
        fields=fields,
        pair_curves=pair_curves,
        local_curves=local_curves,
        spreads=spreads,
        summaries=summaries,
    )
    audit_png, audit_pdf = plot_hcp_old_vs_corrected(
        out_dir=out_dir,
        theta=theta,
        phi=phi,
        old_field=fields["hcp_legacy"],
        new_field=fields["hcp_correct"],
    )

    np.savez_compressed(
        out_dir / "fig4_corrected_projection_fields_and_curves.npz",
        theta=theta.numpy(),
        phi=phi.numpy(),
        fcc_correct_field=fields["fcc_correct"],
        hcp_correct_field=fields["hcp_correct"],
        hcp_legacy_field=fields["hcp_legacy"],
        fcc_pair_mis_deg=pair_curves["fcc_correct"]["mis_deg"],
        fcc_pair_dist=pair_curves["fcc_correct"]["dist"],
        hcp_pair_mis_deg=pair_curves["hcp_correct"]["mis_deg"],
        hcp_pair_dist=pair_curves["hcp_correct"]["dist"],
        fcc_local_mis_deg=local_curves["fcc_correct"]["mis_deg"],
        fcc_local_dist=local_curves["fcc_correct"]["dist"],
        hcp_local_mis_deg=local_curves["hcp_correct"]["mis_deg"],
        hcp_local_dist=local_curves["hcp_correct"]["dist"],
        fcc_spread=spreads["fcc_correct"],
        hcp_spread=spreads["hcp_correct"],
    )
    summary_path = out_dir / "fig4_corrected_projection_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")

    return {
        "output_dir": str(out_dir),
        "fig4_png": str(fig4_png),
        "fig4_pdf": str(fig4_pdf),
        "hcp_audit_png": str(audit_png),
        "hcp_audit_pdf": str(audit_pdf),
        "summary_json": str(summary_path),
        "field_npz": str(out_dir / "fig4_corrected_projection_fields_and_curves.npz"),
        "summaries": summaries,
    }


if __name__ == "__main__":
    result = run_all()
    print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2))
