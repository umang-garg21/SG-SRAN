"""Standalone axis-angle ball visualization for the corrected OCRP embeddings.

This is an exploratory companion to Fig. 4.  The usual Fig. 4a/b sphere fixes
the rotation angle and varies only the rotation axis.  Here we visualize the
full axis-angle ball

    xi = theta * n,   0 <= theta <= pi,

so the direction of xi is the active rotation axis and the radius of xi is the
active rotation angle.  The color is the locally calibrated embedding distance
from identity:

    ||f(R(xi)) - f(I)|| / local_scale.

The boundary of the ball is the 180 degree shell, where antipodal boundary
points represent the same SO(3) rotation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reynolds_qsr_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.fig4_encoder_correct_projection_redesign import (  # noqa: E402
    CORRECT_FCC,
    CORRECT_HCP,
    axis_angle_to_matrix_variable,
    build_embedding,
    calibration_report,
    eval_embedding_batched,
    fibonacci_sphere_directions,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "out" / "axis_angle_embedding_ball"
PAPER_FIG_DIR = REPO_ROOT / "Paper" / "EBSD_SR_Nature_v3" / "figs"
PANEL_COLOR = "#1f2933"
EDGE_COLOR = "#52616b"
NATURE_CMAP = LinearSegmentedColormap.from_list(
    "nature_muted_blue_peach",
    [
        (0.00, "#253043"),  # soft ink navy
        (0.22, "#3f6f9f"),  # muted scientific blue
        (0.44, "#8fb9c7"),  # desaturated blue-cyan
        (0.64, "#f1ede6"),  # warm paper highlight
        (0.82, "#d79a83"),  # muted peach
        (1.00, "#9b5a62"),  # subdued rose upper tail
    ],
    N=256,
)


def axis_angle_vectors_to_distance(
    embedding,
    xi: torch.Tensor,
    *,
    scale: float,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Evaluate calibrated embedding distance from identity for axis-angle xi."""

    xi = xi.to(dtype=torch.float64)
    angle = xi.norm(dim=-1)
    axis = xi / angle[:, None].clamp_min(1.0e-12)
    # Axis is arbitrary at the origin because theta=0 produces the identity.
    axis = torch.where(angle[:, None] > 1.0e-12, axis, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    R = axis_angle_to_matrix_variable(axis, angle)
    y = eval_embedding_batched(embedding, R, chunk_size=chunk_size)
    y0 = eval_embedding_batched(embedding, torch.eye(3, dtype=torch.float64)[None, :, :], chunk_size=1)
    return (y - y0).norm(dim=-1) / float(scale)


def fibonacci_axis_angle_ball(
    n_points: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Low-discrepancy points in the axis-angle ball of radius pi."""

    n_points = int(n_points)
    axes = fibonacci_sphere_directions(n_points, seed=seed, dtype=dtype)
    i = torch.arange(n_points, dtype=dtype)
    # r^3 stratification gives approximately uniform volume coverage.
    radius = math.pi * ((i + 0.5) / float(n_points)).pow(1.0 / 3.0)
    return axes * radius[:, None]


def slice_axis_angle_grid(
    plane: str,
    *,
    n_grid: int,
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """Return 2D coordinates and 3D xi vectors for a central ball slice."""

    coords = torch.linspace(-math.pi, math.pi, int(n_grid), dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    rr = torch.sqrt(xx * xx + yy * yy)
    mask = (rr <= math.pi).numpy()

    zeros = torch.zeros_like(xx)
    if plane == "x=0":
        xi = torch.stack([zeros, xx, yy], dim=-1)
    elif plane == "y=0":
        xi = torch.stack([xx, zeros, yy], dim=-1)
    elif plane == "z=0":
        xi = torch.stack([xx, yy, zeros], dim=-1)
    else:
        raise ValueError(f"Unknown plane {plane!r}; expected x=0, y=0, or z=0.")

    return xx.numpy() * (180.0 / math.pi), yy.numpy() * (180.0 / math.pi), xi.reshape(-1, 3), mask


def compute_system_data(
    spec,
    *,
    n_cloud: int,
    n_grid: int,
    seed: int,
    chunk_size: int,
) -> Dict[str, object]:
    """Compute 3D cloud values and orthogonal central slices for one system."""

    embedding = build_embedding(spec)
    calibration = calibration_report(embedding)
    scale = float(calibration["scale"])

    cloud_xi = fibonacci_axis_angle_ball(n_cloud, seed=seed)
    cloud_values = axis_angle_vectors_to_distance(embedding, cloud_xi, scale=scale, chunk_size=chunk_size)

    slices = {}
    for plane in ["x=0", "y=0", "z=0"]:
        x_deg, y_deg, flat_xi, mask = slice_axis_angle_grid(plane, n_grid=n_grid)
        flat_values = axis_angle_vectors_to_distance(embedding, flat_xi, scale=scale, chunk_size=chunk_size)
        values = flat_values.reshape(n_grid, n_grid).numpy()
        values[~mask] = np.nan
        slices[plane] = {
            "x_deg": x_deg,
            "y_deg": y_deg,
            "values": values,
            "mask": mask,
        }

    return {
        "embedding_label": spec.label,
        "irreps_full": str(embedding.irreps_full),
        "irreps_a1": str(embedding.irreps_a1),
        "dim_a1": int(embedding.irreps_a1.dim),
        "local_scale": scale,
        "local_metric_eigvals": [float(v) for v in calibration["eigvals"].detach().cpu().tolist()],
        "cloud_xi": cloud_xi.numpy(),
        "cloud_values": cloud_values.numpy(),
        "slices": slices,
    }


def panel_letter(ax, letter: str) -> None:
    """Add a paper-style panel letter to 2D or 3D axes."""

    if hasattr(ax, "text2D"):
        ax.text2D(
            0.01,
            0.985,
            letter,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=PANEL_COLOR,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.7),
        )
    else:
        ax.text(
            0.025,
            0.980,
            letter,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=PANEL_COLOR,
            va="top",
            ha="left",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.7),
        )


def plot_cloud(ax, data: Dict[str, object], *, norm: Normalize, title: str) -> None:
    """Plot a cutaway-looking low-discrepancy cloud in the axis-angle ball."""

    xi = np.asarray(data["cloud_xi"]) * (180.0 / math.pi)
    values = np.asarray(data["cloud_values"])
    # Sort back-to-front by the default view direction for cleaner layering.
    order = np.argsort(xi[:, 0] + 0.6 * xi[:, 1] + 0.8 * xi[:, 2])
    xi = xi[order]
    values = values[order]
    colors = NATURE_CMAP(norm(values))
    ax.scatter(
        xi[:, 0],
        xi[:, 1],
        xi[:, 2],
        c=colors,
        s=2.0,
        alpha=0.26,
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )
    lim = 180.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.set_zticks([-180, 0, 180])
    # The 3D cloud is meant as a qualitative volume cue.  Keeping its tick
    # labels competes with the adjacent quantitative slice panels, so the
    # numeric axes are shown only on the slices.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(length=0, pad=0)
    ax.view_init(elev=24, azim=37)
    ax.set_box_aspect((1, 1, 1), zoom=1.08)
    ax.set_title(title, fontsize=11, color=PANEL_COLOR, pad=2)


def plot_slice(ax, slice_data: Dict[str, object], *, norm: Normalize, title: str, xlabel: str, ylabel: str) -> None:
    """Plot one central disk slice through the axis-angle ball."""

    values = np.asarray(slice_data["values"])
    im = ax.imshow(
        values,
        origin="lower",
        extent=(-180, 180, -180, 180),
        cmap=NATURE_CMAP,
        norm=norm,
        interpolation="bicubic",
    )
    circle = plt.Circle((0, 0), 180, fill=False, color="#2b2b2b", lw=0.9, alpha=0.75)
    ax.add_patch(circle)
    ax.axhline(0, color="white", lw=0.45, alpha=0.50)
    ax.axvline(0, color="white", lw=0.45, alpha=0.50)
    ax.plot(0, 0, marker="o", ms=2.5, color="white", mec="#222222", mew=0.35)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=8, colors=EDGE_COLOR)
    ax.set_xlabel(xlabel, color=PANEL_COLOR, labelpad=1)
    ax.set_ylabel(ylabel, color=PANEL_COLOR, labelpad=1)
    ax.set_title(title, fontsize=10, color=PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.6)
    return im


def plot_axis_angle_figure(
    systems: Dict[str, Dict[str, object]],
    *,
    out_dir: Path,
    paper_fig_dir: Path | None = PAPER_FIG_DIR,
) -> Tuple[Path, Path]:
    """Create the standalone all-angle/all-axis visualization."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if paper_fig_dir is not None:
        paper_fig_dir.mkdir(parents=True, exist_ok=True)

    all_values = []
    for data in systems.values():
        all_values.append(np.asarray(data["cloud_values"]))
        for sl in data["slices"].values():
            all_values.append(np.asarray(sl["values"])[np.isfinite(sl["values"])])
    vmax = float(np.quantile(np.concatenate(all_values), 0.995))
    norm = Normalize(vmin=0.0, vmax=vmax)

    plt.rcParams.update(
        {
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": PANEL_COLOR,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(15.0, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        left=0.045,
        right=0.900,
        top=0.925,
        bottom=0.070,
        width_ratios=[1.14, 1.0, 1.0, 1.0],
        wspace=0.28,
        hspace=0.32,
    )

    plane_meta = [
        ("x=0", r"$\xi_x=0$ slice", r"$\xi_y$ (deg)", r"$\xi_z$ (deg)"),
        ("y=0", r"$\xi_y=0$ slice", r"$\xi_x$ (deg)", r"$\xi_z$ (deg)"),
        ("z=0", r"$\xi_z=0$ slice", r"$\xi_x$ (deg)", r"$\xi_y$ (deg)"),
    ]
    letters = iter("abcdefgh")
    row_info = [
        ("fcc", r"FCC / $O$: $1{\times}4e$"),
        ("hcp", r"HCP / $D_6$: $1{\times}2e+1{\times}4e+2{\times}6e$"),
    ]
    image_axes = []
    for row, (key, row_title) in enumerate(row_info):
        data = systems[key]
        ax_cloud = fig.add_subplot(gs[row, 0], projection="3d")
        panel_letter(ax_cloud, next(letters))
        plot_cloud(ax_cloud, data, norm=norm, title=row_title)

        for col, (plane, title, xlabel, ylabel) in enumerate(plane_meta, start=1):
            ax = fig.add_subplot(gs[row, col])
            panel_letter(ax, next(letters))
            plot_slice(
                ax,
                data["slices"][plane],
                norm=norm,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            image_axes.append(ax)

    sm = ScalarMappable(norm=norm, cmap=NATURE_CMAP)
    cax = fig.add_axes([0.920, 0.190, 0.018, 0.620])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\|f(R(\boldsymbol{\xi}))-f(I)\|/c$")

    png = out_dir / "axis_angle_embedding_ball_corrected_fcc_hcp.png"
    pdf = out_dir / "axis_angle_embedding_ball_corrected_fcc_hcp.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    if paper_fig_dir is not None:
        paper_png = paper_fig_dir / "figure_encoder_axis_angle_ball.png"
        paper_pdf = paper_fig_dir / "figure_encoder_axis_angle_ball.pdf"
        paper_png.write_bytes(png.read_bytes())
        paper_pdf.write_bytes(pdf.read_bytes())

    return png, pdf


def run_all(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    n_cloud: int = 18000,
    n_grid: int = 161,
    chunk_size: int = 8192,
) -> Dict[str, object]:
    """Compute and save the standalone axis-angle figure."""

    out_dir = Path(output_dir)
    systems = {
        "fcc": compute_system_data(CORRECT_FCC, n_cloud=n_cloud, n_grid=n_grid, seed=1100, chunk_size=chunk_size),
        "hcp": compute_system_data(CORRECT_HCP, n_cloud=n_cloud, n_grid=n_grid, seed=2200, chunk_size=chunk_size),
    }
    png, pdf = plot_axis_angle_figure(systems, out_dir=out_dir)

    npz_path = out_dir / "axis_angle_embedding_ball_corrected_fcc_hcp_data.npz"
    np.savez_compressed(
        npz_path,
        fcc_cloud_xi=systems["fcc"]["cloud_xi"],
        fcc_cloud_values=systems["fcc"]["cloud_values"],
        hcp_cloud_xi=systems["hcp"]["cloud_xi"],
        hcp_cloud_values=systems["hcp"]["cloud_values"],
        fcc_x0_slice=systems["fcc"]["slices"]["x=0"]["values"],
        fcc_y0_slice=systems["fcc"]["slices"]["y=0"]["values"],
        fcc_z0_slice=systems["fcc"]["slices"]["z=0"]["values"],
        hcp_x0_slice=systems["hcp"]["slices"]["x=0"]["values"],
        hcp_y0_slice=systems["hcp"]["slices"]["y=0"]["values"],
        hcp_z0_slice=systems["hcp"]["slices"]["z=0"]["values"],
    )
    summary = {
        "axis_angle_convention": "active: xi = theta * n, R(xi) acts on vectors",
        "distance": "||f(R(xi)) - f(I)|| / local_scale",
        "theta_range_deg": [0.0, 180.0],
        "boundary_note": "antipodal points on the 180 degree boundary represent the same SO(3) rotation",
        "n_cloud": int(n_cloud),
        "n_grid": int(n_grid),
        "fcc": {
            "label": systems["fcc"]["embedding_label"],
            "irreps_full": systems["fcc"]["irreps_full"],
            "irreps_a1": systems["fcc"]["irreps_a1"],
            "dim_a1": systems["fcc"]["dim_a1"],
            "local_scale": systems["fcc"]["local_scale"],
            "cloud_distance_range": [
                float(np.nanmin(systems["fcc"]["cloud_values"])),
                float(np.nanmax(systems["fcc"]["cloud_values"])),
            ],
        },
        "hcp": {
            "label": systems["hcp"]["embedding_label"],
            "irreps_full": systems["hcp"]["irreps_full"],
            "irreps_a1": systems["hcp"]["irreps_a1"],
            "dim_a1": systems["hcp"]["dim_a1"],
            "local_scale": systems["hcp"]["local_scale"],
            "cloud_distance_range": [
                float(np.nanmin(systems["hcp"]["cloud_values"])),
                float(np.nanmax(systems["hcp"]["cloud_values"])),
            ],
        },
    }
    summary_path = out_dir / "axis_angle_embedding_ball_corrected_fcc_hcp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    paper_png = PAPER_FIG_DIR / "figure_encoder_axis_angle_ball.png"
    paper_pdf = PAPER_FIG_DIR / "figure_encoder_axis_angle_ball.pdf"
    return {
        "output_dir": str(out_dir),
        "png": str(png),
        "pdf": str(pdf),
        "paper_png": str(paper_png),
        "paper_pdf": str(paper_pdf),
        "data_npz": str(npz_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }


if __name__ == "__main__":
    result = run_all()
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))
