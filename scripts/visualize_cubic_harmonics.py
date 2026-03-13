#!/usr/bin/env python3
"""Visualize cubic harmonics (FCC O-group invariant modes) on the sphere."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    make_subplots = None

try:
    from .plot_fcc_invariant_irreps import (
        _build_fcc_syms_inv_wxyz,
        _mode_field_on_sphere,
        compute_invariant_basis_Ul,
        plot_mode_field,
        plot_mode_lobe_3d,
    )
except ImportError:
    from plot_fcc_invariant_irreps import (
        _build_fcc_syms_inv_wxyz,
        _mode_field_on_sphere,
        compute_invariant_basis_Ul,
        plot_mode_field,
        plot_mode_lobe_3d,
    )


def _parse_ls(ls_text: str) -> tuple[int, ...]:
    ls = tuple(int(x.strip()) for x in ls_text.split(",") if x.strip())
    if not ls:
        raise ValueError("Ls cannot be empty.")
    if any(l < 0 for l in ls):
        raise ValueError(f"All l must be >= 0, got {ls}.")
    return ls


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ls",
        type=str,
        default="4,6,8,10,12",
        help="Comma-separated spherical-harmonic degrees (default: 4,6,8,10,12).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/cubic_harmonics"),
        help="Directory where PNG files are written.",
    )
    parser.add_argument("--n-theta", type=int, default=181, help="Theta grid size (0..pi).")
    parser.add_argument("--n-phi", type=int, default=361, help="Phi grid size (0..2pi).")
    parser.add_argument(
        "--max-modes-per-l",
        type=int,
        default=-1,
        help="Limit number of modes per l; -1 means all modes.",
    )
    parser.add_argument("--lobe-elev", type=float, default=25.0, help="3D lobe elevation in degrees.")
    parser.add_argument("--lobe-azim", type=float, default=35.0, help="3D lobe azimuth in degrees.")
    parser.add_argument("--eig-tol", type=float, default=1e-5, help="Invariant-space eigen tolerance.")
    parser.add_argument("--rel-tol", type=float, default=1e-8, help="Fallback SVD relative tolerance.")
    parser.add_argument("--abs-tol", type=float, default=1e-6, help="Fallback SVD absolute tolerance.")
    parser.add_argument("--skip-heatmap", action="store_true", help="Skip 2D theta/phi heatmap output.")
    parser.add_argument("--skip-lobe3d", action="store_true", help="Skip 3D lobe output.")
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Skip interactive HTML output.",
    )
    parser.add_argument(
        "--interactive-html",
        type=str,
        default="cubic_harmonics_interactive.html",
        help="Interactive HTML filename (placed in --out-dir unless absolute).",
    )
    parser.add_argument(
        "--skip-direct-sum",
        action="store_true",
        help="Skip direct-sum composite mode generation.",
    )
    parser.add_argument(
        "--animate-symmetry",
        action="store_true",
        help="Write GIF animations that apply all FCC O-group symmetry operations.",
    )
    parser.add_argument(
        "--anim-n-theta",
        type=int,
        default=49,
        help="Theta grid size for animation surfaces (smaller is faster).",
    )
    parser.add_argument(
        "--anim-n-phi",
        type=int,
        default=97,
        help="Phi grid size for animation surfaces (smaller is faster).",
    )
    parser.add_argument(
        "--anim-fps",
        type=int,
        default=2,
        help="Animation GIF frames per second.",
    )
    parser.add_argument(
        "--anim-out-dir",
        type=Path,
        default=None,
        help="Animation output dir (defaults to --out-dir).",
    )
    return parser.parse_args()


def _maxabs(x: np.ndarray, eps: float = 1e-12) -> float:
    return max(float(np.max(np.abs(x))), eps)


def _make_direct_sum_field(named_fields: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
    acc = np.zeros_like(named_fields[0][1], dtype=np.float64)
    for _, field in named_fields:
        acc = acc + (field / _maxabs(field))
    return acc / float(len(named_fields))


def _field_to_surface(field: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vmax = _maxabs(field)
    r = np.abs(field) / vmax
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z, vmax


def _build_interactive_html(
    named_fields: Sequence[Tuple[str, np.ndarray]],
    theta: np.ndarray,
    phi: np.ndarray,
    out_path: Path,
) -> None:
    if go is None or make_subplots is None:
        print("plotly not available; skipping interactive HTML output.")
        return

    if len(named_fields) == 0:
        raise ValueError("No fields to plot interactively.")

    phi_deg = np.degrees(phi[0, :])
    theta_deg = np.degrees(theta[:, 0])
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "heatmap"}]],
        column_widths=[0.62, 0.38],
        subplot_titles=("3D lobe (rotate/zoom)", "theta/phi heatmap"),
        horizontal_spacing=0.06,
    )

    labels = [name for name, _ in named_fields]
    n = len(named_fields)

    for idx, (label, field) in enumerate(named_fields):
        x, y, z, vmax = _field_to_surface(field, theta, phi)
        is_visible = idx == 0

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=field,
                cmin=-vmax,
                cmax=vmax,
                colorscale="RdBu",
                reversescale=True,
                showscale=True,
                colorbar={"title": "amp", "x": 0.44, "len": 0.78},
                visible=is_visible,
                name=label,
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                    "<br>amp=%{surfacecolor:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=phi_deg,
                y=theta_deg,
                z=field,
                zmin=-vmax,
                zmax=vmax,
                colorscale="RdBu",
                reversescale=True,
                showscale=False,
                visible=is_visible,
                name=f"{label} heat",
                hovertemplate=(
                    "phi=%{x:.1f} deg<br>theta=%{y:.1f} deg"
                    "<br>amp=%{z:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    buttons = []
    for idx, label in enumerate(labels):
        visible = [False] * (2 * n)
        visible[2 * idx] = True
        visible[2 * idx + 1] = True
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}, {"title": f"FCC O-invariant cubic harmonics: {label}"}],
            }
        )

    fig.update_layout(
        title=f"FCC O-invariant cubic harmonics: {labels[0]}",
        height=760,
        width=1450,
        margin={"l": 10, "r": 10, "b": 10, "t": 45},
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
        },
        xaxis2={"title": "phi (deg)", "range": [0.0, 360.0]},
        yaxis2={"title": "theta (deg)", "range": [0.0, 180.0]},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 1.09,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
                "buttons": buttons,
            }
        ],
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 1.145,
                "text": "mode:",
                "showarrow": False,
                "font": {"size": 12},
            }
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    print(f"Saved interactive HTML: {out_path}")


def _quat_wxyz_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _build_o_group_rotation_mats_active_np() -> np.ndarray:
    syms_bunge_inv = _build_fcc_syms_inv_wxyz(dtype=torch.float64).detach().cpu().numpy()
    syms_active = syms_bunge_inv.copy()
    syms_active[:, 1:] *= -1.0
    syms_active /= np.linalg.norm(syms_active, axis=-1, keepdims=True).clip(min=1e-12)
    mats = np.stack([_quat_wxyz_to_rotmat_np(q) for q in syms_active], axis=0)
    return mats


def _angles_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)
    return theta, phi


def _build_rotated_angle_grids(theta: np.ndarray, phi: np.ndarray, sym_mats: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    dirs = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )
    dirs_flat = dirs.reshape(-1, 3)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for R in sym_mats:
        # Active rotation on object: f_R(n) = f(R^{-1} n). With row-vectors, R^{-1} n => n @ R.
        dirs_in = dirs_flat @ R
        theta_in, phi_in = _angles_from_xyz(dirs_in.reshape(dirs.shape))
        out.append((theta_in, phi_in))
    return out


def _evaluate_mode_field(coeff_col: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    coeff = coeff_col
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    return _mode_field_on_sphere(coeff, 0, theta, phi)


def _operation_label(k: int, R: np.ndarray) -> str:
    trace = float(np.trace(R))
    cos_ang = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = math.degrees(math.acos(cos_ang))
    return f"op {k:02d} | angle={angle:6.2f} deg"


def _safe_label(label: str) -> str:
    return (
        label.replace(" ", "_")
        .replace(",", "")
        .replace("=", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _write_symmetry_animation(
    label: str,
    base_field: np.ndarray,
    transformed_fields: Sequence[np.ndarray],
    sym_mats: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    out_path: Path,
    fps: int,
    elev: float,
    azim: float,
) -> None:
    if len(transformed_fields) != int(sym_mats.shape[0]):
        raise ValueError("transformed_fields length must match number of symmetry matrices.")

    vmax = max(_maxabs(base_field), max(_maxabs(f) for f in transformed_fields))
    base_r = np.abs(base_field) / vmax
    xb = base_r * np.sin(theta) * np.cos(phi)
    yb = base_r * np.sin(theta) * np.sin(phi)
    zb = base_r * np.cos(theta)
    base_facecolors = plt.cm.coolwarm(plt.Normalize(vmin=-vmax, vmax=vmax)(base_field))

    lim = 1.05
    diff_vmax = max(_maxabs(f - base_field) for f in transformed_fields)
    if diff_vmax <= 0.0:
        diff_vmax = 1e-12

    fig = plt.figure(figsize=(11.5, 5.0), dpi=130)
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    diff_im = ax2d.imshow(
        np.zeros_like(base_field),
        origin="lower",
        cmap="coolwarm",
        vmin=-diff_vmax,
        vmax=diff_vmax,
        aspect="auto",
        extent=[0.0, 360.0, 0.0, 180.0],
    )
    cbar = fig.colorbar(diff_im, ax=ax2d, shrink=0.90)
    cbar.set_label("delta amplitude")
    ax2d.set_xlabel("phi (deg)")
    ax2d.set_ylabel("theta (deg)")
    ax2d.set_title("Transformed - original")
    info = ax2d.text(
        0.02,
        1.02,
        "",
        transform=ax2d.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )

    def _draw_3d(k: int) -> None:
        ax3d.cla()
        ft = transformed_fields[k]
        rt = np.abs(ft) / vmax
        xt = rt * np.sin(theta) * np.cos(phi)
        yt = rt * np.sin(theta) * np.sin(phi)
        zt = rt * np.cos(theta)

        ax3d.plot_surface(
            xb,
            yb,
            zb,
            facecolors=base_facecolors,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.48,
        )
        ax3d.plot_surface(
            xt,
            yt,
            zt,
            color="#ff8c00",
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.26,
        )
        ax3d.plot_wireframe(
            xt,
            yt,
            zt,
            rstride=2,
            cstride=2,
            color="black",
            linewidth=0.28,
            alpha=0.78,
        )

        # Faint fixed axes and bold transformed axes show the current operation orientation.
        ax3d.quiver(0, 0, 0, 1.05, 0, 0, color="#cc0000", alpha=0.20)
        ax3d.quiver(0, 0, 0, 0, 1.05, 0, color="#006600", alpha=0.20)
        ax3d.quiver(0, 0, 0, 0, 0, 1.05, color="#0033cc", alpha=0.20)
        R = sym_mats[k]
        ex = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ey = R @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ez = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ax3d.quiver(0, 0, 0, ex[0], ex[1], ex[2], color="#cc0000", linewidth=1.8)
        ax3d.quiver(0, 0, 0, ey[0], ey[1], ey[2], color="#006600", linewidth=1.8)
        ax3d.quiver(0, 0, 0, ez[0], ez[1], ez[2], color="#0033cc", linewidth=1.8)

        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_axis_off()
        ax3d.set_title(f"{label} | {_operation_label(k, R)}", pad=8.0)

    def _update(k: int):
        _draw_3d(k)
        diff = transformed_fields[k] - base_field
        diff_im.set_data(diff)
        resid = float(np.max(np.abs(diff)))
        info.set_text(f"max|delta f| = {resid:.3e}")
        return (diff_im, info)

    _update(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(transformed_fields),
        interval=1000.0 / float(max(int(fps), 1)),
        blit=False,
        repeat=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=animation.PillowWriter(fps=max(int(fps), 1)))
    plt.close(fig)
    print(f"Saved symmetry animation: {out_path}")


def main() -> None:
    args = _parse_args()
    ls = _parse_ls(args.ls)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    syms_inv = _build_fcc_syms_inv_wxyz(dtype=torch.float64)

    theta_1d = np.linspace(0.0, math.pi, int(args.n_theta), dtype=np.float64)
    phi_1d = np.linspace(0.0, 2.0 * math.pi, int(args.n_phi), dtype=np.float64)
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")

    named_fields: List[Tuple[str, np.ndarray]] = []
    mode_coeffs: List[Tuple[str, np.ndarray]] = []

    for l in ls:
        U_l = compute_invariant_basis_Ul(
            l=l,
            sym_quats_wxyz_bunge_inv=syms_inv,
            eig_tol=float(args.eig_tol),
            rel_tol=float(args.rel_tol),
            abs_tol=float(args.abs_tol),
        )
        U_np = U_l.detach().cpu().numpy()

        rank = int(U_np.shape[-1])
        n_modes = rank if int(args.max_modes_per_l) < 0 else min(rank, int(args.max_modes_per_l))
        print(f"l={l}: rank(U_l)={rank}, plotting {n_modes} mode(s)")

        for mode_idx in range(n_modes):
            label = f"l={l}, k={mode_idx}"
            field = _mode_field_on_sphere(U_np, mode_idx, theta, phi)
            named_fields.append((label, field))
            mode_coeffs.append((label, U_np[:, mode_idx : mode_idx + 1]))
            if not bool(args.skip_heatmap):
                heatmap_path = out_dir / f"cubic_harmonic_l{l:02d}_k{mode_idx:02d}.png"
                plot_mode_field(field=field, l=l, mode_idx=mode_idx, out_path=heatmap_path)
            if not bool(args.skip_lobe3d):
                lobe_path = out_dir / f"cubic_harmonic_l{l:02d}_k{mode_idx:02d}_lobe3d.png"
                plot_mode_lobe_3d(
                    field=field,
                    theta=theta,
                    phi=phi,
                    l=l,
                    mode_idx=mode_idx,
                    out_path=lobe_path,
                    elev=float(args.lobe_elev),
                    azim=float(args.lobe_azim),
                )

    direct_sum_field: np.ndarray | None = None
    if not bool(args.skip_direct_sum):
        if len(named_fields) == 0:
            raise RuntimeError("No invariant modes available for direct-sum field.")
        direct_sum_field = _make_direct_sum_field(named_fields)
        named_fields.append(("direct_sum_equal_weight", direct_sum_field))
        if not bool(args.skip_heatmap):
            plot_mode_field(
                field=direct_sum_field,
                l=-1,
                mode_idx=0,
                out_path=out_dir / "cubic_harmonic_direct_sum.png",
            )
        if not bool(args.skip_lobe3d):
            plot_mode_lobe_3d(
                field=direct_sum_field,
                theta=theta,
                phi=phi,
                l=-1,
                mode_idx=0,
                out_path=out_dir / "cubic_harmonic_direct_sum_lobe3d.png",
                elev=float(args.lobe_elev),
                azim=float(args.lobe_azim),
            )

    if not bool(args.skip_interactive):
        html_arg = Path(args.interactive_html)
        html_path = html_arg if html_arg.is_absolute() else (out_dir / html_arg)
        _build_interactive_html(named_fields=named_fields, theta=theta, phi=phi, out_path=html_path)

    if bool(args.animate_symmetry):
        if len(mode_coeffs) == 0:
            raise RuntimeError("No invariant modes available for symmetry animation.")

        anim_out_dir: Path = out_dir if args.anim_out_dir is None else args.anim_out_dir
        anim_out_dir.mkdir(parents=True, exist_ok=True)

        theta_anim_1d = np.linspace(0.0, math.pi, int(args.anim_n_theta), dtype=np.float64)
        phi_anim_1d = np.linspace(0.0, 2.0 * math.pi, int(args.anim_n_phi), dtype=np.float64)
        theta_anim, phi_anim = np.meshgrid(theta_anim_1d, phi_anim_1d, indexing="ij")

        sym_mats = _build_o_group_rotation_mats_active_np()
        rotated_angle_grids = _build_rotated_angle_grids(theta_anim, phi_anim, sym_mats)

        per_mode_anim: List[Tuple[str, np.ndarray, List[np.ndarray], float]] = []
        for label, coeff_col in mode_coeffs:
            base_anim = _evaluate_mode_field(coeff_col, theta_anim, phi_anim)
            transformed_anim = [
                _evaluate_mode_field(coeff_col, theta_in, phi_in)
                for theta_in, phi_in in rotated_angle_grids
            ]
            scale = _maxabs(base_anim)
            per_mode_anim.append((label, base_anim, transformed_anim, scale))
            gif_path = anim_out_dir / f"cubic_harmonic_{_safe_label(label)}_symmetry_O.gif"
            _write_symmetry_animation(
                label=label,
                base_field=base_anim,
                transformed_fields=transformed_anim,
                sym_mats=sym_mats,
                theta=theta_anim,
                phi=phi_anim,
                out_path=gif_path,
                fps=int(args.anim_fps),
                elev=float(args.lobe_elev),
                azim=float(args.lobe_azim),
            )

        if not bool(args.skip_direct_sum):
            if len(per_mode_anim) == 0:
                raise RuntimeError("No invariant modes available for direct-sum animation.")
            n_ops = int(sym_mats.shape[0])
            direct_base = np.zeros_like(per_mode_anim[0][1], dtype=np.float64)
            direct_transformed = [
                np.zeros_like(per_mode_anim[0][1], dtype=np.float64) for _ in range(n_ops)
            ]
            for _, base_anim, transformed_anim, scale in per_mode_anim:
                direct_base += base_anim / scale
                for k in range(n_ops):
                    direct_transformed[k] += transformed_anim[k] / scale

            inv_count = 1.0 / float(len(per_mode_anim))
            direct_base *= inv_count
            direct_transformed = [f * inv_count for f in direct_transformed]

            direct_gif_path = anim_out_dir / "cubic_harmonic_direct_sum_symmetry_O.gif"
            _write_symmetry_animation(
                label="direct_sum_equal_weight",
                base_field=direct_base,
                transformed_fields=direct_transformed,
                sym_mats=sym_mats,
                theta=theta_anim,
                phi=phi_anim,
                out_path=direct_gif_path,
                fps=int(args.anim_fps),
                elev=float(args.lobe_elev),
                azim=float(args.lobe_azim),
            )

    print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
