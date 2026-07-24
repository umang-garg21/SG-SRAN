"""
Interactive single-quaternion FCC orbit viewer.

Panels:
1) upper stereographic,
2) lower stereographic,
3) 3D axis-angle plot,
4) IPF key.

Fixed symmetry convention (repo default):
    q' = s^{-1} ⊗ q
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY"):
    matplotlib.use("WebAgg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button, RadioButtons, Slider
import orix.plot  # registers stereographic/ipf projections
from orix.projections.stereographic import _vector2xy
from orix.quaternion import Orientation
from orix.quaternion.orientation_region import OrientationRegion
from orix.vector import Vector3d

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcc_syms_inv import build_fcc_syms, build_fcc_syms_inv
from utils.symmetry_utils import quaternion_left_matrix
from utils.quat_ops import quat_left_multiply_numpy
from utils.symmetry_utils import resolve_symmetry

DIR_VECTORS = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _normalize_quat(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < eps:
        raise ValueError("Quaternion norm is too small to normalize.")
    return (q / n).astype(np.float32, copy=False)


def _single_angle_deg(q_wxyz: np.ndarray) -> float:
    q = _normalize_quat(q_wxyz)
    w = float(np.clip(abs(q[0]), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def _stereo_xy(
    ori: Orientation, v_ref: Vector3d, pole: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = ori * v_ref
    x, y = _vector2xy(v, pole=pole)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(v.z).reshape(-1)
    vis = z >= 0 if pole == -1 else z <= 0
    return x, y, vis


def _draw_cubic_guides(ax) -> None:
    v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
    v2fold = Vector3d(
        [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [1, 1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            [1, -1, 0],
        ]
    )
    ax.draw_circle(v4fold, color="black", linewidth=0.8, alpha=0.25)
    ax.draw_circle(v3fold, color="black", linewidth=0.8, alpha=0.25)
    ax.draw_circle(v2fold, color="black", linewidth=0.8, alpha=0.25)
    ax.set_labels("RD", "TD", None)
    ax.show_hemisphere_label()


def _build_orbit(q_wxyz: np.ndarray, sym_ops_wxyz: np.ndarray) -> np.ndarray:
    """Fixed Bunge/passive symmetry action: q' = s^{-1} ⊗ q."""
    n = int(sym_ops_wxyz.shape[0])
    q_single = np.asarray(q_wxyz, dtype=np.float32).reshape(1, 4)
    ops_inv = _quat_conjugate(np.asarray(sym_ops_wxyz, dtype=np.float32))
    orbit = quat_left_multiply_numpy(q_single, ops_inv, layout="quat_last")
    return orbit.reshape(n, 4).astype(np.float32, copy=False)


def _compute_plot_data(
    q_wxyz: np.ndarray,
    sym_obj,
    sym_ops: np.ndarray,
    ref_dir: str,
) -> dict[str, np.ndarray | Orientation]:
    orbit = _build_orbit(q_wxyz, sym_ops)
    v_ref = DIR_VECTORS[ref_dir]

    ori_single = Orientation(q_wxyz.reshape(1, 4), symmetry=sym_obj)
    ori_orbit = Orientation(orbit, symmetry=sym_obj)

    ckey = orix.plot.IPFColorKeyTSL(sym_obj.laue)
    ckey.direction = v_ref
    color_single = np.asarray(ckey.orientation2color(ori_single), dtype=np.float32).reshape(1, 3)
    color_orbit = np.asarray(ckey.orientation2color(ori_orbit), dtype=np.float32).reshape(-1, 3)

    xs_u, ys_u, vis_s_u = _stereo_xy(ori_single, v_ref, pole=-1)
    xs_l, ys_l, vis_s_l = _stereo_xy(ori_single, v_ref, pole=1)
    xo_u, yo_u, vis_o_u = _stereo_xy(ori_orbit, v_ref, pole=-1)
    xo_l, yo_l, vis_o_l = _stereo_xy(ori_orbit, v_ref, pole=1)

    return {
        "ori_single": ori_single,
        "ori_orbit": ori_orbit,
        "x_single_upper": xs_u,
        "y_single_upper": ys_u,
        "vis_single_upper": vis_s_u,
        "x_single_lower": xs_l,
        "y_single_lower": ys_l,
        "vis_single_lower": vis_s_l,
        "x_orbit_upper": xo_u,
        "y_orbit_upper": yo_u,
        "vis_orbit_upper": vis_o_u,
        "x_orbit_lower": xo_l,
        "y_orbit_lower": yo_l,
        "vis_orbit_lower": vis_o_l,
        "color_single": color_single,
        "color_orbit": color_orbit,
    }


def _apply_reynolds(q_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (reynolds_orbit [24,4], q_reynolds_normalized [4]) via s_g ⊗ q."""
    fcc_ops = build_fcc_syms().numpy()  # (24, 4)
    # Build 4x4 left-multiply matrices for each op and apply to q as a column vec
    q_col = q_norm.reshape(4, 1).astype(np.float64)
    orbit = np.stack([
        (quaternion_left_matrix(fcc_ops[g].astype(np.float64)) @ q_col).reshape(4)
        for g in range(len(fcc_ops))
    ], axis=0).astype(np.float32)  # (24, 4)
    q_avg = orbit.mean(axis=0)  # Reynolds average
    norm = float(np.linalg.norm(q_avg))
    q_reynolds = (q_avg / norm).astype(np.float32) if norm > 1e-9 else q_avg
    return orbit, q_reynolds


def _plot_q_vs_fcc_syms_inv(q_norm: np.ndarray, sym_obj, ref_dir: str) -> None:
    """Three-panel figure: input q | fcc_syms_inv ⊗ q (24) | Reynolds s_g ⊗ q + average."""
    fcc_inv_ops = build_fcc_syms_inv().numpy()  # (24, 4) [w x y z]
    fcc_inv_orbit = quat_left_multiply_numpy(
        q_norm.reshape(1, 4), fcc_inv_ops, layout="quat_last"
    ).reshape(24, 4).astype(np.float32)

    reynolds_orbit, q_reynolds = _apply_reynolds(q_norm)

    print(f"\nOriginal q [w x y z]: {q_norm.tolist()}")
    print(f"\nfcc_syms_inv ⊗ q  ({len(fcc_inv_ops)} results):")
    for i, r in enumerate(fcc_inv_orbit):
        print(f"  [{i:2d}] {r.tolist()}")
    print(f"\nReynolds orbit  s_g ⊗ q  (24 results, group = fcc_syms):")
    for i, r in enumerate(reynolds_orbit):
        print(f"  [{i:2d}] {r.tolist()}")
    print(f"\nReynolds mean (normalized) [w x y z]: {q_reynolds.tolist()}")

    v_ref = DIR_VECTORS[ref_dir]
    ori_q = Orientation(q_norm.reshape(1, 4), symmetry=sym_obj)
    ori_fcc = Orientation(fcc_inv_orbit, symmetry=sym_obj)
    ori_reyn_orbit = Orientation(reynolds_orbit, symmetry=sym_obj)
    ori_reyn = Orientation(q_reynolds.reshape(1, 4), symmetry=sym_obj)

    ckey = orix.plot.IPFColorKeyTSL(sym_obj.laue)
    ckey.direction = v_ref
    c_q = np.asarray(ckey.orientation2color(ori_q), dtype=np.float32).reshape(1, 3)
    c_fcc = np.asarray(ckey.orientation2color(ori_fcc), dtype=np.float32).reshape(24, 3)
    c_reyn_orbit = np.asarray(ckey.orientation2color(ori_reyn_orbit), dtype=np.float32).reshape(24, 3)
    c_reyn = np.asarray(ckey.orientation2color(ori_reyn), dtype=np.float32).reshape(1, 3)

    fig2, (ax_q, ax_fcc, ax_reyn) = plt.subplots(
        1, 3, subplot_kw={"projection": "stereographic"}, figsize=(15, 5)
    )
    for ax in (ax_q, ax_fcc, ax_reyn):
        _draw_cubic_guides(ax)

    # --- panel 1: input q ---
    xs, ys, vis_q = _stereo_xy(ori_q, v_ref, pole=-1)
    Axes.scatter(
        ax_q, xs[vis_q], ys[vis_q],
        s=180, marker="*", c=c_q[vis_q],
        edgecolors="black", linewidths=0.8, zorder=5,
    )
    ax_q.set_title(
        f"Input q\n[{q_norm[0]:.5f}, {q_norm[1]:.5f}, {q_norm[2]:.5f}, {q_norm[3]:.5f}]",
        fontsize=9,
    )

    # --- panel 2: fcc_syms_inv ⊗ q ---
    xf, yf, vis_f = _stereo_xy(ori_fcc, v_ref, pole=-1)
    Axes.scatter(
        ax_fcc, xf[vis_f], yf[vis_f],
        s=40, c=c_fcc[vis_f],
        edgecolors="none", alpha=0.9, zorder=3,
    )
    ax_fcc.set_title("fcc_syms_inv ⊗ q  (24, upper hemi)", fontsize=9)

    # --- panel 3: Reynolds orbit + averaged projection ---
    xr, yr, vis_r = _stereo_xy(ori_reyn_orbit, v_ref, pole=-1)
    Axes.scatter(
        ax_reyn, xr[vis_r], yr[vis_r],
        s=40, c=c_reyn_orbit[vis_r],
        edgecolors="none", alpha=0.7, zorder=3, label="s_g ⊗ q",
    )
    xrm, yrm, vis_rm = _stereo_xy(ori_reyn, v_ref, pole=-1)
    Axes.scatter(
        ax_reyn, xrm[vis_rm], yrm[vis_rm],
        s=180, marker="D", c=c_reyn[vis_rm],
        edgecolors="black", linewidths=0.9, zorder=5, label="Reynolds mean",
    )
    ax_reyn.set_title(
        f"Reynolds: s_g ⊗ q  +  mean (◆)\n"
        f"[{q_reynolds[0]:.5f}, {q_reynolds[1]:.5f}, {q_reynolds[2]:.5f}, {q_reynolds[3]:.5f}]",
        fontsize=9,
    )

    fig2.suptitle(
        f"Input q  |  fcc_syms_inv ⊗ q  |  Reynolds operator (s_g ⊗ q)  —  ref-dir: {ref_dir}",
        fontsize=11,
    )
    fig2.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive single-quaternion FCC orbit viewer (upper/lower stereographic + axis-angle + IPF key)."
    )
    parser.add_argument(
        "--quat",
        type=float,
        nargs=4,
        default=[0.9238795, 0.3826834, 0.0, 0.0],
        metavar=("W", "X", "Y", "Z"),
        help="Input quaternion [w x y z].",
    )
    parser.add_argument(
        "--sym",
        type=str,
        default="O",
        help="Symmetry group for orbit ops. Default 'O' (proper cubic, 24 ops).",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        choices=("X", "Y", "Z"),
        default="Z",
        help="Reference direction for IPF coloring/projection.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate orbit points with symmetry-op index.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/single_quat_fcc_stereo.png",
        help="Output PNG path for Save button and --save-only mode.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Render once and save PNG without interactive controls.",
    )

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args()


def _offsets(x: np.ndarray, y: np.ndarray, vis: np.ndarray) -> np.ndarray:
    if np.count_nonzero(vis) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([x[vis], y[vis]]).astype(np.float32, copy=False)


def _masked_colors(rgb: np.ndarray, vis: np.ndarray) -> np.ndarray:
    if np.count_nonzero(vis) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(rgb[vis], dtype=np.float32)


def _save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")


def main() -> None:
    args = parse_args()

    sym_obj = resolve_symmetry(args.sym)
    sym_ops = np.asarray(sym_obj.data, dtype=np.float32)

    q_raw = np.asarray(args.quat, dtype=np.float64)
    q_norm = _normalize_quat(q_raw)

    state = {
        "ref_dir": args.ref_dir,
        "annotate": bool(args.annotate),
        "q_raw": q_raw.copy(),
        "q_norm": q_norm.copy(),
    }

    data = _compute_plot_data(
        state["q_norm"],
        sym_obj,
        sym_ops,
        state["ref_dir"],
    )

    _plot_q_vs_fcc_syms_inv(state["q_norm"], sym_obj, state["ref_dir"])

    fig = plt.figure(figsize=(17.2, 8.8), dpi=170)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[4.2, 1.8],
        width_ratios=[1.0, 1.0, 1.0, 0.82],
        hspace=0.30,
        wspace=0.22,
    )
    ax_upper = fig.add_subplot(gs[0, 0], projection="stereographic")
    ax_lower = fig.add_subplot(gs[0, 1], projection="stereographic")
    ax_axis = fig.add_subplot(gs[0, 2], projection="axangle")
    ax_key = fig.add_subplot(gs[0, 3], projection="ipf", symmetry=sym_obj.laue)
    ax_key.plot_ipf_color_key(show_title=False)
    fz_region = OrientationRegion.from_symmetry(sym_obj)
    ax_axis.plot_wireframe(
        fz_region,
        color="black",
        linewidth=0.6,
        alpha=0.25,
    )

    _draw_cubic_guides(ax_upper)
    _draw_cubic_guides(ax_lower)

    art_orbit_up = Axes.scatter(
        ax_upper,
        data["x_orbit_upper"][data["vis_orbit_upper"]],
        data["y_orbit_upper"][data["vis_orbit_upper"]],
        s=32,
        c=data["color_orbit"][data["vis_orbit_upper"]],
        edgecolors="none",
        alpha=0.9,
        zorder=3,
        label="FCC orbit",
    )
    art_single_up = Axes.scatter(
        ax_upper,
        data["x_single_upper"][data["vis_single_upper"]],
        data["y_single_upper"][data["vis_single_upper"]],
        s=140,
        marker="*",
        c=data["color_single"][data["vis_single_upper"]],
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="input q",
    )
    art_orbit_lo = Axes.scatter(
        ax_lower,
        data["x_orbit_lower"][data["vis_orbit_lower"]],
        data["y_orbit_lower"][data["vis_orbit_lower"]],
        s=32,
        c=data["color_orbit"][data["vis_orbit_lower"]],
        edgecolors="none",
        alpha=0.9,
        zorder=3,
        label="FCC orbit",
    )
    art_single_lo = Axes.scatter(
        ax_lower,
        data["x_single_lower"][data["vis_single_lower"]],
        data["y_single_lower"][data["vis_single_lower"]],
        s=140,
        marker="*",
        c=data["color_single"][data["vis_single_lower"]],
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="input q",
    )

    axis_art = {"orbit": None, "single": None, "text": None}
    labels_upper: list = []
    labels_lower: list = []

    def _clear_labels() -> None:
        while labels_upper:
            labels_upper.pop().remove()
        while labels_lower:
            labels_lower.pop().remove()

    def _draw_labels() -> None:
        if not state["annotate"]:
            return
        n = int(data["x_orbit_upper"].shape[0])
        for i in range(n):
            if data["vis_orbit_upper"][i]:
                labels_upper.append(
                    ax_upper.text(
                        float(data["x_orbit_upper"][i]),
                        float(data["y_orbit_upper"][i]),
                        str(i),
                        fontsize=6,
                        ha="left",
                        va="bottom",
                        color="black",
                    )
                )
            if data["vis_orbit_lower"][i]:
                labels_lower.append(
                    ax_lower.text(
                        float(data["x_orbit_lower"][i]),
                        float(data["y_orbit_lower"][i]),
                        str(i),
                        fontsize=6,
                        ha="left",
                        va="bottom",
                        color="black",
                    )
                )

    def _draw_axis_angle() -> None:
        if axis_art["orbit"] is not None:
            axis_art["orbit"].remove()
        if axis_art["single"] is not None:
            axis_art["single"].remove()
        if axis_art["text"] is not None:
            axis_art["text"].remove()

        axis_art["orbit"] = ax_axis.scatter(
            data["ori_orbit"],
            s=26,
            c=data["color_orbit"],
            alpha=0.9,
            edgecolors="none",
            label="orbit axes",
        )
        axis_art["single"] = ax_axis.scatter(
            data["ori_single"],
            s=140,
            marker="*",
            c=data["color_single"],
            edgecolors="black",
            linewidths=0.8,
            label="input axis",
        )
        ang_deg = _single_angle_deg(state["q_norm"])
        axis_art["text"] = ax_axis.text2D(
            0.03,
            0.96,
            f"single angle = {ang_deg:.2f} deg",
            transform=ax_axis.transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )

    def _update_titles() -> None:
        qn = state["q_norm"]
        up_vis = int(np.count_nonzero(data["vis_orbit_upper"]))
        lo_vis = int(np.count_nonzero(data["vis_orbit_lower"]))
        n = int(data["vis_orbit_upper"].size)
        ax_upper.set_title(f"Upper stereographic (IPF-{state['ref_dir']})  visible={up_vis}/{n}", fontsize=10)
        ax_lower.set_title(
            f"Lower stereographic (IPF-{state['ref_dir']})\n"
            f"q=[{qn[0]:.5f}, {qn[1]:.5f}, {qn[2]:.5f}, {qn[3]:.5f}]  visible={lo_vis}/{n}",
            fontsize=10,
        )
        ax_axis.set_title("Orix axis-angle plot (fundamental-zone wiregrid)", fontsize=10)
        ax_key.set_title(f"IPF key (color reference: {state['ref_dir']})", fontsize=10)

    def _refresh_xy_artists() -> None:
        art_orbit_up.set_offsets(_offsets(data["x_orbit_upper"], data["y_orbit_upper"], data["vis_orbit_upper"]))
        art_orbit_up.set_facecolors(_masked_colors(data["color_orbit"], data["vis_orbit_upper"]))
        art_orbit_lo.set_offsets(_offsets(data["x_orbit_lower"], data["y_orbit_lower"], data["vis_orbit_lower"]))
        art_orbit_lo.set_facecolors(_masked_colors(data["color_orbit"], data["vis_orbit_lower"]))
        art_single_up.set_offsets(_offsets(data["x_single_upper"], data["y_single_upper"], data["vis_single_upper"]))
        art_single_up.set_facecolors(_masked_colors(data["color_single"], data["vis_single_upper"]))
        art_single_lo.set_offsets(_offsets(data["x_single_lower"], data["y_single_lower"], data["vis_single_lower"]))
        art_single_lo.set_facecolors(_masked_colors(data["color_single"], data["vis_single_lower"]))

    ax_axis.grid(True, alpha=0.3)

    _draw_labels()
    _draw_axis_angle()
    _update_titles()
    ax_lower.legend(loc="best", fontsize=8)
    fig.suptitle("Single Quaternion FCC Orbit: Upper/Lower + 3D Axis-Angle + IPF Key", fontsize=12)

    if args.save_only:
        _save_figure(fig, Path(args.out).expanduser())
        plt.close(fig)
        print(f"normalized quaternion [w x y z]: {state['q_norm'].tolist()}")
        print(f"symmetry group: {sym_obj.name}, operators: {sym_ops.shape[0]}")
        print(f"reference direction: {state['ref_dir']}")
        return

    # Controls
    slider_w_ax = fig.add_axes([0.07, 0.23, 0.36, 0.026])
    slider_x_ax = fig.add_axes([0.07, 0.19, 0.36, 0.026])
    slider_y_ax = fig.add_axes([0.07, 0.15, 0.36, 0.026])
    slider_z_ax = fig.add_axes([0.07, 0.11, 0.36, 0.026])
    slider_w = Slider(slider_w_ax, "w", -1.0, 1.0, valinit=float(state["q_raw"][0]))
    slider_x = Slider(slider_x_ax, "x", -1.0, 1.0, valinit=float(state["q_raw"][1]))
    slider_y = Slider(slider_y_ax, "y", -1.0, 1.0, valinit=float(state["q_raw"][2]))
    slider_z = Slider(slider_z_ax, "z", -1.0, 1.0, valinit=float(state["q_raw"][3]))

    ax_ref = fig.add_axes([0.47, 0.11, 0.08, 0.14])
    rb_ref = RadioButtons(ax_ref, ("X", "Y", "Z"), active=("X", "Y", "Z").index(state["ref_dir"]))
    ax_reset = fig.add_axes([0.58, 0.23, 0.10, 0.034])
    ax_save = fig.add_axes([0.58, 0.18, 0.10, 0.034])
    ax_annot = fig.add_axes([0.58, 0.13, 0.10, 0.034])
    btn_reset = Button(ax_reset, "Reset q")
    btn_save = Button(ax_save, "Save PNG")
    btn_annot = Button(ax_annot, "Annotate")

    info_text = fig.text(
        0.07,
        0.05,
        "Fixed symmetry action q' = s^{-1}⊗q. Controls: sliders, ref-dir, annotate, save.",
        fontsize=9,
    )

    def _recompute_and_redraw() -> None:
        nonlocal data
        try:
            state["q_norm"] = _normalize_quat(state["q_raw"])
        except ValueError:
            return
        data = _compute_plot_data(
            state["q_norm"],
            sym_obj,
            sym_ops,
            state["ref_dir"],
        )
        _refresh_xy_artists()
        _draw_axis_angle()
        _clear_labels()
        _draw_labels()
        _update_titles()
        fig.canvas.draw_idle()

    def _on_slider(_val) -> None:
        state["q_raw"][:] = np.array(
            [slider_w.val, slider_x.val, slider_y.val, slider_z.val], dtype=np.float64
        )
        _recompute_and_redraw()

    def _on_ref(label: str) -> None:
        state["ref_dir"] = label
        _recompute_and_redraw()

    def _on_reset(_event) -> None:
        q0 = np.asarray(args.quat, dtype=np.float64)
        slider_w.set_val(float(q0[0]))
        slider_x.set_val(float(q0[1]))
        slider_y.set_val(float(q0[2]))
        slider_z.set_val(float(q0[3]))
        state["q_raw"][:] = q0
        _recompute_and_redraw()

    def _on_save(_event) -> None:
        out_path = Path(args.out).expanduser()
        _save_figure(fig, out_path)
        info_text.set_text(f"Saved: {out_path.resolve()}")
        fig.canvas.draw_idle()

    def _on_annot(_event) -> None:
        state["annotate"] = not state["annotate"]
        _clear_labels()
        _draw_labels()
        info_text.set_text(f"Annotate symmetry indices: {state['annotate']}")
        fig.canvas.draw_idle()

    slider_w.on_changed(_on_slider)
    slider_x.on_changed(_on_slider)
    slider_y.on_changed(_on_slider)
    slider_z.on_changed(_on_slider)
    rb_ref.on_clicked(_on_ref)
    btn_reset.on_clicked(_on_reset)
    btn_save.on_clicked(_on_save)
    btn_annot.on_clicked(_on_annot)

    # Keep widget/callback refs alive for notebook/WebAgg.
    fig._widget_refs = {
        "sliders": [slider_w, slider_x, slider_y, slider_z],
        "radios": [rb_ref],
        "buttons": [btn_reset, btn_save, btn_annot],
        "axes": [
            slider_w_ax,
            slider_x_ax,
            slider_y_ax,
            slider_z_ax,
            ax_ref,
            ax_reset,
            ax_save,
            ax_annot,
        ],
    }
    fig._callback_refs = {
        "on_slider": _on_slider,
        "on_ref": _on_ref,
        "on_reset": _on_reset,
        "on_save": _on_save,
        "on_annot": _on_annot,
    }

    print("Interactive mode started.")
    print("Panels: upper/lower stereographic, 3D axis-angle, IPF key.")
    print("Close the plot window (or stop WebAgg) to exit.")
    plt.show()


if __name__ == "__main__":
    main()
