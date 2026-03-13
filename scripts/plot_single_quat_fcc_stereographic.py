#!/usr/bin/env python3
"""
Plot one quaternion on a stereographic projection and its FCC symmetry orbit.

Default convention follows this repo's Bunge/passive rule:
    q' = s^{-1} ⊗ q  (left-multiply inverse symmetry operator)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import orix.plot  # registers "stereographic" projection with matplotlib
from orix.projections.stereographic import _vector2xy
from orix.quaternion import Orientation
from orix.vector import Vector3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _stereo_xy(ori: Orientation, v_ref: Vector3d, pole: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _build_orbit(q_wxyz: np.ndarray, sym_ops_wxyz: np.ndarray, action: str) -> np.ndarray:
    n = int(sym_ops_wxyz.shape[0])
    q_single = np.asarray(q_wxyz, dtype=np.float32).reshape(1, 4)
    ops = np.asarray(sym_ops_wxyz, dtype=np.float32)

    if action == "bunge-left-inv":
        orbit = quat_left_multiply_numpy(q_single, _quat_conjugate(ops), layout="quat_last")
        return orbit.reshape(n, 4).astype(np.float32, copy=False)

    if action == "active-left":
        orbit = quat_left_multiply_numpy(q_single, ops, layout="quat_last")
        return orbit.reshape(n, 4).astype(np.float32, copy=False)

    if action == "active-right":
        # q ⊗ s = conj( conj(s) ⊗ conj(q) )
        q_conj = _quat_conjugate(q_single)
        left = quat_left_multiply_numpy(q_conj, _quat_conjugate(ops), layout="quat_last")
        orbit = _quat_conjugate(left.reshape(n, 4))
        return orbit.astype(np.float32, copy=False)

    raise ValueError(f"Unknown action: {action}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one quaternion and its FCC symmetry orbit in stereographic projection."
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
        help="Reference direction used for stereographic point projection.",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=("bunge-left-inv", "active-left", "active-right"),
        default="bunge-left-inv",
        help="How symmetry is applied to q.",
    )
    parser.add_argument(
        "--pole",
        type=str,
        choices=("upper", "lower"),
        default="upper",
        help="Stereographic hemisphere.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate orbit points with symmetry op index.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/single_quat_fcc_stereo.png",
        help="Output PNG path.",
    )

    # Notebook kernels inject internal argv such as --f=.../kernel-*.json.
    # In that context, ignore unknown args so this can be run directly in cells.
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
        return args

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    q = _normalize_quat(np.array(args.quat, dtype=np.float64))
    sym_obj = resolve_symmetry(args.sym)
    sym_ops = np.asarray(sym_obj.data, dtype=np.float32)

    orbit = _build_orbit(q, sym_ops, args.action)
    pole = -1 if args.pole == "upper" else 1
    v_ref = DIR_VECTORS[args.ref_dir]

    ori_single = Orientation(q.reshape(1, 4), symmetry=sym_obj)
    ori_orbit = Orientation(orbit, symmetry=sym_obj)

    x0, y0, vis0 = _stereo_xy(ori_single, v_ref, pole=pole)
    x_orb, y_orb, vis_orb = _stereo_xy(ori_orbit, v_ref, pole=pole)

    id_idx = int(np.argmin(np.linalg.norm(sym_ops - np.array([1.0, 0.0, 0.0, 0.0]), axis=1)))

    fig = plt.figure(figsize=(12.0, 5.4), dpi=180)
    ax_single = fig.add_subplot(1, 2, 1, projection="stereographic")
    ax_orbit = fig.add_subplot(1, 2, 2, projection="stereographic")

    _draw_cubic_guides(ax_single)
    _draw_cubic_guides(ax_orbit)

    Axes.scatter(
        ax_single,
        x0[vis0],
        y0[vis0],
        s=120,
        marker="o",
        color="tab:blue",
        edgecolors="black",
        linewidths=0.8,
    )

    Axes.scatter(
        ax_orbit,
        x_orb[vis_orb],
        y_orb[vis_orb],
        s=30,
        marker="o",
        color="tab:gray",
        alpha=0.85,
        edgecolors="none",
        label="FCC symmetry orbit",
    )
    Axes.scatter(
        ax_orbit,
        [x_orb[id_idx]],
        [y_orb[id_idx]],
        s=120,
        marker="*",
        color="tab:red",
        edgecolors="black",
        linewidths=0.6,
        label="original (identity op)",
    )

    if args.annotate:
        for i in range(len(x_orb)):
            if not vis_orb[i]:
                continue
            ax_orbit.text(
                float(x_orb[i]),
                float(y_orb[i]),
                str(i),
                fontsize=6,
                ha="left",
                va="bottom",
                color="black",
            )

    ax_single.set_title(
        f"Single quaternion (ref={args.ref_dir}, {args.pole} hemisphere)\nq=[{q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}]"
    )
    ax_orbit.set_title(
        f"Same quaternion with all {sym_obj.name} symmetry ops\naction={args.action}, visible points={int(np.count_nonzero(vis_orb))}/{len(vis_orb)}"
    )
    ax_orbit.legend(loc="best", fontsize=8)

    fig.suptitle("Stereographic Quaternion Plot and FCC Symmetry Orbit", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"normalized quaternion [w x y z]: {q.tolist()}")
    print(f"symmetry group: {sym_obj.name}, operators: {sym_ops.shape[0]}")
    print(f"reference direction: {args.ref_dir}, pole: {args.pole}, action: {args.action}")


if __name__ == "__main__":
    main()
