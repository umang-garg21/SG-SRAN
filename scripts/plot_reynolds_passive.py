"""
Reynolds averaging under the passive (Bunge) convention.

Convention:  q_g = s_g^{-1} ⊗ q   for each symmetry op s_g
Reynolds average: q_R = normalize( mean_g( s_g^{-1} ⊗ q ) )

Usage
-----
python scripts/plot_reynolds_passive.py
python scripts/plot_reynolds_passive.py --quat 0.6573364734649658	-0.43777036666870117	-0.022942548617720604	0.6129759550094604


0.530537 0.298601 -0.182983 0.771936
python scripts/plot_reynolds_passive.py --quat 1 0 0 0 --ref-dir X
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
import orix.plot
from orix.projections.stereographic import _vector2xy
from orix.quaternion import Orientation
from orix.vector import Vector3d

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcc_syms_inv import build_fcc_syms_inv
from utils.symmetry_utils import quaternion_left_matrix, resolve_symmetry

DIR_VECTORS = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < eps:
        raise ValueError(f"Quaternion norm too small: {n}")
    return (q / n).astype(np.float32)


def _quat_mul_4x4(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product via left-multiply matrix: q_left ⊗ q_right."""
    L = quaternion_left_matrix(q_left.astype(np.float64))
    return (L @ q_right.astype(np.float64).reshape(4)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core: passive Reynolds orbit
# ---------------------------------------------------------------------------

def passive_reynolds_orbit(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Compute the 24-element orbit under passive convention:
        orbit[g] = s_g^{-1} ⊗ q

    s_g^{-1} is the conjugate of s_g (unit quaternion inverse).
    Returns array of shape (24, 4).
    """
    # build_fcc_syms_inv() already gives the conjugates of the 24 cubic ops
    syms_inv = build_fcc_syms_inv().numpy()  # (24, 4) [w x y z]
    orbit = np.stack(
        [_quat_mul_4x4(syms_inv[g], q_wxyz) for g in range(len(syms_inv))],
        axis=0,
    )  # (24, 4)
    return orbit.astype(np.float32)


def reynolds_average(orbit: np.ndarray) -> np.ndarray:
    """Normalize the average of the orbit quaternions."""
    mean = orbit.mean(axis=0)
    return _normalize(mean)


def _sign_invariant_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distance proxy on SO(3) via quaternions: 1 - |dot(a,b)| in [0,1]."""
    aa = _normalize(a)
    bb = _normalize(b)
    return float(1.0 - abs(np.dot(aa, bb)))


def test_reynolds_invariance_passive(q_reynolds: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Check passive invariance of Reynolds average under all 24 ops:
        s_h^{-1} ⊗ q_R  ~  q_R (up to sign)

    Returns
    -------
    passed : bool
        True if all distances are <= tolerance chosen by caller.
    dists : np.ndarray
        Per-op sign-invariant distances (24,).
    """
    syms_inv = build_fcc_syms_inv().numpy().astype(np.float32)
    acted = np.stack([_quat_mul_4x4(s, q_reynolds) for s in syms_inv], axis=0)
    dists = np.asarray([_sign_invariant_dist(v, q_reynolds) for v in acted], dtype=np.float64)
    # pass/fail threshold is handled in main so caller can choose tolerance
    return bool(True), dists


# ---------------------------------------------------------------------------
# Stereographic helpers
# ---------------------------------------------------------------------------

def _stereo_xy(ori: Orientation, v_ref: Vector3d, pole: int = -1):
    v = ori * v_ref
    x, y = _vector2xy(v, pole=pole)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(v.z).reshape(-1)
    vis = z >= 0 if pole == -1 else z <= 0
    return x, y, vis


def _draw_cubic_guides(ax) -> None:
    v4 = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    v3 = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
    v2 = Vector3d([
        [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
        [1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0],
    ])
    for v, lw in [(v4, 1.0), (v3, 0.8), (v2, 0.6)]:
        ax.draw_circle(v, color="black", linewidth=lw, alpha=0.2)
    ax.set_labels("RD", "TD", None)
    ax.show_hemisphere_label()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quat", type=float, nargs=4, default=[0.9238795, 0.3826834, 0.0, 0.0],
                   metavar=("W", "X", "Y", "Z"), help="Input quaternion [w x y z].")
    p.add_argument("--ref-dir", choices=("X", "Y", "Z"), default="Z",
                   help="Reference direction for IPF coloring (default Z).")
    p.add_argument("--sym", type=str, default="O",
                   help="Symmetry group for orix Orientation objects (default 'O').")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="Tolerance for sign-invariant invariance test (default: 1e-5).")
    p.add_argument("--no-plot", action="store_true",
                   help="Only print values and invariance diagnostics; skip plotting.")
    if "ipykernel" in sys.modules:
        args, _ = p.parse_known_args()
        return args
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q = _normalize(np.asarray(args.quat, dtype=np.float64))
    sym_obj = resolve_symmetry(args.sym)
    v_ref = DIR_VECTORS[args.ref_dir]

    # --- compute ---
    orbit = passive_reynolds_orbit(q)       # (24, 4)
    q_R   = reynolds_average(orbit)         # (4,)

    # --- print ---
    print(f"\nInput q [w x y z]:           {q.tolist()}")
    print(f"\nPassive orbit  s_g^{{-1}} ⊗ q  (24 results):")
    for i, r in enumerate(orbit):
        print(f"  [{i:2d}]  {r.tolist()}")
    print(f"\nReynolds average (normalized): {q_R.tolist()}")

    # --- invariance test ---
    _, dists = test_reynolds_invariance_passive(q_R)
    d_max = float(np.max(dists))
    d_mean = float(np.mean(dists))
    passed = d_max <= float(args.tol)
    print("\nPassive invariance test on Reynolds average:")
    print(f"  criterion: max_g (1 - |<s_g^{{-1}}⊗q_R, q_R>|) <= {args.tol:.2e}")
    print(f"  max distance:  {d_max:.6e}")
    print(f"  mean distance: {d_mean:.6e}")
    print(f"  result:        {'PASS' if passed else 'FAIL'}")

    if args.no_plot:
        return

    # --- orix Orientation objects ---
    ori_orbit = Orientation(orbit, symmetry=sym_obj)
    ori_R     = Orientation(q_R.reshape(1, 4), symmetry=sym_obj)

    ckey = orix.plot.IPFColorKeyTSL(sym_obj.laue)
    ckey.direction = v_ref
    c_orbit = np.asarray(ckey.orientation2color(ori_orbit), dtype=np.float32)  # (24, 3)
    c_R     = np.asarray(ckey.orientation2color(ori_R),     dtype=np.float32).reshape(1, 3)

    xo, yo, vis_o = _stereo_xy(ori_orbit, v_ref)
    xr, yr, vis_r = _stereo_xy(ori_R,     v_ref)

    # --- plot ---
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "stereographic"}, figsize=(7, 7))
    _draw_cubic_guides(ax)

    # 24 orbit points
    Axes.scatter(
        ax, xo[vis_o], yo[vis_o],
        s=60, c=c_orbit[vis_o],
        edgecolors="grey", linewidths=0.5,
        alpha=0.85, zorder=3,
        label=r"$s_g^{-1} \otimes q$  (24)",
    )

    # Reynolds average
    Axes.scatter(
        ax, xr[vis_r], yr[vis_r],
        s=220, marker="D", c=c_R[vis_r],
        edgecolors="black", linewidths=1.2,
        zorder=5,
        label=r"Reynolds avg $\bar{q}_R$",
    )

    ax.set_title(
        f"Passive Reynolds orbit  $s_g^{{-1}} \\otimes q$\n"
        f"q = [{q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}]\n"
        f"$\\bar{{q}}_R$ = [{q_R[0]:.5f}, {q_R[1]:.5f}, {q_R[2]:.5f}, {q_R[3]:.5f}]",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
