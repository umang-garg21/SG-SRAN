# import numpy as np
# import matplotlib.pyplot as plt

# from orix.quaternion import Orientation, symmetry
# from orix.plot import IPFColorKeyTSL
# from orix.vector import Vector3d


# # ----------------------------
# # Basic quaternion ops (wxyz)
# # ----------------------------
# def quat_normalize(q):
#     q = np.asarray(q, dtype=np.float64)
#     n = np.linalg.norm(q, axis=-1, keepdims=True)
#     n = np.where(n == 0, 1.0, n)
#     return q / n

# def quat_conj(q):
#     q = np.asarray(q, dtype=np.float64)
#     out = q.copy()
#     out[..., 1:] *= -1.0
#     return out

# def quat_mul(q1, q2):
#     """Hamilton product for scalar-first quaternions (w,x,y,z). Vectorized."""
#     q1 = np.asarray(q1, dtype=np.float64)
#     q2 = np.asarray(q2, dtype=np.float64)

#     w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
#     w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
#     return np.stack([w, x, y, z], axis=-1)

# def enforce_w_positive(q):
#     q = np.asarray(q, dtype=np.float64)
#     return np.where(q[..., :1] < 0, -q, q)


# # ----------------------------
# # Load scalar-last (x,y,z,w) -> wxyz Orientation
# # ----------------------------
# def load_xyzw_npy_to_wxyz(npy_path: str) -> np.ndarray:
#     arr = np.load(npy_path)
#     if arr.ndim != 3:
#         raise ValueError(f"Expected 3D array, got shape {arr.shape}")

#     if arr.shape[-1] == 4:
#         d = arr
#     elif arr.shape[0] == 4:
#         d = np.transpose(arr, (1, 2, 0))
#     else:
#         raise ValueError(f"No axis of size 4 found in {arr.shape}")

#     d = d.astype(np.float64)  # (H,W,4) in xyzw

#     # xyzw -> wxyz
#     q_wxyz = np.stack([d[..., 3], d[..., 0], d[..., 1], d[..., 2]], axis=-1)
#     q_wxyz = quat_normalize(q_wxyz)
#     q_wxyz = enforce_w_positive(q_wxyz)
#     return q_wxyz


# # ----------------------------
# # IPF coloring (always pass symmetry explicitly)
# # ----------------------------
# _DIRS = {"X": Vector3d((1, 0, 0)), "Y": Vector3d((0, 1, 0)), "Z": Vector3d((0, 0, 1))}

# def ipf_rgb(q_wxyz_hw4: np.ndarray, laue_sym: symmetry.Symmetry, ref_dir="Z") -> np.ndarray:
#     """Return (H,W,3) RGB using TSL key. q is (H,W,4) wxyz."""
#     H, W, _ = q_wxyz_hw4.shape
#     ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)

#     ckey = IPFColorKeyTSL(laue_sym)
#     ckey.direction = _DIRS[ref_dir.upper()]
#     rgb = ckey.orientation2color(ori)
#     return rgb


# # ----------------------------
# # Plot: apply symmetry operators and show maps
# # ----------------------------
# def plot_symmetry_ipf_maps_bunge(q_wxyz_hw4: np.ndarray, which_ops=None, ref_dir="Z"):
#     """
#     Bunge convention confirmed: q is specimen->crystal.
#     Correct crystal symmetry action: q' = s^{-1} ⊗ q  (left multiply inverse).
#     """
#     H, W, _ = q_wxyz_hw4.shape

#     # Proper cubic rotations (24): symmetry.O
#     # Laue group for coloring: symmetry.Oh (adds inversion)
#     S = symmetry.O
#     laue = symmetry.Oh

#     if which_ops is None:
#         which_ops = list(range(12))  # show first 12 by default

#     rgb0 = ipf_rgb(q_wxyz_hw4, laue, ref_dir=ref_dir)

#     rgbs = []
#     labels = []
#     for k in which_ops:
#         s_wxyz = S.data[k]              # (4,) quaternion (wxyz)  :contentReference[oaicite:0]{index=0}
#         s_inv = quat_conj(s_wxyz)       # inverse for unit quat
#         q_prime = quat_mul(s_inv, q_wxyz_hw4)   # s^{-1} ⊗ q
#         q_prime = quat_normalize(q_prime)
#         q_prime = enforce_w_positive(q_prime)

#         rgbs.append(ipf_rgb(q_prime, laue, ref_dir=ref_dir))
#         labels.append(f"op {k}: s⁻¹ ⊗ q")

#     # Plot grid
#     n = 1 + len(rgbs)
#     ncols = 4
#     nrows = int(np.ceil(n / ncols))

#     fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows), dpi=120)
#     axes = np.array(axes).reshape(-1)

#     axes[0].imshow(rgb0)
#     axes[0].set_title(f"Original IPF-{ref_dir.upper()}")
#     axes[0].axis("off")

#     for i, (rgb, lab) in enumerate(zip(rgbs, labels), start=1):
#         axes[i].imshow(rgb)
#         axes[i].set_title(lab)
#         axes[i].axis("off")

#     for j in range(n, len(axes)):
#         axes[j].axis("off")

#     plt.tight_layout()
#     plt.show()


# # ----------------------------
# # Plot diagnostic: correct vs wrong side (Bunge)
# # ----------------------------
# def plot_bunge_correct_vs_wrong(q_wxyz_hw4: np.ndarray, op_index=5, ref_dir="Z"):
#     """
#     Correct for Bunge: q' = s^{-1} ⊗ q
#     Wrong-side (common mistake): q' = q ⊗ s
#     """
#     S = symmetry.O
#     laue = symmetry.Oh

#     s = S.data[op_index]          # quaternion for symmetry op
#     s_inv = quat_conj(s)

#     q_correct = quat_mul(s_inv, q_wxyz_hw4)     # s^{-1} ⊗ q
#     q_wrong   = quat_mul(q_wxyz_hw4, s)         # q ⊗ s

#     q_correct = enforce_w_positive(quat_normalize(q_correct))
#     q_wrong   = enforce_w_positive(quat_normalize(q_wrong))

#     rgb0 = ipf_rgb(q_wxyz_hw4, laue, ref_dir=ref_dir)
#     rgbC = ipf_rgb(q_correct,  laue, ref_dir=ref_dir)
#     rgbW = ipf_rgb(q_wrong,    laue, ref_dir=ref_dir)

#     fig, ax = plt.subplots(1, 3, figsize=(14, 5), dpi=120)
#     ax[0].imshow(rgb0); ax[0].set_title(f"Original IPF-{ref_dir.upper()}"); ax[0].axis("off")
#     ax[1].imshow(rgbC); ax[1].set_title(f"Bunge-correct: s⁻¹ ⊗ q (op {op_index})"); ax[1].axis("off")
#     ax[2].imshow(rgbW); ax[2].set_title(f"Wrong-side: q ⊗ s (op {op_index})"); ax[2].axis("off")
#     plt.tight_layout()
#     plt.show()


# # ----------------------------
# # RUN
# # ----------------------------
# if __name__ == "__main__":
#     npy_path = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"

#     q_wxyz = load_xyzw_npy_to_wxyz(npy_path)

#     # 1) Show original + symmetry-applied maps (correct Bunge action)
#     plot_symmetry_ipf_maps_bunge(q_wxyz, which_ops=[0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 21, 23], ref_dir="Z")

#     # 2) Diagnostic: correct vs wrong side
#     plot_bunge_correct_vs_wrong(q_wxyz, op_index=5, ref_dir="Z")




#     # ---------------------------------------------------------------------------
#     # Other
#     # ---------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt

# from orix.quaternion import Orientation, symmetry
# from orix.plot import IPFColorKeyTSL
# from orix.vector import Vector3d


# # ----------------------------
# # Quaternion ops (wxyz)
# # ----------------------------
# def quat_normalize(q):
#     q = np.asarray(q, dtype=np.float64)
#     n = np.linalg.norm(q, axis=-1, keepdims=True)
#     n = np.where(n == 0, 1.0, n)
#     return q / n

# def quat_conj(q):
#     q = np.asarray(q, dtype=np.float64)
#     out = q.copy()
#     out[..., 1:] *= -1.0
#     return out

# def quat_mul(q1, q2):
#     """Hamilton product for scalar-first quaternions (w,x,y,z). Vectorized."""
#     q1 = np.asarray(q1, dtype=np.float64)
#     q2 = np.asarray(q2, dtype=np.float64)

#     w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
#     w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
#     return np.stack([w, x, y, z], axis=-1)

# def enforce_w_positive(q):
#     q = np.asarray(q, dtype=np.float64)
#     return np.where(q[..., :1] < 0, -q, q)


# # ----------------------------
# # Load scalar-last (x,y,z,w) -> wxyz
# # ----------------------------
# def load_xyzw_npy_to_wxyz(npy_path: str) -> np.ndarray:
#     arr = np.load(npy_path)
#     if arr.ndim != 3:
#         raise ValueError(f"Expected 3D array, got shape {arr.shape}")

#     if arr.shape[-1] == 4:
#         d = arr
#     elif arr.shape[0] == 4:
#         d = np.transpose(arr, (1, 2, 0))
#     else:
#         raise ValueError(f"No axis of size 4 found in {arr.shape}")

#     d = d.astype(np.float64)
#     q_wxyz = np.stack([d[..., 3], d[..., 0], d[..., 1], d[..., 2]], axis=-1)
#     q_wxyz = enforce_w_positive(quat_normalize(q_wxyz))
#     return q_wxyz


# # ----------------------------
# # IPF coloring
# # ----------------------------
# _DIRS = {"X": Vector3d((1, 0, 0)), "Y": Vector3d((0, 1, 0)), "Z": Vector3d((0, 0, 1))}

# def ipf_rgb(q_wxyz_hw4: np.ndarray, laue_sym, ref_dir="Z") -> np.ndarray:
#     H, W, _ = q_wxyz_hw4.shape
#     ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
#     ckey = IPFColorKeyTSL(laue_sym)
#     ckey.direction = _DIRS[ref_dir.upper()]
#     return ckey.orientation2color(ori)


# # ----------------------------
# # Multi-op comparison plot
# # ----------------------------
# def plot_symmetry_action_grid(
#     q_wxyz_hw4: np.ndarray,
#     op_indices=None,
#     ref_dir: str = "Z",
#     include_original: bool = True,
# ):
#     """
#     For each symmetry operator index k, show:
#       - Bunge-correct:   q' = s^{-1} ⊗ q
#       - Left no-inv:     q' = s      ⊗ q    (you asked for this explicitly)
#       - Right multiply:  q' = q      ⊗ s

#     Uses proper cubic group S = symmetry.O (24 ops).
#     Colors using Laue group symmetry.Oh in the TSL IPF key.
#     """
#     H, W, _ = q_wxyz_hw4.shape

#     S = symmetry.O      # 24 proper cubic rotations
#     laue = symmetry.Oh  # Laue group for coloring

#     if op_indices is None:
#         op_indices = list(range(24))  # all ops by default
#     else:
#         op_indices = list(op_indices)

#     # Original
#     rgb0 = ipf_rgb(q_wxyz_hw4, laue, ref_dir=ref_dir) if include_original else None

#     # Prepare all panels
#     panels = []
#     titles = []

#     if include_original:
#         panels.append(rgb0)
#         titles.append(f"Original IPF-{ref_dir.upper()}")

#     for k in op_indices:
#         s = S.data[k]             # (4,) wxyz
#         s_inv = quat_conj(s)

#         q_bunge_correct = enforce_w_positive(quat_normalize(quat_mul(s_inv, q_wxyz_hw4)))  # s^{-1} ⊗ q
#         q_left_noinv    = enforce_w_positive(quat_normalize(quat_mul(s,     q_wxyz_hw4)))  # s ⊗ q
#         q_right         = enforce_w_positive(quat_normalize(quat_mul(q_wxyz_hw4, s)))      # q ⊗ s

#         panels.extend([
#             ipf_rgb(q_bunge_correct, laue, ref_dir=ref_dir),
#             ipf_rgb(q_left_noinv,    laue, ref_dir=ref_dir),
#             ipf_rgb(q_right,         laue, ref_dir=ref_dir),
#         ])
#         titles.extend([
#             f"op {k}:  s⁻¹ ⊗ q   (Bunge-correct)",
#             f"op {k}:  s  ⊗ q   (NO inv, left)",
#             f"op {k}:  q  ⊗ s   (right)",
#         ])

#     # Plot in a grid:
#     # Use 3 columns per op (plus one optional original), so make ncols=3
#     ncols = 3
#     n = len(panels)
#     nrows = int(np.ceil(n / ncols))

#     fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 4.0*nrows), dpi=120)
#     axes = np.array(axes).reshape(-1)

#     for i in range(len(axes)):
#         if i < n:
#             axes[i].imshow(panels[i])
#             axes[i].set_title(titles[i], fontsize=9)
#             axes[i].axis("off")
#         else:
#             axes[i].axis("off")

#     fig.suptitle(
#         f"FCC cubic symmetry action comparison (proper 24 ops), IPF-{ref_dir.upper()}",
#         fontsize=12
#     )
#     plt.tight_layout()
#     plt.show()


# # ----------------------------
# # RUN
# # ----------------------------
# if __name__ == "__main__":
#     npy_path = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"

#     q_wxyz = load_xyzw_npy_to_wxyz(npy_path)

#     # Choose a subset (faster to view). Use range(24) for all.
#     ops_to_show = range(24) # [0, 1, 2, 3, 4, 5, 8, 12]  # edit as desired

#     plot_symmetry_action_grid(
#         q_wxyz,
#         op_indices=ops_to_show,
#         ref_dir="Z",
#         include_original=True,
#     )

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from orix.quaternion import Orientation, symmetry
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ============================================================
# Quaternion ops (wxyz scalar-first)
# ============================================================
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n

def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul(q1, q2):
    """Hamilton product (w,x,y,z). Vectorized."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def enforce_w_positive(q):
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)

def misorientation_angle_deg(qA, qB):
    """
    Misorientation angle between two orientation fields represented by unit quaternions.
    angle = 2*acos(|w_rel|) in degrees, where q_rel = inv(qA) ⊗ qB.
    """
    qA = quat_normalize(qA)
    qB = quat_normalize(qB)
    q_rel = quat_mul(quat_conj(qA), qB)
    w = np.clip(np.abs(q_rel[..., 0]), -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return ang * (180.0 / np.pi)


# ============================================================
# Load scalar-last (x,y,z,w) -> scalar-first (w,x,y,z)
# ============================================================
def load_xyzw_npy_to_wxyz(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    if arr.shape[-1] == 4:
        d = arr
    elif arr.shape[0] == 4:
        d = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"No axis of size 4 found in {arr.shape}")

    d = d.astype(np.float64)  # (H,W,4) xyzw
    q = np.stack([d[..., 3], d[..., 0], d[..., 1], d[..., 2]], axis=-1)  # wxyz
    q = enforce_w_positive(quat_normalize(q))
    return q


# ============================================================
# IPF coloring helpers
# ============================================================
_DIRS = {
    "X": Vector3d((1, 0, 0)),
    "Y": Vector3d((0, 1, 0)),
    "Z": Vector3d((0, 0, 1)),
}

def ipf_rgb(q_wxyz_hw4, laue_sym, ref_dir="Z"):
    H, W, _ = q_wxyz_hw4.shape
    ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
    ckey = IPFColorKeyTSL(laue_sym)
    ckey.direction = _DIRS[ref_dir.upper()]
    return ckey.orientation2color(ori)  # (H,W,3)


# ============================================================
# Main: plot IPF X/Y/Z for each op and compute misorientation stats
# ============================================================
def analyze_symmetry_actions_all_ops(
    q_wxyz_hw4: np.ndarray,
    out_dir: str,
    max_ops: int | None = None,
    show: bool = False,
):
    """
    For each cubic proper symmetry op s (24 ops in symmetry.O), compute:
      - q_correct = s^{-1} ⊗ q   (Bunge correct)
      - q_left    = s ⊗ q        (left no-inv)
      - q_right   = q ⊗ s        (right multiply)

    Plot IPF-{X,Y,Z} rows with columns:
      Original | s^{-1}⊗q | s⊗q | q⊗s

    Compute misorientation (deg) maps:
      err_left  = misorientation(q_correct, q_left)
      err_right = misorientation(q_correct, q_right)

    Save per-op figure + print a summary table to identify incorrect action.
    """
    os.makedirs(out_dir, exist_ok=True)

    S = symmetry.O     # 24 proper cubic rotations
    laue = symmetry.Oh # Laue group for coloring (TSL key uses Laue)

    n_ops = len(S.data)
    if max_ops is not None:
        n_ops = min(n_ops, max_ops)

    print("\n=== Symmetry action check (Bunge assumed) ===")
    print("Correct:  q' = s^{-1} ⊗ q")
    print("Compare:  left no-inv  (s ⊗ q)  and  right (q ⊗ s)\n")

    header = f"{'op':>3} | {'mean(err_left)':>14} {'p95(err_left)':>14} {'max(err_left)':>14} || {'mean(err_right)':>15} {'p95(err_right)':>15} {'max(err_right)':>15}"
    print(header)
    print("-" * len(header))

    summary = []

    for k in range(n_ops):
        s = S.data[k]         # (4,) wxyz
        s_inv = quat_conj(s)  # inverse for unit quaternion

        # --- three action modes ---
        q0 = q_wxyz_hw4
        q_correct = enforce_w_positive(quat_normalize(quat_mul(s_inv, q0)))  # s^{-1} ⊗ q
        q_left    = enforce_w_positive(quat_normalize(quat_mul(s,     q0)))  # s ⊗ q
        q_right   = enforce_w_positive(quat_normalize(quat_mul(q0,    s)))   # q ⊗ s

        # --- misorientation error maps vs correct ---
        err_left  = misorientation_angle_deg(q_correct, q_left)
        err_right = misorientation_angle_deg(q_correct, q_right)

        # stats
        def stats(a):
            a = a[np.isfinite(a)]
            return float(np.mean(a)), float(np.quantile(a, 0.95)), float(np.max(a))

        mL, pL, xL = stats(err_left)
        mR, pR, xR = stats(err_right)

        print(f"{k:3d} | {mL:14.6e} {pL:14.6e} {xL:14.6e} || {mR:15.6e} {pR:15.6e} {xR:15.6e}")
        summary.append((k, mL, pL, xL, mR, pR, xR))

        # --- render IPF maps for all directions ---
        # Rows: X,Y,Z ; Cols: original, correct, left, right
        dirs = ["X", "Y", "Z"]
        cols = [
            ("Original", q0),
            ("s^{-1} ⊗ q (correct)", q_correct),
            ("s ⊗ q (left, no inv)", q_left),
            ("q ⊗ s (right)", q_right),
        ]

        fig, axes = plt.subplots(len(dirs), len(cols), figsize=(4.2*len(cols), 4.0*len(dirs)), dpi=150)
        fig.suptitle(f"FCC cubic symmetry op {k} (proper group O, 24 ops) — IPF X/Y/Z", fontsize=12)

        for r, dname in enumerate(dirs):
            for c, (ctitle, qq) in enumerate(cols):
                rgb = ipf_rgb(qq, laue, ref_dir=dname)
                ax = axes[r, c]
                ax.imshow(rgb)
                ax.axis("off")
                if r == 0:
                    ax.set_title(ctitle, fontsize=9)
                if c == 0:
                    ax.text(0.02, 0.92, f"IPF-{dname}", transform=ax.transAxes,
                            fontsize=10, color="w",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

        # Optionally add error heatmaps in a separate figure (very useful)
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        ax2[0].imshow(err_left, vmin=0, vmax=np.quantile(err_left, 0.99))
        ax2[0].set_title(f"err_left = misori(correct, s⊗q)\nmean={mL:.2e}°, p95={pL:.2e}°, max={xL:.2e}°")
        ax2[0].axis("off")

        ax2[1].imshow(err_right, vmin=0, vmax=np.quantile(err_right, 0.99))
        ax2[1].set_title(f"err_right = misori(correct, q⊗s)\nmean={mR:.2e}°, p95={pR:.2e}°, max={xR:.2e}°")
        ax2[1].axis("off")

        plt.tight_layout()

        f1 = os.path.join(out_dir, f"ipf_xyz_op_{k:02d}.png")
        f2 = os.path.join(out_dir, f"misori_op_{k:02d}.png")
        fig.savefig(f1, bbox_inches="tight")
        fig2.savefig(f2, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig2)


    df = pd.DataFrame(
        summary,
        columns=[
            "op",
            "mean_err_left_deg", "p95_err_left_deg", "max_err_left_deg",
            "mean_err_right_deg", "p95_err_right_deg", "max_err_right_deg",
        ],
    )

    csv_path = os.path.join(out_dir, "symmetry_action_summary.csv")
    tex_path = os.path.join(out_dir, "symmetry_action_summary.tex")

    df.to_csv(csv_path, index=False)

    # A nice LaTeX table (booktabs style). Requires \usepackage{booktabs}.
    df.to_latex(
        tex_path,
        index=False,
        float_format="%.6g",
        caption="Misorientation error statistics comparing symmetry application modes (degrees).",
        label="tab:symmetry_action_errors",
    )

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {tex_path}")

    return df


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    npy_path = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"
    out_dir = "./symmetry_action_debug_out"

    q_wxyz = load_xyzw_npy_to_wxyz(npy_path)


    Orientation(q_wxyz.reshape(-1, 4),symmetry=symmetry.Oh).reduce

    q_reduced, s_l, s_r = Orientation(q_wxyz.reshape(-1, 4),symmetry=symmetry.Oh).map_into_symmetry_reduced_zone_with_ops(verbose=True)


    Orientation(q_wxyz.reshape(-1, 4),symmetry=symmetry.Oh).map_into_symmetry_reduced_zone()

    # Set max_ops=None to do all 24; or e.g. 8 for faster.
    df = analyze_symmetry_actions_all_ops(q_wxyz, out_dir=out_dir, max_ops=None, show=False)

    # --------------------------------------------------
    # Settings
    # --------------------------------------------------
    out_dir = "./symmetry_action_debug_out"
    csv_path = os.path.join(out_dir, "symmetry_action_summary.csv")

    K = 6   # number of operators to show (edit as desired)

    # --------------------------------------------------
    # Load summary table
    # --------------------------------------------------
    df = pd.read_csv(csv_path)

    # Sort by wrong-side error (largest mean right error first)
    worst = df.sort_values("mean_err_right_deg", ascending=False).head(K)

    print("Operators selected (largest mean_err_right_deg):")
    print(worst[["op", "mean_err_right_deg"]])

    # --------------------------------------------------
    # Build montage of IPF-only figures
    # --------------------------------------------------
    nrows = K
    ncols = 1   # one wide IPF figure per row

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8*K), dpi=150)

    if K == 1:
        axes = [axes]

    for ax, row in zip(axes, worst.itertuples(index=False)):
        op = int(row.op)
        ipf_path = os.path.join(out_dir, f"ipf_xyz_op_{op:02d}.png")

        img = mpimg.imread(ipf_path)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"FCC symmetry op {op:02d}  |  mean(right error) = {row.mean_err_right_deg:.3g}°",
            fontsize=11,
        )

    plt.tight_layout()

    montage_path = os.path.join(out_dir, "ipf_only_worst_ops_montage.png")
    fig.savefig(montage_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved montage to: {montage_path}")