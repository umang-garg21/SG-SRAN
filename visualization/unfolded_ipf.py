import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from copy import deepcopy
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from utils.quat_ops import (
    normalize_quaternions,
    enforce_hemisphere,
    reduce_to_fz_min_angle,
)

import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from utils.quat_ops import (
    normalize_quaternions,
    enforce_hemisphere,
    reduce_to_fz_min_angle,
)


def plot_fz_ipf_helper(ax, q_flat, sym_class, ref_dir="Z", tol_deg=0.5, label=""):
    """
    Plot the FZ-IPF for quaternion data with:
      - Inside FZ originally (IPF-colored circles, semi-transparent)
      - Mapped from outside FZ (IPF-colored triangles)
      - Outside FZ originally (IPF-colored diamonds with black edge)
      - Blue symmetry axes for reference
    """
    # -------------------------------------------------------------
    # Normalize and reduce quaternions to Fundamental Zone (FZ)
    # -------------------------------------------------------------
    sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class
    q_flat = normalize_quaternions(q_flat, axis=-1)
    q_flat = enforce_hemisphere(q_flat, scalar_first=True)

    q_fz, op_map = reduce_to_fz_min_angle(
        q_flat,
        sym=sym,
        normalize=False,
        hemisphere=False,
        return_op_map=True,
    )

    ori_fz = Orientation(q_fz, symmetry=sym).map_into_symmetry_reduced_zone()
    ori_orig = Orientation(q_flat, symmetry=sym)
    outside_mask = op_map != 0
    frac_outside = outside_mask.mean() * 100

    # -------------------------------------------------------------
    # Reference direction & pole figure vectors
    # -------------------------------------------------------------
    ref_map = {
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
        "Z": Vector3d.zvector(),
    }
    v_ref = ref_map[ref_dir.upper()]
    v_plot = ori_fz * v_ref
    v_orig = ori_orig * v_ref

    # -------------------------------------------------------------
    # IPF colors
    # -------------------------------------------------------------
    from orix import plot as orix_plot

    ckey = orix_plot.IPFColorKeyTSL(sym.laue)
    ckey.direction = v_ref
    colors = ckey.orientation2color(ori_fz)

    # -------------------------------------------------------------
    # Plot categories
    # -------------------------------------------------------------
    ax.set_title(
        f"{label} — Outside FZ: {frac_outside:.2f}%",
        pad=20,
        fontsize=11,
    )

    # 1. Inside FZ originally — circle, transparent
    ax.scatter(
        v_plot[~outside_mask],
        c=colors[~outside_mask],
        s=12,
        alpha=0.7,
        marker="o",
        edgecolors="none",
        label="Inside FZ originally",
    )

    # 2. Mapped from outside FZ — triangle
    if np.any(outside_mask):
        ax.scatter(
            v_plot[outside_mask],
            c=colors[outside_mask],
            s=20,
            alpha=0.95,
            marker="^",
            edgecolors="black",
            linewidths=0.4,
            label="Mapped from outside FZ",
        )

    # 3. Original outside positions — diamond with black edge
    if np.any(outside_mask):
        ax.scatter(
            v_orig[outside_mask],
            c=colors[outside_mask],
            s=24,
            alpha=0.95,
            marker="D",  # diamond shape
            edgecolors="black",
            linewidths=0.5,
            label="Outside FZ originally",
        )

    # -------------------------------------------------------------
    # Blue symmetry reference axes
    # -------------------------------------------------------------
    v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    ax.draw_circle(v4fold, color="blue")

    v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
    ax.draw_circle(v3fold, color="blue")

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
    ax.draw_circle(v2fold, color="blue")
    # sector = sym.fundamental_sector
    # original_pole = deepcopy(sector._pole)
    # sector._pole = ax.pole
    # edges = sector.edges
    # sector._pole = original_pole
    # x, y, _ = ax._pretransform_input((edges,))
    # patch = mpatches.PathPatch(
    #     mpath.Path(np.column_stack([x, y]), closed=True),
    #     facecolor="none",
    #     edgecolor="black",
    #     linewidth=1.5,
    #     alpha=0.9,
    #     zorder=5,
    # )
    # ax.add_patch(patch)
    # -------------------------------------------------------------
    # Axis formatting
    # -------------------------------------------------------------
    ax.set_labels("RD", "TD", None)
    ax.show_hemisphere_label()
    ax.legend(loc="upper right", fontsize=8)

    return frac_outside


def fz_ipf_sr_hr_side_by_side(
    sr_quat: np.ndarray,
    hr_quat: np.ndarray,
    sym_class="Oh",
    ref_dir="Z",
    max_points: int = 5000,
    tol_deg: float = 0.5,
    out_png: str = "fz_ipf_sr_hr.png",
    overwrite: bool = True,
):
    """
    Render SR and HR FZ-IPF plots side by side into a single PNG.
    Uses shared logic with `plot_fz_ipf` for rendering.

    Parameters
    ----------
    sr_quat : ndarray
        SR quaternion array (H,W,4) or (4,H,W).
    hr_quat : ndarray
        HR quaternion array (H,W,4) or (4,H,W).
    sym_class : str or orix symmetry
        Symmetry group.
    ref_dir : str
        Reference direction ("X","Y","Z").
    max_points : int
        Max orientations to plot for speed.
    tol_deg : float
        Misorientation tolerance.
    out_png : str
        Output file path.
    overwrite : bool
        Skip if file exists and overwrite=False.

    Returns
    -------
    dict
        {"SR_frac_outside": float, "HR_frac_outside": float, "out_png": str}
    """
    if out_png and not overwrite and os.path.exists(out_png):
        print(f"Skipping existing {out_png}")
        return {"out_png": out_png}

    # ------------------------------------------------------------------
    # Downsample
    # ------------------------------------------------------------------
    sr_flat = sr_quat.reshape(-1, 4)
    hr_flat = hr_quat.reshape(-1, 4)

    if max_points and sr_flat.shape[0] > max_points:
        sr_flat = sr_flat[np.random.choice(sr_flat.shape[0], max_points, replace=False)]
    if max_points and hr_flat.shape[0] > max_points:
        hr_flat = hr_flat[np.random.choice(hr_flat.shape[0], max_points, replace=False)]

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "stereographic"},
        figsize=(12, 6),
        constrained_layout=True,
    )

    # Plot SR FZ-IPF
    sr_frac = plot_fz_ipf_helper(
        axes[0], sr_flat, sym_class, ref_dir, tol_deg, label="SR"
    )

    # Plot HR FZ-IPF
    hr_frac = plot_fz_ipf_helper(
        axes[1], hr_flat, sym_class, ref_dir, tol_deg, label="HR"
    )

    # ------------------------------------------------------------------
    # Save plot
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved SR-HR FZ IPF plot → {out_png}")

    return {
        "SR_frac_outside": sr_frac,
        "HR_frac_outside": hr_frac,
        "out_png": out_png,
    }


# def plot_fz_ipf_helper(ax, q_flat, sym_class, ref_dir="Z", tol_deg=0.5, label=""):
#     """
#     Helper function to plot the FZ-IPF for SR and HR quaternion data.
#     This is shared by both `fz_ipf_render` and `plot_fz_ipf` functions.
#     """
#     # Normalize quaternions and reduce to the Fundamental Zone (FZ)
#     sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class
#     q_flat = normalize_quaternions(q_flat, axis=-1)
#     q_flat = enforce_hemisphere(q_flat, scalar_first=True)

#     q_fz, op_map = reduce_to_fz_min_angle(
#         q_flat,
#         sym=sym,
#         normalize=False,
#         hemisphere=False,
#         return_op_map=True,
#     )

#     ori_fz = Orientation(q_fz, symmetry=sym).map_into_symmetry_reduced_zone()
#     ori_orig = Orientation(q_flat, symmetry=sym)

#     # Outside mask (non-identity operator)
#     outside_mask = op_map != 0
#     frac_outside = outside_mask.mean() * 100

#     # Reference direction (vector)
#     ref_map = {
#         "X": Vector3d.xvector(),
#         "Y": Vector3d.yvector(),
#         "Z": Vector3d.zvector(),
#     }
#     v_ref = ref_map[ref_dir.upper()]
#     v_plot = ori_fz * v_ref
#     v_orig = ori_orig * v_ref

#     # Plot the title with fraction outside FZ
#     ax.set_title(
#         f"{label} — Outside FZ: {frac_outside:.2f}%",
#         pad=20,
#         fontsize=11,
#     )

#     # Plot original (unreduced) orientations in red
#     ax.scatter(v_orig, c="red", s=5, alpha=0.35, label="Original (unreduced)")

#     # Plot inside FZ (originally) in grey
#     ax.scatter(
#         v_plot[~outside_mask], c="grey", s=5, alpha=0.7, label="Inside FZ originally"
#     )

#     # Plot mapped from outside FZ in green
#     if np.any(outside_mask):
#         ax.scatter(
#             v_plot[outside_mask],
#             c="green",
#             s=5,
#             alpha=0.9,
#             label="Mapped from outside FZ",
#         )

#     v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
#     ax.draw_circle(v4fold, color="blue")

#     v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
#     ax.draw_circle(v3fold, color="blue")

#     v2fold = Vector3d(
#         [
#             [1, 0, 1],
#             [0, 1, 1],
#             [-1, 0, 1],
#             [0, -1, 1],
#             [1, 1, 0],
#             [-1, -1, 0],
#             [-1, 1, 0],
#             [1, -1, 0],
#         ]
#     )
#     ax.draw_circle(v2fold, color="blue")
#     # Draw the symmetry boundary (FZ boundary)
#     # sector = sym.fundamental_sector
#     # original_pole = deepcopy(sector._pole)
#     # sector._pole = ax.pole
#     # edges = sector.edges
#     # sector._pole = original_pole
#     # x, y, _ = ax._pretransform_input((edges,))
#     # patch = mpatches.PathPatch(
#     #     mpath.Path(np.column_stack([x, y]), closed=True),
#     #     facecolor="none",
#     #     edgecolor="black",
#     #     linewidth=2.0,
#     #     alpha=1.0,
#     #     zorder=5,
#     # )
#     # ax.add_patch(patch)

#     # Set labels and show hemisphere labels
#     ax.set_labels("RD", "TD", None)
#     ax.show_hemisphere_label()
#     ax.legend(loc="upper right", fontsize=8)

#     return frac_outside


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.path as mpath

# from copy import deepcopy
# from orix.quaternion import Orientation, symmetry
# from orix.vector import Vector3d
# from utils.quat_ops import (
#     normalize_quaternions,
#     enforce_hemisphere,
#     reduce_to_fz_min_angle,
# )


# def _plot_single_fz_ipf(ax, q_flat, sym_class, ref_dir="Z", tol_deg=0.5, label=""):
#     """
#     Render a single FZ-IPF plot on a given axis.
#     """
#     # ------------------------------------------------------------------
#     # Symmetry and reduction
#     # ------------------------------------------------------------------
#     sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class

#     q_flat = normalize_quaternions(q_flat, axis=-1)
#     q_flat = enforce_hemisphere(q_flat, scalar_first=True)

#     q_fz, op_map = reduce_to_fz_min_angle(q_flat, sym=sym, return_op_map=True)

#     ori_fz = Orientation(q_fz, symmetry=sym).map_into_symmetry_reduced_zone()
#     ori_orig = Orientation(q_flat, symmetry=sym)

#     outside_mask = op_map != 0
#     frac_outside = outside_mask.mean() * 100

#     ref_map = {
#         "X": Vector3d.xvector(),
#         "Y": Vector3d.yvector(),
#         "Z": Vector3d.zvector(),
#     }
#     v_ref = ref_map[ref_dir.upper()]
#     v_plot = ori_fz * v_ref
#     v_orig = ori_orig * v_ref

#     # ------------------------------------------------------------------
#     # Scatter plots
#     # ------------------------------------------------------------------
#     ax.set_title(
#         f"{label} — Outside FZ: {frac_outside:.2f}%",
#         pad=20,
#         fontsize=11,
#     )

#     # 🔴 Original unreduced
#     ax.scatter(v_orig, c="red", s=5, alpha=0.35, label="Original (unreduced)")

#     # 🩶 Inside FZ originally
#     ax.scatter(
#         v_plot[~outside_mask], c="grey", s=5, alpha=0.7, label="Inside FZ originally"
#     )

#     # 🟢 Mapped from outside
#     if np.any(outside_mask):
#         ax.scatter(
#             v_plot[outside_mask],
#             c="green",
#             s=5,
#             alpha=0.9,
#             label="Mapped from outside FZ",
#         )

#     # ------------------------------------------------------------------
#     # FZ boundary overlay
#     # ------------------------------------------------------------------
#     sector = sym.fundamental_sector
#     original_pole = deepcopy(sector._pole)
#     sector._pole = ax.pole
#     edges = sector.edges
#     sector._pole = original_pole
#     x, y, _ = ax._pretransform_input((edges,))
#     patch = mpatches.PathPatch(
#         mpath.Path(np.column_stack([x, y]), closed=True),
#         facecolor="none",
#         edgecolor="black",
#         linewidth=2.0,
#         alpha=1.0,
#         zorder=5,
#     )
#     ax.add_patch(patch)

#     ax.set_labels("RD", "TD", None)
#     ax.show_hemisphere_label()
#     ax.legend(loc="upper right", fontsize=8)

#     return frac_outside


# def fz_ipf_render(
#     sr_quat: np.ndarray,
#     hr_quat: np.ndarray,
#     sym_class="Oh",
#     ref_dir="Z",
#     max_points: int = 5000,
#     tol_deg: float = 0.5,
#     out_png: str = "fz_ipf_sr_hr.png",
#     overwrite: bool = True,
# ):
#     """
#     Render SR and HR FZ-IPF plots side by side into a single PNG.

#     Parameters
#     ----------
#     sr_quat : ndarray
#         SR quaternion array (H,W,4) or (4,H,W).
#     hr_quat : ndarray
#         HR quaternion array (H,W,4) or (4,H,W).
#     sym_class : str or orix symmetry
#         Symmetry group.
#     ref_dir : str
#         Reference direction ("X","Y","Z").
#     max_points : int
#         Max orientations to plot for speed.
#     tol_deg : float
#         Misorientation tolerance.
#     out_png : str
#         Output file path.
#     overwrite : bool
#         Skip if file exists and overwrite=False.

#     Returns
#     -------
#     dict
#         {"SR_frac_outside": float, "HR_frac_outside": float, "out_png": str}
#     """
#     if out_png and not overwrite and os.path.exists(out_png):
#         print(f"Skipping existing {out_png}")
#         return {"out_png": out_png}

#     # ------------------------------------------------------------------
#     # Downsample
#     # ------------------------------------------------------------------
#     sr_flat = sr_quat.reshape(-1, 4)
#     hr_flat = hr_quat.reshape(-1, 4)

#     if max_points and sr_flat.shape[0] > max_points:
#         sr_flat = sr_flat[np.random.choice(sr_flat.shape[0], max_points, replace=False)]
#     if max_points and hr_flat.shape[0] > max_points:
#         hr_flat = hr_flat[np.random.choice(hr_flat.shape[0], max_points, replace=False)]

#     # ------------------------------------------------------------------
#     # Figure setup
#     # ------------------------------------------------------------------
#     fig, axes = plt.subplots(
#         1,
#         2,
#         subplot_kw={"projection": "stereographic"},
#         figsize=(12, 6),
#         constrained_layout=True,
#     )

#     sr_frac = _plot_single_fz_ipf(
#         axes[0], sr_flat, sym_class, ref_dir, tol_deg, label="SR"
#     )
#     hr_frac = _plot_single_fz_ipf(
#         axes[1], hr_flat, sym_class, ref_dir, tol_deg, label="HR"
#     )

#     # ------------------------------------------------------------------
#     # Save
#     # ------------------------------------------------------------------
#     os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#     fig.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"✅ Saved SR-HR FZ IPF plot → {out_png}")

#     return {
#         "SR_frac_outside": sr_frac,
#         "HR_frac_outside": hr_frac,
#         "out_png": out_png,
#     }


# # # -*-coding:utf-8 -*-
# # """
# # File:        unfolded_ipf.py
# # Created at:  2025/10/18 20:52:10
# # Author:      Warren Zamudio
# # Contact:     wzamudio@ucsb.edu
# # Description: Plot orientations inside the Fundamental Zone (FZ) using custom FZ reduction
# #              and align visualization with ORIX's canonical FZ boundaries.
# # """
# # from pathlib import Path
# # from typing import Optional
# # import numpy as np

# # import matplotlib.pyplot as plt
# # import matplotlib.patches as mpatches
# # import matplotlib.path as mpath
# # from copy import deepcopy

# # from orix.quaternion import Orientation, symmetry
# # from orix.vector import Vector3d

# # from utils.quat_ops import (
# #     assert_quaternion_shape,
# #     to_spatial_quat,
# #     normalize_quaternions,
# #     enforce_hemisphere,
# #     reduce_to_fz_min_angle,
# # )


# def plot_fz_ipf(
#     arr_hw4: np.ndarray,
#     sym_class="Oh",
#     ref_dir="Z",
#     max_points: int | None = None,
#     tol_deg: float = 0.5,
#     out_png: str = "fz_ipf_plot.png",
# ):
#     """
#     Plot orientations inside the Fundamental Zone (FZ) with FZ boundary overlay.
#     Uses custom reduce_to_fz_min_angle and then canonicalizes to ORIX's FZ sector.

#     Parameters
#     ----------
#     arr_hw4 : ndarray
#         Quaternion array, (H,W,4) or (4,H,W).
#     sym_class : str or orix symmetry, default="Oh"
#         Symmetry class.
#     ref_dir : str, default="Z"
#         Reference direction to project ("X","Y","Z").
#     max_points : int, optional
#         Maximum orientations to plot (for large datasets).
#     tol_deg : float, default=0.5
#         Misorientation tolerance (deg) for classifying outside-FZ.
#     out_png : str
#         Output PNG path.

#     Returns
#     -------
#     dict
#         {
#             "frac_outside": float,
#             "n_points": int,
#             "out_png": str
#         }
#     """
#     # ------------------------------------------------------------------
#     # Quaternion preprocessing
#     # ------------------------------------------------------------------
#     assert_quaternion_shape(arr_hw4)

#     # Ensure (H,W,4)
#     q_spatial = to_spatial_quat(arr_hw4)
#     q_spatial = normalize_quaternions(q_spatial, axis=-1)
#     q_spatial = enforce_hemisphere(q_spatial, scalar_first=True)

#     q_flat = q_spatial.reshape(-1, 4)
#     n_total = q_flat.shape[0]

#     # Downsample for speed
#     if max_points and n_total > max_points:
#         idx = np.random.choice(n_total, max_points, replace=False)
#         q_flat = q_flat[idx]

#     # ------------------------------------------------------------------
#     # Symmetry + FZ reduction
#     # ------------------------------------------------------------------
#     sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class
#     q_fz, op_map = reduce_to_fz_min_angle(q_flat, sym=sym, return_op_map=True)

#     # Canonicalize into ORIX's standard FZ sector
#     ori_fz = Orientation(q_fz, symmetry=sym).map_into_symmetry_reduced_zone()
#     ori_orig = Orientation(q_flat, symmetry=sym)

#     # Outside mask (non-identity operator)
#     outside_mask = op_map != 0
#     frac_outside = outside_mask.mean() * 100

#     # ------------------------------------------------------------------
#     # Project along reference direction
#     # ------------------------------------------------------------------
#     ref_map = {
#         "X": Vector3d.xvector(),
#         "Y": Vector3d.yvector(),
#         "Z": Vector3d.zvector(),
#     }
#     v_ref = ref_map[ref_dir.upper()]
#     v_plot = ori_fz * v_ref

#     # Optional: background of original (unreduced) orientations
#     v_orig = ori_orig * v_ref

#     # ------------------------------------------------------------------
#     # Plot
#     # ------------------------------------------------------------------
#     fig, ax = plt.subplots(subplot_kw={"projection": "stereographic"}, figsize=(6, 6))
#     ax.set_title(
#         f"{sym.name} FZ IPF ({ref_dir}) — Outside FZ before mapping: {frac_outside:.2f}%",
#         pad=20,
#     )

#     # Show original in red
#     ax.scatter(v_orig, c="red", s=5, alpha=0.9, label="Original (unreduced)")

#     # Inside/outside mapped
#     ax.scatter(
#         v_plot[~outside_mask], c="grey", s=5, alpha=0.7, label="Inside FZ originally"
#     )
#     if np.any(outside_mask):
#         ax.scatter(
#             v_plot[outside_mask],
#             c="green",
#             s=5,
#             alpha=0.9,
#             label="Mapped from outside FZ",
#         )

#     # ------------------------------------------------------------------
#     # Symmetry axes (cubic example)
#     # ------------------------------------------------------------------
#     v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
#     ax.draw_circle(v4fold, color="blue")

#     v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
#     ax.draw_circle(v3fold, color="blue")

#     v2fold = Vector3d(
#         [
#             [1, 0, 1],
#             [0, 1, 1],
#             [-1, 0, 1],
#             [0, -1, 1],
#             [1, 1, 0],
#             [-1, -1, 0],
#             [-1, 1, 0],
#             [1, -1, 0],
#         ]
#     )
#     ax.draw_circle(v2fold, color="blue")

#     # ------------------------------------------------------------------
#     # Draw canonical FZ boundary
#     # ------------------------------------------------------------------
#     # sector = sym.fundamental_sector
#     # original_pole = deepcopy(sector._pole)
#     # sector._pole = ax.pole
#     # edges = sector.edges
#     # sector._pole = original_pole
#     # x, y, _ = ax._pretransform_input((edges,))
#     # patch = mpatches.PathPatch(
#     #     mpath.Path(np.column_stack([x, y]), closed=True),
#     #     facecolor="none",
#     #     edgecolor="black",
#     #     linewidth=2.0,
#     #     alpha=1.0,
#     #     zorder=5,
#     # )
#     # ax.add_patch(patch)

#     ax.set_labels("RD", "TD", None)
#     ax.show_hemisphere_label()
#     ax.legend(loc="upper right", fontsize=9)

#     fig.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     return {
#         "frac_outside": frac_outside,
#         "n_points": len(q_flat),
#         "out_png": out_png,
#     }


# # def plot_sr_hr_fz_ipf(
# #     sr_quat: np.ndarray,
# #     hr_quat: np.ndarray,
# #     sym_class="Oh",
# #     ref_dir="Z",
# #     max_points: Optional[int] = 5000,
# #     out_dir: str = "fz_plots",
# #     tol_deg: float = 0.5,
# #     prefix: str = "sample",
# # ):
# #     """
# #     Plot FZ IPF coverage for SR and HR quaternion arrays side-by-side.

# #     Parameters
# #     ----------
# #     sr_quat : ndarray
# #         SR quaternion array (H, W, 4) or (4, H, W).
# #     hr_quat : ndarray
# #         HR quaternion array (H, W, 4) or (4, H, W).
# #     sym_class : str or orix symmetry, default="Oh"
# #         Symmetry class.
# #     ref_dir : str, default="Z"
# #         Reference direction for IPF projection.
# #     max_points : int, optional
# #         Maximum orientations to plot.
# #     out_dir : str, default="fz_plots"
# #         Output directory for saved plots.
# #     tol_deg : float, default=0.5
# #         Tolerance angle (deg).
# #     prefix : str, default="sample"
# #         File prefix for naming output figures.

# #     Returns
# #     -------
# #     dict
# #         {"SR": sr_result, "HR": hr_result}
# #     """
# #     out_dir = Path(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     sr_out = out_dir / f"{prefix}_SR_fz_ipf.png"
# #     hr_out = out_dir / f"{prefix}_HR_fz_ipf.png"

# #     print(f" Plotting SR FZ-IPF → {sr_out}")
# #     sr_result = plot_fz_ipf(
# #         sr_quat,
# #         sym_class=sym_class,
# #         ref_dir=ref_dir,
# #         max_points=max_points,
# #         tol_deg=tol_deg,
# #         out_png=str(sr_out),
# #     )

# #     print(f"Plotting HR FZ-IPF → {hr_out}")
# #     hr_result = plot_fz_ipf(
# #         hr_quat,
# #         sym_class=sym_class,
# #         ref_dir=ref_dir,
# #         max_points=max_points,
# #         tol_deg=tol_deg,
# #         out_png=str(hr_out),
# #     )

# #     return {"SR": sr_result, "HR": hr_result}


# # # ----------------------------------------------------------------------
# # # Example usage
# # # ----------------------------------------------------------------------
# # if __name__ == "__main__":
# #     from orix.quaternion import symmetry as SYM

# #     q = np.random.randn(128, 128, 4).astype(np.float32)

# #     result = plot_fz_ipf(
# #         q,
# #         sym_class=SYM.O,
# #         ref_dir="Z",
# #         max_points=5000,
# #         tol_deg=0.5,
# #         out_png="fz_ipf_debug.png",
# #     )

# #     print(result)
