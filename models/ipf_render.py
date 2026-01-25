# -*-coding:utf-8 -*-
"""
File:        ipf_render.py
Created at:  2025/10/17 16:39:07
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""


import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Orientation
from orix.vector import Vector3d
from orix import plot as orix_plot

from utils.quat_ops import format_quaternions, assert_quaternion_shape

# Predefined reference directions
_DIRS = {
    "X": Vector3d((1, 0, 0)),
    "Y": Vector3d((0, 1, 0)),
    "Z": Vector3d((0, 0, 1)),
}


# ======================================================================
# Core RGB rendering
# ======================================================================


def render_ipf_rgb(
    q_arr: np.ndarray,
    sym_class,
    ref_dir: str = "ALL",
) -> np.ndarray | list[np.ndarray]:
    """
    Render quaternion orientation array to RGB image(s) for IPF coloring.

    Parameters
    ----------
    q_arr : ndarray
        Quaternion array of shape (H, W, 4), scalar-first or scalar-last. TODO: add check (should only be scalar first?)
    sym_class : orix symmetry
        Symmetry object for coloring.
    ref_dir : {"X","Y","Z","ALL"}, default="ALL"
        Reference direction(s) for coloring.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        RGB array(s) of shape (H, W, 3).
        If ref_dir="ALL", returns list of three arrays [X,Y,Z].
    """
    assert_quaternion_shape(q_arr)

    ori = Orientation(q_arr)
    ori.symmetry = sym_class
    ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

    ref_dir_lc = ref_dir.lower()
    show_all = ref_dir_lc == "all"
    directions = ("X", "Y", "Z") if show_all else (ref_dir.upper(),)

    rgb_list = []
    for d in directions:
        if d not in _DIRS:
            raise ValueError(f"Invalid ref_dir '{d}'. Must be one of X,Y,Z,ALL.")
        ckey.direction = _DIRS[d]
        rgb_list.append(ckey.orientation2color(ori))

    return rgb_list if show_all else rgb_list[0]


# ======================================================================
# Full IPF image rendering
# ======================================================================


def render_ipf_image(
    q_arr: np.ndarray,
    sym_class,
    out_png: Optional[str] = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    format_input: bool = True,
):
    """
    Render quaternion orientation array to an IPF image with consistent formatting.

    Parameters
    ----------
    q_arr : ndarray
        Quaternion array of shape (H, W, 4), any scalar order.
    sym_class : orix symmetry
        Symmetry object for coloring.
    out_png : str, optional
        Output PNG file path. If None, figure is not saved.
    ref_dir : {"X","Y","Z","ALL"}, default="ALL"
        Reference direction(s) for coloring.
    include_key : bool, default=True
        Whether to include IPF color key panel.
    overwrite : bool, default=False
        If False, skip rendering if file exists.
    format_input : bool, default=True
        If True, canonicalize quaternions via `format_quaternions`.

    Returns
    -------
    out_png : str or None
        Saved file path if provided, else None.
    """
    # Skip if file already exists
    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    # Canonicalize quaternion array
    if format_input:
        q_arr = format_quaternions(
            q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            quat_first=False,
        )

    # Render RGB map(s)
    rgb_out = render_ipf_rgb(q_arr, sym_class, ref_dir=ref_dir)
    show_all = isinstance(rgb_out, list)
    ncols = 3 if show_all else 1
    key_cols = 1 if include_key else 0
    fig_cols = ncols + key_cols
    wr = [1] * ncols + ([0.9] if include_key else [])

    fig = plt.figure(
        constrained_layout=False,
        figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
    )
    gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

    if show_all:
        for name, rgb, ax in zip(("X", "Y", "Z"), rgb_out, axes):
            ax.imshow(rgb)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"IPF-{name}")
            ax.axis("off")
    else:
        ref = ref_dir.upper()
        axes[0].imshow(rgb_out)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title(f"IPF-{ref}")
        axes[0].axis("off")

    if include_key:
        ax_ipf = fig.add_subplot(gs[0, -1], projection="ipf", symmetry=sym_class.laue)
        ax_ipf.plot_ipf_color_key()
        ax_ipf.set_title("")

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_png
    else:
        return None


# def render_sr_hr_side_by_side(
#     sr_q_arr: np.ndarray,
#     hr_q_arr: np.ndarray,
#     sym_class,
#     out_png: Optional[str] = None,
#     ref_dir: str = "ALL",
#     include_key: bool = True,
#     overwrite: bool = False,
#     format_input: bool = True,
#     dpi: int = 300,
# ):
#     """
#     Render SR (Super-Resolution) and HR (High-Resolution) quaternion orientation maps
#     in a 2-row layout (SR on top, HR on bottom), optionally with IPF key.

#     Parameters
#     ----------
#     sr_q_arr : ndarray
#         SR quaternion array of shape (H, W, 4).
#     hr_q_arr : ndarray
#         HR quaternion array of shape (H, W, 4).
#     sym_class : orix symmetry
#         Symmetry object for IPF coloring.
#     out_png : str, optional
#         Output PNG file path. If None, figure is not saved.
#     ref_dir : {"X","Y","Z","ALL"}, default="ALL"
#         Reference direction(s) for coloring.
#     include_key : bool, default=True
#         Whether to include IPF color key panel.
#     overwrite : bool, default=False
#         If False, skip rendering if file exists.
#     format_input : bool, default=True
#         If True, canonicalize quaternions.
#     dpi : int, default=300
#         Figure DPI for saved PNG.
#     """
#     # -------------------------------------------------------------------------
#     # Early exit if file already exists
#     # -------------------------------------------------------------------------
#     if out_png and not overwrite and os.path.exists(out_png):
#         return out_png

#     # -------------------------------------------------------------------------
#     # Format quaternions (reduce to FZ, normalize, hemisphere, etc.)
#     # -------------------------------------------------------------------------
#     if format_input:
#         sr_q_arr = format_quaternions(
#             sr_q_arr,
#             normalize=True,
#             hemisphere=True,
#             reduce_fz=True,
#             sym=sym_class,
#             scalar_first=True,
#             quat_first=False,
#         )
#         hr_q_arr = format_quaternions(
#             hr_q_arr,
#             normalize=True,
#             hemisphere=True,
#             reduce_fz=True,
#             sym=sym_class,
#             scalar_first=True,
#             quat_first=False,
#         )

#     # -------------------------------------------------------------------------
#     # Convert to IPF RGB maps
#     # -------------------------------------------------------------------------
#     sr_rgb = render_ipf_rgb(sr_q_arr, sym_class, ref_dir=ref_dir)
#     hr_rgb = render_ipf_rgb(hr_q_arr, sym_class, ref_dir=ref_dir)

#     multi_ref = isinstance(sr_rgb, list)
#     ncols = 3 if multi_ref else 1
#     key_cols = 1 if include_key else 0
#     total_cols = ncols + key_cols
#     total_rows = 2  # SR top, HR bottom

#     # -------------------------------------------------------------------------
#     # Figure setup
#     # -------------------------------------------------------------------------
#     base_w = 5.0
#     key_w = 2.6 if include_key else 0
#     fig_w = base_w * ncols + key_w
#     fig_h = 2 * 4.5
#     fig = plt.figure(figsize=(fig_w, fig_h))
#     gs = fig.add_gridspec(
#         total_rows,
#         total_cols,
#         width_ratios=[1] * ncols + ([0.9] if include_key else []),
#         height_ratios=[1, 1],
#         hspace=0.25,
#         wspace=0.25,
#     )

#     def _imshow(ax, img, title):
#         ax.imshow(img)
#         ax.set_aspect("equal", adjustable="box")
#         ax.set_title(title, fontsize=10)
#         ax.axis("off")

#     # -------------------------------------------------------------------------
#     # Plot SR (top row) and HR (bottom row)
#     # -------------------------------------------------------------------------
#     if multi_ref:
#         for j, (name, img) in enumerate(zip(("X", "Y", "Z"), hr_rgb)):
#             _imshow(fig.add_subplot(gs[0, j]), img, f"HR IPF-{name}")
#         for j, (name, img) in enumerate(zip(("X", "Y", "Z"), sr_rgb)):
#             _imshow(fig.add_subplot(gs[1, j]), img, f"SR IPF-{name}")
#     else:
#         _imshow(fig.add_subplot(gs[0, 0]), sr_rgb, f"SR IPF-{ref_dir.upper()}")
#         _imshow(fig.add_subplot(gs[1, 0]), hr_rgb, f"HR IPF-{ref_dir.upper()}")

#     # -------------------------------------------------------------------------
#     # IPF color key
#     # -------------------------------------------------------------------------
#     if include_key:
#         ax_key = fig.add_subplot(gs[:, -1], projection="ipf", symmetry=sym_class.laue)
#         ax_key.plot_ipf_color_key()
#         ax_key.set_title("")

#     # -------------------------------------------------------------------------
#     # Save figure
#     # -------------------------------------------------------------------------
#     if out_png:
#         os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#         fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
#         plt.close(fig)
#         print(f"Saved SR-HR comparison to: {out_png}")
#         return out_png

#     plt.show()
#     return None
