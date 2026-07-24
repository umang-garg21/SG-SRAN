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


def _expand_ipf_key_limits_for_labels(
    ax,
    texts,
    *,
    pad_frac: float = 0.04,
) -> None:
    """
    Expand the IPF key axes so native crystallographic labels fit cleanly.

    The default orix label anchors are material-aware, but they sit very close
    to the boundary of the stereographic triangle. Expanding the view limits
    slightly keeps the labels in the surrounding white space without forcing
    hard-coded coordinates that only work for one symmetry family.
    """
    visible_texts = [txt for txt in texts if txt.get_visible() and str(txt.get_text()).strip()]
    if not visible_texts:
        return

    canvas = getattr(getattr(ax, "figure", None), "canvas", None)
    if canvas is None:
        return

    try:
        canvas.draw()
        renderer = canvas.get_renderer()
    except Exception:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xmin, xmax = sorted((float(x0), float(x1)))
    ymin, ymax = sorted((float(y0), float(y1)))

    for txt in visible_texts:
        try:
            bbox_disp = txt.get_window_extent(renderer=renderer).expanded(1.06, 1.10)
            bbox_data = ax.transData.inverted().transform(bbox_disp.get_points())
        except Exception:
            continue
        xmin = min(xmin, float(np.min(bbox_data[:, 0])))
        xmax = max(xmax, float(np.max(bbox_data[:, 0])))
        ymin = min(ymin, float(np.min(bbox_data[:, 1])))
        ymax = max(ymax, float(np.max(bbox_data[:, 1])))

    span_x = max(abs(float(x1) - float(x0)), 1e-6)
    span_y = max(abs(float(y1) - float(y0)), 1e-6)
    pad_x = float(pad_frac) * span_x
    pad_y = float(pad_frac) * span_y

    if x0 <= x1:
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
    else:
        ax.set_xlim(xmax + pad_x, xmin - pad_x)
    if y0 <= y1:
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
    else:
        ax.set_ylim(ymax + pad_y, ymin - pad_y)


def style_ipf_key_axes(
    ax,
    *,
    title: str | None = "IPF Key",
    title_fontsize: float = 10.0,
    label_fontsize: float = 8.0,
    label_margin_frac: float = 0.04,
) -> None:
    """
    Draw a clean IPF key while preserving the correct symmetry-specific labels.

    Unlike the previous implementation, this does not hard-code cubic label
    strings or positions. It keeps the native orix crystallographic labels so
    the key stays correct for cubic, hexagonal, and other supported materials.
    """
    ax.plot_ipf_color_key(show_title=False)
    if title:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight="semibold",
            pad=6,
            color="#111111",
        )
    else:
        ax.set_title("")

    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    try:
        ax.set_xticks([])
        ax.set_yticks([])
    except Exception:
        pass

    for spine in getattr(ax, "spines", {}).values():
        try:
            spine.set_visible(False)
        except Exception:
            pass

    for line in getattr(ax, "lines", []):
        try:
            line.set_color("none")
            line.set_linewidth(0.0)
            line.set_clip_on(True)
        except Exception:
            pass

    for patch in getattr(ax, "patches", []):
        try:
            patch.set_linewidth(0.0)
            patch.set_edgecolor("none")
            patch.set_joinstyle("round")
            patch.set_clip_on(True)
        except Exception:
            pass

    for coll in getattr(ax, "collections", []):
        try:
            coll.set_linewidths(0.0)
        except Exception:
            pass
        try:
            coll.set_edgecolor("face")
        except Exception:
            pass

    label_texts = []
    for txt in list(getattr(ax, "texts", [])):
        label = str(txt.get_text()).strip()
        if not label:
            try:
                txt.set_visible(False)
            except Exception:
                pass
            continue
        try:
            txt.set_fontsize(label_fontsize)
            txt.set_fontweight("normal")
            txt.set_color("#111111")
            txt.set_clip_on(False)
            txt.set_zorder(5)
        except Exception:
            pass
        label_texts.append(txt)

    _expand_ipf_key_limits_for_labels(ax, label_texts, pad_frac=label_margin_frac)


def add_ipf_key_panel(
    fig,
    subplot_spec,
    sym_class,
    *,
    title: str | None = "IPF Key",
    title_fontsize: float = 10.0,
    label_fontsize: float = 8.0,
    title_height_ratio: float = 0.28,
):
    """
    Add a dedicated white IPF-key panel with a separate title band.

    Splitting the title from the key itself keeps the crystallographic labels
    from competing with the title, which makes the layout more robust across
    different materials and figure sizes.
    """
    if title:
        title_ratio = float(np.clip(title_height_ratio, 0.15, 0.45))
        key_cell = subplot_spec.subgridspec(
            2,
            1,
            height_ratios=[title_ratio, 1.0 - title_ratio],
            hspace=0.0,
        )
        ax_title = fig.add_subplot(key_cell[0, 0])
        ax_title.set_facecolor("white")
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.45,
            title,
            fontsize=title_fontsize,
            fontweight="semibold",
            color="#111111",
            ha="center",
            va="center",
        )
        key_spec = key_cell[1, 0]
    else:
        key_spec = subplot_spec

    ax_key = fig.add_subplot(key_spec, projection="ipf", symmetry=sym_class.laue)
    style_ipf_key_axes(
        ax_key,
        title=None,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )
    return ax_key


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
            to_quat_first=False,
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
        add_ipf_key_panel(
            fig,
            gs[0, -1],
            sym_class,
            title="IPF Key",
            title_fontsize=10.0,
            label_fontsize=8.0,
        )

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
