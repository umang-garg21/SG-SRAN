import os
import csv
import sys
import argparse
from datetime import datetime

import matplotlib

# VSCode SSH/remote sessions usually have no GUI display. WebAgg serves plots
# in a local browser via forwarded ports.
if not os.environ.get("DISPLAY"):
    matplotlib.use("WebAgg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
import numpy as np
import orix.plot
from orix.projections.stereographic import _vector2xy
from orix.quaternion import Orientation
from orix.vector import Vector3d

from utils.quat_ops import format_quaternions, quat_conjugate, to_scalar_first, to_spatial_quat
from utils.symmetry_utils import resolve_symmetry


FP = "/data/warren/materials/materials_data_mount/x_slice_0100.npy"

DIR_VECTORS = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}


def build_formatted_quaternions(
    fp: str,
    sym: str | object = "Oh",
    scalar_first: bool = False,
    quat_spatial: bool = False,
    apply_format: bool = True,
    quaternion_convention: str = "bunge",
) -> tuple[np.ndarray, object]:
    """
    Load a quaternion field from .npy and return plotting-ready quaternions.

    Parameters
    ----------
    fp : str
        Path to the .npy quaternion array.
    sym : str | object
        Symmetry name or orix symmetry object. Strings are resolved with
        utils.symmetry_utils.resolve_symmetry.
    scalar_first : bool
        True if stored as [w, x, y, z], False if stored as [x, y, z, w].
    quat_spatial : bool
        True if stored as (4, *spatial), False if stored as (*spatial, 4).
    apply_format : bool
        If True, apply normalize + hemisphere + FZ reduction.
    quaternion_convention : str
        "bunge" (passive) or "active". Active is converted by quaternion conjugate.
    """
    q_arr = np.load(fp, mmap_mode="r")
    print(f"loaded: {os.path.abspath(fp)}")
    print(f"shape={q_arr.shape}, dtype={q_arr.dtype}")

    # Resolve aliases like "oh", "m-3m", etc. into an orix symmetry object.
    sym_obj = resolve_symmetry(sym)

    # Validate declared storage layout, then normalize to (*spatial, 4).
    if quat_spatial:
        if q_arr.shape[0] != 4:
            raise ValueError(
                f"quat_spatial=True expects quaternion axis first (4, *spatial), got {q_arr.shape}"
            )
    else:
        if q_arr.shape[-1] != 4:
            raise ValueError(
                f"quat_spatial=False expects quaternion axis last (*spatial, 4), got {q_arr.shape}"
            )
    q = to_spatial_quat(q_arr)

    q = np.asarray(q, dtype=np.float32)
    if not q.flags.writeable:
        q = q.copy()

    # Convert to scalar-first [w, x, y, z] if data is stored scalar-last [x, y, z, w].
    if not scalar_first:
        q = to_scalar_first(q)

    conv = str(quaternion_convention).strip().lower()
    if conv == "active":
        q = quat_conjugate(q)
    elif conv not in {"bunge", "passive"}:
        raise ValueError(
            f"quaternion_convention must be 'bunge' or 'active', got {quaternion_convention!r}"
        )

    if apply_format:
        q = format_quaternions(
            q,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_obj,
            to_quat_first=False,
        )
    else:
        # Keep plotting input as (*spatial, 4), scalar-first.
        q = q.astype(np.float32, copy=False)

    return q, sym_obj


def build_direction_cache(
    q_cf: np.ndarray, sym_obj: object
) -> dict[str, dict[str, np.ndarray | Vector3d]]:
    ori_2d = Orientation(q_cf)
    ori_2d.symmetry = sym_obj

    q_flat = q_cf.reshape(-1, 4)
    ori_flat = Orientation(q_flat, symmetry=sym_obj)

    cache: dict[str, dict[str, np.ndarray | Vector3d]] = {}
    for d, v_ref in DIR_VECTORS.items():
        ckey = orix.plot.IPFColorKeyTSL(sym_obj.laue)
        ckey.direction = v_ref

        rgb_map = ckey.orientation2color(ori_2d)
        colors_flat = ckey.orientation2color(ori_flat)

        v_stereo = ori_flat * v_ref
        v_key = v_stereo.in_fundamental_sector(sym_obj)

        cache[d] = {
            "rgb": rgb_map,
            "colors": colors_flat,
            "v_stereo": v_stereo,
            "v_key": v_key,
        }
    return cache


def show_interactive_ipf(q_cf: np.ndarray, sym_obj: object) -> None:
    cache = build_direction_cache(q_cf, sym_obj)

    fig = plt.figure(figsize=(24.8, 7.2))
    # Keep a roomy plotting area; direction controls are adaptively repositioned
    # when IPF zoom gets very tight.
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.62, 1.33, 0.76, 1.20],
        wspace=0.24,
        top=0.86,
        bottom=0.08,
    )
    ax_ipf = fig.add_subplot(gs[0, 0])
    ax_stereo = fig.add_subplot(gs[0, 1], projection="stereographic")
    ax_key = fig.add_subplot(gs[0, 2], projection="ipf", symmetry=sym_obj.laue)
    ax_list = fig.add_subplot(gs[0, 3])
    ax_list.axis("off")
    ax_list.set_title("Selected points", fontsize=10, pad=8)
    ax_key.plot_ipf_color_key(show_title=False)
    key_static_labels = list(ax_key.texts)

    current_dir = {"value": "Z"}
    current_hemi = {"value": "Upper"}
    dir_buttons: dict[str, Button] = {}
    ipf_point_markers = []
    ipf_point_labels = []
    stereo_point_markers = []
    stereo_point_labels = []
    key_point_markers = []
    key_point_labels = []
    h, w = q_cf.shape[:2]
    n_total = h * w
    bg_step = max(n_total // 50000, 1)
    bg_idx = np.arange(0, n_total, bg_step)
    hover_idx = [None]
    selected_idx = [None]

    # Precompute projected coordinates for each direction.
    for d, dcache in cache.items():
        x_st_upper, y_st_upper = _vector2xy(dcache["v_stereo"], pole=-1)
        x_st_lower, y_st_lower = _vector2xy(dcache["v_stereo"], pole=1)
        x_key, y_key = _vector2xy(dcache["v_key"], pole=ax_key.pole)
        dcache["x_st_by_pole"] = {
            -1: np.asarray(x_st_upper).reshape(-1),
            1: np.asarray(x_st_lower).reshape(-1),
        }
        dcache["y_st_by_pole"] = {
            -1: np.asarray(y_st_upper).reshape(-1),
            1: np.asarray(y_st_lower).reshape(-1),
        }
        dcache["x_key"] = np.asarray(x_key).reshape(-1)
        dcache["y_key"] = np.asarray(y_key).reshape(-1)
        z_st = np.asarray(dcache["v_stereo"].z).reshape(-1)
        z_key = np.asarray(dcache["v_key"].z).reshape(-1)
        dcache["vis_st_by_pole"] = {
            -1: z_st >= 0,
            1: z_st <= 0,
        }
        dcache["vis_key"] = z_key >= 0 if ax_key.pole == -1 else z_key <= 0
        dcache["x_st"] = dcache["x_st_by_pole"][-1]
        dcache["y_st"] = dcache["y_st_by_pole"][-1]
        dcache["vis_st"] = dcache["vis_st_by_pole"][-1]

    def _apply_stereo_pole_to_cache(pole: int):
        for dcache in cache.values():
            dcache["x_st"] = dcache["x_st_by_pole"][pole]
            dcache["y_st"] = dcache["y_st_by_pole"][pole]
            dcache["vis_st"] = dcache["vis_st_by_pole"][pole]

    default_pole = -1 if str(current_hemi["value"]).lower() == "upper" else 1
    _apply_stereo_pole_to_cache(default_pole)

    c0 = cache[current_dir["value"]]
    bg_idx0 = bg_idx[c0["vis_st"][bg_idx]]
    im = ax_ipf.imshow(c0["rgb"])
    im.get_cursor_data = lambda event: None
    im.format_cursor_data = lambda data: ""
    ax_ipf.set_title(f"IPF-{current_dir['value']}")
    ax_ipf.axis("off")
    ipf_full_xlim = tuple(float(v) for v in ax_ipf.get_xlim())
    ipf_full_ylim = tuple(float(v) for v in ax_ipf.get_ylim())
    ipf_full_xspan = max(abs(ipf_full_xlim[1] - ipf_full_xlim[0]), 1e-12)
    ipf_full_yspan = max(abs(ipf_full_ylim[1] - ipf_full_ylim[0]), 1e-12)

    bg_scatter = Axes.scatter(
        ax_stereo,
        c0["x_st"][bg_idx0],
        c0["y_st"][bg_idx0],
        c=c0["colors"][bg_idx0],
        s=50,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
    )

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
    stereo_guide_artists = []
    stereo_outer_labels = []

    def _safe_remove_artists(items):
        while items:
            artist = items.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def _draw_stereo_guides():
        _safe_remove_artists(stereo_guide_artists)
        for vv in (v4fold, v3fold, v2fold):
            out = ax_stereo.draw_circle(vv, color="black", linewidth=0.8, alpha=0.35)
            if out is None:
                continue
            if isinstance(out, (list, tuple)):
                stereo_guide_artists.extend(list(out))
            else:
                stereo_guide_artists.append(out)

    def _refresh_stereo_outer_labels():
        _safe_remove_artists(stereo_outer_labels)
        stereo_label_start = len(ax_stereo.texts)
        ax_stereo.set_labels("RD", "TD", None)
        ax_stereo.show_hemisphere_label()
        stereo_outer_labels.extend(list(ax_stereo.texts[stereo_label_start:]))
        for txt in stereo_outer_labels:
            txt.set_clip_on(True)
            txt.set_clip_path(ax_stereo.patch)

    _draw_stereo_guides()
    ax_stereo.set_title(
        f"Stereographic (IPF-{current_dir['value']} {current_hemi['value']})",
        fontsize=10,
    )
    _refresh_stereo_outer_labels()
    stereo_full_xlim = tuple(float(v) for v in ax_stereo.get_xlim())
    stereo_full_ylim = tuple(float(v) for v in ax_stereo.get_ylim())
    stereo_full_xspan = max(abs(stereo_full_xlim[1] - stereo_full_xlim[0]), 1e-12)
    stereo_full_yspan = max(abs(stereo_full_ylim[1] - stereo_full_ylim[0]), 1e-12)

    def _axis_is_full_view(cur_lim, full_lim, tol_frac: float = 0.03) -> bool:
        cur0, cur1 = float(cur_lim[0]), float(cur_lim[1])
        full0, full1 = float(full_lim[0]), float(full_lim[1])
        cur_span = abs(cur1 - cur0)
        full_span = abs(full1 - full0)
        if full_span <= 1e-12:
            return True
        span_ok = abs(cur_span - full_span) <= tol_frac * full_span
        cur_center = 0.5 * (cur0 + cur1)
        full_center = 0.5 * (full0 + full1)
        center_ok = abs(cur_center - full_center) <= tol_frac * full_span
        return span_ok and center_ok

    def _update_stereo_outer_label_visibility():
        xlim = ax_stereo.get_xlim()
        ylim = ax_stereo.get_ylim()
        show_labels = _axis_is_full_view(xlim, stereo_full_xlim) and _axis_is_full_view(
            ylim, stereo_full_ylim
        )
        for txt in stereo_outer_labels:
            txt.set_visible(show_labels)

    _update_stereo_outer_label_visibility()
    ax_stereo.callbacks.connect("xlim_changed", lambda _ax: _update_stereo_outer_label_visibility())
    ax_stereo.callbacks.connect("ylim_changed", lambda _ax: _update_stereo_outer_label_visibility())
    ax_key.set_title(f"IPF key (ref {current_dir['value']})", fontsize=10)
    key_full_xlim = [tuple(float(v) for v in ax_key.get_xlim())]
    key_full_ylim = [tuple(float(v) for v in ax_key.get_ylim())]
    key_full_locked = [False]
    key_interacted = [False]

    def _update_key_label_visibility():
        xlim = ax_key.get_xlim()
        ylim = ax_key.get_ylim()
        # Lock default key limits from the first stabilized callback after draw.
        if not key_full_locked[0]:
            key_full_xlim[0] = tuple(float(v) for v in xlim)
            key_full_ylim[0] = tuple(float(v) for v in ylim)
            key_full_locked[0] = True
        show_labels = (not key_interacted[0]) or (
            _axis_is_full_view(xlim, key_full_xlim[0], tol_frac=0.10)
            and _axis_is_full_view(ylim, key_full_ylim[0], tol_frac=0.10)
        )
        for txt in key_static_labels:
            txt.set_visible(show_labels)

    _update_key_label_visibility()
    ax_key.callbacks.connect("xlim_changed", lambda _ax: _update_key_label_visibility())
    ax_key.callbacks.connect("ylim_changed", lambda _ax: _update_key_label_visibility())
    ring_edge_color = "#111111"
    ring_edge_width = 2.2

    hover_marker_ipf = Axes.scatter(
        ax_ipf,
        [],
        [],
        s=70,
        facecolors="none",
        edgecolors=ring_edge_color,
        linewidths=ring_edge_width,
        zorder=12,
    )
    hover_marker = Axes.scatter(
        ax_stereo,
        [],
        [],
        s=130,
        facecolors="none",
        edgecolors=ring_edge_color,
        linewidths=ring_edge_width,
        zorder=6,
    )
    selected_marker = Axes.scatter(
        ax_stereo,
        [],
        [],
        marker="D",
        s=180,
        facecolors="none",
        edgecolors="#ff1f1f",
        linewidths=3.4,
        zorder=7,
    )
    hover_marker_key = Axes.scatter(
        ax_key,
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors=ring_edge_color,
        linewidths=ring_edge_width,
        zorder=7,
    )
    selected_marker_key = Axes.scatter(
        ax_key,
        [],
        [],
        marker="D",
        s=145,
        facecolors="none",
        edgecolors="#ff1f1f",
        linewidths=3.2,
        zorder=8,
    )

    def _format_coord(xd, yd):
        if xd is None or yd is None:
            return f"panel=IPF-{current_dir['value']}"
        x = int(np.round(xd))
        y = int(np.round(yd))
        if x < 0 or x >= w or y < 0 or y >= h:
            return f"panel=IPF-{current_dir['value']}"
        q = q_cf[y, x]
        q_str = ", ".join(f"{v:.6f}" for v in q)
        return (
            f"panel=IPF-{current_dir['value']}  "
            f"(x, y)=({x}, {y})  quaternion=({q_str})"
        )

    ax_ipf.format_coord = _format_coord

    info_text = fig.text(
        0.5,
        0.9,
        "Select direction (X/Y/Z); wheel zoom on plots; R/T/K reset zoom; C clears; H toggles shortcuts.",
        ha="center",
        va="center",
        fontsize=10,
    )
    shortcut_text_visible = {"value": True}
    shortcut_text = fig.text(
        0.01,
        0.965,
        "Shortcuts:\n"
        "X/Y/Z: set IPF direction\n"
        "Mouse wheel on plots: zoom in/out\n"
        # "Mouse wheel on stereographic: zoom in/out\n"
        # "Mouse wheel on IPF key: zoom in/out\n"
        "Ctrl+Left-drag on plots: pan\n"
        "R: reset IPF zoom\n"
        "T: reset stereographic zoom\n"
        "K: reset IPF key zoom\n"
        "C: clear selected points\n"
        "E: export selected points CSV\n"
        "[ ]: page list left/right\n"
        "Mouse wheel on list: page list\n"
        "Left click IPF or stereographic: add point\n"
        "Right click or Shift+Left: remove point\n"
        "H or ?: show/hide this help",
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.93, edgecolor="#bbbbbb", boxstyle="round,pad=0.35"),
    )
    selected_points: list[dict[str, object]] = []
    list_state = {"collapsed": False, "page": 0, "page_size": 25}
    dir_button_axes: dict[str, object] = {}
    stereo_hemi_buttons: dict[str, Button] = {}
    stereo_hemi_button_axes: dict[str, object] = {}
    clear_button_ax = [None]
    toggle_button_ax = [None]
    export_button_ax = [None]
    prev_page_ax = [None]
    next_page_ax = [None]
    page_indicator = [None]
    selected_list_text = ax_list.text(
        0.02,
        0.76,
        "Selected points:\n(none)",
        transform=ax_list.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        linespacing=1.2,
        zorder=2,
    )
    selected_row_artists = []
    selected_point_ptr = [None]
    list_view_meta = {"start": 0, "count": 0, "y_top": 0.62, "dy": 0.023}
    list_box_patch = [None]
    pan_state = {
        "active": False,
        "ax": None,
        "press_xy": None,
        "xlim0": None,
        "ylim0": None,
    }
    ctrl_state = {"down": False}

    last_hover = [None]

    def redraw_selected_ipf_points():
        def _safe_remove_all(items):
            while items:
                artist = items.pop()
                try:
                    artist.remove()
                except Exception:
                    # Ignore stale/missing artist handles to keep UI responsive.
                    pass

        _safe_remove_all(ipf_point_markers)
        _safe_remove_all(ipf_point_labels)
        _safe_remove_all(stereo_point_markers)
        _safe_remove_all(stereo_point_labels)
        _safe_remove_all(key_point_markers)
        _safe_remove_all(key_point_labels)

        dcache = cache[current_dir["value"]]
        for i, p in enumerate(selected_points, start=1):
            px, py = int(p["x"]), int(p["y"])
            marker = ax_ipf.plot(
                px,
                py,
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=ring_edge_color,
                markeredgewidth=ring_edge_width,
                markersize=8,
                zorder=10,
            )[0]
            label = ax_ipf.annotate(
                str(i),
                (px, py),
                xytext=(4, 4),
                textcoords="offset points",
                color=ring_edge_color,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=11,
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=0.2),
            )
            ipf_point_markers.append(marker)
            ipf_point_labels.append(label)

            idx = int(p["flat_idx"])
            if dcache["vis_st"][idx]:
                sx = float(dcache["x_st"][idx])
                sy = float(dcache["y_st"][idx])
                sm = Axes.scatter(
                    ax_stereo,
                    [sx],
                    [sy],
                    s=52,
                    facecolors="none",
                    edgecolors=ring_edge_color,
                    linewidths=ring_edge_width,
                    zorder=8,
                )
                sl = ax_stereo.annotate(
                    str(i),
                    (sx, sy),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color=ring_edge_color,
                    fontsize=7,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    zorder=9,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.15),
                )
                stereo_point_markers.append(sm)
                stereo_point_labels.append(sl)

            if dcache["vis_key"][idx]:
                kx = float(dcache["x_key"][idx])
                ky = float(dcache["y_key"][idx])
                km = Axes.scatter(
                    ax_key,
                    [kx],
                    [ky],
                    s=42,
                    facecolors="none",
                    edgecolors=ring_edge_color,
                    linewidths=ring_edge_width,
                    zorder=8,
                )
                kl = ax_key.annotate(
                    str(i),
                    (kx, ky),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color=ring_edge_color,
                    fontsize=7,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    zorder=9,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.15),
                )
                key_point_markers.append(km)
                key_point_labels.append(kl)

    def _total_pages() -> int:
        page_size = max(1, int(list_state["page_size"]))
        return max(1, (len(selected_points) + page_size - 1) // page_size)

    def _set_page(page_idx: int, redraw: bool = True):
        total = _total_pages()
        list_state["page"] = int(np.clip(int(page_idx), 0, total - 1))
        refresh_selected_points_text()
        if redraw:
            fig.canvas.draw_idle()

    def _update_page_indicator():
        txt = page_indicator[0]
        if txt is None:
            return
        total = _total_pages()
        page = int(np.clip(int(list_state["page"]), 0, total - 1))
        txt.set_text(f"Page {page + 1}/{total}")

    def _clear_selected_row_artists():
        while selected_row_artists:
            artist = selected_row_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def refresh_selected_points_text():
        _clear_selected_row_artists()
        if list_state["collapsed"]:
            selected_list_text.set_text("")
            list_view_meta["start"] = 0
            list_view_meta["count"] = 0
            if list_box_patch[0] is not None:
                list_box_patch[0].set_visible(False)
            return

        page_size = max(1, int(list_state["page_size"]))
        total = _total_pages()
        page = int(np.clip(int(list_state["page"]), 0, total - 1))
        list_state["page"] = page
        start = page * page_size
        end = min(len(selected_points), start + page_size)

        # Keep row spacing tied to actual panel pixel height so header/rows stay
        # aligned when figure size changes.
        axis_h_px = max(float(ax_list.bbox.height), 1.0)
        axis_w_px = max(float(ax_list.bbox.width), 1.0)
        instruction_font_size = 8.0
        row_font_size = 8.2
        col_header_font_size = 8.75
        line_h = (row_font_size * 1.2 * float(fig.dpi) / 72.0) / axis_h_px
        col_line_h = (col_header_font_size * 1.2 * float(fig.dpi) / 72.0) / axis_h_px
        instruction_line_h = (instruction_font_size * 1.2 * float(fig.dpi) / 72.0) / axis_h_px
        header_top = 0.80
        header_line_h = instruction_line_h
        dy = line_h * 1.03
        list_view_meta["dy"] = dy

        header_lines = [
            "Select points: Left click IPF/Stereo or click a list row",
            "Remove points: Right click or Shift+Left",
            "Select page: Prev/Next, wheel on panel, or [ ] keys",
        ]
        # Fixed column anchors so headers/rows always align even with
        # different header and row font sizes.
        col_x = {
            "idx": 0.05,
            "dir": 0.15,
            "x": 0.25,
            "y": 0.35,
            "q_w": 0.50,
            "q_x": 0.69,
            "q_y": 0.88,
            "q_z": 1.07,
        }
        row_hl_left = 0.022
        row_hl_right = max(1.20, float(col_x["q_z"]) + 0.13)
        col_specs = (
            ("idx", "idx", "center"),
            ("dir", "dir", "center"),
            ("x", "x", "center"),
            ("y", "y", "center"),
            ("q_w", "q_w", "center"),
            ("q_x", "q_x", "center"),
            ("q_y", "q_y", "center"),
            ("q_z", "q_z", "center"),
        )

        def _fmt_q(v: float) -> str:
            # Space-sign keeps positive/negative values vertically aligned
            # without showing a '+' for positive numbers.
            return format(float(v), " 10.6f")

        if not selected_points:
            list_state["page"] = 0
            selected_point_ptr[0] = None
            header_lines.append("(none)")
            list_view_meta["start"] = 0
            list_view_meta["count"] = 0
            data_y_top = header_top - len(header_lines) * header_line_h - 0.020
        else:
            if (
                selected_point_ptr[0] is None
                or int(selected_point_ptr[0]) < 0
                or int(selected_point_ptr[0]) >= len(selected_points)
            ):
                selected_point_ptr[0] = len(selected_points) - 1

            list_view_meta["start"] = start
            list_view_meta["count"] = end - start

            showing_y = header_top - len(header_lines) * header_line_h - line_h * 0.10 - 0.005
            showing_artist = ax_list.text(
                0.02,
                showing_y,
                f"Showing {start + 1}-{end} of {len(selected_points)}",
                transform=ax_list.transAxes,
                ha="left",
                va="top",
                fontsize=col_header_font_size,
                family="monospace",
                color="#333333",
                fontweight="bold",
                zorder=2,
            )
            selected_row_artists.append(showing_artist)

            col_y = showing_y - max(line_h * 1.05, col_line_h * 1.05) - 0.005
            for key, label, ha in col_specs:
                header_artist = ax_list.text(
                    col_x[key],
                    col_y,
                    label,
                    transform=ax_list.transAxes,
                    ha=ha,
                    va="top",
                    fontsize=col_header_font_size,
                    family="monospace",
                    color="#333333",
                    fontweight="bold",
                    zorder=2,
                )
                selected_row_artists.append(header_artist)

            data_y_top = col_y - max(dy * 0.90, col_line_h * 1.05)
            for rel_i, p in enumerate(selected_points[start:end]):
                global_i = start + rel_i
                y = data_y_top - rel_i * dy
                q = p["q"]
                active = int(selected_point_ptr[0]) == global_i
                row_color = "black" if not active else "#0b3d91"
                row_weight = "normal" if not active else "bold"
                if active:
                    hl = FancyBboxPatch(
                        (row_hl_left, y - dy * 0.92),
                        row_hl_right - row_hl_left,
                        dy * 0.96,
                        transform=ax_list.transAxes,
                        boxstyle="round,pad=0.001,rounding_size=0.002",
                        facecolor="#e8f1ff",
                        edgecolor="#90b6ff",
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=1.8,
                        clip_on=False,
                    )
                    ax_list.add_patch(hl)
                    selected_row_artists.append(hl)

                row_cells = {
                    "idx": f"{global_i + 1}.",
                    "dir": str(p["dir"]),
                    "x": f"{int(p['x'])}",
                    "y": f"{int(p['y'])}",
                    "q_w": _fmt_q(q[0]),
                    "q_x": _fmt_q(q[1]),
                    "q_y": _fmt_q(q[2]),
                    "q_z": _fmt_q(q[3]),
                }
                for key, _label, ha in col_specs:
                    cell_artist = ax_list.text(
                        col_x[key],
                        y,
                        row_cells[key],
                        transform=ax_list.transAxes,
                        ha=ha,
                        va="top",
                        fontsize=row_font_size,
                        family="monospace",
                        color=row_color,
                        fontweight=row_weight,
                        zorder=2,
                    )
                    selected_row_artists.append(cell_artist)

        list_view_meta["y_top"] = data_y_top
        selected_list_text.set_position((0.03, header_top))
        selected_list_text.set_text("\n".join(header_lines))
        _update_page_indicator()

        rows_for_box = int(list_view_meta["count"])
        box_left = 0.02
        # Slightly wider box (~4 monospace chars) to avoid right-edge clipping.
        char_w = (row_font_size * 0.60 * float(fig.dpi) / 72.0) / axis_w_px
        box_right = max(row_hl_right + 0.01, 0.99 + 4.0 * char_w)
        box_top = header_top + line_h * 0.45
        if rows_for_box > 0:
            box_bottom = data_y_top - (rows_for_box - 1) * dy - dy * 0.85
        else:
            box_bottom = header_top - len(header_lines) * header_line_h - line_h * 1.40
        box_bottom = max(0.02, box_bottom)
        box_height = max(0.05, box_top - box_bottom)

        if list_box_patch[0] is not None:
            try:
                list_box_patch[0].remove()
            except Exception:
                pass
            list_box_patch[0] = None

        list_box_patch[0] = FancyBboxPatch(
            (box_left, box_bottom),
            box_right - box_left,
            box_height,
            transform=ax_list.transAxes,
            boxstyle="round,pad=0.010,rounding_size=0.008",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=1.0,
            alpha=0.90,
            zorder=1,
            clip_on=False,
        )
        ax_list.add_patch(list_box_patch[0])

    def _update_selected_ptr_after_remove(removed_idx: int):
        ptr = selected_point_ptr[0]
        if ptr is None:
            return
        ptr = int(ptr)
        if not selected_points:
            selected_point_ptr[0] = None
            return
        if removed_idx == ptr:
            selected_point_ptr[0] = min(removed_idx, len(selected_points) - 1)
            return
        if removed_idx < ptr:
            selected_point_ptr[0] = ptr - 1

    def focus_selected_point(point_idx: int, redraw: bool = True):
        if point_idx < 0 or point_idx >= len(selected_points):
            return

        selected_point_ptr[0] = int(point_idx)
        p = selected_points[int(point_idx)]
        pdir = str(p["dir"])

        if pdir != current_dir["value"]:
            update_direction(pdir, redraw=False)

        update_selected_link_markers()
        refresh_selected_points_text()
        q = np.asarray(p["q"], dtype=float).reshape(4)
        info_text.set_text(
            f"Focused point {point_idx + 1}: dir={pdir},  (x, y)=({p['x']}, {p['y']}),  "
            f"quaternion=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})"
        )
        if redraw:
            fig.canvas.draw_idle()

    def handle_list_click(event) -> bool:
        if list_state["collapsed"] or not selected_points:
            return False
        if event.inaxes is not ax_list:
            return False
        if event.ydata is None:
            return False
        if event.xdata is None:
            return False

        start = int(list_view_meta["start"])
        count = int(list_view_meta["count"])
        if count <= 0:
            return False

        y_top = float(list_view_meta["y_top"])
        dy = float(list_view_meta["dy"])
        y = float(event.ydata)

        rel = int(np.round((y_top - y) / dy))
        if rel < 0 or rel >= count:
            return False
        row_y = y_top - rel * dy
        if abs(y - row_y) > dy * 0.6:
            return False

        global_idx = start + rel
        if global_idx < 0 or global_idx >= len(selected_points):
            return False

        remove_mode = (event.button == 3) or (
            event.button == 1
            and event.key is not None
            and "shift" in str(event.key).lower()
        )
        if remove_mode:
            removed = selected_points.pop(global_idx)
            _update_selected_ptr_after_remove(global_idx)
            list_state["page"] = min(int(list_state["page"]), _total_pages() - 1)
            redraw_selected_ipf_points()
            refresh_selected_points_text()
            update_selected_link_markers()
            info_text.set_text(
                f"Removed point {global_idx + 1}: dir={removed['dir']} "
                f"({removed['x']}, {removed['y']})"
            )
            fig.canvas.draw_idle()
            print(f"Removed selected point {global_idx + 1} from list")
            return True

        focus_selected_point(global_idx, redraw=True)
        return True

    def update_selected_link_markers():
        if not selected_points:
            selected_point_ptr[0] = None
            selected_idx[0] = None
            selected_marker.set_offsets(np.empty((0, 2)))
            selected_marker_key.set_offsets(np.empty((0, 2)))
            return
        ptr = selected_point_ptr[0]
        if ptr is None or int(ptr) < 0 or int(ptr) >= len(selected_points):
            ptr = len(selected_points) - 1
            selected_point_ptr[0] = int(ptr)
        sel = selected_points[int(ptr)]
        idx = int(sel["flat_idx"])
        selected_idx[0] = idx
        dcache = cache[current_dir["value"]]
        if dcache["vis_st"][idx]:
            selected_marker.set_offsets(np.array([[dcache["x_st"][idx], dcache["y_st"][idx]]]))
        else:
            selected_marker.set_offsets(np.empty((0, 2)))
        if dcache["vis_key"][idx]:
            selected_marker_key.set_offsets(np.array([[dcache["x_key"][idx], dcache["y_key"][idx]]]))
        else:
            selected_marker_key.set_offsets(np.empty((0, 2)))

    def clear_selected_points():
        if not selected_points:
            info_text.set_text("Selected points already empty")
            fig.canvas.draw_idle()
            return
        selected_points.clear()
        selected_point_ptr[0] = None
        list_state["page"] = 0
        redraw_selected_ipf_points()
        refresh_selected_points_text()
        update_selected_link_markers()
        info_text.set_text("Cleared all selected points")
        fig.canvas.draw_idle()
        print("Cleared all selected points")

    def export_selected_points():
        if not selected_points:
            info_text.set_text("No selected points to export")
            fig.canvas.draw_idle()
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_csv = os.path.abspath(f"selected_points_{ts}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dir",
                    "x",
                    "y",
                    "flat_idx",
                    "q_w",
                    "q_x",
                    "q_y",
                    "q_z",
                ]
            )
            for p in selected_points:
                q = np.asarray(p["q"], dtype=float).reshape(4)
                writer.writerow(
                    [
                        p["dir"],
                        int(p["x"]),
                        int(p["y"]),
                        int(p["flat_idx"]),
                        float(q[0]),
                        float(q[1]),
                        float(q[2]),
                        float(q[3]),
                    ]
                )

        info_text.set_text(f"Exported {len(selected_points)} selected points to {out_csv}")
        fig.canvas.draw_idle()
        print(f"Exported selected points to {out_csv}")

    def find_nearest_selected_idx(x: int, y: int, max_dist_px: float = 10.0):
        if not selected_points:
            return None
        pts = np.array([[int(p["x"]), int(p["y"])] for p in selected_points], dtype=float)
        d2 = np.sum((pts - np.array([x, y], dtype=float)) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= max_dist_px * max_dist_px:
            return j
        return None

    def find_selected_idx_by_flat(flat_idx: int):
        for i, p in enumerate(selected_points):
            if int(p["flat_idx"]) == int(flat_idx):
                return i
        return None

    def get_pixel_info(event):
        if event.inaxes is not ax_ipf:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        flat_idx = y * w + x
        q = q_cf[y, x]
        q_str = ", ".join(f"{v:.6f}" for v in q)
        return x, y, flat_idx, q, q_str

    def get_stereo_pixel_info(event, max_frac: float = 0.03):
        if event.inaxes is not ax_stereo:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        dcache = cache[current_dir["value"]]
        vis_idx = np.flatnonzero(dcache["vis_st"])
        if vis_idx.size == 0:
            return None

        xd = float(event.xdata)
        yd = float(event.ydata)
        xs = dcache["x_st"][vis_idx]
        ys = dcache["y_st"][vis_idx]
        d2 = (xs - xd) ** 2 + (ys - yd) ** 2
        j = int(np.argmin(d2))
        flat_idx = int(vis_idx[j])

        xlim = ax_stereo.get_xlim()
        ylim = ax_stereo.get_ylim()
        span = max(abs(float(xlim[1]) - float(xlim[0])), abs(float(ylim[1]) - float(ylim[0])))
        max_dist = max(1e-6, max_frac * span)
        if float(d2[j]) > max_dist * max_dist:
            return None

        y = flat_idx // w
        x = flat_idx - y * w
        q = q_cf[y, x]
        q_str = ", ".join(f"{v:.6f}" for v in q)
        return int(x), int(y), flat_idx, q, q_str

    def update_stereo_hemisphere(new_hemi: str, redraw: bool = True):
        hemi = str(new_hemi).strip().lower()
        if hemi not in {"upper", "lower"}:
            return
        hemi_label = "Upper" if hemi == "upper" else "Lower"
        target_pole = -1 if hemi == "upper" else 1

        current_hemi["value"] = hemi_label
        try:
            ax_stereo.pole = target_pole
        except Exception:
            pass
        _apply_stereo_pole_to_cache(target_pole)
        _draw_stereo_guides()
        _refresh_stereo_outer_labels()
        _update_stereo_outer_label_visibility()
        update_direction(current_dir["value"], redraw=False)

        if stereo_hemi_buttons:
            active_color = "#cfe8ff"
            active_hover = "#b9dcff"
            inactive_color = "#e6e6e6"
            inactive_hover = "#d9d9d9"
            for name, btn in stereo_hemi_buttons.items():
                is_active = name.lower() == hemi
                btn.color = active_color if is_active else inactive_color
                btn.hovercolor = active_hover if is_active else inactive_hover
                btn.ax.set_facecolor(btn.color)
                btn.label.set_fontweight("bold" if is_active else "normal")
                for spine in btn.ax.spines.values():
                    spine.set_linewidth(1.4 if is_active else 1.0)
                    spine.set_edgecolor("#2f5f92" if is_active else "#444444")

        info_text.set_text(f"Stereographic hemisphere -> {hemi_label}")
        print(f"Stereographic hemisphere -> {hemi_label}")
        if redraw:
            fig.canvas.draw_idle()

    def update_direction(new_dir: str, redraw: bool = True):
        new_dir = str(new_dir).upper()
        if new_dir not in cache:
            return
        current_dir["value"] = new_dir
        dcache = cache[new_dir]
        im.set_data(dcache["rgb"])
        ax_ipf.set_title(f"IPF-{new_dir}")
        ax_stereo.set_title(
            f"Stereographic (IPF-{new_dir} {current_hemi['value']})",
            fontsize=10,
        )
        ax_key.set_title(f"IPF Key (Ref {new_dir})", fontsize=10)

        bg_idx_dir = bg_idx[dcache["vis_st"][bg_idx]]
        bg_offsets = np.column_stack([dcache["x_st"][bg_idx_dir], dcache["y_st"][bg_idx_dir]])
        bg_scatter.set_offsets(bg_offsets)
        bg_scatter.set_facecolors(dcache["colors"][bg_idx_dir])

        if hover_idx[0] is not None:
            idx = hover_idx[0]
            hy = int(idx) // w
            hx = int(idx) - hy * w
            hover_marker_ipf.set_offsets(np.array([[hx, hy]]))
            if dcache["vis_st"][idx]:
                hover_marker.set_offsets(np.array([[dcache["x_st"][idx], dcache["y_st"][idx]]]))
            else:
                hover_marker.set_offsets(np.empty((0, 2)))
            if dcache["vis_key"][idx]:
                hover_marker_key.set_offsets(np.array([[dcache["x_key"][idx], dcache["y_key"][idx]]]))
            else:
                hover_marker_key.set_offsets(np.empty((0, 2)))
        else:
            hover_marker_ipf.set_offsets(np.empty((0, 2)))
            hover_marker.set_offsets(np.empty((0, 2)))
            hover_marker_key.set_offsets(np.empty((0, 2)))
        if selected_idx[0] is not None:
            idx = selected_idx[0]
            if dcache["vis_st"][idx]:
                selected_marker.set_offsets(np.array([[dcache["x_st"][idx], dcache["y_st"][idx]]]))
            else:
                selected_marker.set_offsets(np.empty((0, 2)))
            if dcache["vis_key"][idx]:
                selected_marker_key.set_offsets(np.array([[dcache["x_key"][idx], dcache["y_key"][idx]]]))
            else:
                selected_marker_key.set_offsets(np.empty((0, 2)))
        redraw_selected_ipf_points()

        info_text.set_text(
            f"Direction={new_dir}  (hover/click IPF to inspect quaternion and linked points)"
        )
        if dir_buttons:
            active_color = "#ffd24a"
            active_hover = "#ffcb2f"
            inactive_color = "#e6e6e6"
            inactive_hover = "#d9d9d9"
            for d, btn in dir_buttons.items():
                is_active = d == new_dir
                btn.color = active_color if is_active else inactive_color
                btn.hovercolor = active_hover if is_active else inactive_hover
                btn.ax.set_facecolor(btn.color)
                btn.label.set_fontweight("bold" if is_active else "normal")
                btn.label.set_color("#111111" if is_active else "#2f2f2f")
                for spine in btn.ax.spines.values():
                    spine.set_linewidth(1.6 if is_active else 1.0)
                    spine.set_edgecolor("#6b4f00" if is_active else "#444444")
        print(f"IPF direction -> {new_dir}")

        if redraw:
            fig.canvas.draw()

    def on_hover(event):
        if pan_state["active"]:
            return
        pixel_info = get_pixel_info(event)
        if pixel_info is None:
            pixel_info = get_stereo_pixel_info(event)
        if pixel_info is None:
            if hover_idx[0] is not None or last_hover[0] is not None:
                hover_idx[0] = None
                last_hover[0] = None
                hover_marker_ipf.set_offsets(np.empty((0, 2)))
                hover_marker.set_offsets(np.empty((0, 2)))
                hover_marker_key.set_offsets(np.empty((0, 2)))
                fig.canvas.draw_idle()
            return
        x, y, flat_idx, _, q_str = pixel_info
        key = (x, y, current_dir["value"])
        if key == last_hover[0]:
            return
        last_hover[0] = key
        hover_idx[0] = flat_idx
        dcache = cache[current_dir["value"]]
        info_text.set_text(
            f"dir={current_dir['value']},  (x, y)=({x}, {y}),  quaternion=({q_str})"
        )
        hover_marker_ipf.set_offsets(np.array([[x, y]]))
        if dcache["vis_st"][flat_idx]:
            hover_marker.set_offsets(np.array([[dcache["x_st"][flat_idx], dcache["y_st"][flat_idx]]]))
        else:
            hover_marker.set_offsets(np.empty((0, 2)))
        if dcache["vis_key"][flat_idx]:
            hover_marker_key.set_offsets(np.array([[dcache["x_key"][flat_idx], dcache["y_key"][flat_idx]]]))
        else:
            hover_marker_key.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()

    def on_click(event):
        def _event_has_ctrl(ev) -> bool:
            if ev is None or ev.key is None:
                return False
            k = str(ev.key).lower()
            return ("ctrl" in k) or ("control" in k)

        if (
            event.button == 1
            and (_event_has_ctrl(event) or ctrl_state["down"])
            and event.inaxes in (ax_ipf, ax_stereo, ax_key)
        ):
            if event.xdata is None or event.ydata is None:
                if event.x is None or event.y is None:
                    return
                px, py = event.inaxes.transData.inverted().transform((event.x, event.y))
            else:
                px, py = float(event.xdata), float(event.ydata)
            pan_state["active"] = True
            pan_state["ax"] = event.inaxes
            pan_state["press_xy"] = (float(px), float(py))
            pan_state["xlim0"] = tuple(float(v) for v in event.inaxes.get_xlim())
            pan_state["ylim0"] = tuple(float(v) for v in event.inaxes.get_ylim())
            return

        if event.x is not None and event.y is not None:
            if event.button == 1:
                # In WebAgg over SSH, widget callbacks can be unreliable. Route
                # UI button clicks directly via axis hit-testing.
                for d in ("X", "Y", "Z"):
                    d_ax = dir_button_axes.get(d)
                    if d_ax is not None and d_ax.bbox.contains(event.x, event.y):
                        update_direction(d)
                        return

                for hemi in ("Upper", "Lower"):
                    h_ax = stereo_hemi_button_axes.get(hemi)
                    if h_ax is not None and h_ax.bbox.contains(event.x, event.y):
                        update_stereo_hemisphere(hemi)
                        return

                if (
                    clear_button_ax[0] is not None
                    and clear_button_ax[0].bbox.contains(event.x, event.y)
                ):
                    clear_selected_points()
                    return

                if (
                    toggle_button_ax[0] is not None
                    and toggle_button_ax[0].bbox.contains(event.x, event.y)
                ):
                    on_toggle(None)
                    return

                if (
                    export_button_ax[0] is not None
                    and export_button_ax[0].bbox.contains(event.x, event.y)
                ):
                    export_selected_points()
                    return

                if (
                    prev_page_ax[0] is not None
                    and prev_page_ax[0].bbox.contains(event.x, event.y)
                ):
                    _set_page(int(list_state["page"]) - 1, redraw=True)
                    return

                if (
                    next_page_ax[0] is not None
                    and next_page_ax[0].bbox.contains(event.x, event.y)
                ):
                    _set_page(int(list_state["page"]) + 1, redraw=True)
                    return

            if handle_list_click(event):
                return

        pixel_info = get_pixel_info(event)
        if pixel_info is None:
            pixel_info = get_stereo_pixel_info(event)
        if pixel_info is None:
            return

        x, y, flat_idx, q, q_str = pixel_info
        remove_mode = (event.button == 3) or (
            event.button == 1
            and event.key is not None
            and "shift" in str(event.key).lower()
        )
        if remove_mode:
            j = find_nearest_selected_idx(x, y)
            if j is None:
                info_text.set_text(f"No selected point near ({x}, {y}) to remove")
                fig.canvas.draw_idle()
                return
            removed = selected_points.pop(j)
            _update_selected_ptr_after_remove(j)
            list_state["page"] = min(int(list_state["page"]), _total_pages() - 1)
            redraw_selected_ipf_points()
            refresh_selected_points_text()
            update_selected_link_markers()
            info_text.set_text(
                f"Removed point {j + 1}: dir={removed['dir']} ({removed['x']}, {removed['y']})"
            )
            fig.canvas.draw_idle()
            print(f"Removed selected point {j + 1}")
            return

        if event.button != 1:
            return

        existing_idx = find_selected_idx_by_flat(flat_idx)
        if existing_idx is not None:
            focus_selected_point(existing_idx, redraw=True)
            info_text.set_text(
                f"Point already selected: #{existing_idx + 1}, dir={selected_points[existing_idx]['dir']}, "
                f"(x, y)=({x}, {y})"
            )
            fig.canvas.draw_idle()
            return

        selected_points.append(
            {
                "dir": current_dir["value"],
                "x": x,
                "y": y,
                "flat_idx": flat_idx,
                "q": np.array(q, dtype=float),
            }
        )
        selected_point_ptr[0] = len(selected_points) - 1
        list_state["page"] = _total_pages() - 1
        redraw_selected_ipf_points()
        refresh_selected_points_text()
        update_selected_link_markers()
        info_text.set_text(
            f"Selected dir={current_dir['value']},  (x, y)=({x}, {y}),  quaternion=({q_str})"
        )
        fig.canvas.draw_idle()

        print(
            f"dir={current_dir['value']}, x={x} y={y}, "
            f"quaternion=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})"
        )

    def _pan_limits_1d(lim0, full_lim, delta: float):
        a0, a1 = float(lim0[0]), float(lim0[1])
        f0, f1 = float(full_lim[0]), float(full_lim[1])
        forward = a1 >= a0
        lo0, hi0 = (a0, a1) if forward else (a1, a0)
        flo, fhi = (f0, f1) if f1 >= f0 else (f1, f0)
        span = hi0 - lo0
        lo = lo0 - float(delta)
        hi = lo + span
        if lo < flo:
            shift = flo - lo
            lo += shift
            hi += shift
        if hi > fhi:
            shift = hi - fhi
            lo -= shift
            hi -= shift
        lo = max(flo, lo)
        hi = min(fhi, hi)
        return (lo, hi) if forward else (hi, lo)

    def on_pan_motion(event):
        if not pan_state["active"]:
            return
        axp = pan_state["ax"]
        if axp is None:
            return
        if event.inaxes is not axp:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = pan_state["press_xy"]
        dx = float(event.xdata) - float(x0)
        dy = float(event.ydata) - float(y0)

        if axp is ax_ipf:
            full_x = ipf_full_xlim
            full_y = ipf_full_ylim
        elif axp is ax_stereo:
            full_x = stereo_full_xlim
            full_y = stereo_full_ylim
        else:
            full_x = key_full_xlim[0]
            full_y = key_full_ylim[0]
            key_interacted[0] = True

        new_xlim = _pan_limits_1d(pan_state["xlim0"], full_x, dx)
        new_ylim = _pan_limits_1d(pan_state["ylim0"], full_y, dy)
        axp.set_xlim(new_xlim)
        axp.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def on_pan_release(event):
        if not pan_state["active"]:
            return
        if event.button != 1:
            return
        pan_state["active"] = False
        pan_state["ax"] = None
        pan_state["press_xy"] = None
        pan_state["xlim0"] = None
        pan_state["ylim0"] = None

    # Disable default Matplotlib key bindings so non X/Y/Z keys do nothing.
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "key_press_handler_id"):
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)

    # Always show in-figure clickable buttons in WebAgg/SSH.
    dir_btn_w = 0.05
    dir_btn_h = 0.05
    dir_btn_gap = 0.012
    dir_total_w = 3 * dir_btn_w + 2 * dir_btn_gap
    dir_title_text = fig.text(
        0.5,
        0.93,
        "IPF Reference Direction",
        fontsize=12,
        ha="center",
        va="center",
    )
    for i, d in enumerate(("X", "Y", "Z")):
        b_ax = fig.add_axes([0.1 + i * 0.06, 0.84, dir_btn_w, dir_btn_h])
        btn = Button(b_ax, d)
        dir_buttons[d] = btn
        dir_button_axes[d] = b_ax

    def _ipf_zoom_ratio() -> float:
        xlim = ax_ipf.get_xlim()
        ylim = ax_ipf.get_ylim()
        xspan = max(abs(float(xlim[1]) - float(xlim[0])), 1e-12)
        yspan = max(abs(float(ylim[1]) - float(ylim[0])), 1e-12)
        return max(ipf_full_xspan / xspan, ipf_full_yspan / yspan)

    def _update_dir_controls_position():
        ipf_pos = ax_ipf.get_position()
        dir_center_x = 0.5 * (ipf_pos.x0 + ipf_pos.x1)
        dir_x0 = dir_center_x - 0.5 * dir_total_w

        # Base placement tracks the current IPF panel top edge.
        base_btn_y = min(max(ipf_pos.y1 + 0.016, 0.83), 0.94 - dir_btn_h)

        # If user zooms very deep, lift controls to avoid any perceived overlap.
        z = _ipf_zoom_ratio()
        lift = 0.0
        if z > 8.0:
            lift = min((z - 8.0) / 48.0, 1.0) * 0.065

        dir_btn_y = min(base_btn_y + lift, 0.94 - dir_btn_h)
        dir_title_y = min(dir_btn_y + dir_btn_h + 0.018, 0.975)
        dir_title_text.set_position((dir_center_x, dir_title_y))

        for i, d in enumerate(("X", "Y", "Z")):
            b_ax = dir_button_axes.get(d)
            if b_ax is None:
                continue
            b_ax.set_position([dir_x0 + i * (dir_btn_w + dir_btn_gap), dir_btn_y, dir_btn_w, dir_btn_h])

    _update_dir_controls_position()
    ax_ipf.callbacks.connect("xlim_changed", lambda _ax: _update_dir_controls_position())
    ax_ipf.callbacks.connect("ylim_changed", lambda _ax: _update_dir_controls_position())

    # Selected-points panel controls (right side): clear and minimize/expand.
    list_pos = ax_list.get_position()
    btn_h = 0.045
    btn_y = list_pos.y1 - btn_h - 0.012
    btn_left = list_pos.x0 + 0.008
    btn_gap = 0.006
    btn_right = list_pos.x1 - 0.008
    btn_w = max(0.04, (btn_right - btn_left - 2 * btn_gap) / 3.0)
    clear_ax = fig.add_axes([btn_left, btn_y, btn_w, btn_h], zorder=20)
    toggle_ax = fig.add_axes([btn_left + btn_w + btn_gap, btn_y, btn_w, btn_h], zorder=20)
    export_ax = fig.add_axes(
        [btn_left + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h],
        zorder=20,
    )
    clear_btn = Button(clear_ax, "Clear All")
    toggle_btn = Button(toggle_ax, "Minimize")
    export_btn = Button(export_ax, "Export CSV")
    clear_button_ax[0] = clear_ax
    toggle_button_ax[0] = toggle_ax
    export_button_ax[0] = export_ax

    nav_y = btn_y - btn_h - 0.008
    nav_w = max(0.04, (btn_right - btn_left - 2 * btn_gap) / 3.0)
    prev_ax = fig.add_axes([btn_left, nav_y, nav_w, btn_h], zorder=20)
    next_ax = fig.add_axes([btn_right - nav_w, nav_y, nav_w, btn_h], zorder=20)
    prev_btn = Button(prev_ax, "Prev")
    next_btn = Button(next_ax, "Next")
    prev_page_ax[0] = prev_ax
    next_page_ax[0] = next_ax
    page_indicator[0] = fig.text(
        (list_pos.x0 + list_pos.x1) * 0.5,
        nav_y + 0.5 * btn_h,
        "Page 1/1",
        ha="center",
        va="center",
        fontsize=9,
    )

    def on_toggle(_event):
        list_state["collapsed"] = not list_state["collapsed"]
        collapsed = list_state["collapsed"]
        selected_list_text.set_visible(not collapsed)
        for artist in selected_row_artists:
            artist.set_visible(not collapsed)
        if list_box_patch[0] is not None:
            list_box_patch[0].set_visible(not collapsed)
        prev_btn.ax.set_visible(not collapsed)
        next_btn.ax.set_visible(not collapsed)
        page_indicator[0].set_visible(not collapsed)
        if collapsed:
            ax_list.set_title("Selected points (minimized)", fontsize=10, pad=8)
            toggle_btn.label.set_text("Expand")
            info_text.set_text("selected points panel minimized")
        else:
            ax_list.set_title("Selected points", fontsize=10, pad=8)
            toggle_btn.label.set_text("Minimize")
            refresh_selected_points_text()
            info_text.set_text("selected points panel expanded")
        fig.canvas.draw_idle()

    # Click handling is routed through on_click() for reliable WebAgg behavior.

    def _scroll_step(event) -> int:
        if hasattr(event, "step") and event.step is not None and event.step != 0:
            return 1 if event.step > 0 else -1
        if event.button == "up":
            return 1
        if event.button == "down":
            return -1
        return 0

    def _zoom_limits_1d(cur_lim, full_lim, center: float, scale: float, min_span: float = 6.0):
        cur0, cur1 = float(cur_lim[0]), float(cur_lim[1])
        full0, full1 = float(full_lim[0]), float(full_lim[1])
        forward = cur1 >= cur0
        lo, hi = (cur0, cur1) if forward else (cur1, cur0)
        full_lo, full_hi = (full0, full1) if full0 <= full1 else (full1, full0)

        cur_span = max(hi - lo, 1e-12)
        full_span = max(full_hi - full_lo, 1e-12)
        new_span = float(np.clip(cur_span * scale, min_span, full_span))

        rel = float(np.clip((center - lo) / cur_span, 0.0, 1.0))
        new_lo = center - rel * new_span
        new_hi = new_lo + new_span

        if new_lo < full_lo:
            shift = full_lo - new_lo
            new_lo += shift
            new_hi += shift
        if new_hi > full_hi:
            shift = new_hi - full_hi
            new_lo -= shift
            new_hi -= shift

        new_lo = max(full_lo, new_lo)
        new_hi = min(full_hi, new_hi)
        return (new_lo, new_hi) if forward else (new_hi, new_lo)

    def reset_ipf_zoom(redraw: bool = True):
        ax_ipf.set_xlim(ipf_full_xlim)
        ax_ipf.set_ylim(ipf_full_ylim)
        if redraw:
            fig.canvas.draw_idle()

    def reset_stereo_zoom(redraw: bool = True):
        ax_stereo.set_xlim(stereo_full_xlim)
        ax_stereo.set_ylim(stereo_full_ylim)
        if redraw:
            fig.canvas.draw_idle()

    def reset_key_zoom(redraw: bool = True):
        key_interacted[0] = False
        ax_key.set_xlim(key_full_xlim[0])
        ax_key.set_ylim(key_full_ylim[0])
        if redraw:
            fig.canvas.draw_idle()

    def _zoom_ipf_from_scroll(event) -> bool:
        if event.x is None or event.y is None:
            return False
        if not ax_ipf.bbox.contains(event.x, event.y):
            return False

        step = _scroll_step(event)
        if step == 0:
            return False

        if event.xdata is None or event.ydata is None:
            xdata, ydata = ax_ipf.transData.inverted().transform((event.x, event.y))
        else:
            xdata = float(event.xdata)
            ydata = float(event.ydata)

        scale = 1.0 / 1.25 if step > 0 else 1.25
        new_xlim = _zoom_limits_1d(ax_ipf.get_xlim(), ipf_full_xlim, float(xdata), scale)
        new_ylim = _zoom_limits_1d(ax_ipf.get_ylim(), ipf_full_ylim, float(ydata), scale)
        ax_ipf.set_xlim(new_xlim)
        ax_ipf.set_ylim(new_ylim)
        fig.canvas.draw_idle()
        return True

    def _zoom_stereo_from_scroll(event) -> bool:
        if event.x is None or event.y is None:
            return False
        if not ax_stereo.bbox.contains(event.x, event.y):
            return False

        step = _scroll_step(event)
        if step == 0:
            return False

        if event.xdata is None or event.ydata is None:
            xdata, ydata = ax_stereo.transData.inverted().transform((event.x, event.y))
        else:
            xdata = float(event.xdata)
            ydata = float(event.ydata)

        scale = 1.0 / 1.25 if step > 0 else 1.25
        new_xlim = _zoom_limits_1d(
            ax_stereo.get_xlim(),
            stereo_full_xlim,
            float(xdata),
            scale,
            min_span=0.08,
        )
        new_ylim = _zoom_limits_1d(
            ax_stereo.get_ylim(),
            stereo_full_ylim,
            float(ydata),
            scale,
            min_span=0.08,
        )
        ax_stereo.set_xlim(new_xlim)
        ax_stereo.set_ylim(new_ylim)
        fig.canvas.draw_idle()
        return True

    def _zoom_key_from_scroll(event) -> bool:
        if event.x is None or event.y is None:
            return False
        if not ax_key.bbox.contains(event.x, event.y):
            return False

        step = _scroll_step(event)
        if step == 0:
            return False

        if event.xdata is None or event.ydata is None:
            xdata, ydata = ax_key.transData.inverted().transform((event.x, event.y))
        else:
            xdata = float(event.xdata)
            ydata = float(event.ydata)

        key_interacted[0] = True
        scale = 1.0 / 1.25 if step > 0 else 1.25
        new_xlim = _zoom_limits_1d(
            ax_key.get_xlim(), key_full_xlim[0], float(xdata), scale, min_span=0.06
        )
        new_ylim = _zoom_limits_1d(
            ax_key.get_ylim(), key_full_ylim[0], float(ydata), scale, min_span=0.06
        )
        ax_key.set_xlim(new_xlim)
        ax_key.set_ylim(new_ylim)
        fig.canvas.draw_idle()
        return True

    def on_scroll_list(event):
        if _zoom_ipf_from_scroll(event):
            return
        if _zoom_stereo_from_scroll(event):
            return
        if _zoom_key_from_scroll(event):
            return

        if list_state["collapsed"]:
            return
        if event.x is None or event.y is None:
            return

        in_panel = ax_list.bbox.contains(event.x, event.y)
        if not in_panel:
            return

        step = _scroll_step(event)
        if step == 0:
            return
        step_dir = -1 if step > 0 else 1
        _set_page(int(list_state["page"]) + step_dir, redraw=True)

    def on_key(event):
        if event.key is None:
            return
        key = str(event.key).lower()
        if key in {"control", "ctrl"}:
            ctrl_state["down"] = True
            return
        if key in {"x", "y", "z"}:
            update_direction(key.upper())
        elif key == "r":
            reset_ipf_zoom(redraw=True)
            info_text.set_text("IPF zoom reset")
        elif key == "t":
            reset_stereo_zoom(redraw=True)
            info_text.set_text("Stereographic zoom reset")
        elif key == "k":
            reset_key_zoom(redraw=True)
            info_text.set_text("IPF key zoom reset")
        elif key == "c":
            clear_selected_points()
        elif key == "e":
            export_selected_points()
        elif key in {"["}:
            _set_page(int(list_state["page"]) - 1, redraw=True)
        elif key in {"]"}:
            _set_page(int(list_state["page"]) + 1, redraw=True)
        elif key in {"h", "?", "shift+/"}:
            shortcut_text_visible["value"] = not shortcut_text_visible["value"]
            shortcut_text.set_visible(shortcut_text_visible["value"])
            fig.canvas.draw_idle()

    def on_key_release(event):
        if event.key is None:
            return
        key = str(event.key).lower()
        if key in {"control", "ctrl"} or "ctrl" in key or "control" in key:
            ctrl_state["down"] = False

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("motion_notify_event", on_pan_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_pan_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll_list)
    fig.canvas.mpl_connect("resize_event", lambda _event: _update_dir_controls_position())
    fig.suptitle(
        "IPF Plot, Stereographic Plot, and IPF key",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    refresh_selected_points_text()
    update_direction("Z", redraw=False)
    plt.show()


def show_interactive_ipf_from_npy(
    npy_path: str,
    sym: str | object = "Oh",
    scalar_first: bool = False,
    quat_spatial: bool = False,
    apply_format: bool = True,
    quaternion_convention: str = "bunge",
) -> None:
    """Convenience entry point: load/prep quaternions then open interactive viewer."""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"input file does not exist: {npy_path}")

    q_cf, sym_obj = build_formatted_quaternions(
        npy_path,
        sym=sym,
        scalar_first=scalar_first,
        quat_spatial=quat_spatial,
        apply_format=apply_format,
        quaternion_convention=quaternion_convention,
    )
    show_interactive_ipf(q_cf, sym_obj)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive IPF + stereographic quaternion viewer",
    )
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=FP,
        help=f"Path to quaternion .npy file (default: {FP})",
    )
    parser.add_argument(
        "--sym",
        default="Oh",
        help="Symmetry name/alias or canonical symbol (e.g. Oh, m-3m, D6h).",
    )
    parser.add_argument(
        "--scalar-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stored quaternion component order is [w,x,y,z].",
    )
    parser.add_argument(
        "--quat-spatial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stored layout is (4,*spatial) instead of (*spatial,4).",
    )
    parser.add_argument(
        "--apply-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply normalize+hemisphere+FZ formatting before plotting.",
    )
    parser.add_argument(
        "--quaternion-convention",
        choices=("bunge", "active", "passive"),
        default="bunge",
        help="Quaternion convention of input data.",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()

    # Notebook/ipykernel injects extra argv; ignore unknown there so this
    # still runs cleanly from interactive notebook environments.
    if argv is not None:
        args = parser.parse_args(argv)
    elif "ipykernel" in sys.modules:
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    show_interactive_ipf_from_npy(
        args.npy_path,
        sym=args.sym,
        scalar_first=bool(args.scalar_first),
        quat_spatial=bool(args.quat_spatial),
        apply_format=bool(args.apply_format),
        quaternion_convention=args.quaternion_convention,
    )


if __name__ == "__main__":
    main()
