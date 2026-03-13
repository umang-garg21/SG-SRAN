import os
import csv
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
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d

from utils.quat_ops import format_quaternions, to_scalar_first


FP = "/data/warren/materials/materials_data_mount/EBSD/datasets/IN718_slices/x/x_slice_0000.npy"
SYM = symmetry.Oh
DIR_VECTORS = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}


def build_formatted_quaternions(fp: str) -> np.ndarray:
    q_arr = np.load(fp, mmap_mode="r")
    print(f"loaded: {fp}")
    print(f"shape={q_arr.shape}, dtype={q_arr.dtype}")

    # Input is expected scalar-last [x, y, z, w], then converted to scalar-first
    # [w, x, y, z] before canonical formatting.
    q_arr_scalar_first = to_scalar_first(q_arr)
    q_cf = format_quaternions(
        q_arr_scalar_first,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=SYM.name,
        to_quat_first=False,
    )
    return q_cf


def build_direction_cache(q_cf: np.ndarray) -> dict[str, dict[str, np.ndarray | Vector3d]]:
    ori_2d = Orientation(q_cf)
    ori_2d.symmetry = SYM

    q_flat = q_cf.reshape(-1, 4)
    ori_flat = Orientation(q_flat, symmetry=SYM)

    cache: dict[str, dict[str, np.ndarray | Vector3d]] = {}
    for d, v_ref in DIR_VECTORS.items():
        ckey = orix.plot.IPFColorKeyTSL(SYM.laue)
        ckey.direction = v_ref

        rgb_map = ckey.orientation2color(ori_2d)
        colors_flat = ckey.orientation2color(ori_flat)

        v_stereo = ori_flat * v_ref
        v_key = v_stereo.in_fundamental_sector(SYM)

        cache[d] = {
            "rgb": rgb_map,
            "colors": colors_flat,
            "v_stereo": v_stereo,
            "v_key": v_key,
        }
    return cache


def show_interactive_ipf(q_cf: np.ndarray) -> None:
    cache = build_direction_cache(q_cf)

    fig = plt.figure(figsize=(24.2, 7.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.66, 1.25, 0.78, 1.20], wspace=0.24)
    ax_ipf = fig.add_subplot(gs[0, 0])
    ax_stereo = fig.add_subplot(gs[0, 1], projection="stereographic")
    ax_key = fig.add_subplot(gs[0, 2], projection="ipf", symmetry=SYM.laue)
    ax_list = fig.add_subplot(gs[0, 3])
    ax_list.axis("off")
    ax_list.set_title("Selected points", fontsize=10, pad=8)
    ax_key.plot_ipf_color_key(show_title=False)

    current_dir = {"value": "Z"}
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
        x_st, y_st = _vector2xy(dcache["v_stereo"], pole=ax_stereo.pole)
        x_key, y_key = _vector2xy(dcache["v_key"], pole=ax_key.pole)
        dcache["x_st"] = np.asarray(x_st).reshape(-1)
        dcache["y_st"] = np.asarray(y_st).reshape(-1)
        dcache["x_key"] = np.asarray(x_key).reshape(-1)
        dcache["y_key"] = np.asarray(y_key).reshape(-1)
        z_st = np.asarray(dcache["v_stereo"].z).reshape(-1)
        z_key = np.asarray(dcache["v_key"].z).reshape(-1)
        dcache["vis_st"] = z_st >= 0 if ax_stereo.pole == -1 else z_st <= 0
        dcache["vis_key"] = z_key >= 0 if ax_key.pole == -1 else z_key <= 0

    c0 = cache[current_dir["value"]]
    bg_idx0 = bg_idx[c0["vis_st"][bg_idx]]
    im = ax_ipf.imshow(c0["rgb"])
    im.get_cursor_data = lambda event: None
    im.format_cursor_data = lambda data: ""
    ax_ipf.set_title(f"IPF-{current_dir['value']}")
    ax_ipf.axis("off")

    bg_scatter = Axes.scatter(
        ax_stereo,
        c0["x_st"][bg_idx0],
        c0["y_st"][bg_idx0],
        c=c0["colors"][bg_idx0],
        s=6,
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
    ax_stereo.draw_circle(v4fold, color="black", linewidth=0.8, alpha=0.35)
    ax_stereo.draw_circle(v3fold, color="black", linewidth=0.8, alpha=0.35)
    ax_stereo.draw_circle(v2fold, color="black", linewidth=0.8, alpha=0.35)
    ax_stereo.set_title(f"Stereographic (IPF-{current_dir['value']} reference)", fontsize=10)
    ax_stereo.set_labels("RD", "TD", None)
    ax_stereo.show_hemisphere_label()
    ax_key.set_title(f"IPF key (ref {current_dir['value']})", fontsize=10)

    hover_marker = Axes.scatter(
        ax_stereo,
        [],
        [],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=6,
    )
    selected_marker = Axes.scatter(
        ax_stereo,
        [],
        [],
        marker="x",
        s=110,
        c="red",
        linewidths=2.0,
        zorder=7,
    )
    hover_marker_key = Axes.scatter(
        ax_key,
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=7,
    )
    selected_marker_key = Axes.scatter(
        ax_key,
        [],
        [],
        marker="x",
        s=100,
        c="red",
        linewidths=2.0,
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
        "Select direction (X/Y/Z); C clears, E exports; Left/Right pages list; H toggles shortcuts.",
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
        "C: clear selected points\n"
        "E: export selected points CSV\n"
        "[ ]: page list left/right\n"
        "Mouse wheel on list: page list\n"
        "Left click IPF: add point\n"
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

        dir_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
        dcache = cache[current_dir["value"]]
        for i, p in enumerate(selected_points, start=1):
            px, py = int(p["x"]), int(p["y"])
            pdir = str(p["dir"])
            color = dir_colors.get(pdir, "k")
            marker = ax_ipf.plot(
                px,
                py,
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=7,
                zorder=10,
            )[0]
            label = ax_ipf.text(
                px + 2,
                py + 2,
                str(i),
                color=color,
                fontsize=8,
                fontweight="bold",
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
                    s=42,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.3,
                    zorder=8,
                )
                sl = Axes.text(
                    ax_stereo,
                    sx + 0.012,
                    sy + 0.012,
                    str(i),
                    color=color,
                    fontsize=7,
                    fontweight="bold",
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
                    edgecolors=color,
                    linewidths=1.3,
                    zorder=8,
                )
                kl = Axes.text(
                    ax_key,
                    kx + 0.012,
                    ky + 0.012,
                    str(i),
                    color=color,
                    fontsize=7,
                    fontweight="bold",
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
            "Select points: Left click IPF or click a list row",
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

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def update_direction(new_dir: str, redraw: bool = True):
        new_dir = str(new_dir).upper()
        if new_dir not in cache:
            return
        current_dir["value"] = new_dir
        dcache = cache[new_dir]
        im.set_data(dcache["rgb"])
        ax_ipf.set_title(f"IPF-{new_dir}")
        ax_stereo.set_title(f"Stereographic (IPF-{new_dir} Reference)", fontsize=10)
        ax_key.set_title(f"IPF Key (Ref {new_dir})", fontsize=10)

        bg_idx_dir = bg_idx[dcache["vis_st"][bg_idx]]
        bg_offsets = np.column_stack([dcache["x_st"][bg_idx_dir], dcache["y_st"][bg_idx_dir]])
        bg_scatter.set_offsets(bg_offsets)
        bg_scatter.set_facecolors(dcache["colors"][bg_idx_dir])

        if hover_idx[0] is not None:
            idx = hover_idx[0]
            if dcache["vis_st"][idx]:
                hover_marker.set_offsets(np.array([[dcache["x_st"][idx], dcache["y_st"][idx]]]))
            else:
                hover_marker.set_offsets(np.empty((0, 2)))
            if dcache["vis_key"][idx]:
                hover_marker_key.set_offsets(np.array([[dcache["x_key"][idx], dcache["y_key"][idx]]]))
            else:
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
            for d, btn in dir_buttons.items():
                is_active = d == new_dir
                btn.ax.set_facecolor("#cfe8ff" if is_active else "#f0f0f0")
                btn.label.set_fontweight("bold" if is_active else "normal")
        print(f"IPF direction -> {new_dir}")

        if redraw:
            fig.canvas.draw()

    def on_hover(event):
        pixel_info = get_pixel_info(event)
        if pixel_info is None:
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
        if event.x is not None and event.y is not None:
            if event.button == 1:
                # In WebAgg over SSH, widget callbacks can be unreliable. Route
                # UI button clicks directly via axis hit-testing.
                for d in ("X", "Y", "Z"):
                    d_ax = dir_button_axes.get(d)
                    if d_ax is not None and d_ax.bbox.contains(event.x, event.y):
                        update_direction(d)
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

    # Disable default Matplotlib key bindings so non X/Y/Z keys do nothing.
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "key_press_handler_id"):
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)

    # Always show in-figure clickable buttons in WebAgg/SSH.
    ipf_pos = ax_ipf.get_position()
    dir_center_x = 0.5 * (ipf_pos.x0 + ipf_pos.x1)
    dir_btn_w = 0.05
    dir_btn_h = 0.05
    dir_btn_gap = 0.012
    dir_total_w = 3 * dir_btn_w + 2 * dir_btn_gap
    dir_x0 = dir_center_x - 0.5 * dir_total_w
    fig.text(
        dir_center_x,
        0.925,
        "IPF Reference Direction",
        fontsize=12,
        ha="center",
        va="center",
    )
    for i, d in enumerate(("X", "Y", "Z")):
        b_ax = fig.add_axes([dir_x0 + i * (dir_btn_w + dir_btn_gap), 0.845, dir_btn_w, dir_btn_h])
        btn = Button(b_ax, d)
        dir_buttons[d] = btn
        dir_button_axes[d] = b_ax

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

    def on_scroll_list(event):
        if list_state["collapsed"]:
            return
        if event.x is None or event.y is None:
            return

        in_panel = ax_list.bbox.contains(event.x, event.y)
        if not in_panel:
            return

        step_dir = 0
        if hasattr(event, "step") and event.step is not None and event.step != 0:
            step_dir = -1 if event.step > 0 else 1
        elif event.button == "up":
            step_dir = -1
        elif event.button == "down":
            step_dir = 1
        if step_dir == 0:
            return

        _set_page(int(list_state["page"]) + step_dir, redraw=True)

    def on_key(event):
        if event.key is None:
            return
        key = str(event.key).lower()
        if key in {"x", "y", "z"}:
            update_direction(key.upper())
        elif key == "c":
            clear_selected_points()
        elif key == "e":
            export_selected_points()
        elif key in {"left", "["}:
            _set_page(int(list_state["page"]) - 1, redraw=True)
        elif key in {"right", "]"}:
            _set_page(int(list_state["page"]) + 1, redraw=True)
        elif key in {"h", "?", "shift+/"}:
            shortcut_text_visible["value"] = not shortcut_text_visible["value"]
            shortcut_text.set_visible(shortcut_text_visible["value"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll_list)
    fig.suptitle(
        "IPF Plot, Stereographic Plot, and IPF key",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    refresh_selected_points_text()
    update_direction("Z", redraw=False)
    plt.show()


def main():
    if not os.path.exists(FP):
        raise FileNotFoundError(f"input file does not exist: {FP}")

    q_cf = build_formatted_quaternions(FP)
    show_interactive_ipf(q_cf)


if __name__ == "__main__":
    main()
