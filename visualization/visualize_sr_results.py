import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from utils.quat_ops import format_quaternions
from visualization.ipf_render import render_ipf_rgb


def render_sr_hr_side_by_side(
    sr_q_arr: np.ndarray,
    hr_q_arr: np.ndarray,
    sym_class,
    out_png: Optional[str] = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    format_input: bool = True,
    dpi: int = 300,
):
    """
    Render SR (Super-Resolution) and HR (High-Resolution) quaternion orientation maps
    in a 2-row layout (SR on top, HR on bottom), optionally with IPF key.

    Parameters
    ----------
    sr_q_arr : ndarray
        SR quaternion array of shape (H, W, 4).
    hr_q_arr : ndarray
        HR quaternion array of shape (H, W, 4).
    sym_class : orix symmetry
        Symmetry object for IPF coloring.
    out_png : str, optional
        Output PNG file path. If None, figure is not saved.
    ref_dir : {"X","Y","Z","ALL"}, default="ALL"
        Reference direction(s) for coloring.
    include_key : bool, default=True
        Whether to include IPF color key panel.
    overwrite : bool, default=False
        If False, skip rendering if file exists.
    format_input : bool, default=True
        If True, canonicalize quaternions.
    dpi : int, default=300
        Figure DPI for saved PNG.
    """
    # -------------------------------------------------------------------------
    # Early exit if file already exists
    # -------------------------------------------------------------------------
    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    # -------------------------------------------------------------------------
    # Format quaternions (reduce to FZ, normalize, hemisphere, etc.)
    # -------------------------------------------------------------------------
    if format_input:
        sr_q_arr = format_quaternions(
            sr_q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            quat_first=False,
        )
        hr_q_arr = format_quaternions(
            hr_q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            quat_first=False,
        )

    # -------------------------------------------------------------------------
    # Convert to IPF RGB maps
    # -------------------------------------------------------------------------
    sr_rgb = render_ipf_rgb(sr_q_arr, sym_class, ref_dir=ref_dir)
    hr_rgb = render_ipf_rgb(hr_q_arr, sym_class, ref_dir=ref_dir)

    multi_ref = isinstance(sr_rgb, list)
    ncols = 3 if multi_ref else 1
    key_cols = 1 if include_key else 0
    total_cols = ncols + key_cols
    total_rows = 2  # SR top, HR bottom

    # -------------------------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------------------------
    base_w = 5.0
    key_w = 2.6 if include_key else 0
    fig_w = base_w * ncols + key_w
    fig_h = 2 * 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[1] * ncols + ([0.9] if include_key else []),
        height_ratios=[1, 1],
        hspace=0.25,
        wspace=0.25,
    )

    def _imshow(ax, img, title):
        ax.imshow(img)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Plot SR (top row) and HR (bottom row)
    # -------------------------------------------------------------------------
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), hr_rgb)):
            _imshow(fig.add_subplot(gs[0, j]), img, f"HR IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), sr_rgb)):
            _imshow(fig.add_subplot(gs[1, j]), img, f"SR IPF-{name}")
    else:
        _imshow(fig.add_subplot(gs[0, 0]), sr_rgb, f"SR IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[1, 0]), hr_rgb, f"HR IPF-{ref_dir.upper()}")

    # -------------------------------------------------------------------------
    # IPF color key
    # -------------------------------------------------------------------------
    if include_key:
        ax_key = fig.add_subplot(gs[:, -1], projection="ipf", symmetry=sym_class.laue)
        ax_key.plot_ipf_color_key()
        ax_key.set_title("")

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved SR-HR comparison to: {out_png}")
        return out_png

    plt.show()
    return None


def render_sr_hr_lr_side_by_side(
    sr_q_arr: np.ndarray,
    hr_q_arr: np.ndarray,
    lr_q_arr: np.ndarray,
    sym_class,
    out_png: Optional[str] = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    format_input: bool = True,
    dpi: int = 300,
    mis_deg_arr: Optional[np.ndarray] = None,
):
    """
    Render LR (Low-Resolution), SR (Super-Resolution), and HR (High-Resolution)
    quaternion orientation maps in a 3-row layout:
        LR (top), SR (middle), HR (bottom)

    Parameters
    ----------
    sr_q_arr, hr_q_arr, lr_q_arr : ndarray
        Quaternion arrays of shape (H, W, 4).
    sym_class : orix symmetry
        Symmetry object for IPF coloring.
    out_png : str, optional
        Output PNG file path. If None, figure is not saved.
    ref_dir : {"X","Y","Z","ALL"}, default="ALL"
        Reference direction(s) for coloring.
    include_key : bool, default=True
        Whether to include IPF color key panel.
    overwrite : bool, default=False
        If False, skip rendering if file exists.
    format_input : bool, default=True
        If True, canonicalize quaternions.
    dpi : int, default=300
        Figure DPI for saved PNG.
    mis_deg_arr : ndarray of shape (H_hr, W_hr), optional
        Per-pixel misorientation angle in degrees (SR vs HR).
        When provided, rendered as a heatmap in row 4.
    """
    # -------------------------------------------------------------------------
    # Early exit if file already exists
    # -------------------------------------------------------------------------
    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    # -------------------------------------------------------------------------
    # Format quaternions (normalize, reduce FZ, etc.)
    # -------------------------------------------------------------------------
    if format_input:

        def _fmt(arr):
            return format_quaternions(
                arr,
                normalize=True,
                hemisphere=True,
                reduce_fz=True,
                sym=sym_class,
                quat_first=False,
            )

        sr_q_arr = _fmt(sr_q_arr)
        hr_q_arr = _fmt(hr_q_arr)
        lr_q_arr = _fmt(lr_q_arr)

    # -------------------------------------------------------------------------
    # Convert to IPF RGB maps
    # -------------------------------------------------------------------------
    sr_rgb = render_ipf_rgb(sr_q_arr, sym_class, ref_dir=ref_dir)
    hr_rgb = render_ipf_rgb(hr_q_arr, sym_class, ref_dir=ref_dir)
    lr_rgb = render_ipf_rgb(lr_q_arr, sym_class, ref_dir=ref_dir)

    multi_ref = isinstance(sr_rgb, list)
    ncols = 3 if multi_ref else 1
    spacer_cols = 1 if include_key else 0  # narrow gap before key
    key_cols = 1 if include_key else 0
    total_cols = ncols + spacer_cols + key_cols
    has_mis_row = mis_deg_arr is not None
    total_rows = 4 if has_mis_row else 3  # LR, SR, HR, [Mis heatmap]

    # -------------------------------------------------------------------------
    # Figure setup — row heights proportional to actual pixel heights so that
    # LR appears small and SR / HR appear large (matching their true resolution).
    # -------------------------------------------------------------------------
    h_sr, w_sr = sr_q_arr.shape[:2]
    h_hr, w_hr = hr_q_arr.shape[:2]
    h_lr, w_lr = lr_q_arr.shape[:2]

    # Anchor: LR row is at least 1.5 inches tall; everything else scales from that.
    min_lr_inches = 1.5
    px_per_inch = h_lr / min_lr_inches
    row_h_sr = h_sr / px_per_inch
    row_h_hr = h_hr / px_per_inch
    row_h_lr = h_lr / px_per_inch
    row_h_mis = row_h_hr * 0.55 if has_mis_row else 0.0  # compact heatmap row
    fig_h = row_h_sr + row_h_hr + row_h_lr + row_h_mis + 0.5

    # Width: scale so the widest image (SR/HR) fills ~5 in per column at its aspect ratio
    max_w = max(w_sr, w_hr, w_lr)
    base_w = max(4.0, max_w / px_per_inch / ncols)
    key_w = 2.6 if include_key else 0
    fig_w = base_w * ncols + key_w

    spacer_ratio = [0.15] if include_key else []
    key_ratio = [0.9] if include_key else []
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [row_h_lr, row_h_sr, row_h_hr]
    if has_mis_row:
        height_ratios.append(row_h_mis)
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[1] * ncols + spacer_ratio + key_ratio,
        height_ratios=height_ratios,
        hspace=0.25,
        wspace=0.1,
    )

    def _imshow(ax, img, title, shape):
        ax.imshow(img)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}  ({shape[1]}×{shape[0]}px)", fontsize=10)
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Plot LR (row 0), SR (row 1), HR (row 2), Misorientation heatmap (row 3)
    # -------------------------------------------------------------------------
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), lr_rgb)):
            _imshow(fig.add_subplot(gs[0, j]), img, f"LR IPF-{name}", (h_lr, w_lr))
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), sr_rgb)):
            _imshow(fig.add_subplot(gs[1, j]), img, f"SR IPF-{name}", (h_sr, w_sr))
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), hr_rgb)):
            _imshow(fig.add_subplot(gs[2, j]), img, f"HR IPF-{name}", (h_hr, w_hr))
    else:
        _imshow(fig.add_subplot(gs[0, 0]), lr_rgb, f"LR IPF-{ref_dir.upper()}", (h_lr, w_lr))
        _imshow(fig.add_subplot(gs[1, 0]), sr_rgb, f"SR IPF-{ref_dir.upper()}", (h_sr, w_sr))
        _imshow(fig.add_subplot(gs[2, 0]), hr_rgb, f"HR IPF-{ref_dir.upper()}", (h_hr, w_hr))

    if has_mis_row:
        # Span the heatmap across all IPF columns for a wide single panel
        ax_mis = fig.add_subplot(gs[3, 0:ncols])
        vmax = float(np.percentile(mis_deg_arr, 99))  # robust max (clip outliers)
        im = ax_mis.imshow(
            mis_deg_arr,
            cmap="inferno",
            vmin=0.0,
            vmax=max(vmax, 1e-3),
            aspect="equal",
            interpolation="nearest",
        )
        ax_mis.set_title(
            f"Misorientation SR→HR  ({h_hr}×{w_hr}px)  "
            f"mean={mis_deg_arr.mean():.2f}°  p99={vmax:.2f}°",
            fontsize=10,
        )
        ax_mis.axis("off")
        cbar = fig.colorbar(im, ax=ax_mis, orientation="horizontal", fraction=0.03, pad=0.08)
        cbar.set_label("Misorientation angle (°)", fontsize=9)

    # -------------------------------------------------------------------------
    # IPF color key
    # -------------------------------------------------------------------------
    if include_key:
        ax_key = fig.add_subplot(gs[:, -1], projection="ipf", symmetry=sym_class.laue)
        ax_key.plot_ipf_color_key()
        ax_key.set_title("")

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved SR-HR-LR comparison to: {out_png}")
        return out_png

    plt.show()
    return None


def render_input_output_side_by_side(
    input_q_arr: np.ndarray,
    output_q_arr: np.ndarray,
    sym_class,
    out_png: Optional[str] = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    format_input: bool = True,
    dpi: int = 300,
):
    """
    Render input and output quaternion orientation maps
    in a 2-row layout (input on top, output on bottom), optionally with IPF key.

    Parameters
    ----------
    input_q_arr : ndarray
        Input quaternion array of shape (H, W, 4). MUST BE SCALAR-FIRST (w, x, y, z) for correct formatting.
    output_q_arr : ndarray
        Output quaternion array of shape (H, W, 4). MUST BE SCALAR-FIRST (w, x, y, z) for correct formatting.
    sym_class : orix symmetry
        Symmetry object for IPF coloring.
    out_png : str, optional
        Output PNG file path. If None, figure is not saved.
    ref_dir : {"X","Y","Z","ALL"}, default="ALL"
        Reference direction(s) for coloring.
    include_key : bool, default=True
        Whether to include IPF color key panel.
    overwrite : bool, default=False
        If False, skip rendering if file exists.
    format_input : bool, default=True
        If True, canonicalize quaternions.
    dpi : int, default=300
        Figure DPI for saved PNG.
    """
    # -------------------------------------------------------------------------
    # Early exit if file already exists
    # -------------------------------------------------------------------------
    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    # -------------------------------------------------------------------------
    # Format quaternions (reduce to FZ, normalize, hemisphere, etc.)
    # -------------------------------------------------------------------------
    if format_input:
        input_q_arr = format_quaternions(
            input_q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            quat_first=False,
        )
        output_q_arr = format_quaternions(
            output_q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            quat_first=False,
        )

    # -------------------------------------------------------------------------
    # Convert to IPF RGB maps
    # -------------------------------------------------------------------------
    input_rgb = render_ipf_rgb(input_q_arr, sym_class, ref_dir=ref_dir)
    output_rgb = render_ipf_rgb(output_q_arr, sym_class, ref_dir=ref_dir)

    multi_ref = isinstance(input_rgb, list)
    ncols = 3 if multi_ref else 1
    key_cols = 1 if include_key else 0
    total_cols = ncols + key_cols
    total_rows = 2  # input top, output bottom

    # -------------------------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------------------------
    base_w = 5.0
    key_w = 2.6 if include_key else 0
    fig_w = base_w * ncols + key_w
    fig_h = 2 * 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[1] * ncols + ([0.9] if include_key else []),
        height_ratios=[1, 1],
        hspace=0.25,
        wspace=0.25,
    )

    def _imshow(ax, img, title):
        ax.imshow(img)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Plot input (top row) and output (bottom row)
    # -------------------------------------------------------------------------
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), output_rgb)):
            _imshow(fig.add_subplot(gs[0, j]), img, f"Output IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), input_rgb)):
            _imshow(fig.add_subplot(gs[1, j]), img, f"Input IPF-{name}")
    else:
        _imshow(fig.add_subplot(gs[0, 0]), input_rgb, f"Input IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[1, 0]), output_rgb, f"Output IPF-{ref_dir.upper()}")

    # -------------------------------------------------------------------------
    # IPF color key
    # -------------------------------------------------------------------------
    if include_key:
        ax_key = fig.add_subplot(gs[:, -1], projection="ipf", symmetry=sym_class.laue)
        ax_key.plot_ipf_color_key()
        ax_key.set_title("")

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved Input-Output comparison to: {out_png}")
        return out_png

    plt.show()
    return None
