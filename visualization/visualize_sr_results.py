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
):
    """
    Render SR (Super-Resolution), HR (High-Resolution), and LR (Low-Resolution)
    quaternion orientation maps in a 3-row layout:
        SR (top), HR (middle), LR (bottom)

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
                scalar_first=True,
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
    key_cols = 1 if include_key else 0
    total_cols = ncols + key_cols
    total_rows = 3  # SR, HR, LR

    # -------------------------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------------------------
    base_w = 5.0
    key_w = 2.6 if include_key else 0
    fig_w = base_w * ncols + key_w
    fig_h = 3 * 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[1] * ncols + ([0.9] if include_key else []),
        height_ratios=[1, 1, 1],
        hspace=0.3,
        wspace=0.25,
    )

    def _imshow(ax, img, title):
        ax.imshow(img)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Plot SR (row 0), HR (row 1), LR (row 2)
    # -------------------------------------------------------------------------
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), sr_rgb)):
            _imshow(fig.add_subplot(gs[0, j]), img, f"SR IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), hr_rgb)):
            _imshow(fig.add_subplot(gs[1, j]), img, f"HR IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), lr_rgb)):
            _imshow(fig.add_subplot(gs[2, j]), img, f"LR IPF-{name}")
    else:
        _imshow(fig.add_subplot(gs[0, 0]), sr_rgb, f"SR IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[1, 0]), hr_rgb, f"HR IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[2, 0]), lr_rgb, f"LR IPF-{ref_dir.upper()}")

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
