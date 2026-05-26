import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from utils.quat_ops import (
    enforce_hemisphere,
    format_quaternions,
    normalize_quaternions,
    quat_conjugate,
    quat_mul_np,
)
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb


def _project_lr_rgb_to_hr_sparse(
    lr_rgb: np.ndarray,
    lr_hw: tuple[int, int],
    hr_hw: tuple[int, int],
) -> np.ndarray:
    """Embed LR RGB pixels on an HR canvas; non-sampled pixels are black."""
    h_lr, w_lr = int(lr_hw[0]), int(lr_hw[1])
    h_hr, w_hr = int(hr_hw[0]), int(hr_hw[1])
    if lr_rgb.shape[:2] != (h_lr, w_lr):
        raise ValueError(
            f"LR RGB shape {tuple(lr_rgb.shape[:2])} does not match lr_hw={(h_lr, w_lr)}"
        )
    if h_lr <= 0 or w_lr <= 0 or h_hr <= 0 or w_hr <= 0:
        raise ValueError(f"Invalid shapes lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}")

    scale_y = h_hr // h_lr
    scale_x = w_hr // w_lr
    if h_lr * scale_y != h_hr or w_lr * scale_x != w_hr:
        raise ValueError(
            "HR shape must be an integer multiple of LR shape for sparse projection: "
            f"lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}"
        )

    out = np.zeros((h_hr, w_hr, 3), dtype=lr_rgb.dtype)
    y_idx = np.arange(h_lr, dtype=np.int64) * scale_y + (scale_y // 2)
    x_idx = np.arange(w_lr, dtype=np.int64) * scale_x + (scale_x // 2)
    y_idx = np.clip(y_idx, 0, h_hr - 1)
    x_idx = np.clip(x_idx, 0, w_hr - 1)
    out[np.ix_(y_idx, x_idx)] = lr_rgb
    return out


def _upsample_lr_rgb_to_hr_nearest(
    lr_rgb: np.ndarray,
    lr_hw: tuple[int, int],
    hr_hw: tuple[int, int],
) -> np.ndarray:
    """Upsample LR RGB to the HR canvas with nearest-neighbor replication."""
    h_lr, w_lr = int(lr_hw[0]), int(lr_hw[1])
    h_hr, w_hr = int(hr_hw[0]), int(hr_hw[1])
    if lr_rgb.shape[:2] != (h_lr, w_lr):
        raise ValueError(
            f"LR RGB shape {tuple(lr_rgb.shape[:2])} does not match lr_hw={(h_lr, w_lr)}"
        )
    if h_lr <= 0 or w_lr <= 0 or h_hr <= 0 or w_hr <= 0:
        raise ValueError(f"Invalid shapes lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}")

    scale_y = h_hr // h_lr
    scale_x = w_hr // w_lr
    if h_lr * scale_y != h_hr or w_lr * scale_x != w_hr:
        raise ValueError(
            "HR shape must be an integer multiple of LR shape for nearest upsampling: "
            f"lr={(h_lr, w_lr)} hr={(h_hr, w_hr)}"
        )
    return np.repeat(np.repeat(lr_rgb, scale_y, axis=0), scale_x, axis=1)


def _pad_rgb_to_canvas(
    rgb: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Center-pad an RGB image on a black canvas without resampling."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    th, tw = int(target_hw[0]), int(target_hw[1])
    if h > th or w > tw:
        raise ValueError(f"Image {(h, w)} is larger than target canvas {(th, tw)}")
    out = np.zeros((th, tw, 3), dtype=rgb.dtype)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rgb
    return out


def _proper_sym_ops_wxyz(sym_class) -> np.ndarray:
    """Return proper crystal symmetry ops (wxyz), normalized, with identity at index 0."""
    qcs_all = np.asarray(sym_class.data, dtype=np.float64)
    qcs = qcs_all[:24] if getattr(sym_class, "name", "") == "m-3m" else qcs_all
    qcs = normalize_quaternions(qcs, axis=-1)

    id_idx = int(np.argmin(np.linalg.norm(qcs - np.array([1.0, 0.0, 0.0, 0.0]), axis=1)))
    if id_idx != 0:
        qcs[[0, id_idx]] = qcs[[id_idx, 0]]
    return qcs


def _align_output_to_input_branch(
    output_q_arr: np.ndarray,
    input_q_arr: np.ndarray,
    sym_class,
    global_refine_iters: int = 2,
) -> np.ndarray:
    """
    Symmetry-match each output quaternion to the closest equivalent branch of the input.

    This prevents global color-palette shifts in IPF visualization when output is correct
    up to a crystal symmetry operator.
    """
    q_in = np.asarray(input_q_arr, dtype=np.float64)
    q_out = np.asarray(output_q_arr, dtype=np.float64)
    if q_in.shape != q_out.shape:
        raise ValueError(
            f"Input/output quaternion shapes must match for branch alignment: {q_in.shape} vs {q_out.shape}"
        )

    spatial_shape = q_in.shape[:-1]
    q_in_flat = q_in.reshape(-1, 4)
    q_out_flat = q_out.reshape(-1, 4)

    qcs = _proper_sym_ops_wxyz(sym_class)
    qcs_inv = quat_conjugate(qcs)  # (M,4)

    def _sym_match(q_flat: np.ndarray) -> np.ndarray:
        # Left action (Bunge/passive): s^{-1} ⊗ q_out
        cand_left = quat_mul_np(qcs_inv[None, :, :], q_flat[:, None, :])  # (N,M,4)
        score_left = np.abs(np.einsum("nmi,ni->nm", cand_left, q_in_flat))
        idx_left = np.argmax(score_left, axis=1)
        best_left = cand_left[np.arange(q_flat.shape[0]), idx_left]
        best_left_score = score_left[np.arange(q_flat.shape[0]), idx_left]

        # Right-action fallback for convention mismatch.
        cand_right = quat_mul_np(q_flat[:, None, :], qcs_inv[None, :, :])  # (N,M,4)
        score_right = np.abs(np.einsum("nmi,ni->nm", cand_right, q_in_flat))
        idx_right = np.argmax(score_right, axis=1)
        best_right = cand_right[np.arange(q_flat.shape[0]), idx_right]
        best_right_score = score_right[np.arange(q_flat.shape[0]), idx_right]

        use_left = best_left_score >= best_right_score
        out = np.where(use_left[:, None], best_left, best_right)

        sign = np.einsum("ni,ni->n", out, q_in_flat)
        out[sign < 0.0] *= -1.0
        return out

    q_aligned_flat = _sym_match(q_out_flat)

    # Refine a residual global frame offset (palette shift) then rematch symmetry.
    iters = max(0, int(global_refine_iters))
    for _ in range(iters):
        delta = quat_mul_np(q_in_flat, quat_conjugate(q_aligned_flat))
        delta = normalize_quaternions(delta, axis=-1)
        delta[delta[:, 0] < 0.0] *= -1.0
        delta_mean = delta.mean(axis=0, keepdims=True)
        delta_mean = normalize_quaternions(delta_mean, axis=-1)

        q_aligned_flat = quat_mul_np(
            np.broadcast_to(delta_mean, q_aligned_flat.shape),
            q_aligned_flat,
        )
        q_aligned_flat = normalize_quaternions(q_aligned_flat, axis=-1)
        q_aligned_flat = _sym_match(q_aligned_flat)

    q_aligned = q_aligned_flat.reshape(*spatial_shape, 4)
    q_aligned = normalize_quaternions(q_aligned, axis=-1)
    q_aligned = enforce_hemisphere(q_aligned, scalar_first=True)
    return q_aligned.astype(np.float32, copy=False)


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
            to_quat_first=False,
        )
        hr_q_arr = format_quaternions(
            hr_q_arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            to_quat_first=False,
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
        ax.imshow(img, interpolation="nearest", resample=False)
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
        add_ipf_key_panel(
            fig,
            gs[:, -1],
            sym_class,
            title="IPF Key",
            title_fontsize=10.0,
            label_fontsize=8.0,
        )

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
    pixels_per_image_pixel: int = 1,
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
                to_quat_first=False,
            )

        sr_q_arr = _fmt(sr_q_arr)
        hr_q_arr = _fmt(hr_q_arr)
        lr_q_arr = _fmt(lr_q_arr)

    # -------------------------------------------------------------------------
    # Convert to IPF RGB maps
    # -------------------------------------------------------------------------
    lr_h, lr_w = lr_q_arr.shape[:2]
    sr_h, sr_w = sr_q_arr.shape[:2]
    hr_h, hr_w = hr_q_arr.shape[:2]

    sr_rgb = render_ipf_rgb(sr_q_arr, sym_class, ref_dir=ref_dir)
    hr_rgb = render_ipf_rgb(hr_q_arr, sym_class, ref_dir=ref_dir)
    lr_rgb = render_ipf_rgb(lr_q_arr, sym_class, ref_dir=ref_dir)

    multi_ref = isinstance(sr_rgb, list)

    if multi_ref:
        lr_rgb = [
            _upsample_lr_rgb_to_hr_nearest(img, (lr_h, lr_w), (hr_h, hr_w))
            for img in lr_rgb
        ]
    else:
        lr_rgb = _upsample_lr_rgb_to_hr_nearest(lr_rgb, (lr_h, lr_w), (hr_h, hr_w))

    target_h = max(int(hr_h), int(sr_h))
    target_w = max(int(hr_w), int(sr_w))

    if multi_ref:
        lr_rgb = [_pad_rgb_to_canvas(img, (target_h, target_w)) for img in lr_rgb]
        sr_rgb = [_pad_rgb_to_canvas(img, (target_h, target_w)) for img in sr_rgb]
        hr_rgb = [_pad_rgb_to_canvas(img, (target_h, target_w)) for img in hr_rgb]
    else:
        lr_rgb = _pad_rgb_to_canvas(lr_rgb, (target_h, target_w))
        sr_rgb = _pad_rgb_to_canvas(sr_rgb, (target_h, target_w))
        hr_rgb = _pad_rgb_to_canvas(hr_rgb, (target_h, target_w))

    ncols = 3 if multi_ref else 1
    label_cols = 1
    key_cols = 1 if include_key else 0
    total_cols = label_cols + ncols + key_cols
    total_rows = 3  # LR, SR, HR

    # -------------------------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------------------------
    scale = max(1, int(pixels_per_image_pixel))
    panel_h = int(target_h) * scale
    panel_w = int(target_w) * scale
    label_w = int(round(0.90 * panel_w))
    key_w = int(round(1.15 * panel_w)) if include_key else 0
    fig_px_w = label_w + panel_w * ncols + key_w
    fig_px_h = panel_h * total_rows
    fig = plt.figure(
        figsize=(fig_px_w / float(dpi), fig_px_h / float(dpi)),
        dpi=dpi,
        facecolor="white",
    )
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[label_w] + [panel_w] * ncols + ([key_w] if include_key else []),
        height_ratios=[panel_h, panel_h, panel_h],
        hspace=0.015,
        wspace=0.015,
    )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.995, wspace=0.015, hspace=0.015)

    def _imshow(ax, img):
        ax.imshow(img, interpolation="nearest", resample=False)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # Dedicated label panel on white background to keep text off the image content.
    ax_labels = fig.add_subplot(gs[:, 0])
    ax_labels.set_facecolor("white")
    ax_labels.set_xlim(0.0, 1.0)
    ax_labels.set_ylim(0.0, 1.0)
    ax_labels.axis("off")

    # Convert desired pixel-sized typography to points so text remains stable at high DPI.
    pt_per_px = 72.0 / float(max(int(dpi), 1))
    row_title_fs = float(np.clip(0.090 * panel_h, 16.0, 26.0) * pt_per_px)
    row_detail_fs = float(np.clip(0.060 * panel_h, 10.0, 16.0) * pt_per_px)
    col_title_fs = float(np.clip(0.060 * panel_h, 11.0, 16.0) * pt_per_px)

    row_labels = [
        ("LR", f"display upsampled\n{lr_h}x{lr_w} -> {hr_h}x{hr_w}"),
        ("SR", f"{sr_h}x{sr_w}"),
        ("HR", f"{hr_h}x{hr_w}"),
    ]
    row_y = [5.0 / 6.0, 0.5, 1.0 / 6.0]

    for y, (title, detail) in zip(row_y, row_labels):
        ax_labels.text(
            0.06,
            y + 0.055,
            title,
            fontsize=row_title_fs,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="center",
        )
        ax_labels.text(
            0.06,
            y - 0.020,
            detail,
            fontsize=row_detail_fs,
            fontweight="normal",
            color="#4a4a4a",
            ha="left",
            va="center",
            linespacing=1.12,
        )

    def _set_panel_title(ax, title: str) -> None:
        ax.set_title(
            title,
            fontsize=col_title_fs,
            fontweight="semibold",
            color="#111111",
            pad=6,
        )

    # -------------------------------------------------------------------------
    # Plot LR (row 0), SR (row 1), HR (row 2)
    # -------------------------------------------------------------------------
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), lr_rgb)):
            ax = fig.add_subplot(gs[0, j + 1])
            _imshow(ax, img)
            _set_panel_title(ax, f"IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), sr_rgb)):
            _imshow(fig.add_subplot(gs[1, j + 1]), img)
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), hr_rgb)):
            _imshow(fig.add_subplot(gs[2, j + 1]), img)
    else:
        ax = fig.add_subplot(gs[0, 1])
        _imshow(ax, lr_rgb)
        _set_panel_title(ax, f"IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[1, 1]), sr_rgb)
        _imshow(fig.add_subplot(gs[2, 1]), hr_rgb)

    # -------------------------------------------------------------------------
    # IPF color key
    # -------------------------------------------------------------------------
    if include_key:
        key_title_fs = float(np.clip(0.070 * panel_h, 11.0, 18.0) * pt_per_px)
        key_label_fs = float(np.clip(0.055 * panel_h, 9.0, 14.0) * pt_per_px)
        add_ipf_key_panel(
            fig,
            gs[1, ncols + 1],
            sym_class,
            title="IPF Key",
            title_fontsize=key_title_fs,
            label_fontsize=key_label_fs,
        )

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor())
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
    align_output_to_input: bool = False,
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
    align_output_to_input : bool, default=False
        If True, symmetry-match output to input before IPF rendering to avoid
        branch/palette shifts due to equivalent cubic orientations.
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
            to_quat_first=False,
        )
        if align_output_to_input:
            output_q_arr = format_quaternions(
                output_q_arr,
                normalize=True,
                hemisphere=True,
                reduce_fz=False,
                sym=sym_class,
                to_quat_first=False,
            )
            output_q_arr = _align_output_to_input_branch(
                output_q_arr=output_q_arr,
                input_q_arr=input_q_arr,
                sym_class=sym_class,
            )
        else:
            output_q_arr = format_quaternions(
                output_q_arr,
                normalize=True,
                hemisphere=True,
                reduce_fz=True,
                sym=sym_class,
                to_quat_first=False,
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
        ax.imshow(img, interpolation="nearest", resample=False)
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
        add_ipf_key_panel(
            fig,
            gs[:, -1],
            sym_class,
            title="IPF Key",
            title_fontsize=10.0,
            label_fontsize=8.0,
        )

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
