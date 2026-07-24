# -*-coding:utf-8 -*-
"""
File:        dataset_utils.py
Created at:  2025/10/17 13:20:23
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Dataset helper functions for quaternion SR dataset construction.
"""

import os
import re
import random
import numpy as np
from typing import List, Tuple, Optional

# -----------------------------------
# Regex for numeric sorting of files
# -----------------------------------
_LAST_INT_RE = re.compile(r"(\d+)(?=\.npy$)")


# -----------------------------------
# File and path helpers
# -----------------------------------
def last_int_key(fp: str) -> int:
    """
    Extract the last integer from a filename (before `.npy`) for natural sorting.

    Example:
        "file_003.npy" -> 3
    """
    m = _LAST_INT_RE.search(os.path.basename(fp))
    return int(m.group(1)) if m else -1


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_npy(path: str, arr: np.ndarray) -> None:
    """
    Save numpy array to disk in float32 C-contiguous format.

    Parameters
    ----------
    path : str
        Output path for .npy file.
    arr : np.ndarray
        Array to save.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    np.save(path, arr)


# -----------------------------------
# Patch size helpers
# -----------------------------------
def pick_patch_size_all(
    files: List[str], scale: int, cap: Optional[int] = None
) -> Tuple[int, ...]:
    """
    Compute largest power-of-2 patch size that fits all quaternion images and
    is divisible by `scale`. Uses memmap headers only (no data loading).

    Supports (4,H,W), (H,W,4), (4,H,W,D), (H,W,D,4), etc.

    Parameters
    ----------
    files : list of str
        List of file paths.
    scale : int
        Downsampling scale (must divide patch size evenly).
    cap : int, optional
        Upper bound for patch size.

    Returns
    -------
    patch_shape : tuple[int]
        Equal-length patch dimensions (P,), (P,P), or (P,P,P).
    """
    min_side = np.inf
    ndim = None

    for fp in files:
        mm = np.lib.format.open_memmap(fp, mode="r")
        shp = mm.shape
        del mm

        # Identify quaternion axis (dim=4)
        if shp[0] == 4:
            spatial = shp[1:]
        elif shp[-1] == 4:
            spatial = shp[:-1]
        else:
            axes = [i for i, s in enumerate(shp) if s == 4]
            if not axes:
                raise ValueError(f"No quaternion axis (size=4) found in shape {shp}")
            ax = axes[0]
            spatial = tuple(s for i, s in enumerate(shp) if i != ax)

        if len(spatial) < 1:
            raise ValueError(f"No spatial dimensions found in {fp}, shape={shp}")

        ndim = len(spatial) if ndim is None else ndim
        if len(spatial) != ndim:
            raise ValueError(
                f"Inconsistent dimensionality across files ({len(spatial)} vs {ndim})"
            )

        min_side = min(min_side, *spatial)

    if not np.isfinite(min_side):
        raise ValueError("No valid spatial shapes found in dataset")

    # Cap and quantize to nearest power of 2 divisible by scale
    lim = int(min(min_side, cap)) if cap else int(min_side)
    P = 1 << (lim.bit_length() - 1)
    while P % scale != 0:
        P //= 2
    if P <= 0:
        raise ValueError(f"No valid patch size for scale={scale}")

    return tuple([P] * ndim)


# -----------------------------------
# Patch extraction helpers
# -----------------------------------
def random_aligned_patches(
    q: np.ndarray, patch_shape: Tuple[int, ...], scale: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a random aligned HR/LR quaternion patch from (4,*spatial).

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4,H,W,...) or similar.
    patch_shape : tuple[int]
        Patch shape to extract.
    scale : int
        Downsampling scale factor.

    Returns
    -------
    lr_patch : np.ndarray
        Low-resolution quaternion patch (4, H/scale, W/scale, ...).
    hr_patch : np.ndarray
        High-resolution quaternion patch (4, H, W, ...).
    """
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion-first (4,...), got {q.shape}")

    ndim = q.ndim - 1
    if len(patch_shape) != ndim:
        raise ValueError(f"Patch shape {patch_shape} incompatible with {q.shape}")

    starts = [
        random.randrange(0, q.shape[d + 1] - patch_shape[d] + 1) for d in range(ndim)
    ]
    slices_hr = (slice(None),) + tuple(
        slice(st, st + ps) for st, ps in zip(starts, patch_shape)
    )
    hr = q[slices_hr]
    stride = (slice(None),) + tuple(slice(None, None, scale) for _ in range(ndim))
    lr = hr[stride]

    # Ensure proper dtype and memory layout
    return (
        np.asarray(lr, dtype=np.float32, order="C"),
        np.asarray(hr, dtype=np.float32, order="C"),
    )
