# -*-coding:utf-8 -*-
"""
File:        quat_ops.py
Created at:  2025/10/17 18:33:35
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import numpy as np
import torch
from orix.quaternion import Orientation, symmetry as SYM
from orix.quaternion.orientation_region import OrientationRegion
from utils.symmetry_utils import resolve_symmetry
from orix.sampling import get_sample_fundamental


def assert_quaternion_shape(q: np.ndarray):
    """
    Ensure the input array has exactly one quaternion axis of size 4.

    Parameters
    ----------
    q : np.ndarray

    Raises
    ------
    ValueError
        If no axis of size 4 is found or if multiple exist.
    """
    if not isinstance(q, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(q)}")

    quat_axes = [i for i, s in enumerate(q.shape) if s == 4]
    if len(quat_axes) == 0:
        raise ValueError(f"No quaternion axis (size=4) found in shape {q.shape}")
    if len(quat_axes) > 1:
        raise ValueError(f"Multiple quaternion axes found in shape {q.shape}")


# =====================================================
# Quaternion axis helpers
# =====================================================


def to_spatial_quat(arr: np.ndarray) -> np.ndarray:
    """
    Reorder array so the quaternion axis (size=4) is last: (*spatial, 4).

    Accepts layouts such as:
        (4, H, W)
        (H, W, 4)
        (4, H, W, D)
        (H, W, D, 4)
        etc.

    Parameters
    ----------
    arr : np.ndarray
        Quaternion array containing exactly one axis of length 4.

    Returns
    -------
    q : np.ndarray
        Quaternion-last array (*spatial, 4), dtype float32.

    Raises
    ------
    ValueError
        If no quaternion axis (size=4) found or multiple exist.
    """
    assert_quaternion_shape(arr)

    shape = arr.shape
    if shape[-1] == 4:
        q = arr
    elif shape[0] == 4:
        q = np.moveaxis(arr, 0, -1)
    else:
        quat_axes = [i for i, s in enumerate(shape) if s == 4]
        q = np.moveaxis(arr, quat_axes[0], -1)

    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)
    return q


def to_quat_spatial(arr: np.ndarray) -> np.ndarray:
    """
    Reorder array so the quaternion channel (size 4) comes first: (4, *spatial).

    Common layouts:
        (4, H, W, ...)
        (H, W, ..., 4)

    Falls back to scanning all axes for the quaternion dimension if not found
    in the first or last position.

    Parameters
    ----------
    arr : np.ndarray
        Quaternion array containing exactly one axis of length 4.

    Returns
    -------
    q : np.ndarray
        Quaternion-first array (4, *spatial), float32 dtype preserved.

    Raises
    ------
    ValueError
        If no dimension of length 4 exists or if multiple do.
    """
    assert_quaternion_shape(arr)

    shape = arr.shape
    if shape[0] == 4:
        q = arr
    elif shape[-1] == 4:
        q = np.moveaxis(arr, -1, 0)
    else:
        quat_axes = [i for i, s in enumerate(shape) if s == 4]
        q = np.moveaxis(arr, quat_axes[0], 0)

    return q.astype(np.float32, copy=False)


# =====================================================
# Scalar position helpers
# =====================================================


# def is_scalar_first(q: np.ndarray) -> bool:
#     """
#     Return True if the quaternion array is scalar-first ([s,x,y,z]),
#     and False if it is scalar-last ([x,y,z,s]).

#     Works for both (4, H, W, ...) and (H, W, ..., 4) layouts.

#     Parameters
#     ----------
#     q : np.ndarray
#         Quaternion array.

#     Returns
#     -------
#     bool
#         True if scalar-first, False otherwise.
#     """
#     assert_quaternion_shape(q)

#     if q.shape[0] == 4:  # quat-first
#         return np.mean(np.abs(q[0])) >= np.mean(np.abs(q[-1]))
#     else:  # spatial-last
#         return np.mean(np.abs(q[..., 0])) >= np.mean(np.abs(q[..., -1]))


def to_scalar_first(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion array from [x,y,z,s] (scalar-last)
    to [s,x,y,z] (scalar-first).

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4, ...) or (..., 4): MUST BE SCALAR LAST.

    Returns
    -------
    np.ndarray
        Scalar-first quaternion array.
    """

    if q.shape[0] == 4:
        out = np.empty_like(q, dtype=np.float32)
        out[0] = q[-1]
        out[1:] = q[:-1]
        return out
    elif q.shape[-1] == 4:
        out = np.empty_like(q, dtype=np.float32)
        out[..., 0] = q[..., -1]
        out[..., 1:] = q[..., :-1]
        return out

    raise ValueError(
        f"[to_quat_scalar_first] Quaternion dimension not found in {q.shape}"
    )


def to_scalar_last(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion array from [s,x,y,z] (scalar-first)
    to [x,y,z,s] (scalar-last).

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4, ...) or (..., 4). MUST BE SCALAR FIRST.

    Returns
    -------
    np.ndarray
        Scalar-last quaternion array.
    """

    if q.shape[0] == 4:
        out = np.empty_like(q, dtype=np.float32)
        out[:-1] = q[1:]
        out[-1] = q[0]
        return out
    elif q.shape[-1] == 4:
        out = np.empty_like(q, dtype=np.float32)
        out[..., :-1] = q[..., 1:]
        out[..., -1] = q[..., 0]
        return out

    raise ValueError(
        f"[to_quat_scalar_last] Quaternion dimension not found in {q.shape}"
    )


# =====================================================
# Normalization & hemisphere helpers
# =====================================================


def safe_norm(
    x: np.ndarray,
    axis: int = 0,
    keepdims: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute vector norm with epsilon guard to avoid division by zero.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, default=-1
        Axis to compute norm over.
    keepdims : bool, default=True
        Whether to keep dimensions (for safe broadcasting).
    eps : float, default=1e-12
        Minimum threshold for norm.

    Returns
    -------
    np.ndarray
        Norm array (broadcastable to input).
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    np.maximum(norms, eps, out=norms)  # in-place clamp for speed
    return norms


def normalize_quaternions(
    q: np.ndarray,
    axis: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize quaternion arrays to unit length using safe_norm.
    """
    if not q.flags.writeable:
        q = q.copy()
    q /= safe_norm(q, axis=axis, keepdims=True, eps=eps)
    return q


def enforce_hemisphere(
    q: np.ndarray,
    scalar_first: bool = True,
) -> np.ndarray:
    """
    Enforce scalar part >= 0 (canonical hemisphere) for quaternion arrays.

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4, *spatial) or (*spatial, 4).
    scalar_first : bool, optional, default=True
        True if scalar is the first component, False if scalar is last.

    Returns
    -------
    np.ndarray
        Hemisphere-aligned quaternion array (same layout, in-place when possible).
    """
    assert_quaternion_shape(q)

    # Make writeable if it's a memmap or read-only
    if not q.flags.writeable:
        q = q.copy()

    idx = 0 if scalar_first else -1

    if q.shape[0] == 4:
        # quaternion-first layout (4, H, W)
        mask = q[idx] < 0
        if np.any(mask):
            q[:, mask] *= -1.0
    else:
        # spatial-last layout (H, W, 4)
        mask = q[..., idx] < 0
        if np.any(mask):
            q[mask] *= -1.0
    return q


# ----------------------
# Multiplication
# ----------------------


def quat_left_multiply_numpy(
    q_right: np.ndarray,
    q_left: np.ndarray,
    eps: float = 1e-12,
    normalize: bool = True,
    layout: str = "quat_first",
) -> np.ndarray:
    """
    Left multiply a set of symmetry operators with a quaternion field (NumPy).

    Computes:  q_out = q_left ⊗ q_right

    Parameters
    ----------
    q_right : np.ndarray
        Quaternion array of shape (4, *spatial) or (*spatial, 4).
    q_left : np.ndarray
        Operator quaternions of shape (M, 4).
    eps : float, default=1e-12
        Numerical floor for normalization.
    normalize : bool, default=True
        If True, normalize output quaternions.
    layout : {"quat_first", "quat_last"}, default="quat_first"
        Output layout:
        - "quat_first" → (M, 4, *spatial)
        - "quat_last"  → (M, *spatial, 4)

    Returns
    -------
    np.ndarray
        Quaternion array with left-multiplied operators, shape depends on `layout`.
    """
    assert_quaternion_shape(q_right)
    q_right = np.asarray(q_right, dtype=np.float32)
    q_left = np.asarray(q_left, dtype=np.float32)

    # Convert to (*spatial, 4)
    q_spatial = to_spatial_quat(q_right)

    spatial_shape = q_spatial.shape[:-1]
    N = int(np.prod(spatial_shape))
    M = q_left.shape[0]

    flat = q_spatial.reshape(N, 4)

    # Components
    w0, x0, y0, z0 = q_left[:, 0], q_left[:, 1], q_left[:, 2], q_left[:, 3]
    w1, x1, y1, z1 = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]

    out = np.empty((M, N, 4), dtype=np.float32)
    out[:, :, 0] = (
        w0[:, None] * w1 - x0[:, None] * x1 - y0[:, None] * y1 - z0[:, None] * z1
    )
    out[:, :, 1] = (
        w0[:, None] * x1 + x0[:, None] * w1 + y0[:, None] * z1 - z0[:, None] * y1
    )
    out[:, :, 2] = (
        w0[:, None] * y1 - x0[:, None] * z1 + y0[:, None] * w1 + z0[:, None] * x1
    )
    out[:, :, 3] = (
        w0[:, None] * z1 + x0[:, None] * y1 - y0[:, None] * x1 + z0[:, None] * w1
    )

    # Normalize
    if normalize:
        norms = safe_norm(out, axis=2, keepdims=True, eps=eps)
        out /= norms

    # Reshape to spatial layout
    out = out.reshape(M, *spatial_shape, 4)

    # Reorder if needed
    if layout == "quat_first":
        out = np.moveaxis(out, -1, 1)  # (M, 4, *spatial)
    elif layout == "quat_last":
        pass  # already in (*spatial, 4)
    else:
        raise ValueError(
            f"Invalid layout '{layout}'. Expected 'quat_first' or 'quat_last'."
        )

    return out


def quat_left_multiply_torch(
    q_right: torch.Tensor,
    q_left: torch.Tensor,
    eps: float = 1e-12,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Left multiply a set of symmetry operators with a quaternion field (Torch).
    Optimized for training/inference on GPU.

    Parameters
    ----------
    q_right : torch.Tensor
        Quaternion tensor of shape (4,*spatial) or (*spatial,4),
        already on the correct device and dtype=torch.float32.
    q_left : torch.Tensor
        Operator quaternions of shape (M, 4),
        on the same device as q_right.
    eps : float
        Numerical floor for normalization.
    normalize : bool
        If True, normalize output quaternions.

    Returns
    -------
    out : torch.Tensor
        Quaternion tensor of shape (M, 4, *spatial).
    """
    # Ensure layout (*spatial, 4)
    if q_right.shape[0] == 4:
        q_right = torch.moveaxis(q_right, 0, -1)

    orig_spatial = q_right.shape[:-1]
    N = int(torch.prod(torch.tensor(orig_spatial)).item())
    M = q_left.shape[0]

    flat = q_right.reshape(N, 4)

    # Components (already float32, no casting)
    w0, x0, y0, z0 = q_left[:, 0], q_left[:, 1], q_left[:, 2], q_left[:, 3]
    w1 = flat[:, 0].unsqueeze(0)
    x1 = flat[:, 1].unsqueeze(0)
    y1 = flat[:, 2].unsqueeze(0)
    z1 = flat[:, 3].unsqueeze(0)

    w0 = w0.unsqueeze(1)
    x0 = x0.unsqueeze(1)
    y0 = y0.unsqueeze(1)
    z0 = z0.unsqueeze(1)

    out = torch.empty((M, N, 4), dtype=torch.float32, device=q_right.device)
    out[:, :, 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    out[:, :, 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    out[:, :, 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    out[:, :, 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    if normalize:
        norms = torch.linalg.norm(out, dim=2, keepdim=True)
        out = out / torch.clamp(norms, min=eps)

    out = out.view(M, *orig_spatial, 4)
    out = torch.moveaxis(out, -1, 1)  # (M, 4, *spatial)
    return out


def is_in_fz(q: np.ndarray, sym) -> np.ndarray:
    """
    Check if each quaternion is inside the Fundamental Zone (FZ).

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4, *spatial) or (*spatial, 4).
    sym : orix.quaternion.symmetry.Symmetry
        Symmetry group.

    Returns
    -------
    mask : np.ndarray
        Array of shape (*spatial,) containing 1 if quaternion is in FZ,
        and 0 otherwise (dtype=np.uint8).
    """
    # 1. Standardize layout
    q_spatial = to_spatial_quat(q)  # (*spatial, 4)
    flat = q_spatial.reshape(-1, 4)

    # 2. Define region
    region = OrientationRegion.from_symmetry(sym)

    # 3. Membership check (vectorized)
    inside = Orientation(flat, sym) < region

    # 4. Reshape back to spatial shape, convert to int (0/1)
    mask = inside.reshape(q_spatial.shape[:-1]).astype(np.uint8)

    return mask


def reduce_to_fz_min_angle(
    q: np.ndarray,
    sym,
    normalize: bool = True,
    hemisphere: bool = True,
    return_op_map: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Reduce quaternions to the Fundamental Zone (FZ) of a given symmetry group
    using the minimum misorientation angle criterion.

    This implementation is fully vectorized and uses `quat_left_multiply_numpy`
    to apply symmetry operators efficiently.

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4, *spatial) or (*spatial, 4).
    sym : orix.quaternion.symmetry.Symmetry
        Symmetry group for FZ reduction.
    normalize : bool, default=True
        Whether to normalize quaternions to unit norm before reduction.
    hemisphere : bool, default=True
        Whether to enforce hemisphere (scalar ≥ 0) before reduction.
    return_op_map : bool, default=False
        If True, return also the operator index map used for each quaternion.
    eps : float, default=1e-12
        Numerical floor for normalization.

    Returns
    -------
    q_fz : np.ndarray
        Quaternion array reduced to FZ, same layout as input.
    op_map : np.ndarray, optional
        Operator index map (if return_op_map=True).
    """
    orig_first = q.shape[0] == 4
    q_spatial = q if not orig_first else to_spatial_quat(q)
    q_spatial = np.ascontiguousarray(q_spatial, dtype=np.float32)

    # 1. Optional normalize + hemisphere
    if normalize:
        normalize_quaternions(q_spatial, axis=-1, eps=eps)
    if hemisphere:
        enforce_hemisphere(q_spatial, scalar_first=True)

    if isinstance(sym, str):
        sym = resolve_symmetry(sym)

    flat = q_spatial.reshape(-1, 4)
    N = flat.shape[0]
    region = OrientationRegion.from_symmetry(sym)

    # Early exit if already in FZ
    if (Orientation(flat, sym) < region).all():
        q_out = q if orig_first else q_spatial
        if return_op_map:
            return q_out, np.zeros(q_spatial.shape[:-1], dtype=np.int32)
        return q_out

    # 2. Apply symmetry ops (vectorized)
    # Bunge convention: s⁻¹ ⊗ q  (unit quat inverse = conjugate: negate vector part)
    ops = sym.data.astype(np.float32, copy=False)
    ops_inv = ops.copy()
    ops_inv[:, 1:] *= -1.0
    M = ops_inv.shape[0]
    cand = quat_left_multiply_numpy(
        q_spatial, ops_inv, eps=eps, normalize=True, layout="quat_last"
    )
    cand_flat = cand.reshape(M * N, 4)

    # 3. Check FZ membership
    inside_mask = (Orientation(cand_flat, sym) < region).reshape(M, N)

    # 4. Select candidate with max scalar (min misorientation angle)
    cand_2d = cand.reshape(M, N, 4)
    w_vals = cand_2d[..., 0]
    w_vals[~inside_mask] = -np.inf

    best_idx = np.argmax(w_vals, axis=0)
    best_idx_exp = best_idx[np.newaxis, :, np.newaxis]
    q_fz = np.take_along_axis(cand_2d, best_idx_exp, axis=0).squeeze(0)
    q_fz = q_fz.reshape(q_spatial.shape)

    # 5. Return in original layout
    q_fz_out = to_quat_spatial(q_fz) if orig_first else q_fz
    return (
        (q_fz_out, best_idx.reshape(q_spatial.shape[:-1]))
        if return_op_map
        else q_fz_out
    )


def format_quaternions(
    q: np.ndarray,
    normalize: bool = True,
    hemisphere: bool = True,
    reduce_fz: bool = False,
    sym=None,
    quat_first: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Canonicalize quaternion arrays to a consistent scalar position and layout.

    FZ reduction is always performed with scalar-first quaternions internally.

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (...,4) or (4,...): MUST BE SCALAR FIRST.
    normalize : bool, default=True
        Normalize each quaternion to unit norm.
    hemisphere : bool, default=True
        Flip quaternions so scalar part (s) >= 0.
    reduce_fz : bool, default=False
        If True, reduce quaternions to the fundamental zone.
    sym : str or orix.quaternion.symmetry.Symmetry or None
        Symmetry to use for FZ reduction if reduce_fz=True.
    quat_first : bool, default=False
        If True, return layout (4,*spatial); if False, return (*spatial,4).
    eps : float, default=1e-12
        Numerical floor for normalization.

    Returns
    -------
    q_out : np.ndarray
        Canonicalized quaternion array with requested layout.
    """
    assert_quaternion_shape(q)

    # 1. Force quaternion-first and scalar-first internally
    q_out = to_quat_spatial(q)

    # 2. Normalization & hemisphere (skip if FZ reduction)
    if not reduce_fz:
        if normalize:
            normalize_quaternions(q_out, axis=0, eps=eps)
        if hemisphere:
            enforce_hemisphere(q_out, scalar_first=True)

    # 3. FZ reduction (always scalar-first)
    if reduce_fz:
        if sym is None:
            raise ValueError("`sym` must be provided when reduce_fz=True")
        if isinstance(sym, str):
            sym = resolve_symmetry(sym)

        q_fz = reduce_to_fz_min_angle(
            q_out,
            sym=sym,
            normalize=normalize,
            hemisphere=hemisphere,
            return_op_map=False,
            eps=eps,
        )

    # 5. Final layout
    if quat_first:
        q_out = to_quat_spatial(q_out)
    else:
        q_out = to_spatial_quat(q_out)

    return q_out.astype(np.float32, copy=False)


def to_torch_quat_spatial(arr: np.ndarray) -> torch.Tensor:
    """
    Convert numpy quaternion array to torch.Tensor (4,*spatial),
    ensuring float32 and C-contiguous layout.
    """
    assert_quaternion_shape(arr)

    if (
        arr.dtype != np.float32
        or not arr.flags["C_CONTIGUOUS"]
        or not arr.flags["WRITEABLE"]
    ):
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(arr)


def to_torch_quat_spatial(arr: np.ndarray) -> torch.Tensor:
    """
    Convert numpy quaternion array to torch.Tensor (4,*spatial),
    ensuring float32 and C-contiguous layout.
    """
    assert_quaternion_shape(arr)

    if (
        arr.dtype != np.float32
        or not arr.flags["C_CONTIGUOUS"]
        or not arr.flags["WRITEABLE"]
    ):
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(arr)


def torch_to_numpy_quat(t: torch.Tensor, channel_last: bool = True) -> np.ndarray:
    """
    Convert a torch quaternion tensor (B,C,H,W) or (B,H,W,C) to numpy (H,W,4) or (B,H,W,4).

    Parameters
    ----------
    t : torch.Tensor
        Tensor containing quaternion components. Shape can be:
        - (B, 4, H, W)  (channel-first, e.g. model output)
        - (B, H, W, 4)  (channel-last)
    channel_last : bool, default=True
        If True, output is channel-last (H,W,4).
        If False, output stays channel-first (4,H,W).

    Returns
    -------
    np.ndarray
        Quaternion array with shape (B,H,W,4) if batch > 1 else (H,W,4).
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")

    # Move to CPU if needed
    t = t.detach().cpu()

    # Ensure channel-last ordering for visualization
    if t.ndim == 4 and not channel_last:
        # (B,4,H,W) -> (B,H,W,4)
        t = t.permute(0, 2, 3, 1)

    # Remove batch dimension if B=1
    np_arr = t.numpy()
    if np_arr.shape[0] == 1:
        np_arr = np_arr[0]

    return np_arr


def get_dummy_quats(resolution_deg: float = 3.0, pg=None):
    """
    Return stable, uniformly sampled quaternions inside the FZ
    of a chosen point group for use in tests.
    """
    if pg is None:
        pg = SYM.O  # cubic 432 by default
    rot = get_sample_fundamental(
        resolution_deg,
        point_group=pg,
        method="cubochoric",  # deterministic for a given resolution
    )
    return rot.data  # (N, 4) array
