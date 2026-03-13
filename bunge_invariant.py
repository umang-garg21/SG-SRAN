"""Invariant feature encoders for Bunge-convention quaternions.

This module assumes scalar-first quaternions in [w, x, y, z] ordering and
passive (Bunge) convention. Crystal symmetry is applied as:

    q_equiv = s^{-1} ⊗ q

for proper cubic symmetry operators s.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Avoid OpenMP shared-memory failures in restricted environments.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def build_fcc_syms_wxyz() -> np.ndarray:
    """Return 24 proper cubic symmetry operators [w,x,y,z]."""
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [0, inv_sqrt_2, inv_sqrt_2, 0],
            [0, inv_sqrt_2, 0, inv_sqrt_2],
            [0, 0, inv_sqrt_2, inv_sqrt_2],
            [0, inv_sqrt_2, -inv_sqrt_2, 0],
            [0, 0, inv_sqrt_2, -inv_sqrt_2],
            [0, inv_sqrt_2, 0, -inv_sqrt_2],
            [half, half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, -half],
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True, dtype=np.float64)
    out[..., 1:] *= -1.0
    return out


def build_fcc_syms_inv_wxyz() -> np.ndarray:
    """Return s^{-1} for the 24 proper cubic operators."""
    return quat_conjugate(build_fcc_syms_wxyz())


def normalize_quaternions(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return q / n


def enforce_hemisphere_w_positive(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    sgn = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sgn


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for arrays (...,4), broadcast-compatible."""
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (...,4) [w,x,y,z] to rotation matrix (...,3,3)."""
    q = normalize_quaternions(np.asarray(q, dtype=np.float64))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[..., 0, 1] = 2.0 * (x * y - z * w)
    r[..., 0, 2] = 2.0 * (x * z + y * w)
    r[..., 1, 0] = 2.0 * (x * y + z * w)
    r[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[..., 1, 2] = 2.0 * (y * z - x * w)
    r[..., 2, 0] = 2.0 * (x * z - y * w)
    r[..., 2, 1] = 2.0 * (y * z + x * w)
    r[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (...,3,3) to quaternion (...,4) [w,x,y,z]."""
    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape[:-2] + (4,), dtype=np.float64)
    it = np.ndindex(r.shape[:-2])
    for idx in it:
        m = r[idx]
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        out[idx] = np.array([w, x, y, z], dtype=np.float64)

    return normalize_quaternions(out)


def project_to_so3(r: np.ndarray) -> np.ndarray:
    """Project matrix (...,3,3) to nearest proper rotation via SVD."""
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r)
    it = np.ndindex(r.shape[:-2])
    for idx in it:
        u, _, vt = np.linalg.svd(r[idx], full_matrices=False)
        m = u @ vt
        if np.linalg.det(m) < 0.0:
            u[:, -1] *= -1.0
            m = u @ vt
        out[idx] = m
    return out


def quaternion_axis_to_last(arr: np.ndarray) -> np.ndarray:
    """Move the single quaternion axis (size 4) to the last position."""
    arr = np.asarray(arr)
    axes = [i for i, s in enumerate(arr.shape) if s == 4]
    if len(axes) != 1:
        raise ValueError(
            f"Expected exactly one quaternion axis of size 4, got shape {arr.shape}"
        )
    q_axis = axes[0]
    if q_axis == arr.ndim - 1:
        return arr
    return np.moveaxis(arr, q_axis, -1)


def bunge_left_orbit(q_flat: np.ndarray, syms_inv: np.ndarray | None = None) -> np.ndarray:
    """Generate left crystal orbit for Bunge quaternions: s^{-1} ⊗ q."""
    q_flat = normalize_quaternions(np.asarray(q_flat, dtype=np.float64))
    if q_flat.ndim != 2 or q_flat.shape[1] != 4:
        raise ValueError(f"Expected (N,4) quaternions, got {q_flat.shape}")

    if syms_inv is None:
        syms_inv = build_fcc_syms_inv_wxyz()
    syms_inv = normalize_quaternions(np.asarray(syms_inv, dtype=np.float64))
    if syms_inv.ndim != 2 or syms_inv.shape[1] != 4:
        raise ValueError(f"Expected (G,4) symmetry operators, got {syms_inv.shape}")

    orbit = quat_mul(syms_inv[None, :, :], q_flat[:, None, :])  # (N,G,4)
    return normalize_quaternions(orbit)


def reduce_to_fz_bunge(
    q_flat: np.ndarray, syms_inv: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Bunge FZ canonicalization by max |w| over left orbit."""
    orbit = bunge_left_orbit(q_flat, syms_inv=syms_inv)  # (N,G,4)
    best = np.argmax(np.abs(orbit[..., 0]), axis=1)
    q_fz = orbit[np.arange(orbit.shape[0]), best]
    q_fz = enforce_hemisphere_w_positive(normalize_quaternions(q_fz))
    return q_fz, best


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shift)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def soft_orbit_canonical_bunge(
    q_flat: np.ndarray,
    beta: float = 64.0,
    syms_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth orbit-based canonicalization for Bunge quaternions.

    Weights each orbit element by softmax(beta * |w|), then averages.
    """
    orbit = bunge_left_orbit(q_flat, syms_inv=syms_inv)  # (N,G,4)
    scores = float(beta) * np.abs(orbit[..., 0])
    weights = _stable_softmax(scores, axis=1)  # (N,G)

    # Average in SO(3) space (sign-invariant), then project back to unit quaternions.
    rot_orbit = quat_to_matrix(orbit)  # (N,G,3,3)
    rot_avg = np.sum(weights[..., None, None] * rot_orbit, axis=1)  # (N,3,3)
    rot_proj = project_to_so3(rot_avg)
    q_soft = matrix_to_quat(rot_proj)
    q_soft = enforce_hemisphere_w_positive(q_soft)
    return q_soft, weights, orbit


def quaternion_log_map(q_flat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log map of unit quaternions to R^3; uses hemisphere w>=0."""
    q = enforce_hemisphere_w_positive(normalize_quaternions(q_flat, eps=eps))
    w = np.clip(q[:, 0:1], -1.0, 1.0)
    v = q[:, 1:]
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    theta = 2.0 * np.arctan2(v_norm, np.clip(w, eps, None))
    scale = np.full_like(v_norm, 2.0)
    np.divide(theta, np.clip(v_norm, eps, None), out=scale, where=v_norm > eps)
    return v * scale


def soft_orbit_moment_features(orbit: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Second-order orbit descriptors (invariant summary statistics)."""
    abs_w = np.abs(orbit[..., 0])  # (N,G)
    mean_abs_w = np.sum(weights * abs_w, axis=1, keepdims=True)
    var_abs_w = np.sum(weights * (abs_w - mean_abs_w) ** 2, axis=1, keepdims=True)
    std_abs_w = np.sqrt(np.clip(var_abs_w, 0.0, None))

    g = orbit.shape[1]
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=1, keepdims=True)
    entropy = entropy / math.log(float(g))

    v = orbit[..., 1:]  # (N,G,3)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    m_xx = np.sum(weights * vx * vx, axis=1, keepdims=True)
    m_yy = np.sum(weights * vy * vy, axis=1, keepdims=True)
    m_zz = np.sum(weights * vz * vz, axis=1, keepdims=True)
    m_xy = np.sum(weights * vx * vy, axis=1, keepdims=True)
    m_xz = np.sum(weights * vx * vz, axis=1, keepdims=True)
    m_yz = np.sum(weights * vy * vz, axis=1, keepdims=True)

    return np.concatenate(
        [mean_abs_w, std_abs_w, entropy, m_xx, m_yy, m_zz, m_xy, m_xz, m_yz],
        axis=1,
    )


def encode_bunge_invariant_features(
    q: np.ndarray,
    method: str = "soft_orbit",
    beta: float = 64.0,
    syms_inv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Encode quaternion field to symmetry-invariant features.

    Methods
    -------
    - fz_logmap:
        Exact cubic invariance via FZ canonicalization + log map (3 dims).
    - soft_orbit:
        Smooth invariant via orbit soft-pooling + moment descriptors (12 dims).
    - hybrid:
        Concatenate exact and smooth embeddings (15 dims).
    """
    q_last = quaternion_axis_to_last(q)
    spatial_shape = q_last.shape[:-1]
    q_flat = normalize_quaternions(q_last.reshape(-1, 4))

    if syms_inv is None:
        syms_inv = build_fcc_syms_inv_wxyz()
    syms_inv = normalize_quaternions(syms_inv)

    method_key = str(method).strip().lower()
    if method_key == "fz_logmap":
        q_fz, op_idx = reduce_to_fz_bunge(q_flat, syms_inv=syms_inv)
        feat = quaternion_log_map(q_fz)
        out = {
            "features": feat,
            "q_fz": q_fz,
            "sym_index": op_idx.astype(np.int64, copy=False),
        }
    elif method_key == "soft_orbit":
        q_soft, weights, orbit = soft_orbit_canonical_bunge(
            q_flat, beta=beta, syms_inv=syms_inv
        )
        mom = soft_orbit_moment_features(orbit, weights)
        feat = np.concatenate(
            [quaternion_log_map(q_soft), mom],
            axis=1,
        )
        out = {
            "features": feat,
            "q_soft": q_soft,
            "orbit_entropy": mom[:, 2],
        }
    elif method_key == "hybrid":
        q_fz, op_idx = reduce_to_fz_bunge(q_flat, syms_inv=syms_inv)
        q_soft, weights, orbit = soft_orbit_canonical_bunge(
            q_flat, beta=beta, syms_inv=syms_inv
        )
        feat = np.concatenate(
            [
                quaternion_log_map(q_fz),
                quaternion_log_map(q_soft),
                soft_orbit_moment_features(orbit, weights),
            ],
            axis=1,
        )
        out = {
            "features": feat,
            "q_fz": q_fz,
            "q_soft": q_soft,
            "sym_index": op_idx.astype(np.int64, copy=False),
        }
    else:
        raise ValueError(f"Unknown method '{method}'.")

    feature_dim = out["features"].shape[1]
    out["features"] = out["features"].reshape(*spatial_shape, feature_dim).astype(
        np.float32, copy=False
    )

    if "q_fz" in out:
        out["q_fz"] = out["q_fz"].reshape(*spatial_shape, 4).astype(np.float32, copy=False)
    if "q_soft" in out:
        out["q_soft"] = out["q_soft"].reshape(*spatial_shape, 4).astype(np.float32, copy=False)
    if "sym_index" in out:
        out["sym_index"] = out["sym_index"].reshape(*spatial_shape).astype(
            np.int64, copy=False
        )
    if "orbit_entropy" in out:
        out["orbit_entropy"] = out["orbit_entropy"].reshape(*spatial_shape).astype(
            np.float32, copy=False
        )

    out["method"] = method_key
    out["beta"] = float(beta)
    return out


def random_unit_quaternions(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q = rng.normal(size=(int(n), 4))
    return normalize_quaternions(q)
