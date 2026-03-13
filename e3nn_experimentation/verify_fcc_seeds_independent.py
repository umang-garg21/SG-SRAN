#!/usr/bin/env python3
"""
Independent FCC seed verification for L=4 and L=6.

This script does NOT use e3nn. It builds the cubic group representation
numerically in a real tesseral-harmonic basis and applies a Reynolds projector
to recover the invariant seed vectors. It then compares against:
  1) hard-coded Python constants used in models/autoencoder.py
  2) exact closed forms:
       L=4:  s4[4] = sqrt(7/12), s4[8] = sqrt(5/12)
       L=6:  s6[6] = 1/sqrt(8),  s6[10] = -sqrt(7/8)
"""

from __future__ import annotations

import os
from math import sqrt

# Avoid Intel OpenMP SHM issues in restricted environments.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import special


def _complex_sph_harm(m: int, l: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
    """Return complex Y_l^m using SciPy, handling API changes."""
    if hasattr(special, "sph_harm_y"):
        # New API: sph_harm_y(n=l, m, theta=polar, phi=azimuth)
        return special.sph_harm_y(l, m, polar, azimuth)
    # Old API (deprecated in newer SciPy): sph_harm(m, n=l, theta=azimuth, phi=polar)
    return special.sph_harm(m, l, azimuth, polar)


def _cubic_quats_wxyz() -> np.ndarray:
    inv_sqrt_2 = 1.0 / sqrt(2.0)
    half = 0.5
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [0, inv_sqrt_2, inv_sqrt_2, 0],
            [0, inv_sqrt_2, -inv_sqrt_2, 0],
            [0, inv_sqrt_2, 0, inv_sqrt_2],
            [0, inv_sqrt_2, 0, -inv_sqrt_2],
            [0, 0, inv_sqrt_2, inv_sqrt_2],
            [0, 0, inv_sqrt_2, -inv_sqrt_2],
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


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _real_sh_basis(l: int, xyz: np.ndarray) -> np.ndarray:
    """Real tesseral basis ordered by m=-l..l."""
    x, y, z = xyz.T
    r = np.sqrt(x * x + y * y + z * z)
    az = np.mod(np.arctan2(y, x), 2 * np.pi)
    pol = np.arccos(np.clip(z / r, -1.0, 1.0))

    cols: list[np.ndarray] = []
    for m in range(-l, l + 1):
        if m > 0:
            Yp = _complex_sph_harm(m, l, az, pol)
            Yn = _complex_sph_harm(-m, l, az, pol)
            cols.append(np.real(((-1) ** m * Yp + Yn) / np.sqrt(2.0)))
        elif m < 0:
            mp = -m
            Yp = _complex_sph_harm(mp, l, az, pol)
            Yn = _complex_sph_harm(-mp, l, az, pol)
            cols.append(np.real((((-1) ** mp * Yp - Yn) / (1j * np.sqrt(2.0)))))
        else:
            cols.append(np.real(_complex_sph_harm(0, l, az, pol)))
    return np.stack(cols, axis=1)


def _representation_matrix_numeric(l: int, rot: np.ndarray, sphere_pts: np.ndarray) -> np.ndarray:
    # Solve B * D = B_rot in least-squares sense.
    B = _real_sh_basis(l, sphere_pts)
    B_rot = _real_sh_basis(l, sphere_pts @ rot.T)
    D, *_ = np.linalg.lstsq(B, B_rot, rcond=None)
    return D


def _compute_seed(l: int, rotations: np.ndarray, sphere_pts: np.ndarray) -> tuple[np.ndarray, float]:
    d = 2 * l + 1
    Ds = np.stack(
        [_representation_matrix_numeric(l, R, sphere_pts) for R in rotations],
        axis=0,
    )
    P = Ds.mean(axis=0)
    P = 0.5 * (P + P.T)

    evals, evecs = np.linalg.eigh(P)
    seed = evecs[:, np.argmax(evals)]
    if seed[l] < 0:
        seed = -seed
    seed[np.abs(seed) < 1e-10] = 0.0
    seed = seed / np.linalg.norm(seed)
    return seed, float(np.max(evals))


def _compare(name: str, s: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    x = s.copy()
    if np.dot(x, ref) < 0:
        x = -x
    l2 = float(np.linalg.norm(x - ref))
    mx = float(np.max(np.abs(x - ref)))
    print(f"  vs {name:<10} | L2 = {l2:.12g} | max abs = {mx:.12g}")
    return l2, mx


def main() -> None:
    q = _cubic_quats_wxyz()
    R = _quat_to_matrix(q)

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(1400, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    s4, e4 = _compute_seed(4, R, pts)
    s6, e6 = _compute_seed(6, R, pts)

    py4 = np.zeros(9)
    py4[4] = 0.7638
    py4[8] = 0.6455
    py4 /= np.linalg.norm(py4)

    py6 = np.zeros(13)
    py6[6] = 0.3536
    py6[10] = -0.9354
    py6 /= np.linalg.norm(py6)

    ex4 = np.zeros(9)
    ex4[4] = np.sqrt(7.0 / 12.0)
    ex4[8] = np.sqrt(5.0 / 12.0)
    ex4 /= np.linalg.norm(ex4)

    ex6 = np.zeros(13)
    ex6[6] = 1.0 / np.sqrt(8.0)
    ex6[10] = -np.sqrt(7.0 / 8.0)
    ex6 /= np.linalg.norm(ex6)

    print("Independent FCC seed verification (no e3nn)")
    print("=" * 72)
    print("L=4")
    print(f"  max eigenvalue  = {e4:.16f}")
    print(f"  nonzero indices = {np.where(np.abs(s4) > 1e-8)[0].tolist()}")
    print(f"  seed values     = {[float(v) for v in s4[np.abs(s4) > 1e-8]]}")
    _compare("python", s4, py4)
    _compare("exact", s4, ex4)
    print()

    print("L=6")
    print(f"  max eigenvalue  = {e6:.16f}")
    print(f"  nonzero indices = {np.where(np.abs(s6) > 1e-8)[0].tolist()}")
    print(f"  seed values     = {[float(v) for v in s6[np.abs(s6) > 1e-8]]}")
    _compare("python", s6, py6)
    _compare("exact", s6, ex6)
    print("=" * 72)


if __name__ == "__main__":
    main()

