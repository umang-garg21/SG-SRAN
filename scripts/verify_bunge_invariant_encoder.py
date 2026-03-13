#!/usr/bin/env python3
"""Numerically verify cubic invariance of Bunge quaternion encoders."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bunge_invariant import (
    build_fcc_syms_inv_wxyz,
    encode_bunge_invariant_features,
    normalize_quaternions,
    quat_mul,
    quaternion_axis_to_last,
    random_unit_quaternions,
)


def _load_input(path: Path, key: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        if key is not None:
            if key not in z:
                raise KeyError(f"Key '{key}' not in {path}. Keys: {list(z.keys())}")
            return z[key]
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"No arrays in {path}")
        return z[keys[0]]
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _sample_quaternions(
    input_path: Path | None,
    key: str | None,
    samples: int,
    seed: int,
) -> np.ndarray:
    if input_path is None:
        return random_unit_quaternions(samples, seed=seed)

    arr = _load_input(input_path, key)
    q = quaternion_axis_to_last(arr).reshape(-1, 4)
    q = normalize_quaternions(q)
    if q.shape[0] <= samples:
        return q

    rng = np.random.default_rng(seed)
    idx = rng.choice(q.shape[0], size=samples, replace=False)
    return q[idx]


def _max_mean_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a - b)
    return float(np.max(d)), float(np.mean(d))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=None, help="Optional .npy/.npz quaternions")
    p.add_argument("--key", type=str, default=None, help="Key for .npz input")
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=64.0)
    p.add_argument(
        "--method",
        type=str,
        default="soft_orbit",
        choices=["fz_logmap", "soft_orbit", "hybrid"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q = _sample_quaternions(args.input, args.key, args.samples, args.seed)  # (N,4)

    base = encode_bunge_invariant_features(q, method=args.method, beta=args.beta)["features"]
    base = base.reshape(q.shape[0], -1)

    syms_inv = build_fcc_syms_inv_wxyz()
    sym_max = []
    sym_mean = []

    # Bunge equivalence: q ~ s^{-1} ⊗ q (left action by inverse crystal op)
    for g in syms_inv:
        q_g = quat_mul(np.broadcast_to(g, q.shape), q)
        q_g = normalize_quaternions(q_g)
        f_g = encode_bunge_invariant_features(q_g, method=args.method, beta=args.beta)[
            "features"
        ].reshape(q.shape[0], -1)
        mx, mn = _max_mean_diff(base, f_g)
        sym_max.append(mx)
        sym_mean.append(mn)

    # Sign-equivalence check: q and -q same rotation
    q_neg = -q
    f_neg = encode_bunge_invariant_features(q_neg, method=args.method, beta=args.beta)[
        "features"
    ].reshape(q.shape[0], -1)
    sign_max, sign_mean = _max_mean_diff(base, f_neg)

    print("Bunge Invariant Encoder Verification")
    print("=" * 64)
    print(f"method: {args.method}")
    print(f"beta: {args.beta}")
    print(f"num_quaternions: {q.shape[0]}")
    if args.input is not None:
        print(f"input: {args.input}")
    else:
        print("input: random unit quaternions")
    print("-" * 64)
    print(f"symmetry action max |Δfeature|   : {max(sym_max):.6e}")
    print(f"symmetry action mean |Δfeature|  : {np.mean(sym_mean):.6e}")
    print(f"sign flip max |Δfeature|         : {sign_max:.6e}")
    print(f"sign flip mean |Δfeature|        : {sign_mean:.6e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
