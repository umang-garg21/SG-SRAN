#!/usr/bin/env python3
"""Encode Bunge-convention quaternions into cubic-invariant features.

Input assumptions
-----------------
- Quaternions are scalar-first [w, x, y, z].
- Convention is Bunge/passive.
- Quaternion axis can be first, last, or any single axis of size 4.

Output
------
Saves a compressed .npz containing:
- features          : invariant embedding (..., F)
- optional q_fz     : exact FZ canonical quaternions (..., 4)
- optional q_soft   : smooth orbit-canonical quaternions (..., 4)
- optional sym_index: selected symmetry index for FZ mode
- meta_json         : JSON metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bunge_invariant import encode_bunge_invariant_features


def _load_quaternion_array(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), "npy:data"
    if suffix == ".npz":
        z = np.load(path)
        if key is not None:
            if key not in z:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(z.keys())}")
            return z[key], f"npz:{key}"
        keys = list(z.keys())
        if len(keys) == 0:
            raise ValueError(f"No arrays found in {path}")
        return z[keys[0]], f"npz:{keys[0]}"
    raise ValueError(f"Unsupported input extension '{suffix}'. Use .npy or .npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Input .npy or .npz file")
    p.add_argument("--output", required=True, type=Path, help="Output .npz file")
    p.add_argument("--key", type=str, default=None, help="Array key if --input is .npz")
    p.add_argument(
        "--method",
        type=str,
        default="soft_orbit",
        choices=["fz_logmap", "soft_orbit", "hybrid"],
        help="Invariant encoding method",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=64.0,
        help="Soft-orbit temperature factor (used in soft_orbit/hybrid)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arr, source = _load_quaternion_array(args.input, args.key)

    result = encode_bunge_invariant_features(
        arr,
        method=args.method,
        beta=float(args.beta),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"features": result["features"]}
    for key in ("q_fz", "q_soft", "sym_index", "orbit_entropy"):
        if key in result:
            payload[key] = result[key]

    meta = {
        "input_path": str(args.input),
        "input_source": source,
        "output_path": str(args.output),
        "method": result["method"],
        "beta": float(result["beta"]),
        "input_shape": list(arr.shape),
        "feature_shape": list(result["features"].shape),
        "quaternion_convention": "Bunge passive, scalar-first [w,x,y,z]",
    }
    payload["meta_json"] = np.asarray(json.dumps(meta, indent=2))

    np.savez_compressed(args.output, **payload)

    print(f"saved: {args.output}")
    print(f"method: {result['method']}")
    print(f"input_shape: {tuple(arr.shape)}")
    print(f"feature_shape: {tuple(result['features'].shape)}")


if __name__ == "__main__":
    main()
