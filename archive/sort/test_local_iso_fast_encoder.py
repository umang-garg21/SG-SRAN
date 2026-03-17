import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.local_iso_embedding_test_slow import (
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)
from models.local_iso_embedding import (
    build_fast_local_iso_fcc_encoder,
    build_fast_local_iso_hcp_encoder,
)


def _rand_quats(n: int, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=dtype, device=device)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _timeit(fn, x: torch.Tensor, n_warmup: int, n_iter: int) -> float:
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = fn(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = fn(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        t1 = time.perf_counter()
    return (t1 - t0) / float(n_iter)


def _run_one(group: str, n: int, dtype: torch.dtype, device: str) -> None:
    if group == "O":
        ref = build_local_iso_fcc_embedding(dtype=dtype, device=device)
        fast = build_fast_local_iso_fcc_encoder(dtype=dtype, device=device)
    elif group == "D6":
        ref = build_local_iso_hcp_embedding(dtype=dtype, device=device, d6_convention="z_axis")
        fast = build_fast_local_iso_hcp_encoder(dtype=dtype, device=device, d6_convention="z_axis")
    else:
        raise ValueError(group)

    q = _rand_quats(n, dtype=dtype, device=device)

    with torch.no_grad():
        y_ref = ref.forward_from_quaternions(q)
        y_fast = fast.forward_from_quaternions(q)

    err = _max_abs(y_ref, y_fast)
    print(f"[{group}] irreps_out: {fast.irreps_out}")
    print(f"[{group}] shape ref/fast: {tuple(y_ref.shape)} / {tuple(y_fast.shape)}")
    print(f"[{group}] max |ref-fast|: {err:.3e}")

    t_ref = _timeit(ref.forward_from_quaternions, q, n_warmup=3, n_iter=20)
    t_fast = _timeit(fast.forward_from_quaternions, q, n_warmup=3, n_iter=20)
    speedup = t_ref / max(t_fast, 1e-12)

    print(f"[{group}] ref   avg forward: {1e3 * t_ref:.3f} ms")
    print(f"[{group}] fast  avg forward: {1e3 * t_fast:.3f} ms")
    print(f"[{group}] speedup          : {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reference vs fast local-iso irreps encoders")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"device={args.device}, dtype={dtype}, batch={args.batch}")
    _run_one("O", n=args.batch, dtype=dtype, device=args.device)
    print()
    _run_one("D6", n=args.batch, dtype=dtype, device=args.device)
