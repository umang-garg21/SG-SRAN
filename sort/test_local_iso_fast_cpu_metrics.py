import os
import sys
import time

import torch
from e3nn import o3

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
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(q[:, :1] < 0.0, -q, q)


@torch.no_grad()
def gram_at_identity_fast(encoder, eps: float = 1e-7) -> torch.Tensor:
    device = encoder.group_mats.device
    dtype = encoder.group_mats.dtype

    s1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    s2 = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    s3 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )

    tangents = []
    for s in (s1, s2, s3):
        rp = torch.matrix_exp(eps * s)
        rm = torch.matrix_exp(-eps * s)
        ep = encoder.forward_from_matrices(rp.unsqueeze(0))[0]
        em = encoder.forward_from_matrices(rm.unsqueeze(0))[0]
        tangents.append((ep - em) / (2.0 * eps))

    return torch.stack(
        [torch.stack([(vi * vj).sum() for vj in tangents], dim=0) for vi in tangents],
        dim=0,
    )


@torch.no_grad()
def right_invariance_error_fast(
    encoder,
    *,
    n_samples: int = 24,
    seed: int = 0,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    device = encoder.group_mats.device
    dtype = encoder.group_mats.dtype

    q = _rand_quats(n_samples, dtype=dtype, device=device)
    r = o3.quaternion_to_matrix(q)
    e = encoder.forward_from_matrices(r)

    errs = []
    for g in encoder.group_mats:
        erg = encoder.forward_from_matrices(r @ g)
        errs.append((erg - e).abs().max())

    err = torch.stack(errs)
    return float(err.mean().item()), float(err.max().item())


@torch.no_grad()
def benchmark_forward(fn, x: torch.Tensor, *, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        _ = fn(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(x)
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


@torch.no_grad()
def run_group(name: str, fast_enc, ref_enc, *, batch: int = 128) -> None:
    device = str(fast_enc.group_mats.device)
    dtype = fast_enc.group_mats.dtype

    print(f"\n=== {name} (CPU) ===", flush=True)
    print("irreps_out:", fast_enc.irreps_out, flush=True)

    # 1) Gram metric at identity
    g_fast = gram_at_identity_fast(fast_enc, eps=1e-7)
    eye = torch.eye(3, dtype=dtype, device=fast_enc.group_mats.device)
    gram_err = float((g_fast - eye).abs().max().item())
    print("G_fast =")
    print(g_fast)
    print(f"||G_fast - I||_max: {gram_err:.3e}", flush=True)

    # 2) Right-invariance error
    inv_mean, inv_max = right_invariance_error_fast(fast_enc, n_samples=24, seed=0)
    print(f"right-invariance |E(Rg)-E(R)| mean={inv_mean:.3e}, max={inv_max:.3e}", flush=True)

    # 3) Speed (fast vs reference)
    q = _rand_quats(batch, dtype=dtype, device=device)
    t_fast = benchmark_forward(fast_enc, q, warmup=3, iters=20)
    t_ref = benchmark_forward(ref_enc.forward_from_quaternions, q, warmup=3, iters=20)
    print(f"forward time fast : {1e3 * t_fast:.3f} ms / batch {batch}", flush=True)
    print(f"forward time ref  : {1e3 * t_ref:.3f} ms / batch {batch}", flush=True)
    print(f"speedup (ref/fast): {t_ref / max(t_fast, 1e-12):.2f}x", flush=True)


if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)

    # Explicitly CPU-only as requested.
    device = "cpu"
    dtype = torch.float64

    print(f"device={device}, dtype={dtype} (CUDA disabled intentionally)", flush=True)

    fcc_fast = build_fast_local_iso_fcc_encoder(device=device, dtype=dtype)
    hcp_fast = build_fast_local_iso_hcp_encoder(device=device, dtype=dtype, d6_convention="z_axis")

    fcc_ref = build_local_iso_fcc_embedding(device=device, dtype=dtype)
    hcp_ref = build_local_iso_hcp_embedding(device=device, dtype=dtype, d6_convention="z_axis")

    run_group("FCC/O", fcc_fast, fcc_ref, batch=128)
    run_group("HCP/D6", hcp_fast, hcp_ref, batch=128)
