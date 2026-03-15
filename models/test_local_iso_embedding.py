"""Tests for LocalIsoCrystalEmbedding.

This file separates:
1) ML/runtime tests in the model's native dtype/device
2) geometry/isometry tests in a rebuilt float64 model

Usage:
    python test_local_iso_embedding_strict.py
"""

import math
from dataclasses import dataclass

import torch
from e3nn import o3

# ---------------------------------------------------------------------
# Model import
# ---------------------------------------------------------------------
from local_iso_embedding import (
    LocalIsoCrystalEmbedding,
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def random_quaternions(
    n: int,
    *,
    dtype: torch.dtype,
    device,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(n, 4, dtype=dtype, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[:, :1] < 0.0, -q, q)
    return q


def random_rotation_matrices(
    n: int,
    *,
    dtype: torch.dtype,
    device,
    seed: int = 0,
) -> torch.Tensor:
    q = random_quaternions(n, dtype=dtype, device=device, seed=seed)
    return o3.quaternion_to_matrix(q)


@dataclass
class TestResult:
    name: str
    passed: bool
    value: float
    tol: float


def print_result(res: TestResult) -> None:
    status = "PASS" if res.passed else "FAIL"
    print(f"[{status}] {res.name:34s} value={res.value:.6e}  tol={res.tol:.6e}")


# ---------------------------------------------------------------------
# Runtime / ML-path tests
# ---------------------------------------------------------------------


@torch.no_grad()
def check_right_invariance_runtime(
    model: LocalIsoCrystalEmbedding,
    *,
    n_trials: int = 32,
    tol: float = 1e-6,
    use_raw: bool = False,
    seed: int = 0,
) -> TestResult:
    """
    Test E(R g) = E(R) on the model exactly as used in training/inference.
    """
    embed_fn = model.forward_raw if use_raw else model.forward_irreps
    device = model.group_mats.device
    dtype = model.group_mats.dtype

    R_batch = random_rotation_matrices(
        n_trials,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    ER = embed_fn(R_batch)
    errs = []
    for g in model.group_mats:
        ERg = embed_fn(R_batch @ g)
        errs.append((ERg - ER).abs().amax())

    err = float(torch.stack(errs).amax().item())
    return TestResult(
        name=f"right_invariance_runtime({'raw' if use_raw else 'irreps'})",
        passed=(err < tol),
        value=err,
        tol=tol,
    )


# ---------------------------------------------------------------------
# Strict float64 geometric tests
# ---------------------------------------------------------------------


def rebuild_float64_model(
    src_model: LocalIsoCrystalEmbedding,
) -> LocalIsoCrystalEmbedding:
    """
    Rebuild the same model in float64 on the same device for diagnostics.
    """
    device = src_model.group_mats.device

    if src_model.group_name == "O":
        model64 = build_local_iso_fcc_embedding(
            dtype=torch.float64,
            device=device,
        )
    elif src_model.group_name == "D6":
        model64 = build_local_iso_hcp_embedding(
            d6_convention=src_model.d6_convention,
            dtype=torch.float64,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported group_name={src_model.group_name}")

    return model64.eval()


@torch.no_grad()
def gram_at_identity_strict(
    src_model: LocalIsoCrystalEmbedding,
    *,
    eps: float = 1e-7,
    use_raw: bool = False,
) -> torch.Tensor:
    """
    Local-isometry Gram matrix computed using a separately rebuilt
    float64 copy of the model.
    """
    model64 = rebuild_float64_model(src_model)
    embed_fn = model64.forward_raw if use_raw else model64.forward_irreps
    device = model64.group_mats.device
    dtype = torch.float64

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
    basis = [s1, s2, s3]

    tangents = []
    for S in basis:
        Rp = torch.matrix_exp(eps * S)
        Rm = torch.matrix_exp(-eps * S)
        Ep = embed_fn(Rp.unsqueeze(0))[0]
        Em = embed_fn(Rm.unsqueeze(0))[0]
        tangents.append((Ep - Em) / (2.0 * eps))

    G = torch.stack(
        [torch.stack([(vi * vj).sum() for vj in tangents], dim=0) for vi in tangents],
        dim=0,
    )
    return G


@torch.no_grad()
def check_local_isometry_strict(
    model: LocalIsoCrystalEmbedding,
    *,
    tol: float = 5e-7,
    eps: float = 1e-7,
    use_raw: bool = False,
) -> tuple[TestResult, torch.Tensor]:
    """
    Check ||G - I||_max using a float64 rebuilt model.
    """
    G = gram_at_identity_strict(model, eps=eps, use_raw=use_raw)
    I = torch.eye(3, dtype=G.dtype, device=G.device)
    err = float((G - I).abs().amax().item())

    return (
        TestResult(
            name=f"local_isometry_strict({'raw' if use_raw else 'irreps'})",
            passed=(err < tol),
            value=err,
            tol=tol,
        ),
        G,
    )


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run_one_model(
    label: str,
    model: LocalIsoCrystalEmbedding,
    *,
    inv_tol: float,
    gram_tol: float,
    eps: float,
    seed: int,
    n_trials: int,
) -> bool:
    print("=" * 90)
    print(label)
    print(f"runtime device = {model.group_mats.device}")
    print(f"runtime dtype  = {model.group_mats.dtype}")
    print(f"group_name     = {model.group_name}")
    if model.group_name == "D6":
        print(f"d6_convention  = {model.d6_convention}")
    print(f"irreps_out     = {model.irreps_out}")
    print("-" * 90)

    ok = True

    # Runtime invariance
    r1 = check_right_invariance_runtime(
        model,
        n_trials=n_trials,
        tol=inv_tol,
        use_raw=False,
        seed=seed,
    )
    print_result(r1)
    ok &= r1.passed

    r2 = check_right_invariance_runtime(
        model,
        n_trials=n_trials,
        tol=inv_tol,
        use_raw=True,
        seed=seed,
    )
    print_result(r2)
    ok &= r2.passed

    # float64 Gram tests
    g1_res, G1 = check_local_isometry_strict(
        model,
        tol=gram_tol,
        eps=eps,
        use_raw=False,
    )
    print_result(g1_res)
    print("Gram (irreps):")
    print(G1)
    ok &= g1_res.passed

    g2_res, G2 = check_local_isometry_strict(
        model,
        tol=gram_tol,
        eps=eps,
        use_raw=True,
    )
    print_result(g2_res)
    print("Gram (raw):")
    print(G2)
    ok &= g2_res.passed

    print()
    return ok


def main():
    torch.set_printoptions(precision=10, sci_mode=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # Build training-style models in float32
    # -------------------------------------------------------------
    models = [
        (
            "FCC / cubic O",
            build_local_iso_fcc_embedding(dtype=torch.float32, device=device).eval(),
        ),
        (
            "HCP / D6 (z_axis)",
            build_local_iso_hcp_embedding(
                d6_convention="z_axis",
                dtype=torch.float32,
                device=device,
            ).eval(),
        ),
        (
            "HCP / D6 (paper)",
            build_local_iso_hcp_embedding(
                d6_convention="paper",
                dtype=torch.float32,
                device=device,
            ).eval(),
        ),
    ]

    # Runtime float32 invariance tolerance
    inv_tol = 1e-5

    # float64 local-isometry tolerance
    gram_tol = 5e-7
    eps = 1e-7

    seed = 1234
    n_trials = 32

    all_ok = True
    for label, model in models:
        ok = run_one_model(
            label,
            model,
            inv_tol=inv_tol,
            gram_tol=gram_tol,
            eps=eps,
            seed=seed,
            n_trials=n_trials,
        )
        all_ok &= ok

    print("=" * 90)
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
