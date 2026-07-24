"""
test_so3_equivariance_spatial_conv.py
======================================
Validates SO(3) equivariance of EquivariantSpatialConv.

Mathematical property tested
----------------------------
For any R ∈ SO(3) with Wigner-D matrices D_in = I_in.D(R), D_out = I_out.D(R):

    conv( x @ D_in.T, shape )  ≈  conv( x, shape ) @ D_out.T

Tested configurations
---------------------
  - FCC irreps  (irreps_a1 = 1x4e,         irreps_full = 1x2e+1x4e)
  - HCP irreps  (irreps_a1 = 2x2e+1x4e+1x6e)
  - a1 → a1 (identity residual), kernel 3 and 9
  - with_residual=True and False

Run:  python eqv_inv_tests/test_so3_equivariance_spatial_conv.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from e3nn import o3
from e3nn.o3 import Irreps

from _helpers import (
    rotate_features, rel_error, report, section, summary,
    rand_quaternions,
)
from models.SR_double_conv_SRattn_a1 import EquivariantSpatialConv, LocalIsoCrystalEncoder

# ── config ────────────────────────────────────────────────────────────────────
H, W       = 12, 10
N_TRIALS   = 16
TOL        = 1e-4
DEVICE     = torch.device("cpu")
SEED       = 42


def _run_so3_equivariance(
    conv: EquivariantSpatialConv,
    x: torch.Tensor,
    irreps_in: Irreps,
    irreps_out: Irreps,
    n_trials: int,
    seed: int,
) -> tuple[float, float]:
    """
    Returns (mean_rel_err, max_rel_err) over n_trials random SO(3) rotations.
    For each rotation R:
        y_rotate_then_conv  = conv( x @ D_in.T )
        y_conv_then_rotate  = conv( x ) @ D_out.T
        error = rel_error(y_rotate_then_conv, y_conv_then_rotate)
    """
    img_shape = (H, W)
    with torch.no_grad():
        y_ref = conv(x, img_shape)

    errors = []
    torch.manual_seed(seed)
    for _ in range(n_trials):
        R      = o3.rand_matrix(dtype=torch.float32).to(DEVICE)
        D_in   = irreps_in.D_from_matrix(R).to(DEVICE)
        D_out  = irreps_out.D_from_matrix(R).to(DEVICE)

        x_rot = rotate_features(x, D_in)          # rotate input features
        with torch.no_grad():
            y_in_rot  = conv(x_rot, img_shape)     # conv after rotating input
            y_out_rot = rotate_features(y_ref, D_out)  # rotate conv output

        errors.append(rel_error(y_in_rot, y_out_rot))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max())


def _make_conv(irreps_in, irreps_out, kernel_size, use_residual, seed):
    torch.manual_seed(seed)
    return EquivariantSpatialConv(
        kernel_size=kernel_size,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        use_residual=use_residual,
    ).to(DEVICE).eval()


def _rand_features(irreps: Irreps, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, H * W, irreps.dim, dtype=torch.float32, device=DEVICE)


def main() -> None:
    results: list[bool] = []

    for crystal in ("fcc", "hcp"):
        enc = LocalIsoCrystalEncoder(crystal=crystal, dtype=torch.float32, device=DEVICE).eval()
        irr_a1   = enc.irreps_a1
        irr_full = enc.irreps_full

        section(f"Crystal={crystal.upper()}  irreps_a1={irr_a1}  irreps_full={irr_full}")

        cases = [
            # (label,               irreps_in, irreps_out, kernel, residual, seed)
            ("a1→a1  k=3  no-res",  irr_a1,   irr_a1,    3, False, 10),
            ("a1→a1  k=3  residual",irr_a1,   irr_a1,    3, True,  11),
            ("a1→a1  k=9  no-res",  irr_a1,   irr_a1,    9, False, 12),
            ("a1→a1  k=9  residual",irr_a1,   irr_a1,    9, True,  13),
        ]
        # add full→full only if irreps differ
        if irr_a1 != irr_full:
            cases += [
                ("full→full  k=3  no-res",  irr_full, irr_full, 3, False, 20),
                ("full→full  k=3  residual",irr_full, irr_full, 3, True,  21),
                ("a1→full    k=3  no-res",  irr_a1,   irr_full, 3, False, 30),
            ]

        for label, irr_in, irr_out, k, res, seed in cases:
            conv = _make_conv(irr_in, irr_out, k, res, seed)
            x    = _rand_features(irr_in, seed + 100)
            mean_err, max_err = _run_so3_equivariance(
                conv, x, irr_in, irr_out, N_TRIALS, SEED
            )
            ok = report(
                f"{crystal.upper()} {label}",
                max_err, TOL,
                extra=f"mean={mean_err:.2e}",
            )
            results.append(ok)

    summary(results)


if __name__ == "__main__":
    main()
