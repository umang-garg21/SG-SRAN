"""
test_so3_equivariance_transpose_conv.py
========================================
Validates SO(3) equivariance of EquivariantTransposeConv.

Mathematical property tested
----------------------------
For any R ∈ SO(3) with D_in = I_in.D(R), D_out = I_out.D(R):

    upsamp( x @ D_in.T, lr_shape )[0]  ≈  upsamp( x, lr_shape )[0] @ D_out.T

Also verifies that the output HR shape equals (H*r, W*r) for upsample_factor r.

Run:  python eqv_inv_tests/test_so3_equivariance_transpose_conv.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from e3nn import o3
from e3nn.o3 import Irreps

from _helpers import rotate_features, rel_error, report, section, summary
from models.SR_double_conv_SRattn_a1 import EquivariantTransposeConv, LocalIsoCrystalEncoder

# ── config ────────────────────────────────────────────────────────────────────
H, W            = 8, 8
UPSAMPLE_FACTOR = 4
N_TRIALS        = 16
TOL             = 1e-4
DEVICE          = torch.device("cpu")
SEED            = 42


def _run_so3_equivariance(
    conv: EquivariantTransposeConv,
    x: torch.Tensor,
    lr_shape: tuple[int, int],
    irreps_in: Irreps,
    irreps_out: Irreps,
    n_trials: int,
    seed: int,
) -> tuple[float, float]:
    """
    Returns (mean_rel_err, max_rel_err) over n_trials random SO(3) rotations.
    """
    with torch.no_grad():
        y_ref, hr_shape = conv(x, lr_shape)

    errors = []
    torch.manual_seed(seed)
    for _ in range(n_trials):
        R     = o3.rand_matrix(dtype=torch.float32).to(DEVICE)
        D_in  = irreps_in.D_from_matrix(R).to(DEVICE)
        D_out = irreps_out.D_from_matrix(R).to(DEVICE)

        x_rot = rotate_features(x, D_in)
        with torch.no_grad():
            y_in_rot, _ = conv(x_rot, lr_shape)
            y_out_rot   = rotate_features(y_ref, D_out)

        errors.append(rel_error(y_in_rot, y_out_rot))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max())


def _check_output_shape(
    conv: EquivariantTransposeConv,
    x: torch.Tensor,
    lr_shape: tuple[int, int],
) -> bool:
    """Output spatial dims must be (H*r, W*r)."""
    with torch.no_grad():
        out, (Hr, Wr) = conv(x, lr_shape)
    H, W = lr_shape
    r = conv.upsample_factor
    expected_N = H * r * W * r
    ok = (out.shape[-2] == expected_N) and (Hr == H * r) and (Wr == W * r)
    return ok


def _make_conv(irreps_in, irreps_out, upsample_factor, use_residual, seed):
    torch.manual_seed(seed)
    return EquivariantTransposeConv(
        kernel_size=3,
        upsample_factor=upsample_factor,
        use_residual=use_residual,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    ).to(DEVICE).eval()


def _rand_features(irreps: Irreps, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, H * W, irreps.dim, dtype=torch.float32, device=DEVICE)


def main() -> None:
    results: list[bool] = []
    lr_shape = (H, W)

    for crystal in ("fcc", "hcp"):
        enc    = LocalIsoCrystalEncoder(crystal=crystal, dtype=torch.float32, device=DEVICE).eval()
        irr_a1 = enc.irreps_a1

        section(f"Crystal={crystal.upper()}  irreps_a1={irr_a1}  upsample_factor={UPSAMPLE_FACTOR}")

        cases = [
            # (label,                use_residual, seed)
            ("a1→a1  no-res",        False,        10),
            ("a1→a1  residual",      True,         11),
        ]

        for label, res, seed in cases:
            conv = _make_conv(irr_a1, irr_a1, UPSAMPLE_FACTOR, res, seed)
            x    = _rand_features(irr_a1, seed + 100)

            # shape check
            shape_ok = _check_output_shape(conv, x, lr_shape)
            results.append(shape_ok)
            status = "PASS" if shape_ok else "FAIL"
            print(f"  [\033[92m{status}\033[0m] {crystal.upper()} {label:<40s}  output_shape=({H*UPSAMPLE_FACTOR},{W*UPSAMPLE_FACTOR})")

            # equivariance check
            mean_err, max_err = _run_so3_equivariance(
                conv, x, lr_shape, irr_a1, irr_a1, N_TRIALS, SEED
            )
            ok = report(
                f"{crystal.upper()} {label} SO(3)-equivariance",
                max_err, TOL,
                extra=f"mean={mean_err:.2e}",
            )
            results.append(ok)

    summary(results)


if __name__ == "__main__":
    main()
