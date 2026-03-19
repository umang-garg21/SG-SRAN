"""
test_o_equivariance_spatial_layers.py
=======================================
Validates equivariance of EquivariantSpatialConv and EquivariantTransposeConv
specifically under the CRYSTAL GROUP (O for FCC, D6 for HCP) acting on features
via Wigner-D matrices.

Mathematical context
--------------------
The crystal group O (FCC, 24 elements) is a finite subgroup of SO(3).
Each element g ∈ O has a rotation matrix R(g) and therefore acts on the irrep
feature space via D = irreps.D_from_matrix(R(g)).

Since the layers are SO(3)-equivariant (see test_so3_equivariance_*.py),
they are *automatically* O-equivariant.  This test verifies that property
explicitly for every concrete crystal group element, using the actual sym_ops
quaternions stored in the encoder.

Property tested for each layer f and each g ∈ crystal group:

    f( x @ D_in(g).T, shape )  ≈  f( x, shape ) @ D_out(g).T

This is distinct from test_encoder_crystal_invariance.py which tests
invariance on orientation (quaternion) inputs.

Run:  python eqv_inv_tests/test_o_equivariance_spatial_layers.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from e3nn.o3 import Irreps

from _helpers import (
    quat_to_matrix, rotate_features, rel_error,
    report, section, summary, rand_quaternions,
)
from models.SR_double_conv_SRattn_a1 import (
    EquivariantSpatialConv,
    EquivariantTransposeConv,
    LocalIsoCrystalEncoder,
)

# ── config ────────────────────────────────────────────────────────────────────
H, W            = 8, 8
UPSAMPLE_FACTOR = 4
TOL             = 1e-4
DEVICE          = torch.device("cpu")
SEED            = 42


def _group_rotation_matrices(sym_ops: torch.Tensor) -> list[torch.Tensor]:
    """Convert crystal sym_ops quaternions to 3x3 rotation matrices."""
    return [quat_to_matrix(sym_ops[i]) for i in range(sym_ops.shape[0])]


def _test_conv_o_equivariance(
    conv: EquivariantSpatialConv,
    x: torch.Tensor,
    R_mats: list[torch.Tensor],
    irreps_in: Irreps,
    irreps_out: Irreps,
) -> tuple[float, float, list[float]]:
    """
    Returns (mean_err, max_err, per_op_errors) over all crystal group elements.
    """
    img_shape = (H, W)
    with torch.no_grad():
        y_ref = conv(x, img_shape)

    errors = []
    for R in R_mats:
        D_in  = irreps_in.D_from_matrix(R).to(DEVICE)
        D_out = irreps_out.D_from_matrix(R).to(DEVICE)

        x_rot = rotate_features(x, D_in)
        with torch.no_grad():
            y_in_rot  = conv(x_rot, img_shape)
            y_out_rot = rotate_features(y_ref, D_out)

        errors.append(rel_error(y_in_rot, y_out_rot))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max()), errors


def _test_upsamp_o_equivariance(
    conv: EquivariantTransposeConv,
    x: torch.Tensor,
    R_mats: list[torch.Tensor],
    irreps_in: Irreps,
    irreps_out: Irreps,
) -> tuple[float, float, list[float]]:
    lr_shape = (H, W)
    with torch.no_grad():
        y_ref, _ = conv(x, lr_shape)

    errors = []
    for R in R_mats:
        D_in  = irreps_in.D_from_matrix(R).to(DEVICE)
        D_out = irreps_out.D_from_matrix(R).to(DEVICE)

        x_rot = rotate_features(x, D_in)
        with torch.no_grad():
            y_in_rot, _  = conv(x_rot, lr_shape)
            y_out_rot    = rotate_features(y_ref, D_out)

        errors.append(rel_error(y_in_rot, y_out_rot))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max()), errors


def main() -> None:
    all_results: list[bool] = []

    for crystal in ("fcc", "hcp"):
        enc     = LocalIsoCrystalEncoder(crystal=crystal, dtype=torch.float32, device=DEVICE).eval()
        irr_a1  = enc.irreps_a1
        sym_ops = enc.sym_ops
        n_ops   = sym_ops.shape[0]
        R_mats  = _group_rotation_matrices(sym_ops)

        section(
            f"Crystal={crystal.upper()}  |S|={n_ops}  "
            f"irreps_a1={irr_a1}  (each element tested explicitly)"
        )

        torch.manual_seed(SEED)
        x = torch.randn(1, H * W, irr_a1.dim, dtype=torch.float32, device=DEVICE)

        # ── EquivariantSpatialConv (a1→a1, k=3) ───────────────────────────────
        for res in (False, True):
            torch.manual_seed(SEED + 1)
            conv = EquivariantSpatialConv(
                kernel_size=3, irreps_in=irr_a1, irreps_out=irr_a1, use_residual=res,
            ).to(DEVICE).eval()

            mean_e, max_e, per_op = _test_conv_o_equivariance(conv, x, R_mats, irr_a1, irr_a1)
            label = f"{crystal.upper()} SpatialConv(res={res}) O-equivariance"
            ok = report(label, max_e, TOL, extra=f"mean={mean_e:.2e}  over {n_ops} ops")
            all_results.append(ok)
            if not ok:
                print(f"    Per-op: {[f'{e:.2e}' for e in per_op]}")

        # ── EquivariantTransposeConv (a1→a1) ──────────────────────────────────
        for res in (False, True):
            torch.manual_seed(SEED + 2)
            tconv = EquivariantTransposeConv(
                kernel_size=3, upsample_factor=UPSAMPLE_FACTOR,
                use_residual=res, irreps_in=irr_a1, irreps_out=irr_a1,
            ).to(DEVICE).eval()

            mean_e, max_e, per_op = _test_upsamp_o_equivariance(tconv, x, R_mats, irr_a1, irr_a1)
            label = f"{crystal.upper()} TransposeConv(res={res}) O-equivariance"
            ok = report(label, max_e, TOL, extra=f"mean={mean_e:.2e}  over {n_ops} ops")
            all_results.append(ok)
            if not ok:
                print(f"    Per-op: {[f'{e:.2e}' for e in per_op]}")

    summary(all_results)


if __name__ == "__main__":
    main()
