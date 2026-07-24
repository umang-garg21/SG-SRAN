from __future__ import annotations

"""Symmetry checks for the equivariant spatial convolution block."""

import torch

from models.SR_double_conv_SRattn import EquivariantSpatialConv
from symmetry_tests._symmetry_utils import (
    assert_rel_or_rms,
    best_feature_action_variant,
    choose_group_symmetry,
    choose_so3_probe_quaternion,
    left_action_quaternions,
    random_unit_quats,
    right_action_quaternions,
)


def test_equivariant_conv_right_and_left_symmetry_behavior(encoder) -> None:
    """Conv block should preserve encoder-stage right invariance and left equivariance."""
    device = encoder.embedding.group_mats.device
    dtype = encoder.embedding.group_mats.dtype
    # Small grid keeps test fast while still exercising spatial neighborhood logic.
    h, w = 4, 5
    n = h * w

    # Isolated block under test, configured to match encoder A1 feature irreps.
    block = EquivariantSpatialConv(
        kernel_size=3,
        irreps_in=encoder.irreps_a1,
        irreps_out=encoder.irreps_a1,
        use_residual=True,
    ).to(device=device, dtype=dtype).eval()

    q_active = random_unit_quats(n, seed=303, dtype=dtype, device=device)
    g = choose_group_symmetry(encoder.sym_ops)
    so3_probe = choose_so3_probe_quaternion(encoder.sym_ops)

    with torch.no_grad():
        # Encode once, then run the block as baseline.
        feat_base = encoder.forward_a1_active(q_active)
        y_base = block(feat_base, (h, w))

        # Right action branch.
        feat_right = encoder.forward_a1_active(right_action_quaternions(q_active, g))
        y_right = block(feat_right, (h, w))

        # Left action by crystal group element.
        feat_left_g = encoder.forward_a1_active(left_action_quaternions(q_active, g))
        y_left_g = block(feat_left_g, (h, w))

        # Left action by general SO(3) probe element.
        feat_left_so3 = encoder.forward_a1_active(left_action_quaternions(q_active, so3_probe))
        y_left_so3 = block(feat_left_so3, (h, w))

    # Right action by crystal symmetry G should be invariant at A1-encoded features.
    assert_rel_or_rms(y_base, y_right, rel_tol=5e-4, rms_tol=5e-5)

    # Left action by G should produce an equivariant channel transform.
    # Calibrate representation convention before asserting output behavior.
    _, a_g, rel_g, rms_g = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_g,
        irreps=encoder.irreps_a1,
        sym=g,
    )
    assert rel_g < 3e-4 or rms_g < 3e-5
    assert_rel_or_rms(y_base @ a_g, y_left_g, rel_tol=7e-4, rms_tol=7e-5)

    # Left action by a generic SO(3) rotation should also be equivariant.
    _, a_so3, rel_so3, rms_so3 = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_so3,
        irreps=encoder.irreps_a1,
        sym=so3_probe,
    )
    assert rel_so3 < 4e-4 or rms_so3 < 4e-5
    # Apply the calibrated feature action to baseline block output.
    assert_rel_or_rms(y_base @ a_so3, y_left_so3, rel_tol=9e-4, rms_tol=9e-5)
