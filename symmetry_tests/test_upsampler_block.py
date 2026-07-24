from __future__ import annotations

"""Symmetry checks for the equivariant transpose-convolution upsampler block."""

import torch

from models.SR_double_conv_SRattn import EquivariantTransposeConv
from symmetry_tests._symmetry_utils import (
    assert_rel_or_rms,
    best_feature_action_variant,
    choose_group_symmetry,
    choose_so3_probe_quaternion,
    left_action_quaternions,
    random_unit_quats,
    right_action_quaternions,
)


def test_upsampler_right_and_left_symmetry_behavior(encoder) -> None:
    """Upsampler should preserve right invariance and left equivariance behavior."""
    device = encoder.embedding.group_mats.device
    dtype = encoder.embedding.group_mats.dtype
    # Small LR shape to keep the test quick while still testing scale change.
    h, w = 4, 4
    n = h * w

    # Configure block to operate within A1 feature space.
    block = EquivariantTransposeConv(
        kernel_size=3,
        upsample_factor=2,
        use_residual=True,
        irreps_in=encoder.irreps_a1,
        irreps_out=encoder.irreps_a1,
    ).to(device=device, dtype=dtype).eval()

    q_active = random_unit_quats(n, seed=404, dtype=dtype, device=device)
    g = choose_group_symmetry(encoder.sym_ops)
    so3_probe = choose_so3_probe_quaternion(encoder.sym_ops)

    with torch.no_grad():
        feat_base = encoder.forward_a1_active(q_active)
        # Baseline upsampling output and HR shape.
        y_base, hr_shape = block(feat_base, (h, w))

        # Right-action path.
        feat_right = encoder.forward_a1_active(right_action_quaternions(q_active, g))
        y_right, hr_shape_right = block(feat_right, (h, w))

        # Left action by group element.
        feat_left_g = encoder.forward_a1_active(left_action_quaternions(q_active, g))
        y_left_g, hr_shape_left_g = block(feat_left_g, (h, w))

        # Left action by generic SO(3) probe.
        feat_left_so3 = encoder.forward_a1_active(left_action_quaternions(q_active, so3_probe))
        y_left_so3, hr_shape_left_so3 = block(feat_left_so3, (h, w))

    # All branches must produce the same spatial upsampled shape.
    assert hr_shape == hr_shape_right == hr_shape_left_g == hr_shape_left_so3
    # Right-invariance expected for A1-encoded features.
    assert_rel_or_rms(y_base, y_right, rel_tol=8e-4, rms_tol=8e-5)

    # Determine correct D vs D^T convention from encoder behavior.
    _, a_g, rel_g, rms_g = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_g,
        irreps=encoder.irreps_a1,
        sym=g,
    )
    assert rel_g < 3e-4 or rms_g < 3e-5
    # Left equivariance under discrete crystal group G.
    assert_rel_or_rms(y_base @ a_g, y_left_g, rel_tol=1e-3, rms_tol=1e-4)

    _, a_so3, rel_so3, rms_so3 = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_so3,
        irreps=encoder.irreps_a1,
        sym=so3_probe,
    )
    assert rel_so3 < 4e-4 or rms_so3 < 4e-5
    # Left equivariance under a generic SO(3) probe rotation.
    assert_rel_or_rms(y_base @ a_so3, y_left_so3, rel_tol=1.2e-3, rms_tol=1.2e-4)
