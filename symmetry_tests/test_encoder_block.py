from __future__ import annotations

"""Symmetry checks for the encoder block only."""

import torch

from symmetry_tests._symmetry_utils import (
    assert_rel_or_rms,
    best_feature_action_variant,
    choose_group_symmetry,
    choose_so3_probe_quaternion,
    left_action_quaternions,
    random_unit_quats,
    right_action_quaternions,
)


def test_encoder_right_invariance_under_group_g(encoder) -> None:
    """A1 encoder output should be invariant to right action by crystal symmetry g in G."""
    # Use encoder-native dtype/device to avoid precision/device mismatch noise.
    device = encoder.embedding.group_mats.device
    dtype = encoder.embedding.group_mats.dtype
    # Sample active-convention quaternions used by forward_a1_active.
    q_active = random_unit_quats(96, seed=101, dtype=dtype, device=device)
    # Pick one deterministic element from the discrete crystal group.
    g = choose_group_symmetry(encoder.sym_ops)

    with torch.no_grad():
        # Baseline encoding.
        feat_base = encoder.forward_a1_active(q_active)
        # Right action q -> q ⊗ g, then encode.
        feat_right = encoder.forward_a1_active(right_action_quaternions(q_active, g))

    # Right-invariance means both encoded feature sets should match.
    assert_rel_or_rms(feat_base, feat_right, rel_tol=2e-4, rms_tol=3e-5)


def test_encoder_left_equivariance_for_group_g_and_so3(encoder) -> None:
    """A1 encoder should transform equivariantly under left action for G and SO(3)."""
    device = encoder.embedding.group_mats.device
    dtype = encoder.embedding.group_mats.dtype
    q_active = random_unit_quats(96, seed=202, dtype=dtype, device=device)
    g = choose_group_symmetry(encoder.sym_ops)
    # Choose a probe rotation likely outside the discrete crystal group.
    so3_probe = choose_so3_probe_quaternion(encoder.sym_ops)

    with torch.no_grad():
        feat_base = encoder.forward_a1_active(q_active)
        # Left action with group element g.
        feat_left_g = encoder.forward_a1_active(left_action_quaternions(q_active, g))
        # Left action with generic SO(3) probe rotation.
        feat_left_so3 = encoder.forward_a1_active(left_action_quaternions(q_active, so3_probe))

    # Calibrate whether row-vector convention is x@D or x@D^T for this stage.
    _, a_g, rel_g, rms_g = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_g,
        irreps=encoder.irreps_a1,
        sym=g,
    )
    # Calibration itself should already show small mismatch.
    assert rel_g < 3e-4 or rms_g < 3e-5
    # Final equivariance check for G.
    assert_rel_or_rms(feat_base @ a_g, feat_left_g, rel_tol=3e-4, rms_tol=3e-5)

    _, a_so3, rel_so3, rms_so3 = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_so3,
        irreps=encoder.irreps_a1,
        sym=so3_probe,
    )
    assert rel_so3 < 4e-4 or rms_so3 < 4e-5
    # Final equivariance check for generic SO(3) rotation.
    assert_rel_or_rms(feat_base @ a_so3, feat_left_so3, rel_tol=4e-4, rms_tol=4e-5)
