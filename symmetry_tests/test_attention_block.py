from __future__ import annotations

"""Symmetry checks for the block-local attention residual module."""

import torch

from models.SR_double_conv_SRattn import AttentionBlock
from symmetry_tests._symmetry_utils import (
    assert_rel_or_rms,
    best_feature_action_variant,
    choose_group_symmetry,
    choose_so3_probe_quaternion,
    left_action_quaternions,
    make_block_distance_matrix,
    random_unit_quats,
    right_action_quaternions,
)


def _run_attention_delta(
    block: AttentionBlock,
    feat: torch.Tensor,
    *,
    h: int,
    w: int,
    block_h: int,
    block_w: int,
) -> torch.Tensor:
    """Run one attention block and return only the residual delta tensor.

    The production pipeline uses `feat = feat + delta`; this helper isolates `delta`
    so we can test the attention operator itself.
    """
    # Distance matrix drives the learned positional bias term.
    d_block = make_block_distance_matrix(
        block_h,
        block_w,
        dtype=feat.dtype,
        device=feat.device,
    )
    return block(feat.unsqueeze(0), d_block, h, w, block_h, block_w).squeeze(0)


def test_attention_right_and_left_symmetry_behavior(encoder) -> None:
    """Attention delta should be right-invariant and left-equivariant."""
    device = encoder.embedding.group_mats.device
    dtype = encoder.embedding.group_mats.dtype
    h, w = 4, 4
    n = h * w
    # Use multiple blocks so attention partitioning logic is exercised.
    block_h, block_w = 2, 2

    # Match attention irreps to encoder A1 features.
    block = AttentionBlock(
        irreps_feat=encoder.irreps_a1,
        num_channels=2,
    ).to(device=device, dtype=dtype).eval()

    with torch.no_grad():
        # Force non-trivial residual behavior.
        torch.manual_seed(17)
        block.lin_out.weight.normal_(mean=0.0, std=0.05)
        # Non-zero positional bias ensures pairwise distance path is active.
        block.pos_bias.weight.fill_(0.10)
        block.pos_bias.bias.fill_(0.01)

    q_active = random_unit_quats(n, seed=505, dtype=dtype, device=device)
    g = choose_group_symmetry(encoder.sym_ops)
    so3_probe = choose_so3_probe_quaternion(encoder.sym_ops)

    with torch.no_grad():
        feat_base = encoder.forward_a1_active(q_active)
        # Baseline attention delta.
        delta_base = _run_attention_delta(block, feat_base, h=h, w=w, block_h=block_h, block_w=block_w)

        # Right action branch.
        feat_right = encoder.forward_a1_active(right_action_quaternions(q_active, g))
        delta_right = _run_attention_delta(block, feat_right, h=h, w=w, block_h=block_h, block_w=block_w)

        # Left action by group element.
        feat_left_g = encoder.forward_a1_active(left_action_quaternions(q_active, g))
        delta_left_g = _run_attention_delta(block, feat_left_g, h=h, w=w, block_h=block_h, block_w=block_w)

        # Left action by generic SO(3) probe.
        feat_left_so3 = encoder.forward_a1_active(left_action_quaternions(q_active, so3_probe))
        delta_left_so3 = _run_attention_delta(
            block,
            feat_left_so3,
            h=h,
            w=w,
            block_h=block_h,
            block_w=block_w,
        )

    # Right-invariance under G.
    assert_rel_or_rms(delta_base, delta_right, rel_tol=8e-4, rms_tol=8e-5)

    # Calibrate representation action (D vs D^T) before output assertion.
    _, a_g, rel_g, rms_g = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_g,
        irreps=encoder.irreps_a1,
        sym=g,
    )
    assert rel_g < 3e-4 or rms_g < 3e-5
    # Left equivariance under G.
    assert_rel_or_rms(delta_base @ a_g, delta_left_g, rel_tol=1.1e-3, rms_tol=1.1e-4)

    _, a_so3, rel_so3, rms_so3 = best_feature_action_variant(
        base=feat_base,
        transformed=feat_left_so3,
        irreps=encoder.irreps_a1,
        sym=so3_probe,
    )
    assert rel_so3 < 4e-4 or rms_so3 < 4e-5
    # Left equivariance under generic SO(3) probe.
    assert_rel_or_rms(delta_base @ a_so3, delta_left_so3, rel_tol=1.3e-3, rms_tol=1.3e-4)
