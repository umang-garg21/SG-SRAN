from __future__ import annotations

"""Symmetry checks for the feature-to-quaternion decoder block."""

from contextlib import contextmanager

import torch

import models.SR_double_conv_SRattn_a1 as sr_a1
from models.SR_double_conv_SRattn import CubochoricOptimizingLocalIsoDecoder, LocalIsoCrystalEncoder
from symmetry_tests._symmetry_utils import (
    assert_quat_rel_or_rms,
    choose_group_symmetry,
    choose_so3_probe_quaternion,
    left_action_quaternions,
    passive_right_action_for_active_left,
    random_unit_quats,
    right_action_quaternions,
    to_passive_quaternions,
)


@contextmanager
def _patched_sampler(table_quats_passive: torch.Tensor):
    """Temporarily replace orix FZ sampler with a deterministic in-memory table.

    This keeps decoder tests self-contained and avoids runtime dependency issues
    from the external sampling stack.
    """
    original = sr_a1._sample_fz_quaternions_passive

    def _fake_sample(  # noqa: PLR0913
        group_name: str,
        resolution: int,
        method: str,
        dtype: torch.dtype,
        device: torch.device,
        max_rows: int | None = None,
    ) -> torch.Tensor:
        # Signature matches the production sampler but only table values matter.
        del group_name, resolution, method
        out = table_quats_passive.to(device=device, dtype=dtype)
        if max_rows is not None:
            out = out[: int(max_rows)]
        return out

    sr_a1._sample_fz_quaternions_passive = _fake_sample
    try:
        yield
    finally:
        sr_a1._sample_fz_quaternions_passive = original


def _build_decoder_without_orix_sampling(
    crystal: str,
) -> tuple[
    LocalIsoCrystalEncoder,
    CubochoricOptimizingLocalIsoDecoder,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Construct encoder+decoder with a synthetic lookup table and no orix sampling."""
    # Encoder under test for selected crystal family.
    encoder = LocalIsoCrystalEncoder(
        crystal=crystal,
        dtype=torch.float32,
        device="cpu",
    ).eval()

    g = choose_group_symmetry(encoder.sym_ops)
    so3_probe = choose_so3_probe_quaternion(encoder.sym_ops)
    # Base active quaternions that define the deterministic query set.
    q_base_active = random_unit_quats(
        32,
        seed=606,
        dtype=encoder.embedding.group_mats.dtype,
        device=encoder.embedding.group_mats.device,
    )

    # Build a compact synthetic table that contains the exact queries used below,
    # so decoder checks stay deterministic and lightweight.
    q_table_active = torch.cat(
        [
            # Baseline orientations.
            q_base_active,
            # Left action by one element of G.
            left_action_quaternions(q_base_active, g),
            # Left action by generic SO(3) probe.
            left_action_quaternions(q_base_active, so3_probe),
        ],
        dim=0,
    )
    # Decoder table is expected in passive convention.
    q_table_passive = to_passive_quaternions(q_table_active)

    # Build the decoder while sampler is patched to our synthetic table.
    with _patched_sampler(q_table_passive):
        decoder = CubochoricOptimizingLocalIsoDecoder(
            encoder=encoder,
            cubochoric_resolution=1,
            method="cubochoric",
            num_starts=1,
            steps=0,
            lr=0.05,
            target_irreps="a1",
            max_table_rows=None,
            table_cache_dir=None,
        ).eval()

    return encoder, decoder, g, so3_probe, q_base_active


def test_decoder_right_invariance_and_left_equivariance(crystal) -> None:
    """Decoded quaternions should satisfy right-invariance and left-equivariance rules."""
    encoder, decoder, g, so3_probe, q_base_active = _build_decoder_without_orix_sampling(crystal)

    # Build transformed active inputs.
    q_right_active = right_action_quaternions(q_base_active, g)
    q_left_g_active = left_action_quaternions(q_base_active, g)
    q_left_so3_active = left_action_quaternions(q_base_active, so3_probe)

    # Encoder.forward_a1 expects passive quaternions.
    q_base_passive = to_passive_quaternions(q_base_active)
    q_right_passive = to_passive_quaternions(q_right_active)
    q_left_g_passive = to_passive_quaternions(q_left_g_active)
    q_left_so3_passive = to_passive_quaternions(q_left_so3_active)

    with torch.no_grad():
        # Encode transformed inputs to A1 features.
        feat_base = encoder.forward_a1(q_base_passive)
        feat_right = encoder.forward_a1(q_right_passive)
        feat_left_g = encoder.forward_a1(q_left_g_passive)
        feat_left_so3 = encoder.forward_a1(q_left_so3_passive)

        # Decode back to passive quaternions.
        q_dec_base = decoder(feat_base)
        q_dec_right = decoder(feat_right)
        q_dec_left_g = decoder(feat_left_g)
        q_dec_left_so3 = decoder(feat_left_so3)

    # Right action by G should be invariant in A1 feature space.
    assert_quat_rel_or_rms(q_dec_base, q_dec_right, rel_tol=2e-4, rms_tol=2e-4)

    # Left action in active convention corresponds to right action by inverse in passive convention.
    q_left_g_expected = passive_right_action_for_active_left(q_dec_base, g)
    q_left_so3_expected = passive_right_action_for_active_left(q_dec_base, so3_probe)

    # Verify equivariance for discrete group G and generic SO(3) probe.
    assert_quat_rel_or_rms(q_left_g_expected, q_dec_left_g, rel_tol=6e-4, rms_tol=6e-4)
    assert_quat_rel_or_rms(q_left_so3_expected, q_dec_left_so3, rel_tol=8e-4, rms_tol=8e-4)
