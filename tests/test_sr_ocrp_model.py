from __future__ import annotations

import pytest
import torch

pytest.importorskip("e3nn")

from models.SR_ocrp import (
    OCRPPatchUpsampler,
    IsoEmbeddingSROCRP,
    WithinSlotInvariantPool,
    _build_patch_token_coords,
)


def _random_unit_quats(n: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    q = torch.randn(n, 4, generator=g, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q


def test_macro_tile_size_one_matches_pixel_path() -> None:
    common = dict(
        crystal="fcc",
        device="cpu",
        upsample_factor=(4, 1),
        window_size=5,
        cluster_connectivity=8,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_lr=0.05,
    )
    pixel = IsoEmbeddingSROCRP(
        **common,
        ocrp_mode="pixel_patch",
        macro_lr_tile_size=3,
        ocrp_token_conditioned_member_bias=True,
    ).eval()
    macro = IsoEmbeddingSROCRP(
        **common,
        ocrp_mode="macro_tile",
        macro_lr_tile_size=1,
        ocrp_token_conditioned_member_bias=True,
    ).eval()
    macro.load_state_dict(pixel.state_dict(), strict=True)

    lr_shape = (3, 4)
    q_lr = _random_unit_quats(lr_shape[0] * lr_shape[1], seed=11)

    with torch.no_grad():
        feat_lr_pixel = pixel.encode(q_lr)
        feat_hr_pixel, hr_shape_pixel = pixel._forward_sr_features(
            q_lr,
            feat_lr_pixel,
            lr_shape=lr_shape,
            return_aux=False,
        )

        feat_lr_macro = macro.encode(q_lr)
        feat_hr_macro, hr_shape_macro = macro._forward_sr_features(
            q_lr,
            feat_lr_macro,
            lr_shape=lr_shape,
            return_aux=False,
        )

    assert hr_shape_pixel == hr_shape_macro == (lr_shape[0] * 4, lr_shape[1] * 1)
    assert torch.allclose(feat_hr_pixel, feat_hr_macro, atol=1e-6, rtol=1e-5)


def test_macro_tile_mode_support_grid_and_output_shape() -> None:
    model = IsoEmbeddingSROCRP(
        crystal="fcc",
        device="cpu",
        upsample_factor=(4, 1),
        window_size=5,
        cluster_connectivity=8,
        ocrp_mode="macro_tile",
        macro_lr_tile_size=3,
        use_lr_conv1=False,
        use_hr_conv1=False,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_lr=0.05,
    ).eval()

    lr_shape = (5, 7)
    q_lr = _random_unit_quats(lr_shape[0] * lr_shape[1], seed=22)

    with torch.no_grad():
        feat_lr = model.encode(q_lr)
        feat_hr, hr_shape, aux = model._forward_sr_features(
            q_lr,
            feat_lr,
            lr_shape=lr_shape,
            return_aux=True,
        )

    assert hr_shape == (20, 7)
    assert feat_hr.shape == (20 * 7, model.feature_dim)
    assert aux["ocrp_mode"] == "macro_tile"
    assert aux["macro_lr_tile_size"] == 3
    assert aux["hr_patch_size"] == (12, 3)
    assert aux["hr_patch_shape"] == (12, 3)
    assert aux["hr_patch_tokens"] == 36
    assert aux["token_conditioned_member_bias"] is True
    assert aux["support_grid_shape"] == (2, 3)
    assert aux["lr_padded_shape"] == (6, 9)
    assert aux["bank_q"].shape == (6, 25, 4)
    assert aux["patch_out"].shape == (6, 36, model.feature_dim)


def test_macro_tile_assembly_is_contiguous_without_overlap() -> None:
    upsampler = OCRPPatchUpsampler(
        irreps_feat="1x0e",
        sym_ops_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        upsample_factor=(4, 1),
        window_size=5,
        ocrp_mode="macro_tile",
        macro_lr_tile_size=3,
    ).eval()

    patch_tokens = upsampler.hr_patch_shape[0] * upsampler.hr_patch_shape[1]
    patch_out = torch.zeros((1, 4, patch_tokens, 1), dtype=torch.float32)
    for tile_idx in range(4):
        patch_out[0, tile_idx, :, 0] = float(tile_idx + 1)

    feat_hr = upsampler._assemble_macro_patch_tokens(
        patch_out,
        grid_shape=(2, 2),
        hr_crop_shape=(24, 6),
    )
    img = feat_hr.view(24, 6, 1).squeeze(-1)

    assert torch.all(img[:12, :3] == 1.0)
    assert torch.all(img[:12, 3:] == 2.0)
    assert torch.all(img[12:, :3] == 3.0)
    assert torch.all(img[12:, 3:] == 4.0)


def test_patch_token_coords_cover_full_normalized_patch() -> None:
    coords = _build_patch_token_coords(
        16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coords.shape == (16, 2)
    assert torch.allclose(coords[0], torch.tensor([-1.0, -1.0], dtype=torch.float32))
    assert torch.allclose(coords[-1], torch.tensor([1.0, 1.0], dtype=torch.float32))


def test_patch_token_coords_support_rectangular_patch() -> None:
    coords = _build_patch_token_coords(
        (4, 1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coords.shape == (4, 2)
    assert torch.allclose(coords[0], torch.tensor([-1.0, 0.0], dtype=torch.float32))
    assert torch.allclose(coords[-1], torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_within_slot_pool_geometry_bias_varies_across_hr_tokens() -> None:
    pool = WithinSlotInvariantPool(
        irreps_feat="1x0e",
        meta_dim=9,
        phase_dim=4,
        window_size=3,
        hidden_dim=8,
        chunk_size=1,
        token_conditioned_member_bias=True,
    ).eval()

    with torch.no_grad():
        for layer in pool.member_key:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        for layer in pool.phase_query:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        for layer in pool.token_bias_ctrl:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        final_linear = pool.token_bias_ctrl[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.bias[1] = 1.0

    slot_anchor_ctx = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    slot_meta = torch.zeros((1, 1, 1, 9), dtype=torch.float32)
    bank_f = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1)
    slot_mask = torch.ones((1, 1, 1, 9), dtype=torch.bool)
    phase_grid = torch.zeros((4, 4), dtype=torch.float32)

    _, alpha = pool(
        slot_anchor_ctx=slot_anchor_ctx,
        slot_meta=slot_meta,
        bank_f=bank_f,
        slot_mask=slot_mask,
        phase_grid=phase_grid,
        return_alpha=True,
    )

    assert alpha is not None
    top_token_alpha = alpha[0, 0, 0, 0]
    bottom_token_alpha = alpha[0, 0, 0, 2]
    assert not torch.allclose(top_token_alpha, bottom_token_alpha)
