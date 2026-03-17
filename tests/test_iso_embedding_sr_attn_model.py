from __future__ import annotations

import pytest
import torch

pytest.importorskip("e3nn")
pytest.importorskip("orix")

from models.SR_double_conv_SRattn import (
    IsoEmbeddingSRAttn,
    LearnableA1QuaternionDecoder,
)


def _random_unit_quats(n: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    q = torch.randn(n, 4, generator=g, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q


@pytest.fixture(scope="module")
def small_model() -> IsoEmbeddingSRAttn:
    model = IsoEmbeddingSRAttn(
        crystal="fcc",
        device="cpu",
        upsample_factor=2,
        num_hr_attn_blocks=1,
        hr_attn_num_channels=4,
        hr_attn_block_size=4,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_lr=0.05,
        decoder_method="cubochoric",
    )
    return model


@pytest.fixture(scope="module")
def small_model_learnable_decoder() -> IsoEmbeddingSRAttn:
    model = IsoEmbeddingSRAttn(
        crystal="fcc",
        device="cpu",
        upsample_factor=2,
        num_hr_attn_blocks=1,
        hr_attn_num_channels=4,
        hr_attn_block_size=4,
        decoder_backend="learnable",
        decoder_learnable_hidden_dim=64,
        decoder_learnable_num_layers=2,
        decoder_learnable_dropout=0.0,
    )
    return model


def test_architecture_contracts(small_model: IsoEmbeddingSRAttn) -> None:
    model = small_model

    assert not hasattr(model, "lift_layer")
    assert not hasattr(model, "conv_hr2")

    assert model.conv_lr1.tp.irreps_in1 == model.irreps_a1
    assert model.conv_lr1.tp.irreps_in2 == model.irreps_a1
    assert model.conv_lr1.tp.irreps_out == model.irreps_full

    assert model.conv_lr2.tp.irreps_in1 == model.irreps_full
    assert model.conv_lr2.tp.irreps_in2 == model.irreps_full
    assert model.conv_lr2.tp.irreps_out == model.irreps_full

    assert model.upsample_conv.tp.irreps_in1 == model.irreps_full
    assert model.upsample_conv.tp.irreps_in2 == model.irreps_full
    assert model.upsample_conv.tp.irreps_out == model.irreps_full

    assert model.conv_hr1.tp.irreps_in1 == model.irreps_full
    assert model.conv_hr1.tp.irreps_in2 == model.irreps_full
    assert model.conv_hr1.tp.irreps_out == model.irreps_full

    assert model.final_proj.irreps_in == model.irreps_full
    assert model.final_proj.irreps_out == model.irreps_a1


def test_forward_sr_shape_and_norm(small_model: IsoEmbeddingSRAttn) -> None:
    model = small_model.eval()
    lr_shape = (2, 2)
    q_lr = _random_unit_quats(4, seed=11)

    with torch.no_grad():
        q_sr = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=True)

    assert q_sr.shape == (16, 4)
    norms = q_sr.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
    assert torch.all(q_sr[:, 0] >= -1e-6)


def test_forward_with_img_shape_matches_forward_sr(small_model: IsoEmbeddingSRAttn) -> None:
    model = small_model.eval()
    lr_shape = (2, 2)
    q_lr = _random_unit_quats(4, seed=22)

    with torch.no_grad():
        a = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=True)
        b = model.forward(q_lr, img_shape=lr_shape, normalize_input=True)

    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_feature_loss_scalar_and_backward(small_model: IsoEmbeddingSRAttn) -> None:
    model = small_model.train()
    model.zero_grad(set_to_none=True)

    lr_shape = (2, 2)
    q_lr = _random_unit_quats(4, seed=33)
    q_hr = _random_unit_quats(16, seed=44)

    loss = model.feature_loss_sr(q_lr, q_hr, lr_shape=lr_shape, normalize_input=True)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()
    assert model.conv_lr1.spatial_weights.grad is not None


def test_learnable_decoder_forward_sr(
    small_model_learnable_decoder: IsoEmbeddingSRAttn,
) -> None:
    model = small_model_learnable_decoder.eval()
    assert isinstance(model.decoder, LearnableA1QuaternionDecoder)

    lr_shape = (2, 2)
    q_lr = _random_unit_quats(4, seed=55)

    with torch.no_grad():
        q_sr = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=True)

    assert q_sr.shape == (16, 4)
    assert torch.isfinite(q_sr).all().item()
    norms = q_sr.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_learnable_decoder_backward(
    small_model_learnable_decoder: IsoEmbeddingSRAttn,
) -> None:
    model = small_model_learnable_decoder.train()
    model.decoder.zero_grad(set_to_none=True)

    feat = torch.randn(8, model.feature_dim_a1, dtype=torch.float32, requires_grad=True)
    q = model.decoder(feat)
    loss = q.pow(2).mean()
    loss.backward()

    grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
