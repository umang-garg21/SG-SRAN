from __future__ import annotations

import pytest
import torch

pytest.importorskip("e3nn")
pytest.importorskip("orix")

from models.SR_double_conv_SRattn import (
    EquivariantTransposeConv,
    IsoEmbeddingSRAttn,
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


def test_non_optimizing_decoder_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Only decoder_backend='optimizing' is supported"):
        IsoEmbeddingSRAttn(
            crystal="fcc",
            device="cpu",
            decoder_backend="learnable",
        )


def test_ablation_no_lr_convs_no_attention_forward() -> None:
    model = IsoEmbeddingSRAttn(
        crystal="hcp",
        device="cpu",
        upsample_factor=2,
        use_lr_conv1=False,
        use_lr_conv2=False,
        use_attention=False,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_lr=0.05,
        decoder_method="cubochoric",
    ).eval()

    lr_shape = (2, 2)
    q_lr = _random_unit_quats(4, seed=77)

    with torch.no_grad():
        q_sr = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=True)

    assert len(model.attention_blocks) == 0
    assert q_sr.shape == (16, 4)
    norms = q_sr.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_upsample_transpose_kernels_are_tied_within_irrep_copy(
    small_model: IsoEmbeddingSRAttn,
) -> None:
    # Structural test:
    # every channel belonging to the same irrep copy must use exactly
    # the same transpose kernel (to avoid m-dependent kernels).
    up = small_model.upsample_conv
    w = up._expanded_transpose_weight().detach()
    idx = up.channel_to_copy_idx.detach()

    for copy_id in range(int(up.num_irrep_copies)):
        ch = torch.nonzero(idx == copy_id, as_tuple=False).view(-1)
        assert int(ch.numel()) >= 1
        base = w[int(ch[0])]
        for ci in ch[1:]:
            assert torch.allclose(w[int(ci)], base, atol=0.0, rtol=0.0)


def test_upsample_stage_equivariant_with_tied_m_kernels() -> None:
    """
    Functional test for the tied-kernel upsampling stage.

    We isolate the transpose-conv path, then check that applying a rotation
    in feature space before upsampling matches rotating the upsampled output.
    """
    from e3nn import o3

    up = EquivariantTransposeConv(
        kernel_size=3,
        upsample_factor=2,
        use_residual=True,
        irreps_in="2x1o",
        irreps_out="2x1o",
    ).eval()

    with torch.no_grad():
        # Make kernels different across irrep copies so we are testing true
        # copy-wise tying (not the trivial "all kernels equal" case).
        g = torch.Generator(device="cpu")
        g.manual_seed(123)
        up.transpose_kernels.copy_(
            torch.randn(up.transpose_kernels.shape, generator=g, dtype=up.transpose_kernels.dtype)
        )
        assert not torch.allclose(up.transpose_kernels[0], up.transpose_kernels[1])

        # Isolate transpose-conv stage:
        # set TP/context branch to zero so the output is exactly the upsampled
        # residual feature map.
        for p in up.tp.parameters():
            p.zero_()
        up.spatial_weights.zero_()

    H, W = 4, 5
    N = H * W
    x = torch.randn(N, up.in_dim, dtype=torch.float32)
    y, _ = up(x, (H, W))

    R = o3.rand_matrix().to(dtype=x.dtype)
    D = up.irreps_in.D_from_matrix(R).to(x.dtype)

    # Depending on row/column convention, either D or D^T can be the correct
    # right action for row-vector features; accept either.
    rels = []
    for A in (D, D.T):
        x_t = x @ A
        y_t, _ = up(x_t, (H, W))
        y_exp = y @ A
        rel_t = torch.linalg.norm(y_t - y_exp) / (torch.linalg.norm(y_exp) + 1e-12)
        rel = float(rel_t.detach().item())
        rels.append(rel)

    assert min(rels) < 1e-5


def test_upsample_accepts_irreps_in_and_irreps_out() -> None:
    up = EquivariantTransposeConv(
        kernel_size=3,
        upsample_factor=2,
        use_residual=True,
        irreps_in="2x1o",
        irreps_out="3x0e + 1x1o",
    ).eval()

    H, W = 3, 4
    N = H * W
    x = torch.randn(N, up.in_dim, dtype=torch.float32)
    y, hr_shape = up(x, (H, W))

    assert hr_shape == (H * 2, W * 2)
    assert y.shape == (H * 2 * W * 2, up.out_dim)
    assert up.residual_proj is not None
