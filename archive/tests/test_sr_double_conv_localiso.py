from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "models" / "SR_double_conv_SRattn.py"
try:
    import orix  # noqa: F401
    HAS_ORIX = True
except Exception:
    HAS_ORIX = False


def _load_module():
    spec = importlib.util.spec_from_file_location("sr_double_localiso_test_mod", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rand_unit_quats(n: int) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(q[:, :1] < 0, -q, q)


@pytest.mark.skipif(not HAS_ORIX, reason="orix is required for cubochoric optimizing decoder")
def test_iso_embedding_sr_attn_fcc_irreps_and_forward():
    mod = _load_module()
    model = mod.IsoEmbeddingSRAttn(
        crystal="fcc",
        device="cpu",
        upsample_factor=2,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=3,
        decoder_steps=2,
        decoder_lr=0.05,
        num_hr_attn_blocks=1,
    ).eval()

    assert not hasattr(model, "lift_layer")
    assert str(model.conv_lr1.tp.irreps_in1) == str(model.irreps_a1)
    assert str(model.conv_lr1.tp.irreps_in2) == str(model.irreps_a1)
    assert str(model.conv_lr1.tp.irreps_out) == str(model.irreps_full)

    assert str(model.conv_lr2.tp.irreps_in1) == str(model.irreps_full)
    assert str(model.conv_lr2.tp.irreps_in2) == str(model.irreps_full)
    assert str(model.conv_lr2.tp.irreps_out) == str(model.irreps_full)

    assert str(model.upsample_conv.tp.irreps_in1) == str(model.irreps_full)
    assert str(model.upsample_conv.tp.irreps_in2) == str(model.irreps_full)
    assert str(model.upsample_conv.tp.irreps_out) == str(model.irreps_full)

    assert str(model.conv_hr1.tp.irreps_in1) == str(model.irreps_full)
    assert str(model.conv_hr1.tp.irreps_in2) == str(model.irreps_full)
    assert str(model.conv_hr1.tp.irreps_out) == str(model.irreps_full)

    assert not hasattr(model, "conv_hr2")
    assert str(model.final_proj.irreps_in) == str(model.irreps_full)
    assert str(model.final_proj.irreps_out) == str(model.irreps_a1)
    assert model.decoder.target_irreps == "a1"

    lr_shape = (2, 2)
    q_lr = _rand_unit_quats(lr_shape[0] * lr_shape[1])
    q_hr = _rand_unit_quats((lr_shape[0] * 2) * (lr_shape[1] * 2))

    q_out = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=True)
    assert q_out.shape == (16, 4)
    assert torch.isfinite(q_out).all()

    loss = model.feature_loss_sr(
        q_lr,
        q_hr,
        lr_shape=lr_shape,
        normalize_input=True,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.skipif(not HAS_ORIX, reason="orix is required for cubochoric optimizing decoder")
def test_iso_embedding_sr_attn_hcp_runs():
    mod = _load_module()
    model = mod.IsoEmbeddingSRAttn(
        crystal="hcp",
        device="cpu",
        upsample_factor=2,
        decoder_cubochoric_resolution=1,
        decoder_num_starts=2,
        decoder_steps=1,
        decoder_lr=0.05,
        num_hr_attn_blocks=1,
    ).eval()
    q_lr = _rand_unit_quats(4)
    q_out = model.forward_sr(q_lr, lr_shape=(2, 2))
    assert q_out.shape == (16, 4)
    assert torch.isfinite(q_out).all()
