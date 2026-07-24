from __future__ import annotations

import pytest
import torch

from models.classic_sr_baselines import stable_sqrtm
from training.train_jangid_baseline import (
    build_loss,
    build_model,
    forward_baseline_active_from_passive,
)


@pytest.mark.parametrize(
    "model_type,scale,n_resgroups",
    [
        ("rcan_1d", [4, 1], 2),
        ("rcan_2d", [4, 4], 2),
        ("san_1d", [4, 1], 1),
        ("san_2d", [4, 4], 1),
        ("han_1d", [4, 1], 10),
        ("han_2d", [4, 4], 10),
    ],
)
def test_classic_baseline_output_shape_and_backward(model_type, scale, n_resgroups):
    model = build_model(
        {
            "model_type": model_type,
            "scale": scale,
            "n_resgroups": n_resgroups,
            "n_resblocks": 1,
            "n_feats": 8,
            "reduction": 4,
        }
    )
    height, width = ((4, 8) if scale == [4, 1] else (4, 4))
    lr = torch.randn(1, 4, height, width)
    lr = lr / lr.norm(dim=1, keepdim=True).clamp_min(1e-6)
    sr = forward_baseline_active_from_passive(model, lr)
    assert sr.shape == (1, 4, height * scale[0], width * scale[1])
    assert torch.allclose(sr.norm(dim=1), torch.ones_like(sr[:, 0]), atol=1e-5)
    sr.square().mean().backward()


def test_crystal_loss_uses_proper_rotation_subgroups():
    assert build_loss("Oh").syms.shape == (24, 4)
    assert build_loss("D6h").syms.shape == (12, 4)


def test_stable_sqrtm_has_finite_forward_and_backward_on_rank_deficient_covariance():
    samples = torch.randn(2, 64, 16, requires_grad=True)
    centered = samples - samples.mean(dim=-1, keepdim=True)
    covariance = centered.bmm(centered.transpose(1, 2)) / samples.shape[-1]
    root = stable_sqrtm(covariance, 5)
    assert torch.isfinite(root).all()
    root.square().mean().backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()
