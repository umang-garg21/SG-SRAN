from __future__ import annotations

import torch
import torch.nn.functional as F

from models import QuaternionBicubicFInterpolateSR


def _random_unit_quats(h: int, w: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q = torch.randn(h, w, 4, generator=gen, dtype=torch.float32)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_forward_sr_matches_manual_torch_bicubic_then_renormalize() -> None:
    lr = _random_unit_quats(3, 4, seed=123)
    model = QuaternionBicubicFInterpolateSR(
        upsample_factor=(2, 3),
        align_corners=False,
        normalize_output=True,
        canonicalize_output=False,
    ).eval()

    out = model.forward_sr(lr.reshape(-1, 4), lr_shape=lr.shape[:2], normalize_input=False)
    out_img = out.reshape(6, 12, 4)

    manual = F.interpolate(
        lr.permute(2, 0, 1).unsqueeze(0),
        scale_factor=(2, 3),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)
    manual = manual / manual.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    assert out_img.shape == (6, 12, 4)
    assert torch.allclose(out_img, manual, atol=1e-6, rtol=1e-5)


def test_forward_alias_matches_forward_sr() -> None:
    lr = _random_unit_quats(2, 2, seed=456).reshape(-1, 4)
    model = QuaternionBicubicFInterpolateSR(upsample_factor=2).eval()

    a = model.forward_sr(lr, lr_shape=(2, 2), normalize_input=True)
    b = model.forward(lr, img_shape=(2, 2), normalize_input=True)

    assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)


def test_output_can_be_sign_canonicalized() -> None:
    lr = _random_unit_quats(2, 3, seed=789).reshape(-1, 4)
    model = QuaternionBicubicFInterpolateSR(
        upsample_factor=2,
        normalize_output=True,
        canonicalize_output=True,
    ).eval()

    out = model.forward_sr(lr, lr_shape=(2, 3), normalize_input=True)

    assert out.shape == (24, 4)
    assert torch.all(out[:, 0] >= -1e-7)
