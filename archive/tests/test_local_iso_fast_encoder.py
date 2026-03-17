import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.local_iso_embedding_test_slow import (
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)
from models.local_iso_embedding import (
    build_fast_local_iso_fcc_encoder,
    build_fast_local_iso_hcp_encoder,
)


def _rand_quats(n: int, dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_fast_encoder_fcc_matches_reference_shape_and_values():
    dtype = torch.float64
    n = 8

    ref = build_local_iso_fcc_embedding(dtype=dtype, device="cpu")
    fast = build_fast_local_iso_fcc_encoder(dtype=dtype, device="cpu")

    q = _rand_quats(n, dtype)
    y_ref = ref.forward_from_quaternions(q)
    y_fast = fast.forward_from_quaternions(q)

    assert tuple(y_ref.shape) == (n, 14)
    assert tuple(y_fast.shape) == (n, 14)
    assert torch.allclose(y_ref, y_fast, atol=1e-5, rtol=1e-5)


def test_fast_encoder_hcp_matches_reference_shape_and_values():
    dtype = torch.float64
    n = 8

    ref = build_local_iso_hcp_embedding(dtype=dtype, device="cpu", d6_convention="z_axis")
    fast = build_fast_local_iso_hcp_encoder(dtype=dtype, device="cpu", d6_convention="z_axis")

    q = _rand_quats(n, dtype)
    y_ref = ref.forward_from_quaternions(q)
    y_fast = fast.forward_from_quaternions(q)

    assert tuple(y_ref.shape) == (n, 32)
    assert tuple(y_fast.shape) == (n, 32)
    assert torch.allclose(y_ref, y_fast, atol=1e-5, rtol=1e-5)
