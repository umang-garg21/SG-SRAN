import os
import sys

import torch
from e3nn import o3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.local_iso_embedding_test_slow import (
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


def _announce(message: str) -> None:
    print(f"[TEST] {message}")


def test_local_iso_irreps_layout():
    _announce("LocalIsoCrystalEmbedding exposes expected irreps layouts for O and D6")
    emb_o = build_local_iso_fcc_embedding(device="cpu")
    emb_d6 = build_local_iso_hcp_embedding(device="cpu", d6_convention="z_axis")

    assert str(emb_o.irreps_out) == "1x2e+1x4e"
    assert str(emb_d6.irreps_out) == "2x2e+1x4e+1x6e"


def test_local_iso_raw_gram_near_identity():
    _announce("Raw HL embedding has near-identity Gram at identity for O and D6")
    emb_o = build_local_iso_fcc_embedding(device="cpu")
    emb_d6 = build_local_iso_hcp_embedding(device="cpu", d6_convention="z_axis")

    eye_o = torch.eye(3, dtype=torch.float64)
    eye_d6 = torch.eye(3, dtype=torch.float64)

    g_o = emb_o.gram_at_identity(use_raw=True, eps=1e-7)
    g_d6 = emb_d6.gram_at_identity(use_raw=True, eps=1e-7)

    err_o = float((g_o - eye_o).abs().max().item())
    err_d6 = float((g_d6 - eye_d6).abs().max().item())

    # D6 can be slightly noisier numerically; both should still be close.
    assert err_o < 1e-8, f"O raw Gram too far from identity: {err_o}"
    assert err_d6 < 1e-6, f"D6 raw Gram too far from identity: {err_d6}"


def test_local_iso_raw_right_invariance():
    _announce("Raw HL embedding is right-invariant under crystal group action")
    emb_o = build_local_iso_fcc_embedding(device="cpu")
    emb_d6 = build_local_iso_hcp_embedding(device="cpu", d6_convention="z_axis")

    err_o = emb_o.right_invariance_error(use_raw=True, n_trials=6, seed=0)
    err_d6 = emb_d6.right_invariance_error(use_raw=True, n_trials=6, seed=0)

    assert err_o < 1e-10, f"O right-invariance error too large: {err_o}"
    assert err_d6 < 1e-10, f"D6 right-invariance error too large: {err_d6}"


def test_local_iso_feature_shapes():
    _announce("Feature dimensions match expected raw/irreps sizes for O and D6")
    emb_o = build_local_iso_fcc_embedding(device="cpu")
    emb_d6 = build_local_iso_hcp_embedding(device="cpu", d6_convention="z_axis")

    r = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    y_o_raw = emb_o.forward_raw(r)
    y_o_ir = emb_o.forward_irreps(r)
    y_d6_raw = emb_d6.forward_raw(r)
    y_d6_ir = emb_d6.forward_irreps(r)

    assert tuple(y_o_raw.shape) == (1, 81)
    assert tuple(y_o_ir.shape) == (1, 14)
    assert tuple(y_d6_raw.shape) == (1, 738)
    assert tuple(y_d6_ir.shape) == (1, 32)


def test_local_iso_forward_from_quaternions_matches_matrix_path():
    _announce("Quaternion forward path matches matrix forward path")
    emb_o = build_local_iso_fcc_embedding(device="cpu")

    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.92387953, 0.0, 0.38268343, 0.0]],
        dtype=torch.float64,
    )
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    y_q = emb_o.forward_from_quaternions(q, raw=False)
    y_r = emb_o.forward_irreps(o3.quaternion_to_matrix(q))

    assert torch.allclose(y_q, y_r, atol=1e-10, rtol=1e-10)
