import numpy as np
import torch
import pytest
from orix.quaternion import Orientation
from utils.quat_ops import (
    assert_quaternion_shape,
    to_spatial_quat,
    to_quat_spatial,
    is_scalar_first,
    to_scalar_first,
    to_scalar_last,
    safe_norm,
    normalize_quaternions,
    enforce_hemisphere,
    quat_left_multiply_numpy,
    quat_left_multiply_torch,
    is_in_fz,
    reduce_to_fz_min_angle,
    format_quaternions,
    get_dummy_quats,
)
from utils.symmetry_utils import resolve_symmetry


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def stable_quats():
    """Deterministic quaternions inside the FZ for cubic 432."""
    q = get_dummy_quats(resolution_deg=3.0)  # shape (N, 4)
    N = int(np.sqrt(q.shape[0]))
    q = q[: N * N].reshape(N, N, 4)
    return q.astype(np.float32)


@pytest.fixture
def stable_quats_quat_first(stable_quats):
    """Quaternion-first version of stable_quats."""
    return to_quat_spatial(stable_quats)


@pytest.fixture
def rand_quat_spatial():
    """Random normalized quaternion field in spatial-last layout."""
    q = np.random.randn(32, 32, 4).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


@pytest.fixture
def rand_quat_first():
    """Random normalized quaternion field in quaternion-first layout."""
    q = np.random.randn(4, 32, 32).astype(np.float32)
    q /= np.linalg.norm(q, axis=0, keepdims=True)
    return q


@pytest.fixture
def sym_O():
    """Cubic (432) symmetry group."""
    return resolve_symmetry("O")


# ============================================================
# Shape and layout tests
# ============================================================


def test_assert_quaternion_shape_valid(stable_quats):
    assert_quaternion_shape(stable_quats)  # should not raise


def test_assert_quaternion_shape_invalid():
    bad = np.zeros((8, 8))
    with pytest.raises(ValueError):
        assert_quaternion_shape(bad)


def test_to_spatial_quat_and_back(stable_quats):
    qf = to_quat_spatial(stable_quats)
    qs = to_spatial_quat(qf)
    assert qs.shape == stable_quats.shape
    np.testing.assert_allclose(
        np.sort(np.abs(qs).ravel()), np.sort(np.abs(stable_quats).ravel())
    )


def test_to_quat_spatial_and_back(stable_quats_quat_first):
    qs = to_spatial_quat(stable_quats_quat_first)
    qf = to_quat_spatial(qs)
    assert qf.shape == stable_quats_quat_first.shape
    np.testing.assert_allclose(
        np.sort(np.abs(qf).ravel()), np.sort(np.abs(stable_quats_quat_first).ravel())
    )


# ============================================================
# Scalar position helpers
# ============================================================


def test_is_scalar_first_consistency(stable_quats_quat_first):
    assert is_scalar_first(stable_quats_quat_first)


def test_to_scalar_first_and_last(stable_quats):
    q_last = to_scalar_last(stable_quats)
    q_first = to_scalar_first(q_last)
    np.testing.assert_allclose(np.abs(q_first), np.abs(stable_quats))


# ============================================================
# Normalization and hemisphere
# ============================================================


def test_safe_norm(rand_quat_spatial):
    norms = safe_norm(rand_quat_spatial, axis=-1)
    assert norms.shape == (32, 32, 1)
    assert np.all(norms > 0)


def test_normalize_quaternions():
    q = np.random.randn(4, 10).astype(np.float32)
    qn = normalize_quaternions(q.copy(), axis=0)
    norms = np.linalg.norm(qn, axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_enforce_hemisphere_scalar_first(rand_quat_first):
    q = rand_quat_first.copy()
    q[0] *= -1  # force scalar negative
    enforce_hemisphere(q, scalar_first=True)
    assert np.all(q[0] >= 0)


# ============================================================
# Multiplication tests
# ============================================================


def test_quat_left_multiply_numpy_layouts(stable_quats, sym_O):
    ops = sym_O.data.astype(np.float32)
    out_qf = quat_left_multiply_numpy(stable_quats, ops, layout="quat_first")
    out_qs = quat_left_multiply_numpy(stable_quats, ops, layout="quat_last")
    assert out_qf.shape[1] == 4
    assert out_qs.shape[-1] == 4
    assert out_qf.shape[0] == out_qs.shape[0]


def test_quat_left_multiply_torch_matches_numpy(stable_quats, sym_O):
    ops = sym_O.data.astype(np.float32)
    q_np = quat_left_multiply_numpy(stable_quats, ops, layout="quat_last")

    q_t = torch.tensor(stable_quats, dtype=torch.float32)
    ops_t = torch.tensor(ops, dtype=torch.float32)
    q_torch = quat_left_multiply_torch(q_t, ops_t)
    q_torch_np = torch.moveaxis(q_torch, 1, -1).cpu().numpy()

    np.testing.assert_allclose(q_np, q_torch_np, atol=1e-6, rtol=1e-6)


# ============================================================
# FZ operations
# ============================================================


def test_is_in_fz_fraction(stable_quats, sym_O):
    q_fz = reduce_to_fz_min_angle(stable_quats, sym_O)
    mask = is_in_fz(q_fz, sym_O)
    frac = mask.mean()
    assert frac > 0.99


def test_reduce_to_fz_min_angle_idempotent(stable_quats, sym_O):
    q_fz1 = reduce_to_fz_min_angle(stable_quats, sym_O)
    q_fz2 = reduce_to_fz_min_angle(q_fz1, sym_O)
    np.testing.assert_allclose(q_fz1, q_fz2, atol=1e-6)


# ============================================================
# Format quaternions
# ============================================================


def test_format_quaternions_spatial_and_quat_first(stable_quats):
    q_fmt_spatial = format_quaternions(stable_quats, quat_first=False)
    q_fmt_quat = format_quaternions(stable_quats, quat_first=True)
    assert q_fmt_spatial.shape[-1] == 4
    assert q_fmt_quat.shape[0] == 4


def test_format_quaternions_with_fz(stable_quats, sym_O):
    q_fmt = format_quaternions(
        stable_quats, normalize=True, hemisphere=True, reduce_fz=True, sym=sym_O
    )
    mask = is_in_fz(q_fmt, sym_O)
    assert mask.mean() > 0.99
