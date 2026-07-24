from __future__ import annotations

"""Shared math and assertion helpers for block-level symmetry tests.

This module centralizes:
- Quaternion algebra and convention helpers.
- Discrete group and SO(3) probe selection utilities.
- Feature-space action matrix construction for e3nn irreps.
- Error metrics/assertions used consistently across all block tests.
"""

import torch
from e3nn.o3 import Irreps


def normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternions and enforce a canonical sign (w >= 0)."""
    # Clamp avoids division by zero for near-degenerate values.
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    # q and -q represent the same rotation; canonicalize for stable comparisons.
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 with output re-normalization."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return normalize_quaternions(out)


def random_unit_quats(
    n: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate deterministic random unit quaternions on the requested device/dtype."""
    # Use a CPU generator for stable seeded behavior across environments.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    q = torch.randn(int(n), 4, generator=g, dtype=dtype, device="cpu")
    return normalize_quaternions(q).to(device=device, dtype=dtype)


def quat_to_matrix_active(q: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternion(s) to active 3x3 rotation matrix/matrices."""
    q = normalize_quaternions(q)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    matrix = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )
    return matrix.reshape(*q.shape[:-1], 3, 3)


def right_action_quaternions(q: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Apply right action q -> q ⊗ g for a single group element g."""
    # Broadcast one symmetry operator to all batch rows.
    sym_batch = sym.view(1, 4).expand(q.shape[0], 4)
    return quat_mul(q, sym_batch)


def left_action_quaternions(q: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Apply left action q -> g ⊗ q for a single group element g."""
    sym_batch = sym.view(1, 4).expand(q.shape[0], 4)
    return quat_mul(sym_batch, q)


def to_passive_quaternions(q_active: torch.Tensor) -> torch.Tensor:
    """Convert active quaternions to passive convention via conjugation."""
    return quat_conjugate(q_active)


def _quat_min_dist_to_set(q: torch.Tensor, qset: torch.Tensor) -> float:
    """Return minimum sign-invariant chordal distance from q to any element in qset."""
    qn = normalize_quaternions(q.view(1, 4))
    s = normalize_quaternions(qset).to(device=qn.device, dtype=qn.dtype)
    # Absolute dot handles q ~ -q equivalence.
    dot = torch.abs((qn * s).sum(dim=-1))
    # For unit quaternions, sqrt(2 - 2*|dot|) is a sign-invariant chordal metric.
    d = torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=0.0))
    return float(torch.min(d).item())


def choose_group_symmetry(sym_ops: torch.Tensor, preferred_index: int = 1) -> torch.Tensor:
    """Choose a deterministic test element from discrete crystal symmetry operators."""
    if int(sym_ops.shape[0]) == 0:
        raise ValueError("Expected non-empty symmetry operators.")
    idx = int(preferred_index)
    if idx < 0 or idx >= int(sym_ops.shape[0]):
        idx = 0
    return normalize_quaternions(sym_ops[idx].view(1, 4))[0]


def choose_so3_probe_quaternion(sym_ops: torch.Tensor) -> torch.Tensor:
    """Pick a deterministic SO(3) probe rotation not too close to the discrete group."""
    dtype = sym_ops.dtype
    device = sym_ops.device
    # Fixed candidates keep tests reproducible across runs.
    candidates = [
        torch.tensor([0.913, 0.143, 0.287, 0.258], dtype=dtype, device=device),
        torch.tensor([0.801, 0.211, -0.413, 0.376], dtype=dtype, device=device),
        torch.tensor([0.692, -0.361, 0.492, -0.382], dtype=dtype, device=device),
    ]
    for cand in candidates:
        q = normalize_quaternions(cand.view(1, 4))[0]
        # Prefer a probe clearly outside G so "SO(3) vs G" checks are meaningful.
        if _quat_min_dist_to_set(q, sym_ops) > 1e-2:
            return q
    # Fallback should be rare; still deterministic.
    return normalize_quaternions(candidates[0].view(1, 4))[0]


def feature_action_matrix(
    irreps: Irreps | str,
    sym: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    variant: str,
) -> torch.Tensor:
    """Build feature-space action matrix for the given rotation and convention variant.

    Variants:
    - "D"   : row-vector features expected as x @ D
    - "D_T" : row-vector features expected as x @ D^T
    """
    irreps_obj = Irreps(irreps)
    rot = quat_to_matrix_active(sym.view(1, 4))[0]
    # Compute D on CPU for robustness, then move to target device/dtype.
    d = irreps_obj.D_from_matrix(rot.detach().cpu()).to(device=device, dtype=dtype)
    if variant == "D":
        return d
    if variant == "D_T":
        return d.T
    raise KeyError(f"Unknown feature-action variant: {variant}")


def error_metrics(expected: torch.Tensor, observed: torch.Tensor) -> tuple[float, float]:
    """Return (relative L2 error, RMS error) between tensors."""
    diff = (expected - observed).detach()
    rms = float(torch.sqrt(torch.mean(diff * diff)).item())
    rel = float(torch.linalg.norm(diff).item() / (torch.linalg.norm(observed.detach()).item() + 1e-12))
    return rel, rms


def quaternion_error_metrics(expected: torch.Tensor, observed: torch.Tensor) -> tuple[float, float]:
    """Quaternion-aware error metrics that account for q ~ -q sign ambiguity."""
    exp = normalize_quaternions(expected)
    obs = normalize_quaternions(observed)
    # Align signs before metric computation so equivalent quaternions compare as equal.
    obs_aligned = torch.where((exp * obs).sum(dim=-1, keepdim=True) < 0.0, -obs, obs)
    return error_metrics(exp, obs_aligned)


def best_feature_action_variant(
    *,
    base: torch.Tensor,
    transformed: torch.Tensor,
    irreps: Irreps | str,
    sym: torch.Tensor,
) -> tuple[str, torch.Tensor, float, float]:
    """Pick the better of D and D^T conventions by direct data fit."""
    # Try D^T first because many blocks store row-vector features.
    best_variant = "D_T"
    best_matrix = feature_action_matrix(
        irreps,
        sym,
        device=base.device,
        dtype=base.dtype,
        variant=best_variant,
    )
    best_rel, best_rms = error_metrics(base @ best_matrix, transformed)

    # Also evaluate the direct D action and keep whichever is numerically better.
    alt_variant = "D"
    alt_matrix = feature_action_matrix(
        irreps,
        sym,
        device=base.device,
        dtype=base.dtype,
        variant=alt_variant,
    )
    alt_rel, alt_rms = error_metrics(base @ alt_matrix, transformed)

    if alt_rel < best_rel or (abs(alt_rel - best_rel) < 1e-12 and alt_rms < best_rms):
        return alt_variant, alt_matrix, alt_rel, alt_rms
    return best_variant, best_matrix, best_rel, best_rms


def assert_rel_or_rms(
    expected: torch.Tensor,
    observed: torch.Tensor,
    *,
    rel_tol: float,
    rms_tol: float,
) -> None:
    """Pass if either relative or RMS error is within tolerance."""
    rel, rms = error_metrics(expected, observed)
    assert rel <= rel_tol or rms <= rms_tol, (
        f"Expected rel<={rel_tol:.2e} or rms<={rms_tol:.2e}, got rel={rel:.3e}, rms={rms:.3e}"
    )


def assert_quat_rel_or_rms(
    expected: torch.Tensor,
    observed: torch.Tensor,
    *,
    rel_tol: float,
    rms_tol: float,
) -> None:
    """Quaternion-aware version of assert_rel_or_rms."""
    rel, rms = quaternion_error_metrics(expected, observed)
    assert rel <= rel_tol or rms <= rms_tol, (
        f"Expected quat rel<={rel_tol:.2e} or rms<={rms_tol:.2e}, got rel={rel:.3e}, rms={rms:.3e}"
    )


def make_block_distance_matrix(
    block_h: int,
    block_w: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build pairwise Euclidean distances for one attention block grid."""
    ys = torch.linspace(-1.0, 1.0, int(block_h), device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, int(block_w), device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return torch.cdist(coords, coords, p=2)


def passive_right_action_for_active_left(q_passive: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Map active left action to equivalent passive right action: q -> q ⊗ g^{-1}."""
    sym_inv = quat_conjugate(sym.view(1, 4))[0]
    sym_inv_batch = sym_inv.view(1, 4).expand(q_passive.shape[0], 4)
    return quat_mul(q_passive, sym_inv_batch)
