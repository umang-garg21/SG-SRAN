"""
eqv_inv_tests/_helpers.py
=========================
Shared utilities for all equivariance / invariance test scripts.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── quaternion helpers ────────────────────────────────────────────────────────

def normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternions to unit length; canonical form has w ≥ 0."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a batch of quaternions (w, x, y, z) → (w, -x, -y, -z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2.  Both (..., 4) in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Active rotation matrix from a unit quaternion (w, x, y, z).
    Input: (..., 4), Output: (..., 3, 3).
    """
    q = normalize_quaternions(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def rand_quaternions(n: int, seed: int, device: torch.device) -> torch.Tensor:
    """n uniform random unit quaternions."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    q = torch.randn(n, 4, generator=g, device=device, dtype=torch.float32)
    return normalize_quaternions(q)


# ── feature-space rotation helpers ───────────────────────────────────────────

def rotate_features(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Apply Wigner-D matrix D (dim, dim) to feature vectors in x (..., dim).
    Treats features as row vectors: x_rot = x @ D.T
    """
    return x @ D.T


# ── error metrics ─────────────────────────────────────────────────────────────

def rel_error(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> float:
    """Relative Frobenius error ||pred - ref|| / ||ref||."""
    diff = (pred - ref).detach().float()
    ref_f = ref.detach().float()
    return float(torch.linalg.norm(diff) / (torch.linalg.norm(ref_f) + eps))


def abs_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean absolute error."""
    return float((pred - ref).detach().float().abs().mean())


# ── reporting ─────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def report(label: str, err: float, tol: float, *, extra: str = "") -> bool:
    status = PASS if err < tol else FAIL
    extra_str = f"  ({extra})" if extra else ""
    print(f"  [{status}] {label:<55s}  rel={err:.2e}  tol={tol:.0e}{extra_str}")
    return err < tol


def section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def summary(results: list[bool]) -> None:
    n = len(results)
    passed = sum(results)
    status = PASS if passed == n else FAIL
    print(f"\n{'='*70}")
    print(f"  [{status}]  {passed}/{n} tests passed")
    print(f"{'='*70}\n")
