"""
test_encoder_crystal_invariance.py
====================================
Validates right-crystal-group invariance of LocalIsoCrystalEncoder.

Mathematical property tested
----------------------------
For all symmetry operators s ∈ crystal group S (stored in encoder.sym_ops):

    enc_a1( quat_mul(s, q) )  ≈  enc_a1( q )   ← LEFT-action

This corresponds to how reduce_to_fz uses sym_ops_inv ⊗ q (left-multiplication)
to enumerate the orbit.  Both left- and right-actions are tested; the one that
gives invariance is reported as PASS, the other is informational.

Crystal groups tested
---------------------
  - FCC (O):   24 symmetry operators
  - HCP (D6h): 12 symmetry operators

Both forward_a1 and forward_full are tested.

Run:  python eqv_inv_tests/test_encoder_crystal_invariance.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from _helpers import (
    normalize_quaternions, quat_mul, rand_quaternions,
    rel_error, report, section, summary,
)
from models.SR_double_conv_SRattn_a1 import LocalIsoCrystalEncoder

# ── config ────────────────────────────────────────────────────────────────────
N_QUATS = 64          # number of random orientations to test
TOL_A1  = 1e-4        # A1 features must be invariant
TOL_FULL= 1e-4        # full irreps — same tolerance (informational if fails)
DEVICE  = torch.device("cpu")
SEED    = 42


def _test_invariance(
    enc: LocalIsoCrystalEncoder,
    q: torch.Tensor,
    sym_ops: torch.Tensor,
    forward_fn,
    label: str,
    tol: float,
    left: bool = True,
) -> list[bool]:
    """
    For each s in sym_ops compute the orbit element and compare enc output.
    left=True:  s ⊗ q  (left-action  — matches how reduce_to_fz uses sym_ops)
    left=False: q ⊗ s  (right-action — crystal frame rotation)
    Returns one bool per sym op.
    """
    results = []
    with torch.no_grad():
        feat_ref = forward_fn(q)                            # (N, C)

    n_ops = sym_ops.shape[0]
    max_errs = []

    for i in range(n_ops):
        s = sym_ops[i].unsqueeze(0).expand(q.shape[0], -1)  # (N, 4)
        if left:
            q_sym = normalize_quaternions(quat_mul(s, q))    # left-action
        else:
            q_sym = normalize_quaternions(quat_mul(q, s))    # right-action
        with torch.no_grad():
            feat_sym = forward_fn(q_sym)

        err = rel_error(feat_sym, feat_ref)
        max_errs.append(err)

    overall_max = max(max_errs)
    worst_op    = int(torch.tensor(max_errs).argmax().item())
    ok = report(
        f"{label}  (all {n_ops} ops, max over ops)",
        overall_max, tol,
        extra=f"worst_op={worst_op}  err={max_errs[worst_op]:.2e}",
    )
    results.append(ok)

    # per-op detail for failures
    if not ok:
        print(f"    Per-op errors:")
        for i, e in enumerate(max_errs):
            flag = " ← FAIL" if e >= tol else ""
            print(f"      op {i:02d}: {e:.2e}{flag}")

    return results


def main() -> None:
    all_results: list[bool] = []

    for crystal in ("fcc", "hcp"):
        enc = LocalIsoCrystalEncoder(
            crystal=crystal, dtype=torch.float32, device=DEVICE
        ).eval()

        sym_ops = enc.sym_ops                               # (n_ops, 4) quaternions
        n_ops   = sym_ops.shape[0]

        section(
            f"Crystal={crystal.upper()}  |S|={n_ops}  "
            f"irreps_a1={enc.irreps_a1}  irreps_full={enc.irreps_full}"
        )

        q = rand_quaternions(N_QUATS, SEED, DEVICE)

        for fn_name, fn, tol in [
            ("forward_a1   (must be invariant)", enc.forward_a1,   TOL_A1),
            ("forward_full (informational)",      enc.forward_full, TOL_FULL),
        ]:
            print(f"  {fn_name}:")
            # left-action: s ⊗ q  (matches reduce_to_fz convention)
            res = _test_invariance(
                enc, q, sym_ops, fn,
                label=f"{crystal.upper()} {fn_name} left-action",
                tol=tol, left=True,
            )
            all_results.extend(res)
            # right-action: q ⊗ s  (informational — may not give invariance)
            _test_invariance(
                enc, q, sym_ops, fn,
                label=f"{crystal.upper()} {fn_name} right-action (info)",
                tol=tol, left=False,
            )

    summary(all_results)


if __name__ == "__main__":
    main()
