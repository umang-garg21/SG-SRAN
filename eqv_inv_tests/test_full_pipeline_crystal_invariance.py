"""
test_full_pipeline_crystal_invariance.py
==========================================
Validates end-to-end crystal-group invariance of the SR feature loss
under BOTH left- and right-actions of the crystal group.

Mathematical properties tested
------------------------------
LEFT-action (s ⊗ q):
    feature_loss_sr( s⊗q_lr, s⊗q_hr ) ≈ feature_loss_sr( q_lr, q_hr )
    Reason: enc_a1 is left-invariant → features unchanged → loss unchanged.

RIGHT-action (q ⊗ s):
    feature_loss_sr( q_lr⊗s, q_hr⊗s ) ≈ feature_loss_sr( q_lr, q_hr )
    Reason: enc_a1 is right-EQUIVARIANT (enc(q⊗s) = D(s)·enc(q)).
            SR network is also equivariant → D(s) factors out.
            D(s) is orthogonal → MSE preserved: ‖D(s)·a − D(s)·b‖ = ‖a − b‖.

Both actions leave the scalar loss invariant, but for distinct reasons.

Run:  python eqv_inv_tests/test_full_pipeline_crystal_invariance.py
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
from models.SR_double_conv_SRattn_a1 import IsoEmbeddingSRAttn, LocalIsoCrystalEncoder

# ── config ────────────────────────────────────────────────────────────────────
LR_H, LR_W = 4, 4
SCALE       = 4
N_QUATS_LR  = LR_H * LR_W
N_QUATS_HR  = N_QUATS_LR * SCALE * SCALE
TOL         = 1e-4
DEVICE      = torch.device("cpu")
SEED        = 42


def _build_model(crystal: str, device: torch.device) -> IsoEmbeddingSRAttn:
    return IsoEmbeddingSRAttn(
        crystal=crystal,
        device=device,
        upsample_factor=SCALE,
        upsample_residual=True,
        use_lr_conv1=True,
        use_lr_conv2=True,
        use_attention=False,
        decoder_cubochoric_resolution=1,
        decoder_steps=0,
        decoder_table_cache_dir=None,
    ).eval()


def _test_loss_invariance(
    model: IsoEmbeddingSRAttn,
    q_lr: torch.Tensor,
    q_hr: torch.Tensor,
    sym_ops: torch.Tensor,
    lr_shape: tuple[int, int],
    left: bool,
) -> tuple[float, float, list[float]]:
    """
    Tests loss invariance under left (s⊗q) or right (q⊗s) crystal group action.
    Returns (mean_rel_err, max_rel_err, per_op_errors).
    """
    with torch.no_grad():
        loss_ref = float(
            model.feature_loss_sr(q_lr, q_hr, lr_shape=lr_shape, normalize_input=True).item()
        )

    errors = []
    for i in range(sym_ops.shape[0]):
        s = sym_ops[i].unsqueeze(0)
        if left:
            q_lr_sym = normalize_quaternions(quat_mul(s.expand(q_lr.shape[0], -1), q_lr))
            q_hr_sym = normalize_quaternions(quat_mul(s.expand(q_hr.shape[0], -1), q_hr))
        else:
            q_lr_sym = normalize_quaternions(quat_mul(q_lr, s.expand(q_lr.shape[0], -1)))
            q_hr_sym = normalize_quaternions(quat_mul(q_hr, s.expand(q_hr.shape[0], -1)))

        with torch.no_grad():
            loss_sym = float(
                model.feature_loss_sr(q_lr_sym, q_hr_sym, lr_shape=lr_shape, normalize_input=True).item()
            )
        errors.append(abs(loss_sym - loss_ref) / (abs(loss_ref) + 1e-12))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max()), errors


def main() -> None:
    all_results: list[bool] = []
    lr_shape = (LR_H, LR_W)

    for crystal in ("fcc", "hcp"):
        enc     = LocalIsoCrystalEncoder(crystal=crystal, dtype=torch.float32, device=DEVICE).eval()
        sym_ops = enc.sym_ops
        n_ops   = sym_ops.shape[0]

        section(
            f"Crystal={crystal.upper()}  |S|={n_ops}  "
            f"LR={LR_H}x{LR_W}  HR={LR_H*SCALE}x{LR_W*SCALE}"
        )

        model = _build_model(crystal, DEVICE)
        q_lr  = rand_quaternions(N_QUATS_LR, SEED,       DEVICE)
        q_hr  = rand_quaternions(N_QUATS_HR, SEED + 100, DEVICE)

        for action, left in [("left  s⊗q  (enc invariant)",    True),
                              ("right q⊗s  (enc equivariant)",  False)]:
            mean_e, max_e, per_op = _test_loss_invariance(
                model, q_lr, q_hr, sym_ops, lr_shape, left=left
            )
            ok = report(
                f"{crystal.upper()} feature_loss_sr {action}",
                max_e, TOL,
                extra=f"mean={mean_e:.2e}  over {n_ops} ops",
            )
            all_results.append(ok)
            if not ok:
                print(f"    Per-op: {[f'{e:.2e}' for e in per_op]}")

    summary(all_results)


if __name__ == "__main__":
    main()
