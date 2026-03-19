"""
test_full_pipeline_crystal_invariance.py
==========================================
Validates end-to-end crystal-group invariance of the SR feature loss.

Mathematical property tested
----------------------------
For all s ∈ crystal group S (FCC=24, HCP=12):

    feature_loss_sr( quat_mul(s, q_lr), quat_mul(s, q_hr), lr_shape )
        ≈  feature_loss_sr( q_lr, q_hr, lr_shape )

Uses LEFT-action (s ⊗ q), consistent with enc_a1 LEFT-invariance confirmed
by test_encoder_crystal_invariance.py.

This holds because:
  1. enc_a1 is left-crystal-group invariant  →  same LR/HR features
  2. D is orthogonal  →  MSE is preserved under equivariant feature rotation
  3. The spatial layers are equivariant  →  SR features rotate consistently
  4. The MSE between two equivariantly-rotated tensors equals the original MSE

The test uses decoder_steps=0 to skip the expensive quaternion optimiser
(the feature_loss_sr does not use the decoder).

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
LR_H, LR_W  = 4, 4
SCALE        = 4
N_QUATS_LR  = LR_H * LR_W
N_QUATS_HR  = N_QUATS_LR * SCALE * SCALE
TOL          = 1e-4
DEVICE       = torch.device("cpu")
SEED         = 42


def _build_model(crystal: str, device: torch.device) -> IsoEmbeddingSRAttn:
    return IsoEmbeddingSRAttn(
        crystal=crystal,
        device=device,
        upsample_factor=SCALE,
        upsample_residual=True,
        use_lr_conv1=True,
        use_lr_conv2=True,
        use_attention=False,            # skip attention for speed
        decoder_cubochoric_resolution=1,
        decoder_steps=0,               # skip decoder optimisation
        decoder_table_cache_dir=None,
    ).eval()


def _test_loss_crystal_invariance(
    model: IsoEmbeddingSRAttn,
    q_lr: torch.Tensor,
    q_hr: torch.Tensor,
    sym_ops: torch.Tensor,
    lr_shape: tuple[int, int],
) -> tuple[float, float, list[float]]:
    """
    For each s ∈ sym_ops: apply right-action to both LR and HR quaternions,
    compute feature_loss_sr, compare to original loss.
    Returns (mean_rel_err, max_rel_err, per_op_errors).
    """
    with torch.no_grad():
        loss_ref = float(
            model.feature_loss_sr(q_lr, q_hr, lr_shape=lr_shape, normalize_input=True)
            .item()
        )

    errors = []
    n_ops = sym_ops.shape[0]

    for i in range(n_ops):
        s = sym_ops[i].unsqueeze(0)

        q_lr_sym = normalize_quaternions(
            quat_mul(s.expand(q_lr.shape[0], -1), q_lr)   # left-action
        )
        q_hr_sym = normalize_quaternions(
            quat_mul(s.expand(q_hr.shape[0], -1), q_hr)   # left-action
        )

        with torch.no_grad():
            loss_sym = float(
                model.feature_loss_sr(
                    q_lr_sym, q_hr_sym, lr_shape=lr_shape, normalize_input=True
                ).item()
            )

        # relative difference in scalar loss values
        err = abs(loss_sym - loss_ref) / (abs(loss_ref) + 1e-12)
        errors.append(err)

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

        torch.manual_seed(SEED)
        q_lr = rand_quaternions(N_QUATS_LR, SEED,       DEVICE)
        q_hr = rand_quaternions(N_QUATS_HR, SEED + 100, DEVICE)

        mean_e, max_e, per_op = _test_loss_crystal_invariance(
            model, q_lr, q_hr, sym_ops, lr_shape
        )
        ok = report(
            f"{crystal.upper()} feature_loss_sr right-crystal-group invariance",
            max_e, TOL,
            extra=f"mean={mean_e:.2e}  ref_loss≠0  over {n_ops} ops",
        )
        all_results.append(ok)

        if not ok:
            print(f"    Per-op errors: {[f'{e:.2e}' for e in per_op]}")

    summary(all_results)


if __name__ == "__main__":
    main()
