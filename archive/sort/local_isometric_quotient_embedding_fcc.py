import math

import torch

# Reuse the generic quotient-embedding machinery; this script only supplies FCC symmetries
# and prints FCC-specific diagnostics.
try:
    from scripts.local_isometric_quotient_embedding import (
        QuotientLocalIsometricEmbedding,
        invariance_check,
        metric_from_embedding,
    )
except ImportError:
    from local_isometric_quotient_embedding import (  # type: ignore
        QuotientLocalIsometricEmbedding,
        invariance_check,
        metric_from_embedding,
    )


def build_fcc_syms() -> torch.Tensor:
    """24 proper cubic rotations (group O) as unit quaternions [w, x, y, z]."""
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [inv_sqrt_2, inv_sqrt_2, 0.0, 0.0],
            [inv_sqrt_2, 0.0, inv_sqrt_2, 0.0],
            [inv_sqrt_2, 0.0, 0.0, inv_sqrt_2],
            [inv_sqrt_2, -inv_sqrt_2, 0.0, 0.0],
            [inv_sqrt_2, 0.0, -inv_sqrt_2, 0.0],
            [inv_sqrt_2, 0.0, 0.0, -inv_sqrt_2],
            [0.0, inv_sqrt_2, inv_sqrt_2, 0.0],
            [0.0, inv_sqrt_2, 0.0, inv_sqrt_2],
            [0.0, 0.0, inv_sqrt_2, inv_sqrt_2],
            [0.0, inv_sqrt_2, -inv_sqrt_2, 0.0],
            [0.0, 0.0, inv_sqrt_2, -inv_sqrt_2],
            [0.0, inv_sqrt_2, 0.0, -inv_sqrt_2],
            [half, half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, -half],
        ],
        dtype=torch.float64,
    )


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # FCC quotient symmetry set used by the Reynolds operator.
    syms = build_fcc_syms().to(device=device, dtype=dtype)

    # Build quotient embedding with local-isometry calibration enabled.
    # For FCC, l=4 and l=6 each have a 1D invariant block.
    emb = QuotientLocalIsometricEmbedding(
        syms,
        l_values=(4, 6),
        tau=1e-6,
        reorthonormalize=True,
        device=device,
        dtype=dtype,
        fit_scalar_weights_first=True,
        calibrate_local_isometry=True,
        fd_eps=1e-6,
    )

    print(f"device={device}, dtype={dtype}")
    for i, block in enumerate(emb.blocks):
        top = float(block.evals[-1].item())
        dim = int(block.U.shape[1])
        print(
            f"L={block.l}: top_eval={top:.6f}, invariant_dim={dim}, "
            f"weight={float(emb.weights[i]):.6f}"
        )
        if block.l in (4, 6) and dim != 1:
            print(f"  WARNING: expected invariant_dim=1 at L={block.l}, got {dim}")

    I = torch.eye(3, dtype=dtype, device=emb.device)

    # Raw pullback metric before the output-space whitener.
    G_raw = emb.predicted_metric_raw()
    print("\nRaw metric (before whitener):\n", G_raw)
    print("||G_raw-I||_F:", float(torch.linalg.norm(G_raw - I).item()))

    # Calibrated metric after whitener: should be ~identity at the base point.
    G_model = emb.predicted_metric()
    G_fd = metric_from_embedding(emb, eps=1e-6)
    print("\nCalibrated metric from model Jacobian:\n", G_model)
    print("||G_model-I||_F:", float(torch.linalg.norm(G_model - I).item()))
    print("\nCalibrated finite-difference metric at identity:\n", G_fd)
    print("||G_fd-I||_F:", float(torch.linalg.norm(G_fd - I).item()))

    # Quotient invariance check under right action q ~ q \otimes g.
    mean_err, max_err = invariance_check(emb, syms, n_samples=512)
    print("\nInvariance check ||Phi(g*q)-Phi(q)||")
    print(f"mean={mean_err:.3e}, max={max_err:.3e}")


if __name__ == "__main__":
    main()
