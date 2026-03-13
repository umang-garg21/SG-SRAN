import math
import torch
from e3nn import o3

# ============================================================
# Quaternion utilities (w,x,y,z) convention
# ============================================================
def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate / inverse for unit quaternions. q (...,4) [w,x,y,z]."""
    out = q.clone()
    out[..., 1:] *= -1
    return out


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product. a,b (...,4) [w,x,y,z]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def random_unit_quats(n: int, device: str) -> torch.Tensor:
    q = torch.randn(n, 4, device=device)
    return quat_normalize(q)


# ============================================================
# e3nn Wigner-D (CUDA safe)
# ============================================================
def wigner_D_cuda(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """CUDA-compatible wrapper for e3nn's wigner_D function (real basis)."""
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device

    alpha = alpha[..., None, None] % (2 * math.pi)
    beta  = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    X = o3._wigner.so3_generators(l).to(device)
    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


def D_from_quats(l: int, q_wxyz: torch.Tensor) -> torch.Tensor:
    """q_wxyz (...,4) -> D_l(..., 2l+1, 2l+1)"""
    R = o3.quaternion_to_matrix(q_wxyz)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    return wigner_D_cuda(l, alpha, beta, gamma)


# ============================================================
# Cubic symmetry operators (24) as quaternions [w,x,y,z]
# These should represent proper rotations of Oh/O.
# ============================================================
def fcc_syms_wxyz(device: str) -> torch.Tensor:
    inv_sqrt_2 = 1 / math.sqrt(2)
    half = 0.5
    syms = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [0, -inv_sqrt_2, -inv_sqrt_2, 0],
            [0, -inv_sqrt_2, 0, -inv_sqrt_2],
            [0, 0, -inv_sqrt_2, -inv_sqrt_2],
            [0, -inv_sqrt_2, inv_sqrt_2, 0],
            [0, 0, -inv_sqrt_2, inv_sqrt_2],
            [0, -inv_sqrt_2, 0, inv_sqrt_2],
            [half, -half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, half],
        ],
        dtype=torch.float32,
        device=device,
    )
    return quat_normalize(syms)


# ============================================================
# Build cubic A1 vector in l-space via projector P = mean_s D_l(s)
# ============================================================
def cubic_A1_vector(l: int, syms_active_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Returns c_l in R^(2l+1) spanning the O-invariant (A1) subspace inside the l irrep,
    computed as the top eigenvector of P = (1/|O|) sum_s D_l(s).

    IMPORTANT: syms_active_wxyz must be in the SAME "active" convention you feed to e3nn.
    """
    D = D_from_quats(l, syms_active_wxyz)   # (24, d, d)
    P = D.mean(dim=0)                       # (d, d)

    # P should be ~ a projector; take principal eigenvector
    evals, evecs = torch.linalg.eigh(P)
    c = evecs[:, -1]
    c = c / (c.norm() + 1e-12)

    # deterministic sign
    if c[0] < 0:
        c = -c
    return c


# ============================================================
# Invariant scalar feature: I_l(q) = c^T D_l(q) c
# ============================================================
def cubic_invariant_scalar(l: int, q_active_wxyz: torch.Tensor, c_l: torch.Tensor) -> torch.Tensor:
    D = D_from_quats(l, q_active_wxyz)              # (..., d, d)
    I = torch.einsum("...ij,i,j->...", D, c_l, c_l) # (...)
    return I


# ============================================================
# Tests
# ============================================================
def test_passive_to_active_transpose(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    If q_b is passive, then q_a = conj(q_b) should satisfy:
      R(q_a) ≈ R(q_b)^T
    """
    q_b = random_unit_quats(256, device)
    q_a = quat_conj(q_b)

    Rb = o3.quaternion_to_matrix(q_b)
    Ra = o3.quaternion_to_matrix(q_a)

    err = (Ra - Rb.transpose(-1, -2)).abs().max().item()
    print(f"[PASSIVE->ACTIVE] max |Ra - Rb^T| = {err:.3e}")
    # This should be tiny if your "Bunge is passive" assumption matches.
    # If it's not tiny, your q_b is probably already active in practice.
    return err


def test_cubic_invariance_active_right_action(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Active convention + crystal symmetry typically acts as right-multiplication:
      q -> q ⊗ s

    We test invariance:
      I_l(q) == I_l(q ⊗ s)
    """
    syms = fcc_syms_wxyz(device)

    # Build cubic A1 vectors
    c4 = cubic_A1_vector(4, syms)
    c6 = cubic_A1_vector(6, syms)

    print(f"[A1 basis] l=4 principal eigenvalue approx: "
          f"{torch.linalg.eigvalsh(D_from_quats(4, syms).mean(0))[-1].item():.6f}")
    print(f"[A1 basis] l=6 principal eigenvalue approx: "
          f"{torch.linalg.eigvalsh(D_from_quats(6, syms).mean(0))[-1].item():.6f}")

    # Random orientations (assume you will feed ACTIVE to e3nn)
    q = random_unit_quats(2048, device)

    # Apply random cubic symmetry on the RIGHT (active convention)
    idx = torch.randint(0, syms.shape[0], (q.shape[0],), device=device)
    s = syms[idx]
    q2 = quat_mul(q, s)

    I4  = cubic_invariant_scalar(4, q,  c4)
    I4b = cubic_invariant_scalar(4, q2, c4)
    I6  = cubic_invariant_scalar(6, q,  c6)
    I6b = cubic_invariant_scalar(6, q2, c6)

    e4 = (I4 - I4b).abs().max().item()
    e6 = (I6 - I6b).abs().max().item()
    print(f"[INVARIANCE active RIGHT] max |I4(q)-I4(q⊗s)| = {e4:.3e}")
    print(f"[INVARIANCE active RIGHT] max |I6(q)-I6(q⊗s)| = {e6:.3e}")

    # Loose tolerances: matrix_exp introduces some numeric noise on GPU
    assert e4 < 5e-4, "l=4 invariant failed (check convention / sym ops)"
    assert e6 < 5e-4, "l=6 invariant failed (check convention / sym ops)"


def test_cubic_invariance_passive_left_action(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    If you insist on staying in passive convention, a common action is left-multiply by inverse:
      q_b -> s^{-1} ⊗ q_b = conj(s) ⊗ q_b

    We can test invariance in passive by converting to active before feature eval:
      q_a = conj(q_b)
      then q_b' = conj(s) ⊗ q_b  corresponds to q_a' = q_a ⊗ s
    """
    syms = fcc_syms_wxyz(device)
    c4 = cubic_A1_vector(4, syms)
    c6 = cubic_A1_vector(6, syms)

    q_b = random_unit_quats(2048, device)  # pretend these are Bunge/passive
    q_a = quat_conj(q_b)                   # convert to active for e3nn

    idx = torch.randint(0, 24, (q_b.shape[0],), device=device)
    s = syms[idx]

    # passive action: q_b' = s^{-1} ⊗ q_b
    q_b2 = quat_mul(quat_conj(s), q_b)
    q_a2 = quat_conj(q_b2)  # should equal q_a ⊗ s

    # invariants computed in active space
    I4  = cubic_invariant_scalar(4, q_a,  c4)
    I4b = cubic_invariant_scalar(4, q_a2, c4)
    I6  = cubic_invariant_scalar(6, q_a,  c6)
    I6b = cubic_invariant_scalar(6, q_a2, c6)

    e4 = (I4 - I4b).abs().max().item()
    e6 = (I6 - I6b).abs().max().item()
    print(f"[INVARIANCE passive LEFT via active] max |I4-I4b| = {e4:.3e}")
    print(f"[INVARIANCE passive LEFT via active] max |I6-I6b| = {e6:.3e}")

    assert e4 < 5e-4, "passive->active invariance failed (check action order/inverses)"
    assert e6 < 5e-4, "passive->active invariance failed (check action order/inverses)"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print("Device:", device)

    # 1) Check whether your "Bunge quats are passive" assumption matches Ra ≈ Rb^T
    test_passive_to_active_transpose(device)

    # 2) Invariance test using ACTIVE convention with RIGHT action q -> q ⊗ s
    test_cubic_invariance_active_right_action(device)

    # 3) Invariance test for PASSIVE left action q_b -> s^{-1} ⊗ q_b,
    #    but computed by converting to active before feeding e3nn
    test_cubic_invariance_passive_left_action(device)

    print("All tests passed.")