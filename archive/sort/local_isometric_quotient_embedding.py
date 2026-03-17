import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from e3nn import o3


def build_hcp_syms_mtex() -> torch.Tensor:
    """12 proper D6 rotations (HCP) as unit quaternions [w, x, y, z]."""
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sqrt3_2, -half, 0.0],
            [sqrt3_2, 0.0, 0.0, half],
            [0.0, half, -sqrt3_2, 0.0],
            [half, 0.0, 0.0, sqrt3_2],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -half, -sqrt3_2, 0.0],
            [-half, 0.0, 0.0, sqrt3_2],
            [0.0, -sqrt3_2, -half, 0.0],
            [-sqrt3_2, 0.0, 0.0, half],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )


def normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def random_quaternions(
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    q = torch.randn((n, 4), device=device, dtype=dtype)
    return normalize_quaternions(q)


def angles_from_quaternions(
    quaternions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot_mats = o3.quaternion_to_matrix(quaternions.to("cpu"))
    return o3.matrix_to_angles(rot_mats)


def reynolds_operator(
    sym_quaternions: torch.Tensor,
    l: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    alpha, beta, gamma = angles_from_quaternions(sym_quaternions)
    D = o3.wigner_D(l, alpha, beta, gamma).to(device=device, dtype=dtype)
    P = D.mean(dim=0)
    return 0.5 * (P + P.transpose(-1, -2))


def canonicalize_basis_signs(U: torch.Tensor) -> torch.Tensor:
    U = U.clone()
    for j in range(U.shape[1]):
        col = U[:, j]
        k = int(torch.argmax(col.abs()).item())
        if float(col[k].item()) < 0.0:
            U[:, j] = -col
    return U


def invariant_basis(
    sym_quaternions: torch.Tensor,
    l: int,
    device: torch.device,
    tau: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    reorthonormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    P = reynolds_operator(sym_quaternions, l=l, device=device, dtype=dtype)
    evals, evecs = torch.linalg.eigh(P)
    mask = evals > (1.0 - float(tau))
    U = evecs[:, mask]
    if U.numel() == 0:
        raise RuntimeError(f"No invariant eigenspace found for l={l}.")
    if reorthonormalize:
        U = torch.linalg.qr(U, mode="reduced").Q
    U = canonicalize_basis_signs(U)
    return U, evals


def quaternion_from_axis_angle(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-12)
    half = 0.5 * float(angle)
    return torch.cat(
        [
            torch.tensor([math.cos(half)], dtype=axis.dtype, device=axis.device),
            axis * math.sin(half),
        ],
        dim=0,
    )


def block_features(quats: torch.Tensor, l: int, U: torch.Tensor) -> torch.Tensor:
    alpha, beta, gamma = angles_from_quaternions(quats)
    D = o3.wigner_D(l, alpha, beta, gamma).to(device=U.device, dtype=U.dtype)
    Y = torch.einsum("nij,jk->nik", D, U)
    return Y.reshape(quats.shape[0], -1)


def block_gram_at_identity(
    l: int,
    U: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    device = U.device
    dtype = U.dtype

    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
    y0 = block_features(q0, l=l, U=U)[0]

    axes = torch.eye(3, dtype=dtype, device=device)
    tangents = []
    for a in range(3):
        dq = quaternion_from_axis_angle(axes[a], eps).unsqueeze(0)
        yp = block_features(dq, l=l, U=U)[0]
        tangents.append((yp - y0) / eps)

    J = torch.stack(tangents, dim=1)  # [feat_dim, 3]
    return J.transpose(0, 1) @ J


def fit_nonnegative_weights(
    gram_blocks: list[torch.Tensor],
    target_metric: torch.Tensor | None = None,
    max_steps: int = 2000,
    tol: float = 1e-12,
) -> torch.Tensor:
    if target_metric is None:
        target_metric = torch.eye(3, dtype=gram_blocks[0].dtype, device=gram_blocks[0].device)

    A = torch.stack([K.reshape(-1) for K in gram_blocks], dim=1)
    b = target_metric.reshape(-1)
    n = A.shape[1]

    c = torch.linalg.lstsq(A, b).solution[:n]
    c = torch.clamp(c, min=0.0)

    AtA = A.transpose(0, 1) @ A
    Atb = A.transpose(0, 1) @ b
    eig_max = torch.linalg.eigvalsh(AtA).max().clamp_min(1e-12)
    step = 1.0 / (2.0 * eig_max)

    for _ in range(max_steps):
        grad = 2.0 * (AtA @ c - Atb)
        c_next = torch.clamp(c - step * grad, min=0.0)
        if torch.norm(c_next - c) < tol:
            c = c_next
            break
        c = c_next

    return c


def output_whitener_from_jacobian(J: torch.Tensor, reg: float = 1e-12) -> torch.Tensor:
    """Build symmetric output-space whitener W with (WJ)^T(WJ)=I."""
    if J.ndim != 2 or J.shape[1] != 3:
        raise ValueError(f"Expected J with shape (N,3), got {tuple(J.shape)}")

    U, S, _ = torch.linalg.svd(J, full_matrices=False)  # U: (N,3)
    s_inv = 1.0 / S.clamp_min(reg)

    W = torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
    W = W + U @ torch.diag(s_inv - 1.0) @ U.transpose(0, 1)
    return W


@dataclass
class InvariantBlock:
    l: int
    U: torch.Tensor
    evals: torch.Tensor
    gram: torch.Tensor


class QuotientLocalIsometricEmbedding(nn.Module):
    """Symmetry-invariant harmonic embedding with local-isometry calibration.

    Construction follows the Reynolds projector workflow and then applies a
    constant output-space whitener so the differential metric at identity is I.
    """

    def __init__(
        self,
        sym_quaternions: torch.Tensor,
        l_values: tuple[int, ...] = (4, 6),
        tau: float = 1e-6,
        reorthonormalize: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        fit_scalar_weights_first: bool = True,
        calibrate_local_isometry: bool = True,
        fd_eps: float = 1e-6,
        whitener_reg: float = 1e-12,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.l_values = tuple(int(l) for l in l_values)
        self.tau = float(tau)
        self.fd_eps = float(fd_eps)

        syms = sym_quaternions.to(self.device, dtype=self.dtype)
        self.blocks: list[InvariantBlock] = []

        gram_blocks: list[torch.Tensor] = []
        for l in self.l_values:
            U, evals = invariant_basis(
                syms,
                l=l,
                device=self.device,
                tau=self.tau,
                dtype=self.dtype,
                reorthonormalize=reorthonormalize,
            )
            K = block_gram_at_identity(l=l, U=U, eps=self.fd_eps)
            gram_blocks.append(K)
            self.blocks.append(InvariantBlock(l=l, U=U, evals=evals, gram=K))
            self.register_buffer(f"U_l{l}", U)

        if fit_scalar_weights_first:
            c = fit_nonnegative_weights(gram_blocks=gram_blocks)
        else:
            c = torch.ones((len(gram_blocks),), dtype=self.dtype, device=self.device)
        self.register_buffer("weights", c)

        J_raw = self.jacobian_at_identity_raw(eps=self.fd_eps)
        self.register_buffer("J_raw", J_raw)

        if calibrate_local_isometry:
            W = output_whitener_from_jacobian(J_raw, reg=whitener_reg)
        else:
            W = torch.eye(J_raw.shape[0], dtype=self.dtype, device=self.device)
        self.register_buffer("output_whitener", W)

        J_cal = W @ J_raw
        self.register_buffer("J_calibrated", J_cal)

    def raw_features(self, quats: torch.Tensor) -> torch.Tensor:
        q = normalize_quaternions(quats.to(self.device, dtype=self.dtype))
        out = []
        for i, block in enumerate(self.blocks):
            U = getattr(self, f"U_l{block.l}")
            y = block_features(q, l=block.l, U=U)
            out.append(torch.sqrt(self.weights[i].clamp_min(0.0)) * y)
        return torch.cat(out, dim=-1)

    def forward(self, quats: torch.Tensor) -> torch.Tensor:
        z_raw = self.raw_features(quats)
        return z_raw @ self.output_whitener.transpose(0, 1)

    def predicted_metric_raw(self) -> torch.Tensor:
        G = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        for i, block in enumerate(self.blocks):
            G = G + self.weights[i].clamp_min(0.0) * block.gram
        return G

    def predicted_metric(self) -> torch.Tensor:
        J = self.J_calibrated
        return J.transpose(0, 1) @ J

    def jacobian_at_identity_raw(self, eps: float = 1e-6) -> torch.Tensor:
        q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=self.dtype, device=self.device)
        z0 = self.raw_features(q0)[0]

        axes = torch.eye(3, dtype=self.dtype, device=self.device)
        tangents = []
        for a in range(3):
            dq = quaternion_from_axis_angle(axes[a], eps).unsqueeze(0)
            za = self.raw_features(dq)[0]
            tangents.append((za - z0) / eps)
        return torch.stack(tangents, dim=1)  # [N, 3]


@torch.no_grad()
def metric_from_embedding(
    emb: QuotientLocalIsometricEmbedding,
    eps: float = 1e-6,
) -> torch.Tensor:
    dtype = emb.dtype
    device = emb.device
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
    z0 = emb(q0)[0]

    axes = torch.eye(3, dtype=dtype, device=device)
    tangents = []
    for a in range(3):
        dq = quaternion_from_axis_angle(axes[a], eps).unsqueeze(0)
        za = emb(dq)[0]
        tangents.append((za - z0) / eps)
    J = torch.stack(tangents, dim=1)
    return J.transpose(0, 1) @ J


@torch.no_grad()
def invariance_check(
    emb: QuotientLocalIsometricEmbedding,
    sym_quaternions: torch.Tensor,
    n_samples: int = 256,
) -> tuple[float, float]:
    syms = normalize_quaternions(sym_quaternions.to(emb.device, dtype=emb.dtype))
    q = random_quaternions(n_samples, device=emb.device, dtype=emb.dtype)
    idx = torch.randint(0, syms.shape[0], (n_samples,), device=emb.device)
    g = syms[idx]

    # Quotient action is right multiplication: q ~ q ⊗ g.
    gq = normalize_quaternions(quat_mul(q, g))

    z = emb(q)
    zg = emb(gq)
    err = torch.norm(zg - z, dim=-1)
    return float(err.mean().item()), float(err.max().item())


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # HCP quotient symmetry set used by the Reynolds operator.
    syms = build_hcp_syms_mtex().to(device=device, dtype=dtype)
    # Build quotient embedding with local-isometry calibration enabled.
    # For HCP, expected invariant dimensions are: dim(I_4)=1, dim(I_6)=2.
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
        if block.l == 6 and dim != 2:
            print(f"  WARNING: expected invariant_dim=2 at L=6, got {dim}")

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
