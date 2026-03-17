import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3


# ============================================================
# Quaternion / rotation helpers
# ============================================================

def normalize_quaternion_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize scalar-first quaternions [w, x, y, z].
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_wxyz_to_matrix_active(q: torch.Tensor) -> torch.Tensor:
    """
    Scalar-first quaternion [w, x, y, z] -> active rotation matrix.

    q rotates vectors by:
        v' = R(q) v
    """
    q = normalize_quaternion_wxyz(q)
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

    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return R


def quat_wxyz_passive_to_matrix_active(q_passive: torch.Tensor) -> torch.Tensor:
    """
    Convert passive/Bunge-style scalar-first quaternion [w, x, y, z]
    to an active rotation matrix by conjugating the quaternion first.

    For unit quaternions:
        q_active = conj(q_passive) = [w, -x, -y, -z]
    """
    q_passive = normalize_quaternion_wxyz(q_passive)
    q_active = q_passive.clone()
    q_active[..., 1:] *= -1.0
    return quat_wxyz_to_matrix_active(q_active)


def random_rotation_matrices(
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Draw random rotation matrices and move to requested device/dtype.
    """
    R = o3.rand_matrix(n)
    if device is not None or dtype is not None:
        R = R.to(device=device, dtype=dtype)
    return R


def make_proper_cubic_group(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build the 24 proper cubic rotation matrices as signed permutation matrices.

    Returns:
        S: (24, 3, 3)
    """
    import itertools

    if dtype is None:
        dtype = torch.get_default_dtype()

    eye = torch.eye(3, device=device, dtype=dtype)
    mats = []

    for perm in itertools.permutations(range(3)):
        P = eye[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            D = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
            M = D @ P
            det = torch.det(M).round().item()
            if det > 0:
                mats.append(M)

    S = torch.stack(mats, dim=0)

    # Robust dedup
    flat = S.reshape(S.shape[0], -1)
    keep = []
    used = torch.zeros(S.shape[0], dtype=torch.bool, device=S.device)
    for i in range(S.shape[0]):
        if used[i]:
            continue
        keep.append(i)
        same = torch.isclose(flat[i:i+1], flat, atol=1e-8, rtol=1e-8).all(dim=-1)
        used |= same

    S = S[keep]

    if S.shape[0] != 24:
        raise RuntimeError(f"Expected 24 proper cubic operators, got {S.shape[0]}")
    return S


# ============================================================
# CUDA-safe Wigner-D helpers
# ============================================================

def wigner_D_device_safe(
    l: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-safe replacement for e3nn's wigner_D path.
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    dtype = alpha.dtype

    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    X = o3._wigner.so3_generators(l).to(device=device, dtype=dtype)

    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


def irrep_D_from_matrix_device_safe(irrep: o3.Irrep, R: torch.Tensor) -> torch.Tensor:
    """
    CUDA-safe replacement for irrep.D_from_matrix(R) for proper rotations.
    """
    alpha, beta, gamma = o3.matrix_to_angles(R)
    return wigner_D_device_safe(irrep.l, alpha, beta, gamma)


def irreps_2e_4e_D_from_matrix_device_safe(R: torch.Tensor) -> torch.Tensor:
    """
    Block diagonal representation for 2e + 4e.

    Returns:
        (..., 14, 14)
    """
    D2 = irrep_D_from_matrix_device_safe(o3.Irrep("2e"), R)  # (..., 5, 5)
    D4 = irrep_D_from_matrix_device_safe(o3.Irrep("4e"), R)  # (..., 9, 9)

    out = torch.zeros(*R.shape[:-2], 14, 14, device=R.device, dtype=R.dtype)
    out[..., :5, :5] = D2
    out[..., 5:, 5:] = D4
    return out


# ============================================================
# Small utilities
# ============================================================

def normalize_features(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize feature vectors along the last dimension.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def default_test_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    """
    Sensible tolerances for tests.
    """
    if dtype == torch.float64:
        return 1e-10, 1e-10
    return 5e-5, 5e-5


# ============================================================
# Output container
# ============================================================

@dataclass
class EmbeddingOutput:
    l4_axes: torch.Tensor      # (..., 9)
    l4_seed: torch.Tensor      # (..., 9)
    full15: torch.Tensor       # (..., 15) = 0e + 2e + 4e
    centered14: torch.Tensor   # (..., 14) = 2e + 4e
    scalar0: torch.Tensor      # (..., 1)
    quad2: torch.Tensor        # (..., 5)
    quartic4: torch.Tensor     # (..., 9)


# ============================================================
# Main embedding module
# ============================================================

class CubicOrientationEmbedding(nn.Module):
    """
    Cubic orientation embedding for SO(3)/O using e3nn.

    Outputs:
      - l4_axes:    compact 9D cubic L=4 descriptor via spherical harmonics
      - l4_seed:    same 9D descriptor via D^4(R) acting on a fixed cubic seed
      - full15:     quartic decomposition into 0e + 2e + 4e
      - centered14: remove scalar block, leaving 2e + 4e (14D)

    Notes:
      - right cubic invariance corresponds to quotienting by O on the right
      - centered14 is the paper-style centered quartic descriptor
      - l4_axes is the compact harmonic descriptor
    """

    def __init__(
        self,
        sh_normalization: Literal["integral", "component", "norm"] = "integral",
        apply_paper_beta_to_full: bool = True,
        use_wigner_seed_transport: bool = True,
    ) -> None:
        super().__init__()

        self.sh_normalization = sh_normalization
        self.apply_paper_beta_to_full = apply_paper_beta_to_full
        self.use_wigner_seed_transport = use_wigner_seed_transport

        # Hielscher–Lippert cubic beta
        self.paper_beta = 3.0 / (2.0 * math.sqrt(2.0))

        axes = torch.eye(3, dtype=torch.get_default_dtype())
        self.register_buffer("axes", axes, persistent=False)

        cubic_ops = make_proper_cubic_group(dtype=torch.get_default_dtype())
        self.register_buffer("cubic_ops", cubic_ops, persistent=False)

        # Fully symmetric rank-4 tensor -> 0e + 2e + 4e
        self.rtp4 = o3.ReducedTensorProducts("ijkl=jikl=ikjl=ijlk", i="1e")

        # Fixed cubic seed in the l=4 space
        with torch.no_grad():
            seed_l4 = o3.spherical_harmonics(
                4,
                axes,
                normalize=True,
                normalization=sh_normalization,
            ).sum(dim=0)
            seed_l4 = seed_l4 / seed_l4.norm().clamp_min(1e-12)

        self.register_buffer("seed_l4", seed_l4, persistent=False)

        self.irrep_l4 = o3.Irrep("4e")
        self.irreps_centered14 = o3.Irreps("1x2e + 1x4e")

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _rotated_axes(self, R: torch.Tensor) -> torch.Tensor:
        """
        U[..., i, :] = R e_i
        shape (..., 3, 3)
        """
        return torch.einsum("...ij,nj->...ni", R, self.axes)

    def _l4_from_axes(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compact cubic descriptor:
            c4(R) = sum_i Y^4(R e_i)
        """
        y4 = o3.spherical_harmonics(
            4,
            U.reshape(-1, 3),
            normalize=True,
            normalization=self.sh_normalization,
        )
        y4 = y4.reshape(*U.shape[:-2], 3, 9)
        return y4.sum(dim=-2)

    def _l4_from_seed(self, R: torch.Tensor) -> torch.Tensor:
        """
        Same cubic descriptor transported by Wigner D:
            c4(R) = D^4(R) s4
        """
        if not self.use_wigner_seed_transport:
            U = self._rotated_axes(R)
            return self._l4_from_axes(U)

        D4 = irrep_D_from_matrix_device_safe(self.irrep_l4, R)
        return torch.einsum("...ij,j->...i", D4, self.seed_l4)

    def _full_rank4_irreps(self, U: torch.Tensor) -> torch.Tensor:
        """
        Build sum_i (R e_i)^{⊗4} and reduce to irreps:
            0e + 2e + 4e = 1 + 5 + 9 = 15 dims

        If apply_paper_beta_to_full=True, multiply by (beta / 3) so this
        matches the paper's orbit-averaged cubic quartic normalization.
        """
        coeffs = 0.0
        for i in range(3):
            ui = U[..., i, :]
            coeffs = coeffs + self.rtp4(ui, ui, ui, ui)

        if self.apply_paper_beta_to_full:
            coeffs = (self.paper_beta / 3.0) * coeffs

        return coeffs

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def forward_matrix(self, R: torch.Tensor) -> EmbeddingOutput:
        """
        R: (..., 3, 3) active rotation matrices
        """
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        U = self._rotated_axes(R)

        l4_axes = self._l4_from_axes(U)
        l4_seed = self._l4_from_seed(R)
        full15 = self._full_rank4_irreps(U)

        # e3nn docs show 0e + 2e + 4e for symmetric rank-4
        scalar0 = full15[..., :1]
        quad2 = full15[..., 1:6]
        quartic4 = full15[..., 6:15]
        centered14 = full15[..., 1:]

        return EmbeddingOutput(
            l4_axes=l4_axes,
            l4_seed=l4_seed,
            full15=full15,
            centered14=centered14,
            scalar0=scalar0,
            quad2=quad2,
            quartic4=quartic4,
        )

    def forward_quat_active_wxyz(self, q_active_wxyz: torch.Tensor) -> EmbeddingOutput:
        R = quat_wxyz_to_matrix_active(q_active_wxyz)
        return self.forward_matrix(R)

    def forward_quat_passive_wxyz(self, q_passive_wxyz: torch.Tensor) -> EmbeddingOutput:
        R = quat_wxyz_passive_to_matrix_active(q_passive_wxyz)
        return self.forward_matrix(R)


# ============================================================
# Tests
# ============================================================

@torch.no_grad()
def test_right_cubic_invariance(
    model: CubicOrientationEmbedding,
    n: int = 16,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Check quotient invariance:
        E(RS) = E(R) for all S in O
    """
    R = random_rotation_matrices(n, device=device, dtype=dtype)

    if atol is None or rtol is None:
        atol, rtol = default_test_tolerances(R.dtype)

    out = model.forward_matrix(R)

    S = model.cubic_ops.to(device=R.device, dtype=R.dtype)   # (24, 3, 3)
    RS = torch.einsum("nij,sjk->nsik", R, S)                 # (n, 24, 3, 3)
    out_RS = model.forward_matrix(RS.reshape(-1, 3, 3))

    l4_ref = out.l4_axes[:, None, :].expand(n, S.shape[0], 9).reshape(-1, 9)
    c14_ref = out.centered14[:, None, :].expand(n, S.shape[0], 14).reshape(-1, 14)

    err_l4 = (out_RS.l4_axes - l4_ref).abs().max().item()
    err_14 = (out_RS.centered14 - c14_ref).abs().max().item()

    return {
        "ok_l4": torch.allclose(out_RS.l4_axes, l4_ref, atol=atol, rtol=rtol),
        "ok_centered14": torch.allclose(out_RS.centered14, c14_ref, atol=atol, rtol=rtol),
        "maxerr_l4": err_l4,
        "maxerr_centered14": err_14,
        "atol": atol,
        "rtol": rtol,
    }


@torch.no_grad()
def test_left_equivariance(
    model: CubicOrientationEmbedding,
    n: int = 16,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Check left SO(3)-equivariance:
        c4(QR) = D^4(Q) c4(R)
        c14(QR) = (D^2(Q) ⊕ D^4(Q)) c14(R)
    """
    R = random_rotation_matrices(n, device=device, dtype=dtype)
    Q = random_rotation_matrices(n, device=device, dtype=dtype)

    if atol is None or rtol is None:
        atol, rtol = default_test_tolerances(R.dtype)

    out_R = model.forward_matrix(R)
    out_QR = model.forward_matrix(torch.einsum("nij,njk->nik", Q, R))

    D4 = irrep_D_from_matrix_device_safe(o3.Irrep("4e"), Q)
    rhs_l4 = torch.einsum("nij,nj->ni", D4, out_R.l4_axes)

    D24 = irreps_2e_4e_D_from_matrix_device_safe(Q)
    rhs_14 = torch.einsum("nij,nj->ni", D24, out_R.centered14)

    err_l4 = (out_QR.l4_axes - rhs_l4).abs().max().item()
    err_14 = (out_QR.centered14 - rhs_14).abs().max().item()

    return {
        "ok_l4": torch.allclose(out_QR.l4_axes, rhs_l4, atol=atol, rtol=rtol),
        "ok_centered14": torch.allclose(out_QR.centered14, rhs_14, atol=atol, rtol=rtol),
        "maxerr_l4": err_l4,
        "maxerr_centered14": err_14,
        "atol": atol,
        "rtol": rtol,
    }


@torch.no_grad()
def test_seed_vs_axes(
    model: CubicOrientationEmbedding,
    n: int = 16,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Compare the two 9D constructions after normalization.
    This is the most meaningful comparison numerically.
    """
    R = random_rotation_matrices(n, device=device, dtype=dtype)

    if atol is None or rtol is None:
        atol, rtol = default_test_tolerances(R.dtype)

    out = model.forward_matrix(R)

    a = normalize_features(out.l4_axes)
    b = normalize_features(out.l4_seed)

    err = (a - b).abs().max().item()
    mean_cos = (a * b).sum(dim=-1).mean().item()

    return {
        "ok": torch.allclose(a, b, atol=atol, rtol=rtol),
        "maxerr": err,
        "mean_cosine": mean_cos,
        "atol": atol,
        "rtol": rtol,
    }


@torch.no_grad()
def run_all_tests(
    model: CubicOrientationEmbedding,
    n: int = 32,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Run the full test suite and print a compact summary.
    """
    r1 = test_right_cubic_invariance(model, n=n, device=device, dtype=dtype)
    r2 = test_left_equivariance(model, n=n, device=device, dtype=dtype)
    r3 = test_seed_vs_axes(model, n=n, device=device, dtype=dtype)

    print("\nRight cubic invariance:")
    print(r1)

    print("\nLeft SO(3) equivariance:")
    print(r2)

    print("\nSeed-vs-axes consistency:")
    print(r3)

    return {"right_invariance": r1, "left_equivariance": r2, "seed_vs_axes": r3}


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    # Toggle this if you want stricter numerical agreement in the tests
    use_float64 = True

    if use_float64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CubicOrientationEmbedding(
        sh_normalization="integral",
        apply_paper_beta_to_full=True,
        use_wigner_seed_transport=True,   # set False to bypass explicit Wigner-D transport
    ).to(device=device, dtype=dtype)

    # Example passive/Bunge scalar-first quaternions [w, x, y, z]
    q_passive = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9238795, 0.0, 0.0, 0.3826834],
        ],
        dtype=dtype,
        device=device,
    )

    out = model.forward_quat_passive_wxyz(q_passive)

    print("l4_axes shape:", out.l4_axes.shape)
    print("l4_seed shape:", out.l4_seed.shape)
    print("full15 shape:", out.full15.shape)
    print("centered14 shape:", out.centered14.shape)

    run_all_tests(model, n=1000, device=device, dtype=dtype)