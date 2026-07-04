import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor


# ============================================================
# Basic helpers
# ============================================================


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)


def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
    return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)


def _quat_to_matrix_active(quats: torch.Tensor) -> torch.Tensor:
    """
    Fast scalar-first quaternion [w,x,y,z] -> active rotation matrix.
    Input:  (..., 4)
    Output: (..., 3, 3)
    """
    q = _normalize_quaternions(quats)
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
    return R.reshape(*q.shape[:-1], 3, 3)


def rot_x(
    theta: float, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=dtype,
        device=device,
    )


def rot_y(
    theta: float, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=dtype,
        device=device,
    )


def rot_z(
    theta: float, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


# ============================================================
# Symmetry groups
# ============================================================


def build_fcc_syms_mtex(
    dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """24 proper cubic rotations (FCC / O) as scalar-first quaternions [w,x,y,z]."""
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
        dtype=dtype,
        device=device,
    )


def build_hcp_syms_mtex(
    dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """12 proper D6 rotations (HCP) as scalar-first quaternions [w,x,y,z]."""
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
        dtype=dtype,
        device=device,
    )


def cubic_group_O(dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    mats = _quat_to_matrix_active(build_fcc_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 24:
        raise RuntimeError(f"Expected 24 cubic rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_zaxis(
    dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    mats = _quat_to_matrix_active(build_hcp_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 12:
        raise RuntimeError(f"Expected 12 D6 rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_paper(
    *, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """
    D6 in the Hielscher-Lippert paper convention:
    - major 6-fold axis || e1
    - 2-fold axis       || e2
    Obtained by conjugating the z-axis realization.
    """
    Gz = dihedral_group_D6_zaxis(dtype=dtype, device=device)

    # Want:
    #   e3 -> e1   (major axis z -> x)
    #   e1 -> e2   (chosen 2-fold axis x -> y)
    #   e2 -> e3
    Q = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )

    return Q @ Gz @ Q.transpose(-1, -2)


# ============================================================
# Tensor helpers
# ============================================================


def _normalize_metric_calibration(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "isometric" if value else "none"
    key = str(value).lower().replace("-", "_")
    if key in {"", "0", "false", "no", "none", "off", "legacy"}:
        return "none"
    if key in {"1", "true", "yes", "on", "iso", "isometric", "local_isometry"}:
        return "isometric"
    raise ValueError(
        "metric_calibration must be one of 'none'/'legacy' or 'isometric', "
        f"got {value!r}"
    )


def full_symmetry_formula(rank: int) -> str:
    letters = list("ijklmnopqrstuvwxyzabcdefgh")
    idx = letters[:rank]
    terms = ["".join(idx)]
    for k in range(rank - 1):
        tmp = idx.copy()
        tmp[k], tmp[k + 1] = tmp[k + 1], tmp[k]
        terms.append("".join(tmp))
    return "=".join(terms)


def _tensor_power_flat_fast(v: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Input:  (..., G, 3)
    Output: (..., G, 3**rank)
    """
    if rank == 1:
        return v

    p2 = (v.unsqueeze(-1) * v.unsqueeze(-2)).reshape(*v.shape[:-1], 9)
    if rank == 2:
        return p2

    p4 = (p2.unsqueeze(-1) * p2.unsqueeze(-2)).reshape(*v.shape[:-1], 81)
    if rank == 4:
        return p4

    if rank == 6:
        return (p4.unsqueeze(-1) * p2.unsqueeze(-2)).reshape(*v.shape[:-1], 729)

    out = v
    for _ in range(rank - 1):
        out = (out.unsqueeze(-1) * v.unsqueeze(-2)).reshape(*out.shape[:-1], -1)
    return out


def _drop_scalar_irreps(
    irreps: o3.Irreps,
    y: torch.Tensor,
) -> tuple[torch.Tensor, o3.Irreps]:
    keep = []
    out_irreps = []
    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        if ir.l != 0:
            keep.append(y[..., sl])
            out_irreps.append((mul, ir))
    y_out = y[..., :0] if len(keep) == 0 else torch.cat(keep, dim=-1)
    return y_out, o3.Irreps(out_irreps).regroup()


def _build_nonscalar_projector(
    rank: int,
    formula: str,
    dtype: torch.dtype,
    device=None,
) -> tuple[torch.Tensor, o3.Irreps]:
    """
    Build linear map:
        flattened symmetric Cartesian tensor -> non-scalar irreps coordinates
    """
    ct = CartesianTensor(formula)
    rtp = ct.reduced_tensor_products()

    flat_dim = 3**rank
    with torch.no_grad():
        basis = torch.eye(flat_dim, dtype=dtype, device=device)
        basis_multi = basis.reshape(flat_dim, *([3] * rank))
        y_full = ct.from_cartesian(basis_multi)  # [flat_dim, full_out_dim]
        y_keep, irreps_no_scalar = _drop_scalar_irreps(rtp.irreps_out, y_full)
        proj = y_keep.contiguous()  # [flat_dim, nonscalar_dim]

    return proj, irreps_no_scalar


def _so3_character_from_angle(
    l: int,
    theta: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Character of the SO(3) irrep of degree l for rotation angle theta.

    chi_l(theta) = sin((l + 1/2) * theta) / sin(theta / 2)
    with the theta -> 0 limit equal to (2l + 1).
    """
    theta = theta.to(dtype=torch.float64)
    half = 0.5 * theta
    denom = torch.sin(half)
    numer = torch.sin((l + 0.5) * theta)
    near_identity = denom.abs() < eps
    safe_denom = torch.where(
        near_identity,
        torch.ones_like(denom),
        denom,
    )
    ratio = numer / safe_denom
    return torch.where(
        near_identity,
        torch.full_like(ratio, float(2 * l + 1)),
        ratio,
    )


def _a1_multiplicity_table_from_group(
    group_mats: torch.Tensor,
    max_l: int,
    rounding_tol: float = 1e-6,
) -> dict[int, int]:
    """
    Compute multiplicity of the trivial irrep A1 inside each SO(3) l-band:
        m_l = (1 / |G|) * sum_{g in G} chi_l(g)

    This uses only the group rotation matrices and SO(3) character formula.
    """
    if group_mats.ndim != 3 or group_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (G,3,3), got {tuple(group_mats.shape)}")

    traces = group_mats.diagonal(dim1=-2, dim2=-1).sum(dim=-1).to(torch.float64)
    cos_theta = ((traces - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    out: dict[int, int] = {}
    for l in range(int(max_l) + 1):
        mean_char = float(_so3_character_from_angle(l, theta).mean().item())
        rounded = int(round(mean_char))
        if abs(mean_char - rounded) > rounding_tol:
            raise RuntimeError(
                f"A1 multiplicity rounding drift for l={l}: "
                f"mean_char={mean_char:.8f}, rounded={rounded}"
            )
        out[l] = max(0, rounded)
    return out


def _canonicalize_basis_signs(
    basis: torch.Tensor,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Make eigenspace bases deterministic by fixing each column's sign.

    Eigenvectors are defined up to sign.  For reproducibility, orient each
    basis vector so its first non-negligible entry is positive.
    """
    basis = basis.clone()
    for col_idx in range(basis.shape[1]):
        col = basis[:, col_idx]
        nz = torch.nonzero(col.abs() > tol, as_tuple=False)
        if nz.numel() == 0:
            continue
        first = int(nz[0, 0].item())
        if float(col[first].item()) < 0.0:
            basis[:, col_idx] = -col
    return basis


def _build_reynolds_projector_basis(
    group_mats: torch.Tensor,
    l: int,
    dtype: torch.dtype,
    device=None,
    eig_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the fixed-subspace basis for the direct Reynolds projector.

    P_l = |G|^{-1} sum_g D^l(g)

    The returned basis has shape (2*l + 1, r_l), where r_l is the rank of
    the projector/eigenvalue-1 multiplicity.  Columns are orthonormal in the
    real tesseral e3nn basis.
    """
    l = int(l)
    ir = o3.Irrep(l, 1)
    # This is an init-time/offline calculation.  Keep it on CPU because some
    # e3nn versions materialize Wigner-D generator buffers on CPU inside
    # D_from_matrix, then move the resulting basis to the requested device.
    out_device = group_mats.device if device is None else torch.device(device)
    group = group_mats.detach().to(dtype=dtype, device="cpu")
    with torch.no_grad():
        Dg = ir.D_from_matrix(group)
        P = Dg.mean(dim=0)
        P = 0.5 * (P + P.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(P)
        keep = evals > (1.0 - float(eig_tol))
        basis = evecs[:, keep].contiguous()
        basis = _canonicalize_basis_signs(basis)
    return basis.contiguous().to(device=out_device), evals.to(device=out_device)


def _fibonacci_sphere_directions(
    num: int,
    dtype: torch.dtype,
    device=None,
) -> torch.Tensor:
    """Deterministic approximately uniform unit directions on S^2."""
    num = int(num)
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    i = torch.arange(num, dtype=dtype, device=device)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / float(num)
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    theta = golden_angle * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def _build_harmonic_synthesis_coefficients(
    basis: torch.Tensor,
    l: int,
    dtype: torch.dtype,
    device=None,
    num_anchors: Optional[int] = None,
    residual_tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Express projector-basis columns as linear combinations of Y_l(anchor).

    This lets runtime evaluate D^l(R) U_l via GPU-safe spherical harmonics:
        U_l[:, j] = sum_a coeff[a, j] Y_l(anchor_a)
        D^l(R)U_l[:, j] = sum_a coeff[a, j] Y_l(R anchor_a)

    Returns:
        anchors: (M, 3)
        coeff:   (M, r_l)
        residual max norm of the reconstruction.
    """
    l = int(l)
    dim_l = 2 * l + 1
    if basis.shape[0] != dim_l:
        raise ValueError(
            f"basis first dimension must be 2*l+1={dim_l}, got {tuple(basis.shape)}"
        )
    if num_anchors is None:
        num_anchors = max(4 * dim_l, 32)

    # Build/solve on CPU for stable init-time least squares, then move buffers.
    out_device = basis.device if device is None else torch.device(device)
    basis_cpu = basis.detach().to(dtype=dtype, device="cpu")
    anchors_cpu = _fibonacci_sphere_directions(
        int(num_anchors),
        dtype=dtype,
        device="cpu",
    )
    y_anchor = o3.spherical_harmonics(
        l,
        anchors_cpu,
        normalize=True,
        normalization="component",
    )  # (M, 2*l + 1)

    # y_anchor.T @ coeff ~= basis.
    coeff = torch.linalg.lstsq(y_anchor.transpose(0, 1), basis_cpu).solution
    recon = y_anchor.transpose(0, 1) @ coeff
    residual = float((recon - basis_cpu).abs().max().item())
    if residual > residual_tol:
        raise RuntimeError(
            f"Could not synthesize l={l} Reynolds basis from spherical anchors: "
            f"max residual={residual:.3e}, tol={residual_tol:.3e}"
        )
    return (
        anchors_cpu.to(device=out_device),
        coeff.to(device=out_device),
        residual,
    )


def _select_irrep_blocks(
    proj: torch.Tensor,
    irreps: o3.Irreps,
    keep_mask: list[bool],
) -> tuple[torch.Tensor, o3.Irreps]:
    if len(keep_mask) != len(irreps):
        raise ValueError(
            f"keep_mask length {len(keep_mask)} must match irreps length {len(irreps)}"
        )

    keep_cols = []
    keep_irreps = []
    for keep, sl, mul_ir in zip(keep_mask, irreps.slices(), irreps):
        if not keep:
            continue
        keep_cols.append(torch.arange(sl.start, sl.stop, device=proj.device))
        keep_irreps.append(mul_ir)

    if len(keep_cols) == 0:
        return proj[:, :0], o3.Irreps()

    col_idx = torch.cat(keep_cols, dim=0)
    return proj.index_select(dim=1, index=col_idx).contiguous(), o3.Irreps(keep_irreps).regroup()


def _prune_irreps_with_no_a1(
    proj: torch.Tensor,
    irreps: o3.Irreps,
    a1_multiplicity_by_l: dict[int, int],
) -> tuple[torch.Tensor, o3.Irreps]:
    """
    Drop whole irrep bands whose l has no A1 content under the crystal group.
    """
    keep_mask = [a1_multiplicity_by_l.get(ir.l, 0) > 0 for _, ir in irreps]
    return _select_irrep_blocks(proj, irreps, keep_mask)


def _prune_irreps_inactive_at_identity(
    proj: torch.Tensor,
    irreps: o3.Irreps,
    x_identity_flat: torch.Tensor,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, o3.Irreps]:
    """
    Drop irrep blocks that evaluate to zero at R = I for this orbit descriptor.
    """
    if irreps.dim == 0 or proj.numel() == 0:
        return proj, irreps

    y0 = x_identity_flat @ proj
    keep_mask = [bool(y0[sl].norm().item() > tol) for sl in irreps.slices()]
    return _select_irrep_blocks(proj, irreps, keep_mask)


# ============================================================
# Main module
# ============================================================


@dataclass
class _RawBlockSpec:
    name: str
    rank: int
    beta: float
    reference_direction: torch.Tensor
    formula: str


@dataclass
class _HarmonicSeedSpec:
    name: str
    beta: float
    reference_direction: torch.Tensor


class LocalIsoCrystalEmbedding(nn.Module):
    """
    ML-oriented HL embedding.

    Fast path:
      quaternion/matrix -> orbit-average tensor -> precomputed linear projection

    Notes:
    - Default dtype is float32 for training.
    - Geometry diagnostics can be run in high precision via the `dtype` argument.
    """

    def __init__(
        self,
        group_name: str,
        d6_convention: str = "z_axis",
        embedding_mode: str = "tensor_product",
        max_harmonic_l: Optional[int] = None,
        metric_calibration: object = "none",
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.group_name = str(group_name)
        self.d6_convention = str(d6_convention)
        mode_key = str(embedding_mode).lower().replace("-", "_")
        if mode_key in {"tensor", "tensor_product", "cartesian", "cartesian_tensor"}:
            mode_key = "tensor_product"
        elif mode_key in {
            "direct",
            "reynolds",
            "direct_reynolds",
            "reynolds_projector",
            "direct_reynolds_projector",
            "projector",
            "projector_basis",
        }:
            mode_key = "direct_reynolds"
        elif mode_key in {
            "direct_orbit_seeds",
            "orbit_seeds",
            "orbit_seed",
            "seed_reynolds",
            "reynolds_harmonic",
            "harmonic",
        }:
            mode_key = "direct_orbit_seeds"
        else:
            raise ValueError(
                "embedding_mode must be 'tensor_product', 'direct_reynolds', "
                "or 'direct_orbit_seeds', "
                f"got {embedding_mode!r}"
            )
        self.embedding_mode = mode_key
        self.max_harmonic_l = None if max_harmonic_l is None else int(max_harmonic_l)
        self.metric_calibration = _normalize_metric_calibration(metric_calibration)
        self.metric_calibration_residual = 0.0
        self.dtype = dtype

        e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        e2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        e3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        specs: list[_RawBlockSpec] = []
        harmonic_seeds: list[_HarmonicSeedSpec] = []

        if self.group_name == "O":
            group_mats = cubic_group_O(dtype=dtype, device=device)
            specs.append(
                _RawBlockSpec(
                    name="rank4",
                    rank=4,
                    beta=3.0 / (2.0 * math.sqrt(2.0)),
                    reference_direction=e1,
                    formula=full_symmetry_formula(4),
                )
            )
            harmonic_seeds.append(
                _HarmonicSeedSpec(
                    name="fcc_axis",
                    beta=3.0 / (2.0 * math.sqrt(2.0)),
                    reference_direction=e1,
                )
            )

        elif self.group_name == "D6":
            if self.d6_convention == "paper":
                group_mats = dihedral_group_D6_paper(dtype=dtype, device=device)
                u2, u6 = e1, e2
            elif self.d6_convention == "z_axis":
                group_mats = dihedral_group_D6_zaxis(dtype=dtype, device=device)
                u2, u6 = e3, e1
            else:
                raise ValueError("d6_convention must be 'z_axis' or 'paper'")

            specs.extend(
                [
                    _RawBlockSpec(
                        name="rank2",
                        rank=2,
                        beta=1.0 / math.sqrt(24.0),
                        reference_direction=u2,
                        formula=full_symmetry_formula(2),
                    ),
                    _RawBlockSpec(
                        name="rank6",
                        rank=6,
                        beta=2.0 * math.sqrt(2.0) / 3.0,
                        reference_direction=u6,
                        formula=full_symmetry_formula(6),
                    ),
                ]
            )
            harmonic_seeds.extend(
                [
                    _HarmonicSeedSpec(
                        name="hcp_axis",
                        beta=1.0 / math.sqrt(24.0),
                        reference_direction=u2,
                    ),
                    _HarmonicSeedSpec(
                        name="hcp_basal",
                        beta=2.0 * math.sqrt(2.0) / 3.0,
                        reference_direction=u6,
                    ),
                ]
            )
        else:
            raise ValueError("group_name must be 'O' or 'D6'")

        self.register_buffer("group_mats", group_mats, persistent=False)

        self.blocks = nn.ModuleList()
        irreps_full_parts = []
        irreps_a1_parts = []

        # Build projectors in high precision first, then cast to runtime dtype.
        compute_dtype = torch.float64 if dtype == torch.float32 else dtype
        if self.embedding_mode == "tensor_product":
            max_rank = max(spec.rank for spec in specs)
            self.a1_multiplicity_by_l = _a1_multiplicity_table_from_group(
                self.group_mats.to(dtype=compute_dtype),
                max_l=max_rank,
            )

            for spec in specs:
                block = nn.Module()
                block.kind = "tensor_product"
                block.name = spec.name
                block.rank = spec.rank
                block.beta = float(spec.beta)

                # Orbit directions {g u}_g from HL notation (reference direction u).
                orbit_directions = torch.matmul(self.group_mats, spec.reference_direction)  # [G, 3]
                orbit_directions_t = orbit_directions.transpose(0, 1).contiguous()  # [3, G]

                proj_full, irreps_no_scalar = _build_nonscalar_projector(
                    spec.rank,
                    spec.formula,
                    dtype=compute_dtype,
                    device=device,
                )

                # First prune l-bands that provably have no A1 under this group.
                proj_in, irreps_active = _prune_irreps_with_no_a1(
                    proj_full,
                    irreps_no_scalar,
                    self.a1_multiplicity_by_l,
                )

                # Then prune any residual dead blocks for this specific orbit descriptor.
                x0_flat = _tensor_power_flat_fast(
                    orbit_directions.to(dtype=compute_dtype).unsqueeze(0),
                    spec.rank,
                ).mean(dim=-2).squeeze(0)
                proj_in, irreps_active = _prune_irreps_inactive_at_identity(
                    proj=proj_in,
                    irreps=irreps_active,
                    x_identity_flat=x0_flat,
                    tol=1e-10 if compute_dtype == torch.float64 else 1e-6,
                )

                block.register_buffer(
                    "orbit_directions_t",
                    orbit_directions_t,
                    persistent=False,
                )
                block.register_buffer("proj_out", proj_full.to(dtype=dtype), persistent=False)
                block.register_buffer(
                    "proj_in",
                    proj_in.to(dtype=dtype),
                    persistent=False,
                )
                block.irreps_full = irreps_no_scalar
                block.irreps_a1 = irreps_active
                block.out_dim = proj_full.shape[-1]
                block.in_dim = proj_in.shape[-1]
                block.raw_dim = 3**spec.rank

                self.blocks.append(block)
                irreps_full_parts += list(irreps_no_scalar)
                irreps_a1_parts += list(irreps_active)
        elif self.embedding_mode == "direct_reynolds":
            default_max_l = 4 if self.group_name == "O" else 6
            max_l = default_max_l if self.max_harmonic_l is None else int(self.max_harmonic_l)
            if max_l < 1:
                raise ValueError(f"max_harmonic_l must be >= 1, got {max_l}")
            self.max_harmonic_l = max_l
            # r_0 is the scalar invariant multiplicity, but scalar features are
            # deliberately not emitted by the non-scalar orientation encoder.
            self.a1_multiplicity_by_l = {0: 1}

            # Direct Reynolds path:
            #   P_l = |G|^{-1} sum_g D^l(g),  U_l = eigenspace(P_l, eigenvalue 1)
            #   R -> D^l(R) U_l, for every surviving l <= max_harmonic_l.
            #
            # This keeps the projector rank r_l itself, not one copy per chosen
            # geometric seed.  Therefore D6 through l=6 gives
            #   1x2e + 1x4e + 2x6e
            # rather than the overcomplete seed-orbit layout
            #   2x2e + 2x4e + 2x6e.
            for l in range(1, max_l + 1):
                basis, evals = _build_reynolds_projector_basis(
                    self.group_mats,
                    l=int(l),
                    dtype=compute_dtype,
                    device=device,
                    eig_tol=1e-6,
                )
                multiplicity = int(basis.shape[1])
                self.a1_multiplicity_by_l[int(l)] = multiplicity
                if multiplicity <= 0:
                    continue

                anchors, coeff, synth_residual = _build_harmonic_synthesis_coefficients(
                    basis,
                    l=int(l),
                    dtype=compute_dtype,
                    device=device,
                )
                anchors_t = anchors.transpose(0, 1).contiguous()

                irreps_l = o3.Irreps(f"{multiplicity}x{int(l)}e")
                block = nn.Module()
                block.kind = "direct_reynolds_projector"
                block.name = f"projector_l{int(l)}"
                block.rank = int(l)
                block.l = int(l)
                # The projector-basis map uses orthonormal Wigner-D bases.  The
                # downstream decoder lookup is built from the same map; local metric
                # calibration can be applied separately when comparing geometries.
                block.beta = 1.0
                block.projector_rank = multiplicity
                block.synthesis_residual = synth_residual
                block.register_buffer("basis", basis.to(dtype=dtype), persistent=False)
                block.register_buffer("projector_evals", evals.to(dtype=dtype), persistent=False)
                block.register_buffer(
                    "anchor_directions_t",
                    anchors_t.to(dtype=dtype),
                    persistent=False,
                )
                block.register_buffer("synthesis_coeff", coeff.to(dtype=dtype), persistent=False)
                block.irreps_full = irreps_l
                block.irreps_a1 = irreps_l
                block.out_dim = int(irreps_l.dim)
                block.in_dim = int(irreps_l.dim)
                block.raw_dim = int(irreps_l.dim)

                self.blocks.append(block)
                irreps_full_parts += list(irreps_l)
                irreps_a1_parts += list(irreps_l)

            if self.metric_calibration == "isometric":
                self._calibrate_direct_reynolds_blocks(compute_dtype)

        else:
            default_max_l = 4 if self.group_name == "O" else 6
            max_l = default_max_l if self.max_harmonic_l is None else int(self.max_harmonic_l)
            if max_l < 1:
                raise ValueError(f"max_harmonic_l must be >= 1, got {max_l}")
            self.max_harmonic_l = max_l
            # r_0 is the scalar invariant multiplicity, but scalar features are
            # deliberately not emitted by the non-scalar orientation encoder.
            self.a1_multiplicity_by_l = {0: 1}

            # Diagnostic seed-orbit Reynolds path:
            #   R -> mean_g Y_l(R g u), l <= max_harmonic_l.
            # A block is kept iff this explicit Reynolds average is nonzero at R = I.
            # This intentionally keeps one copy per geometric seed, so it can be
            # overcomplete relative to rank(P_l).  Use `direct_reynolds` for the
            # minimal projector-basis truncation experiment.
            for spec in harmonic_seeds:
                orbit_directions = torch.matmul(self.group_mats, spec.reference_direction)  # [G, 3]
                orbit_directions_t = orbit_directions.transpose(0, 1).contiguous()  # [3, G]
                orbit_for_prune = orbit_directions.to(dtype=compute_dtype)

                for l in range(1, max_l + 1):
                    y0 = o3.spherical_harmonics(
                        int(l),
                        orbit_for_prune,
                        normalize=True,
                        normalization="component",
                    ).mean(dim=0)
                    if float(y0.norm().item()) <= 1e-10:
                        continue

                    irreps_l = o3.Irreps(f"1x{int(l)}e")
                    block = nn.Module()
                    block.kind = "direct_orbit_seeds"
                    block.name = f"{spec.name}_l{int(l)}"
                    block.rank = int(l)
                    block.l = int(l)
                    block.beta = float(spec.beta)
                    block.identity_norm = float(y0.norm().item())
                    block.register_buffer(
                        "orbit_directions_t",
                        orbit_directions_t,
                        persistent=False,
                    )
                    block.irreps_full = irreps_l
                    block.irreps_a1 = irreps_l
                    block.out_dim = int(irreps_l.dim)
                    block.in_dim = int(irreps_l.dim)
                    block.raw_dim = int(irreps_l.dim)

                    self.blocks.append(block)
                    irreps_full_parts += list(irreps_l)
                    irreps_a1_parts += list(irreps_l)

        self.irreps_full = o3.Irreps(irreps_full_parts).regroup()
        self.irreps_a1 = o3.Irreps(irreps_a1_parts).regroup()
        # String helpers for TP declarations.
        self.irreps_full_str = str(self.irreps_full)
        self.irreps_full_tp = " + ".join(str(part) for part in self.irreps_full)
        self.irreps_a1_str = str(self.irreps_a1)

    @torch.no_grad()
    def _direct_reynolds_block_tangent_gram(
        self,
        block: nn.Module,
        dtype: torch.dtype,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        if eps is None:
            eps = 1e-7 if dtype == torch.float64 else 1e-4

        device = self.group_mats.device
        s1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        s2 = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        s3 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )

        anchors_t = block.anchor_directions_t.to(dtype=dtype)
        coeff = block.synthesis_coeff.to(dtype=dtype)
        tangents = []
        for S in (s1, s2, s3):
            Rp = torch.matrix_exp(eps * S).unsqueeze(0)
            Rm = torch.matrix_exp(-eps * S).unsqueeze(0)
            yp = self._direct_reynolds_projector_feature(Rp, anchors_t, coeff, block.l)[0]
            ym = self._direct_reynolds_projector_feature(Rm, anchors_t, coeff, block.l)[0]
            tangents.append((yp - ym) / (2.0 * eps))

        return torch.stack(
            [
                torch.stack([(vi * vj).sum() for vj in tangents], dim=0)
                for vi in tangents
            ],
            dim=0,
        )

    @torch.no_grad()
    def _calibrate_direct_reynolds_blocks(self, compute_dtype: torch.dtype) -> None:
        """
        Preserve the projected irrep blocks, but scale them so the combined
        direct-Reynolds feature map is locally isometric at the identity.

        We solve for squared scalar block scales x_i in
            sum_i x_i J_i J_i^T ~= I_3.
        If the constraints are underdetermined, use the solution closest to a
        common global scale so no harmonic is arbitrarily emphasized.
        """
        direct_blocks = [
            block
            for block in self.blocks
            if getattr(block, "kind", None) == "direct_reynolds_projector"
        ]
        if not direct_blocks:
            return

        grams = [
            self._direct_reynolds_block_tangent_gram(block, dtype=compute_dtype)
            for block in direct_blocks
        ]
        diag_rows = torch.stack([gram.diag() for gram in grams], dim=1).to(torch.float64)
        target = torch.ones(3, dtype=torch.float64, device=diag_rows.device)

        total_diag = diag_rows.sum(dim=1)
        common = 1.0 / total_diag.mean().clamp_min(1e-12)
        scales_sq0 = torch.full(
            (len(direct_blocks),),
            float(common.item()),
            dtype=torch.float64,
            device=diag_rows.device,
        )

        correction = diag_rows.transpose(0, 1) @ (
            torch.linalg.pinv(diag_rows @ diag_rows.transpose(0, 1))
            @ (target - diag_rows @ scales_sq0)
        )
        scales_sq = scales_sq0 + correction
        if bool((scales_sq <= 0.0).any().item()):
            warnings.warn(
                "Direct-Reynolds isometric calibration produced a non-positive "
                "block scale; falling back to clipped least-squares scales.",
                RuntimeWarning,
                stacklevel=2,
            )
            scales_sq = torch.linalg.lstsq(diag_rows, target).solution.clamp_min(1e-12)

        residual = diag_rows @ scales_sq - target
        self.metric_calibration_residual = float(residual.abs().max().item())
        if self.metric_calibration_residual > 1e-4:
            warnings.warn(
                "Direct-Reynolds blocks cannot form an exactly locally isometric "
                "embedding with scalar block scales; "
                f"max residual={self.metric_calibration_residual:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )

        for block, scale_sq in zip(direct_blocks, scales_sq):
            block.beta = float(torch.sqrt(scale_sq).item())

    def metric_calibration_metadata(self) -> dict[str, object]:
        return {
            "metric_calibration": str(self.metric_calibration),
            "metric_calibration_residual": float(self.metric_calibration_residual),
            "block_betas": [
                {
                    "name": str(getattr(block, "name", f"block_{idx}")),
                    "kind": str(getattr(block, "kind", "tensor_product")),
                    "l": None if not hasattr(block, "l") else int(block.l),
                    "beta": float(getattr(block, "beta", 1.0)),
                }
                for idx, block in enumerate(self.blocks)
            ],
        }

    def get_a1_multiplicity_table(self, max_l: Optional[int] = None) -> dict[int, int]:
        if max_l is None:
            return dict(self.a1_multiplicity_by_l)
        max_l = int(max_l)
        known_max = max(self.a1_multiplicity_by_l.keys(), default=-1)
        if max_l > known_max:
            self.a1_multiplicity_by_l.update(
                _a1_multiplicity_table_from_group(
                    self.group_mats.to(dtype=torch.float64),
                    max_l=max_l,
                )
            )
        return {l: self.a1_multiplicity_by_l[l] for l in range(max_l + 1)}

    # --------------------------------------------------------
    # Fast forward path
    # --------------------------------------------------------

    def _to_device_dtype(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.dtype, device=self.group_mats.device)

    def _orbit_average_flat(
        self,
        R: torch.Tensor,
        orbit_directions_t: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        # R: (..., 3, 3), orbit_directions_t: (3, G)
        # -> v: (..., G, 3)
        v = torch.matmul(R, orbit_directions_t).transpose(-2, -1)
        return _tensor_power_flat_fast(v, rank).mean(dim=-2)

    def _orbit_average_harmonic(
        self,
        R: torch.Tensor,
        orbit_directions_t: torch.Tensor,
        l: int,
    ) -> torch.Tensor:
        # R: (..., 3, 3), orbit_directions_t: (3, G)
        # -> mean_g Y_l(R g u): (..., 2*l + 1)
        v = torch.matmul(R, orbit_directions_t).transpose(-2, -1)
        y = o3.spherical_harmonics(
            int(l),
            v.reshape(-1, 3),
            normalize=True,
            normalization="component",
        )
        y = y.reshape(*v.shape[:-1], 2 * int(l) + 1)
        return y.mean(dim=-2)

    def _direct_reynolds_projector_feature(
        self,
        R: torch.Tensor,
        anchor_directions_t: torch.Tensor,
        synthesis_coeff: torch.Tensor,
        l: int,
    ) -> torch.Tensor:
        # R: (..., 3, 3), anchor_directions_t: (3, M), coeff: (M, r_l)
        # Spherical-harmonic synthesis evaluates D^l(R)U_l without constructing
        # Wigner-D matrices at runtime:
        #   D^l(R)U_l[:, j] = sum_a coeff[a, j] Y_l(R anchor_a).
        # The result is flattened as r_l copies of l, i.e. copy-major:
        #   (..., r_l * (2*l + 1)).
        v = torch.matmul(R, anchor_directions_t).transpose(-2, -1)
        y_anchor = o3.spherical_harmonics(
            int(l),
            v.reshape(-1, 3),
            normalize=True,
            normalization="component",
        )
        y_anchor = y_anchor.reshape(*v.shape[:-1], 2 * int(l) + 1)
        y = torch.einsum("...mc,mr->...cr", y_anchor, synthesis_coeff)
        return y.transpose(-1, -2).reshape(*R.shape[:-2], -1)

    def forward_raw(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            kind = getattr(block, "kind", "tensor_product")
            if kind == "direct_reynolds_projector":
                y = self._direct_reynolds_projector_feature(
                    R,
                    block.anchor_directions_t,
                    block.synthesis_coeff,
                    block.l,
                )
                outs.append(block.beta * y)
            elif kind == "direct_orbit_seeds":
                y = self._orbit_average_harmonic(R, block.orbit_directions_t, block.l)
                outs.append(block.beta * y)
            else:
                x_flat = self._orbit_average_flat(R, block.orbit_directions_t, block.rank)
                outs.append(block.beta * x_flat)
        return torch.cat(outs, dim=-1)

    def forward_irreps(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            kind = getattr(block, "kind", "tensor_product")
            if kind == "direct_reynolds_projector":
                y = self._direct_reynolds_projector_feature(
                    R,
                    block.anchor_directions_t,
                    block.synthesis_coeff,
                    block.l,
                )
            elif kind == "direct_orbit_seeds":
                y = self._orbit_average_harmonic(R, block.orbit_directions_t, block.l)
            else:
                x_flat = self._orbit_average_flat(R, block.orbit_directions_t, block.rank)
                y = x_flat @ block.proj_out
            outs.append(block.beta * y)
        return torch.cat(outs, dim=-1)

    def forward_irreps_a1(self, R: torch.Tensor) -> torch.Tensor:
        """
        Return only active (A1-supported) non-scalar irrep bands.
        """
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            kind = getattr(block, "kind", "tensor_product")
            if kind == "direct_reynolds_projector":
                y = self._direct_reynolds_projector_feature(
                    R,
                    block.anchor_directions_t,
                    block.synthesis_coeff,
                    block.l,
                )
            elif kind == "direct_orbit_seeds":
                y = self._orbit_average_harmonic(R, block.orbit_directions_t, block.l)
            else:
                x_flat = self._orbit_average_flat(R, block.orbit_directions_t, block.rank)
                y = x_flat @ block.proj_in
            outs.append(block.beta * y)
        return torch.cat(outs, dim=-1)

    def forward_irreps_passive(
        self,
        quats_passive: torch.Tensor,
        active_only: bool = False,
    ) -> torch.Tensor:
        """
        Compute irreps embedding from passive quaternions.

        The model uses active rotations internally, so we convert
        passive -> active via quaternion conjugation first.
        """
        q_passive = self._to_device_dtype(quats_passive)
        q_active = _quat_conjugate(q_passive)
        R = _quat_to_matrix_active(q_active)
        return self.forward_irreps_a1(R) if active_only else self.forward_irreps(R)

    def forward_from_quaternions(
        self,
        quats: torch.Tensor,
        raw: bool = False,
        active_only: bool = False,
    ) -> torch.Tensor:
        """
        Quaternion path that expects active quaternions [w, x, y, z].
        """
        q = self._to_device_dtype(quats)
        R = _quat_to_matrix_active(q)
        if raw:
            return self.forward_raw(R)
        return self.forward_irreps_a1(R) if active_only else self.forward_irreps(R)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.forward_irreps(R)

    # --------------------------------------------------------
    # Diagnostics (high precision)
    # --------------------------------------------------------

    @torch.no_grad()
    def gram_at_identity(
        self,
        eps: Optional[float] = None,
        use_raw: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Robust local-isometry diagnostic.
        Runs in high precision by default even if model trains in float32.
        """
        if eps is None:
            eps = 1e-7 if dtype == torch.float64 else 1e-4

        device = self.group_mats.device
        embed_fn: Callable[[torch.Tensor], torch.Tensor]
        embed_fn = self.forward_raw if use_raw else self.forward_irreps

        s1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        s2 = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        s3 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        basis = [s1, s2, s3]

        tangents = []
        for S in basis:
            Rp = torch.matrix_exp(eps * S)
            Rm = torch.matrix_exp(-eps * S)

            # Temporarily evaluate through the same module path but cast input
            Ep = embed_fn(Rp.unsqueeze(0).to(self.group_mats.dtype))[0].to(dtype)
            Em = embed_fn(Rm.unsqueeze(0).to(self.group_mats.dtype))[0].to(dtype)
            tangents.append((Ep - Em) / (2.0 * eps))

        return torch.stack(
            [
                torch.stack([(vi * vj).sum() for vj in tangents], dim=0)
                for vi in tangents
            ],
            dim=0,
        )

    @torch.no_grad()
    def right_invariance_error(
        self,
        n_trials: int = 10,
        use_raw: bool = False,
        seed: int = 0,
    ) -> float:
        embed_fn: Callable[[torch.Tensor], torch.Tensor]
        embed_fn = self.forward_raw if use_raw else self.forward_irreps

        torch.manual_seed(int(seed))
        device = self.group_mats.device
        dtype = self.group_mats.dtype
        errs = []

        for _ in range(int(n_trials)):
            axis = torch.randn(3, dtype=dtype, device=device)
            axis = axis / axis.norm().clamp_min(1e-8)
            angle = 2.0 * math.pi * torch.rand((), dtype=dtype, device=device)

            K = torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=dtype,
                device=device,
            )
            R = torch.matrix_exp(angle * K)

            ER = embed_fn(R.unsqueeze(0))[0]
            for g in self.group_mats:
                ERg = embed_fn((R @ g).unsqueeze(0))[0]
                errs.append((ERg - ER).abs().max())

        return float(torch.stack(errs).max().item())


# ============================================================
# Builders
# ============================================================


def build_local_iso_fcc_embedding(
    dtype: torch.dtype = torch.float32,
    device=None,
    embedding_mode: str = "tensor_product",
    max_harmonic_l: Optional[int] = None,
    metric_calibration: object = "none",
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding(
        "O",
        embedding_mode=embedding_mode,
        max_harmonic_l=max_harmonic_l,
        metric_calibration=metric_calibration,
        dtype=dtype,
        device=device,
    )


def build_local_iso_hcp_embedding(
    d6_convention: str = "z_axis",
    dtype: torch.dtype = torch.float32,
    device=None,
    embedding_mode: str = "tensor_product",
    max_harmonic_l: Optional[int] = None,
    metric_calibration: object = "none",
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding(
        "D6",
        d6_convention=d6_convention,
        embedding_mode=embedding_mode,
        max_harmonic_l=max_harmonic_l,
        metric_calibration=metric_calibration,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "LocalIsoCrystalEmbedding",
    "build_local_iso_fcc_embedding",
    "build_local_iso_hcp_embedding",
    "build_fcc_syms_mtex",
    "build_hcp_syms_mtex",
    "cubic_group_O",
    "dihedral_group_D6_zaxis",
    "dihedral_group_D6_paper",
]
