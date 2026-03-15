import math
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


class LocalIsoCrystalEmbedding(nn.Module):
    """
    ML-oriented HL embedding.

    Fast path:
      quaternion/matrix -> orbit-average tensor -> precomputed linear projection

    Notes:
    - Default dtype is float32 for training.
    - Geometry diagnostics are evaluated in float64 automatically.
    """

    def __init__(
        self,
        group_name: str,
        d6_convention: str = "z_axis",
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.group_name = str(group_name)
        self.d6_convention = str(d6_convention)
        self.dtype = dtype

        e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        e2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        e3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        specs: list[_RawBlockSpec] = []

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
        else:
            raise ValueError("group_name must be 'O' or 'D6'")

        self.register_buffer("group_mats", group_mats, persistent=False)

        self.blocks = nn.ModuleList()
        irreps_out_parts = []

        # Build projectors in float64 first, then cast to training dtype
        build_dtype = torch.float64 if dtype == torch.float32 else dtype

        for spec in specs:
            block = nn.Module()
            block.name = spec.name
            block.rank = spec.rank
            block.beta = float(spec.beta)

            anchors = torch.matmul(self.group_mats, spec.reference_direction)  # [G, 3]
            anchors_t = anchors.transpose(0, 1).contiguous()  # [3, G]

            proj64, irreps_no_scalar = _build_nonscalar_projector(
                spec.rank,
                spec.formula,
                dtype=build_dtype,
                device=device,
            )

            block.register_buffer("anchors_t", anchors_t, persistent=False)
            block.register_buffer("proj", proj64.to(dtype=dtype), persistent=False)
            block.irreps = irreps_no_scalar
            block.out_dim = proj64.shape[-1]
            block.raw_dim = 3**spec.rank

            self.blocks.append(block)
            irreps_out_parts += list(irreps_no_scalar)

        self.irreps_out = o3.Irreps(irreps_out_parts).regroup()

    # --------------------------------------------------------
    # Fast forward path
    # --------------------------------------------------------

    def _to_device_dtype(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.dtype, device=self.group_mats.device)

    def _orbit_average_flat(
        self,
        R: torch.Tensor,
        anchors_t: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        # R: (..., 3, 3), anchors_t: (3, G)
        # -> v: (..., G, 3)
        v = torch.matmul(R, anchors_t).transpose(-2, -1)
        return _tensor_power_flat_fast(v, rank).mean(dim=-2)

    def forward_raw(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            x_flat = self._orbit_average_flat(R, block.anchors_t, block.rank)
            outs.append(block.beta * x_flat)
        return torch.cat(outs, dim=-1)

    def forward_irreps(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            x_flat = self._orbit_average_flat(R, block.anchors_t, block.rank)
            y = x_flat @ block.proj
            outs.append(block.beta * y)
        return torch.cat(outs, dim=-1)

    def forward_irreps_passive(self, quats_passive: torch.Tensor) -> torch.Tensor:
        """
        Compute irreps embedding from passive quaternions.

        The model uses active rotations internally, so we convert
        passive -> active via quaternion conjugation first.
        """
        q_passive = self._to_device_dtype(quats_passive)
        q_active = _quat_conjugate(q_passive)
        R = _quat_to_matrix_active(q_active)
        return self.forward_irreps(R)

    def forward_from_quaternions(
        self, quats: torch.Tensor, raw: bool = False
    ) -> torch.Tensor:
        """
        Quaternion path that expects active quaternions [w, x, y, z].
        """
        q = self._to_device_dtype(quats)
        R = _quat_to_matrix_active(q)
        return self.forward_raw(R) if raw else self.forward_irreps(R)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.forward_irreps(R)

    # --------------------------------------------------------
    # Diagnostics (robust, float64)
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
        Runs in float64 by default even if model trains in float32.
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
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding("O", dtype=dtype, device=device)


def build_local_iso_hcp_embedding(
    d6_convention: str = "z_axis",
    dtype: torch.dtype = torch.float32,
    device=None,
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding(
        "D6", d6_convention=d6_convention, dtype=dtype, device=device
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
