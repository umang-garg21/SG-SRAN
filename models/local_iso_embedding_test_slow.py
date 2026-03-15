import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _group_mats_from_quaternions(sym_quats: torch.Tensor) -> torch.Tensor:
    # e3nn convention: active rotations
    q = _normalize_quaternions(sym_quats)
    return o3.quaternion_to_matrix(q)


def rot_x(theta: float, *, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=dtype,
        device=device,
    )


def rot_y(theta: float, *, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=dtype,
        device=device,
    )


def rot_z(theta: float, *, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def build_fcc_syms_mtex(*, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """24 proper cubic rotations (FCC / group O) as unit quaternions [w, x, y, z]."""
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


def build_hcp_syms_mtex(*, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
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
        dtype=dtype,
        device=device,
    )


def cubic_group_O(*, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """Proper cubic group O as 24 active rotation matrices (from quaternions)."""
    mats = _group_mats_from_quaternions(build_fcc_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 24:
        raise RuntimeError(f"Expected 24 cubic rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_zaxis(*, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """D6 with principal axis aligned to z (from quaternion table)."""
    mats = _group_mats_from_quaternions(build_hcp_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 12:
        raise RuntimeError(f"Expected 12 D6 rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_paper(*, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """D6 in the Hielscher-Lippert paper axis convention."""
    r = rot_x(2.0 * math.pi / 6.0, dtype=dtype, device=device)
    s = rot_y(math.pi, dtype=dtype, device=device)

    mats: list[torch.Tensor] = []

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(Rk.clone())
        Rk = r @ Rk

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(s @ Rk)
        Rk = r @ Rk

    out: list[torch.Tensor] = []
    for R in mats:
        if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in out):
            out.append(R)

    if len(out) != 12:
        raise RuntimeError(f"Expected 12 D6 rotations, got {len(out)}")
    return torch.stack(out, dim=0)


def full_symmetry_formula(rank: int) -> str:
    letters = list("ijklmnopqrstuvwxyzabcdefgh")
    idx = letters[:rank]
    terms = ["".join(idx)]
    for k in range(rank - 1):
        tmp = idx.copy()
        tmp[k], tmp[k + 1] = tmp[k + 1], tmp[k]
        terms.append("".join(tmp))
    return "=".join(terms)


def tensor_power_flat(v: torch.Tensor, rank: int) -> torch.Tensor:
    out = v
    for _ in range(rank - 1):
        out = torch.einsum("...a,...b->...ab", out, v).reshape(*out.shape[:-1], -1)
    return out


def flat_to_multiindex(x: torch.Tensor, rank: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], *([3] * rank))


def drop_scalar_irreps(irreps: o3.Irreps, y: torch.Tensor) -> tuple[torch.Tensor, o3.Irreps]:
    keep = []
    out_irreps = []

    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        if ir.l != 0:
            keep.append(y[..., sl])
            out_irreps.append((mul, ir))

    y_out = y[..., :0] if len(keep) == 0 else torch.cat(keep, dim=-1)
    return y_out, o3.Irreps(out_irreps).regroup()


@dataclass
class _RawBlockSpec:
    name: str
    rank: int
    beta: float
    anchor: torch.Tensor
    formula: str


class LocalIsoCrystalEmbedding(nn.Module):
    """Reusable HL-style local-isometric crystal embedding.

    Supported groups:
    - ``group_name='O'`` (FCC/cubic proper rotations)
    - ``group_name='D6'`` (HCP proper rotations)

    Forward expects rotation matrices ``(..., 3, 3)`` and returns irreps
    coordinates with scalar blocks removed.
    """

    def __init__(
        self,
        group_name: str,
        *,
        d6_convention: str = "z_axis",
        dtype: torch.dtype = torch.float64,
        device=None,
    ):
        super().__init__()
        self.group_name = str(group_name)
        self.d6_convention = str(d6_convention)
        self.dtype = dtype
        self.device = device

        e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        e2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        e3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        specs: list[_RawBlockSpec] = []

        if self.group_name == "O":
            group_quats = build_fcc_syms_mtex(dtype=dtype, device=device)
            group = _group_mats_from_quaternions(group_quats)
            specs.append(
                _RawBlockSpec(
                    name="rank4",
                    rank=4,
                    beta=3.0 / (2.0 * math.sqrt(2.0)),
                    anchor=e1,
                    formula=full_symmetry_formula(4),
                )
            )
        elif self.group_name == "D6":
            if self.d6_convention == "paper":
                group = dihedral_group_D6_paper(dtype=dtype, device=device)
                group_quats = _normalize_quaternions(o3.matrix_to_quaternion(group))
                u2, u6 = e1, e2
            elif self.d6_convention == "z_axis":
                group_quats = build_hcp_syms_mtex(dtype=dtype, device=device)
                group = _group_mats_from_quaternions(group_quats)
                u2, u6 = e3, e1
            else:
                raise ValueError("d6_convention must be 'z_axis' or 'paper'")

            specs.extend(
                [
                    _RawBlockSpec(
                        name="rank2",
                        rank=2,
                        beta=1.0 / math.sqrt(24.0),
                        anchor=u2,
                        formula=full_symmetry_formula(2),
                    ),
                    _RawBlockSpec(
                        name="rank6",
                        rank=6,
                        beta=2.0 * math.sqrt(2.0) / 3.0,
                        anchor=u6,
                        formula=full_symmetry_formula(6),
                    ),
                ]
            )
        else:
            raise ValueError("group_name must be 'O' or 'D6'")

        self.register_buffer("group_quats", _normalize_quaternions(group_quats))
        self.register_buffer("group_mats", group)

        self.blocks = nn.ModuleList()
        irreps_out_parts = []

        for spec in specs:
            block = nn.Module()
            block.name = spec.name
            block.rank = spec.rank
            block.beta = float(spec.beta)

            block.ct = CartesianTensor(spec.formula)
            block.rtp = block.ct.reduced_tensor_products()

            # Orbit anchors: {g u}_g in R^3.
            anchors = torch.einsum("gij,j->gi", self.group_mats, spec.anchor)
            block.register_buffer("anchors", anchors)

            dummy = torch.zeros(*([3] * spec.rank), dtype=dtype, device=device)
            y_full = block.ct.from_cartesian(dummy)
            _, irreps_no_scalar = drop_scalar_irreps(block.rtp.irreps_out, y_full)
            block.irreps = irreps_no_scalar

            self.blocks.append(block)
            irreps_out_parts += list(irreps_no_scalar)

        self.irreps_out = o3.Irreps(irreps_out_parts).regroup()

    def _orbit_average_flat(self, R: torch.Tensor, anchors: torch.Tensor, rank: int) -> torch.Tensor:
        v = torch.einsum("...ij,gj->...gi", R, anchors)
        return tensor_power_flat(v, rank).mean(dim=-2)

    def _to_device_dtype(self, R: torch.Tensor) -> torch.Tensor:
        return R.to(dtype=self.dtype, device=self.group_mats.device)

    def forward_raw(self, R: torch.Tensor) -> torch.Tensor:
        """Raw orbit-averaged Cartesian blocks (concatenated)."""
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            x = self._orbit_average_flat(R, block.anchors, block.rank)
            outs.append(block.beta * x)
        return torch.cat(outs, dim=-1)

    def forward_irreps(self, R: torch.Tensor) -> torch.Tensor:
        """Centered irreps coordinates (scalar blocks dropped)."""
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = self._to_device_dtype(R)
        outs = []
        for block in self.blocks:
            
            x_flat = self._orbit_average_flat(R, block.anchors, block.rank)
            x_multi = flat_to_multiindex(x_flat, block.rank)
            y_full = block.ct.from_cartesian(x_multi)
            y_centered, _ = drop_scalar_irreps(block.rtp.irreps_out, y_full)
            outs.append(block.beta * y_centered)
        return torch.cat(outs, dim=-1)

    def forward_from_quaternions(self, quats: torch.Tensor, *, raw: bool = False) -> torch.Tensor:
        """Convenience path from quaternions (..., 4)."""
        R = o3.quaternion_to_matrix(quats.to(dtype=self.dtype, device=self.group_mats.device))
        return self.forward_raw(R) if raw else self.forward_irreps(R)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.forward_irreps(R)

    @torch.no_grad()
    def gram_at_identity(self, *, eps: float = 1e-7, use_raw: bool = False) -> torch.Tensor:
        embed_fn: Callable[[torch.Tensor], torch.Tensor]
        embed_fn = self.forward_raw if use_raw else self.forward_irreps

        s1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=self.dtype,
            device=self.group_mats.device,
        )
        s2 = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=self.dtype,
            device=self.group_mats.device,
        )
        s3 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=self.dtype,
            device=self.group_mats.device,
        )
        basis = [s1, s2, s3]

        tangents = []
        for S in basis:
            Rp = torch.matrix_exp(eps * S)
            Rm = torch.matrix_exp(-eps * S)
            Ep = embed_fn(Rp.unsqueeze(0))[0]
            Em = embed_fn(Rm.unsqueeze(0))[0]
            tangents.append((Ep - Em) / (2.0 * eps))

        return torch.stack(
            [torch.stack([(vi * vj).sum() for vj in tangents], dim=0) for vi in tangents],
            dim=0,
        )

    @torch.no_grad()
    def right_invariance_error(
        self,
        *,
        n_trials: int = 10,
        use_raw: bool = False,
        seed: int = 0,
    ) -> float:
        embed_fn: Callable[[torch.Tensor], torch.Tensor]
        embed_fn = self.forward_raw if use_raw else self.forward_irreps

        torch.manual_seed(int(seed))
        device = self.group_mats.device
        errs = []

        for _ in range(int(n_trials)):
            axis = torch.randn(3, dtype=self.dtype, device=device)
            axis = axis / axis.norm().clamp_min(1e-12)
            angle = 2.0 * math.pi * torch.rand((), dtype=self.dtype, device=device)

            K = torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=self.dtype,
                device=device,
            )
            R = torch.matrix_exp(angle * K)

            ER = embed_fn(R.unsqueeze(0))[0]
            for g in self.group_mats:
                ERg = embed_fn((R @ g).unsqueeze(0))[0]
                errs.append((ERg - ER).abs().max())

        return float(torch.stack(errs).max().item())


def build_local_iso_fcc_embedding(
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding("O", dtype=dtype, device=device)


def build_local_iso_hcp_embedding(
    *,
    d6_convention: str = "z_axis",
    dtype: torch.dtype = torch.float64,
    device=None,
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding("D6", d6_convention=d6_convention, dtype=dtype, device=device)


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
