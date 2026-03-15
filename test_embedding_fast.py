import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor


# ============================================================
# Basic helpers
# ============================================================

def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _group_mats_from_quaternions(sym_quats: torch.Tensor) -> torch.Tensor:
    return o3.quaternion_to_matrix(sym_quats)


def rot_x(theta: float, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=dtype,
        device=device,
    )


def rot_y(theta: float, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=dtype,
        device=device,
    )


def rot_z(theta: float, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


# ============================================================
# Symmetry groups
# ============================================================

def build_fcc_syms_mtex(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """24 proper cubic rotations (FCC / O) as quaternions [w, x, y, z] in active convention."""
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


def build_hcp_syms_mtex(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """12 proper D6 rotations (HCP) as quaternions [w, x, y, z] in active convention."""
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


def cubic_group_O(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    mats = _group_mats_from_quaternions(build_fcc_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 24:
        raise RuntimeError(f"Expected 24 cubic rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_zaxis(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    mats = _group_mats_from_quaternions(build_hcp_syms_mtex(dtype=dtype, device=device))
    if mats.shape[0] != 12:
        raise RuntimeError(f"Expected 12 D6 rotations, got {mats.shape[0]}")
    return mats


def dihedral_group_D6_paper(*, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """
    D6 in the HL paper frame. This is conjugate to the z-axis version.
    We build it directly; this is init-time only, so speed here is not critical.
    """
    r = rot_x(2.0 * math.pi / 6.0, dtype=dtype, device=device)
    s = rot_y(math.pi, dtype=dtype, device=device)

    mats = []
    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(Rk.clone())
        Rk = r @ Rk

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(s @ Rk)
        Rk = r @ Rk

    out = []
    for R in mats:
        if not any(torch.allclose(R, Q, atol=1e-6, rtol=0.0) for Q in out):
            out.append(R)

    if len(out) != 12:
        raise RuntimeError(f"Expected 12 D6 rotations, got {len(out)}")
    return torch.stack(out, dim=0)


# ============================================================
# Tensor / irreps helpers
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
    Faster tensor powers for the only ranks we actually use (2,4,6).
    Input:  v shape (..., G, 3)
    Output: (..., G, 3**rank)
    """
    if rank == 1:
        return v

    # rank-2
    p2 = (v.unsqueeze(-1) * v.unsqueeze(-2)).reshape(*v.shape[:-1], 9)
    if rank == 2:
        return p2

    # rank-4 = (v⊗v) ⊗ (v⊗v)
    p4 = (p2.unsqueeze(-1) * p2.unsqueeze(-2)).reshape(*v.shape[:-1], 81)
    if rank == 4:
        return p4

    # rank-6 = (v⊗v⊗v⊗v) ⊗ (v⊗v)
    if rank == 6:
        return (p4.unsqueeze(-1) * p2.unsqueeze(-2)).reshape(*v.shape[:-1], 729)

    # Fallback generic path
    out = v
    for _ in range(rank - 1):
        out = (out.unsqueeze(-1) * v.unsqueeze(-2)).reshape(*out.shape[:-1], -1)
    return out


def _flat_to_multiindex(x: torch.Tensor, rank: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], *([3] * rank))


def _drop_scalar_irreps(
    irreps: o3.Irreps, y: torch.Tensor
) -> tuple[torch.Tensor, o3.Irreps]:
    keep = []
    out_irreps = []
    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        if ir.l != 0:
            keep.append(y[..., sl])
            out_irreps.append((mul, ir))
    y_out = y[..., :0] if len(keep) == 0 else torch.cat(keep, dim=-1)
    return y_out, o3.Irreps(out_irreps)


def _build_nonscalar_projector(
    ct: CartesianTensor,
    rtp,
    rank: int,
    *,
    dtype: torch.dtype,
    device=None,
) -> tuple[torch.Tensor, o3.Irreps]:
    """
    Precompute the linear map:
        flat Cartesian tensor (dim 3**rank) -> non-scalar irreps coordinates.
    """
    flat_dim = 3 ** rank
    with torch.no_grad():
        basis = torch.eye(flat_dim, dtype=dtype, device=device)
        basis_multi = basis.reshape(flat_dim, *([3] * rank))
        y_full = ct.from_cartesian(basis_multi)  # [flat_dim, full_irrep_dim]
        y_keep, irreps_no_scalar = _drop_scalar_irreps(rtp.irreps_out, y_full)
        proj = y_keep.contiguous()  # [flat_dim, out_dim]
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
    Faster HL-style local-isometric crystal embedding.

    Training-oriented changes:
    - precompute irreps projection matrices once
    - avoid ct.from_cartesian in every forward pass
    - use float32 by default
    """

    def __init__(
        self,
        group_name: str,
        *,
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

        for spec in specs:
            block = nn.Module()
            block.name = spec.name
            block.rank = spec.rank
            block.beta = float(spec.beta)

            ct = CartesianTensor(spec.formula)
            rtp = ct.reduced_tensor_products()

            # Orbit directions {g u}_g, stored transposed for fast batched matmul
            anchors = torch.matmul(self.group_mats, spec.reference_direction)   # [G, 3]
            anchors_t = anchors.transpose(0, 1).contiguous()                    # [3, G]

            proj, irreps_no_scalar = _build_nonscalar_projector(
                ct,
                rtp,
                spec.rank,
                dtype=dtype,
                device=device,
            )

            block.register_buffer("anchors_t", anchors_t, persistent=False)
            block.register_buffer("proj", proj, persistent=False)
            block.irreps = irreps_no_scalar
            block.out_dim = proj.shape[-1]
            block.raw_dim = 3 ** spec.rank

            self.blocks.append(block)
            irreps_out_parts += list(irreps_no_scalar)

        self.irreps_out = o3.Irreps(irreps_out_parts)

    def _to_device_dtype(self, R: torch.Tensor) -> torch.Tensor:
        return R.to(dtype=self.dtype, device=self.group_mats.device)

    def _orbit_average_flat(
        self,
        R: torch.Tensor,
        anchors_t: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        """
        R:         (..., 3, 3)
        anchors_t: (3, G)
        returns:   (..., 3**rank)
        """
        # (..., 3, G) -> (..., G, 3)
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

    def forward_from_quaternions(self, quats: torch.Tensor, *, raw: bool = False) -> torch.Tensor:
        q = quats.to(dtype=self.dtype, device=self.group_mats.device)
        R = o3.quaternion_to_matrix(q)
        return self.forward_raw(R) if raw else self.forward_irreps(R)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.forward_irreps(R)

    # --------------------------------------------------------
    # Diagnostics (not training-critical)
    # --------------------------------------------------------

    @torch.no_grad()
    def gram_at_identity(self, *, eps: float = 1e-6, use_raw: bool = False) -> torch.Tensor:
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
    dtype: torch.dtype = torch.float32,
    device=None,
) -> LocalIsoCrystalEmbedding:
    return LocalIsoCrystalEmbedding("O", dtype=dtype, device=device)


def build_local_iso_hcp_embedding(
    *,
    d6_convention: str = "z_axis",
    dtype: torch.dtype = torch.float32,
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

"""
Smoke / regression tests for LocalIsoCrystalEmbedding:
1) right-side invariance:   E(R g) = E(R)
2) local isometry:         Gram at identity ~= I_3

Usage:
    python test_local_iso_embedding.py

Adjust MODULE_NAME below to match your file name, e.g.
    from local_iso_embedding import ...
"""

import math
from dataclasses import dataclass

import torch
from e3nn import o3

# ---------------------------------------------------------------------
# CHANGE THIS IMPORT TO MATCH YOUR MODULE FILE NAME
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def random_rotation_matrices(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample random active rotations in SO(3) from normalized random quaternions.
    Returns shape [n, 3, 3].
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(n, 4, dtype=dtype, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[:, :1] < 0.0, -q, q)
    return o3.quaternion_to_matrix(q)


@dataclass
class TestResult:
    name: str
    passed: bool
    value: float
    tol: float


def check_right_invariance(
    model,
    *,
    n_trials: int = 32,
    tol: float = 1e-10,
    use_raw: bool = False,
    seed: int = 0,
) -> TestResult:
    """
    Check max_{R,g} ||E(Rg) - E(R)||_inf
    """
    embed_fn = model.forward_raw if use_raw else model.forward_irreps
    device = model.group_mats.device
    dtype = model.group_mats.dtype

    R_batch = random_rotation_matrices(
        n_trials, dtype=dtype, device=device, seed=seed
    )

    errs = []
    with torch.no_grad():
        ER = embed_fn(R_batch)  # [B, D]
        for g in model.group_mats:
            ERg = embed_fn(R_batch @ g)  # [B, D]
            errs.append((ERg - ER).abs().amax())

    err = float(torch.stack(errs).amax().item())
    return TestResult(
        name=f"right_invariance({'raw' if use_raw else 'irreps'})",
        passed=(err < tol),
        value=err,
        tol=tol,
    )


def check_local_isometry(
    model,
    *,
    tol: float = 1e-7,
    eps: float = 1e-7,
    use_raw: bool = False,
) -> tuple[TestResult, torch.Tensor]:
    """
    Check ||G - I||_max where G is the pullback Gram matrix at identity.
    """
    G = model.gram_at_identity(eps=eps, use_raw=use_raw)
    I = torch.eye(3, dtype=G.dtype, device=G.device)
    err = float((G - I).abs().amax().item())
    return (
        TestResult(
            name=f"local_isometry({'raw' if use_raw else 'irreps'})",
            passed=(err < tol),
            value=err,
            tol=tol,
        ),
        G,
    )


def print_result(res: TestResult) -> None:
    status = "PASS" if res.passed else "FAIL"
    print(f"[{status}] {res.name:28s}  value={res.value:.3e}  tol={res.tol:.3e}")


# ---------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------

def run_one_model(
    label: str,
    model,
    *,
    inv_tol_irreps: float,
    inv_tol_raw: float,
    gram_tol_irreps: float,
    gram_tol_raw: float,
    eps: float,
    n_trials: int,
    seed: int,
) -> bool:
    print("=" * 80)
    print(label)
    print(f"device={model.group_mats.device}, dtype={model.group_mats.dtype}")
    print(f"irreps_out = {model.irreps_out}")
    print("-" * 80)

    ok = True

    # Right invariance
    r1 = check_right_invariance(
        model,
        n_trials=n_trials,
        tol=inv_tol_irreps,
        use_raw=False,
        seed=seed,
    )
    print_result(r1)
    ok &= r1.passed

    r2 = check_right_invariance(
        model,
        n_trials=n_trials,
        tol=inv_tol_raw,
        use_raw=True,
        seed=seed,
    )
    print_result(r2)
    ok &= r2.passed

    # Local isometry
    g1_res, G1 = check_local_isometry(
        model,
        tol=gram_tol_irreps,
        eps=eps,
        use_raw=False,
    )
    print_result(g1_res)
    print("Gram (irreps):")
    print(G1)
    ok &= g1_res.passed

    g2_res, G2 = check_local_isometry(
        model,
        tol=gram_tol_raw,
        eps=eps,
        use_raw=True,
    )
    print_result(g2_res)
    print("Gram (raw):")
    print(G2)
    ok &= g2_res.passed

    print()
    return ok


def main():
    # -------------------------------------------------------------
    # Recommend float64 for geometry tests.
    # If you switch to float32, loosen tolerances by ~1e2 to 1e4.
    # -------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    eps = 1e-3
    n_trials = 1024
    seed = 1234

    # Tolerances for float64
    inv_tol_irreps = 1e-6
    inv_tol_raw = 1e-6
    gram_tol_irreps = 5e-5
    gram_tol_raw = 5e-5

    models = [
        (
            "FCC / cubic O",
            build_local_iso_fcc_embedding(dtype=dtype, device=device),
        ),
        (
            "HCP / D6 (z_axis)",
            build_local_iso_hcp_embedding(
                d6_convention="z_axis", dtype=dtype, device=device
            ),
        ),
        (
            "HCP / D6 (paper)",
            build_local_iso_hcp_embedding(
                d6_convention="paper", dtype=dtype, device=device
            ),
        ),
    ]

    all_ok = True
    for label, model in models:
        ok = run_one_model(
            label,
            model,
            inv_tol_irreps=inv_tol_irreps,
            inv_tol_raw=inv_tol_raw,
            gram_tol_irreps=gram_tol_irreps,
            gram_tol_raw=gram_tol_raw,
            eps=eps,
            n_trials=n_trials,
            seed=seed,
        )
        all_ok &= ok

    print("=" * 80)
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()