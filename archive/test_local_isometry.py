# import math
# from dataclasses import dataclass
# from typing import List, Tuple

# import torch
# import torch.nn as nn
# from e3nn import o3
# from e3nn.io import CartesianTensor


# # ============================================================
# # Basic rotations and finite groups
# # ============================================================

# def rot_x(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
#     c = math.cos(theta)
#     s = math.sin(theta)
#     return torch.tensor(
#         [[1.0, 0.0, 0.0],
#          [0.0, c,   -s ],
#          [0.0, s,    c ]],
#         dtype=dtype, device=device
#     )


# def rot_z(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
#     c = math.cos(theta)
#     s = math.sin(theta)
#     return torch.tensor(
#         [[c,  -s,  0.0],
#          [s,   c,  0.0],
#          [0.0, 0.0, 1.0]],
#         dtype=dtype, device=device
#     )


# def cubic_group_O(*, dtype=torch.float64, device=None) -> List[torch.Tensor]:
#     """
#     Proper cubic group O as all signed permutation matrices with det +1.
#     Size = 24.
#     """
#     mats: List[torch.Tensor] = []
#     eye = torch.eye(3, dtype=dtype, device=device)

#     perms = [
#         (0, 1, 2), (0, 2, 1),
#         (1, 0, 2), (1, 2, 0),
#         (2, 0, 1), (2, 1, 0),
#     ]

#     for p in perms:
#         P = eye[:, list(p)]
#         for sx in (-1, 1):
#             for sy in (-1, 1):
#                 for sz in (-1, 1):
#                     S = torch.diag(torch.tensor([sx, sy, sz], dtype=dtype, device=device))
#                     R = S @ P
#                     if torch.det(R) > 0:
#                         if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in mats):
#                             mats.append(R)

#     assert len(mats) == 24, f"Expected 24 cubic rotations, got {len(mats)}"
#     return mats


# import math
# import torch


# def rot_y(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
#     c = math.cos(theta)
#     s = math.sin(theta)
#     return torch.tensor(
#         [[ c, 0.0,  s ],
#          [0.0, 1.0, 0.0],
#          [-s, 0.0,  c ]],
#         dtype=dtype, device=device
#     )


# def rot_x(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
#     c = math.cos(theta)
#     s = math.sin(theta)
#     return torch.tensor(
#         [[1.0, 0.0, 0.0],
#          [0.0,  c, -s ],
#          [0.0,  s,  c ]],
#         dtype=dtype, device=device
#     )


# def dihedral_group_D6_paper(*, dtype=torch.float64, device=None):
#     """
#     D6 in the convention used by Hielscher–Lippert:
#       - principal 6-fold axis parallel to e1 (x-axis)
#       - second generator is a pi-rotation about e2 (y-axis)

#     Elements: {r^k, s r^k}, k=0,...,5
#     """
#     r = rot_x(2.0 * math.pi / 6.0, dtype=dtype, device=device)   # 60° about x
#     s = rot_y(math.pi, dtype=dtype, device=device)               # 180° about y

#     mats = []

#     Rk = torch.eye(3, dtype=dtype, device=device)
#     for _ in range(6):
#         mats.append(Rk.clone())
#         Rk = r @ Rk

#     Rk = torch.eye(3, dtype=dtype, device=device)
#     for _ in range(6):
#         mats.append(s @ Rk)
#         Rk = r @ Rk

#     out = []
#     for R in mats:
#         if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in out):
#             out.append(R)

#     assert len(out) == 12, f"Expected 12 D6 rotations, got {len(out)}"
#     return out


# # ============================================================
# # Tensor helpers
# # ============================================================

# def full_symmetry_formula(rank: int) -> str:
#     """
#     e3nn symmetry formula for a fully symmetric rank-r tensor.
#     Adjacent transpositions generate the full symmetric group.
#     """
#     letters = list("ijklmnopqrstuvwxyzabcdefgh")
#     idx = letters[:rank]
#     terms = ["".join(idx)]
#     for k in range(rank - 1):
#         tmp = idx.copy()
#         tmp[k], tmp[k + 1] = tmp[k + 1], tmp[k]
#         terms.append("".join(tmp))
#     return "=".join(terms)


# def tensor_power_flat(v: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Raw Cartesian tensor power v^{⊗ rank}, flattened at the end.

#     Input:
#         v : (..., 3)

#     Output:
#         (..., 3**rank)
#     """
#     out = v
#     for _ in range(rank - 1):
#         tmp = torch.einsum("...a,...b->...ab", out, v)
#         out = tmp.reshape(*tmp.shape[:-2], -1)
#     return out


# def flat_to_multiindex(t: torch.Tensor, rank: int) -> torch.Tensor:
#     """
#     Reshape (..., 3**rank) -> (..., 3, 3, ..., 3) with rank copies of 3.
#     """
#     return t.reshape(*t.shape[:-1], *([3] * rank))


# def drop_scalar_irreps(irreps_like, y: torch.Tensor) -> Tuple[torch.Tensor, o3.Irreps]:
#     """
#     Remove l=0 blocks from an e3nn Irreps-like object and the corresponding coordinates.
#     """
#     keep_parts = []
#     out_irreps = []
#     for sl, (mul, ir) in zip(irreps_like.slices(), irreps_like):
#         if ir.l != 0:
#             keep_parts.append(y[..., sl])
#             out_irreps.append((mul, ir))

#     if len(keep_parts) == 0:
#         y_out = y[..., :0]
#     else:
#         y_out = torch.cat(keep_parts, dim=-1)

#     return y_out, o3.Irreps(out_irreps).regroup()


# # ============================================================
# # Cartesian-first locally isometric embedding
# # ============================================================

# @dataclass
# class BlockSpec:
#     name: str
#     rank: int
#     beta: float
#     formula: str
#     u: torch.Tensor


# class LocalIsoCrystalEmbedding(nn.Module):
#     """
#     Cartesian-first implementation of the Hielscher–Lippert front-end.

#     O  : u = e1,       alpha = 4,     beta = 3 / (2 sqrt(2))
#     D6 : u = (e1, e2), alpha = (2,6), beta = (1/sqrt(24), 2 sqrt(2)/3)

#     forward_cartesian(R): raw centered Cartesian embedding
#     forward(R):          centered embedding converted to e3nn irreps coordinates
#     """

#     def __init__(self, group_name: str, *, dtype=torch.float64, device=None):
#         super().__init__()
#         self.group_name = group_name
#         self.dtype = dtype
#         self.device = device

#         e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
#         e2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

#         if group_name == "O":
#             group = cubic_group_O(dtype=dtype, device=device)
#             specs = [
#                 BlockSpec(
#                     name="rank4",
#                     rank=4,
#                     beta=3.0 / (2.0 * math.sqrt(2.0)),
#                     formula=full_symmetry_formula(4),
#                     u=e1,
#                 )
#             ]
#         elif group_name == "D6":
#             group = dihedral_group_D6(dtype=dtype, device=device)
#             specs = [
#                 BlockSpec(
#                     name="rank2",
#                     rank=2,
#                     beta=1.0 / math.sqrt(24.0),
#                     formula=full_symmetry_formula(2),
#                     u=e1,
#                 ),
#                 BlockSpec(
#                     name="rank6",
#                     rank=6,
#                     beta=2.0 * math.sqrt(2.0) / 3.0,
#                     formula=full_symmetry_formula(6),
#                     u=e2,
#                 ),
#             ]
#         else:
#             raise ValueError("group_name must be 'O' or 'D6'")

#         self.group = group
#         self.blocks = nn.ModuleList()
#         irreps_out_parts = []

#         for spec in specs:
#             block = nn.Module()
#             block.name = spec.name
#             block.rank = spec.rank
#             block.beta = spec.beta
#             block.ct = CartesianTensor(spec.formula)

#             # orbit anchors g u, shape (|G|, 3)
#             anchors = torch.stack([g @ spec.u for g in group], dim=0)
#             block.register_buffer("anchors", anchors)

#             # figure out irreps after dropping scalar part
#             dummy = torch.zeros(*([3] * spec.rank), dtype=dtype, device=device)
#             dummy_ir = block.ct.from_cartesian(dummy)
#             _, block_irreps = drop_scalar_irreps(block.ct, dummy_ir)

#             block.irreps = block_irreps
#             self.blocks.append(block)
#             irreps_out_parts += list(block_irreps)

#         self.irreps_out = o3.Irreps(irreps_out_parts).regroup()

#     def _orbit_average_flat(self, R: torch.Tensor, anchors: torch.Tensor, rank: int) -> torch.Tensor:
#         """
#         Compute mean_g (R (g u))^{⊗ rank} as a flattened Cartesian tensor.

#         R      : (..., 3, 3)
#         anchors: (G, 3)

#         returns: (..., 3**rank)
#         """
#         # v shape: (..., G, 3)
#         v = torch.einsum("...ij,gj->...gi", R, anchors)

#         # tensor power per orbit element: (..., G, 3**rank)
#         tp = tensor_power_flat(v, rank)

#         # average over G
#         return tp.mean(dim=-2)

#     def forward_cartesian(self, R: torch.Tensor) -> torch.Tensor:
#         """
#         Raw centered Cartesian embedding, concatenated blockwise.

#         Returns:
#             (..., d_cart_centered)
#         """
#         if R.shape[-2:] != (3, 3):
#             raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

#         R = R.to(dtype=self.dtype, device=self.device)

#         outs = []
#         for block in self.blocks:
#             T_flat = self._orbit_average_flat(R, block.anchors, block.rank)
#             T_multi = flat_to_multiindex(T_flat, block.rank)

#             # Convert to irreps, drop scalar = centered embedding,
#             # then go back to Cartesian coordinates if you want the centered Euclidean block.
#             y_full = block.ct.from_cartesian(T_multi)
#             y_centered, _ = drop_scalar_irreps(block.ct, y_full)

#             # Scale by paper's beta in the centered coordinates.
#             y_centered = block.beta * y_centered

#             outs.append(y_centered)

#         return torch.cat(outs, dim=-1)

#     def forward(self, R: torch.Tensor) -> torch.Tensor:
#         """
#         e3nn irreps coordinates for the centered embedding.

#         Returns:
#             (..., irreps_out.dim)
#         """
#         return self.forward_cartesian(R)


# # ============================================================
# # Local-isometry check at identity
# # ============================================================

# def gram_at_identity(module: nn.Module, eps: float = 1e-7) -> torch.Tensor:
#     """
#     Numerical Gram matrix of the three tangent vectors at identity.

#     For a locally isometric embedding, this should be close to I_3.
#     """
#     I = torch.eye(3, dtype=module.dtype, device=module.device)

#     s1 = torch.tensor(
#         [[0.0, 0.0, 0.0],
#          [0.0, 0.0, -1.0],
#          [0.0, 1.0, 0.0]],
#         dtype=module.dtype, device=module.device
#     )
#     s2 = torch.tensor(
#         [[0.0, 0.0, -1.0],
#          [0.0, 0.0, 0.0],
#          [1.0, 0.0, 0.0]],
#         dtype=module.dtype, device=module.device
#     )
#     s3 = torch.tensor(
#         [[0.0, -1.0, 0.0],
#          [1.0, 0.0, 0.0],
#          [0.0, 0.0, 0.0]],
#         dtype=module.dtype, device=module.device
#     )

#     # Normalize the so(3) basis under the Frobenius metric
#     basis = [s1 / math.sqrt(2.0), s2 / math.sqrt(2.0), s3 / math.sqrt(2.0)]

#     tangents = []
#     for S in basis:
#         Rp = torch.matrix_exp(eps * S)
#         Rm = torch.matrix_exp(-eps * S)

#         Ep = module(Rp.unsqueeze(0))[0]
#         Em = module(Rm.unsqueeze(0))[0]

#         tangents.append((Ep - Em) / (2.0 * eps))

#     G = torch.stack(
#         [torch.stack([(vi * vj).sum() for vj in tangents], dim=0) for vi in tangents],
#         dim=0,
#     )
#     return G


# # ============================================================
# # Example
# # ============================================================

# if __name__ == "__main__":
#     dtype = torch.float64
#     device = "cpu"

#     emb_O = LocalIsoCrystalEmbedding("O", dtype=dtype, device=device)
#     emb_D6 = LocalIsoCrystalEmbedding("D6", dtype=dtype, device=device)

#     print("O irreps_out :", emb_O.irreps_out)    # expected: 1x2e+1x4e
#     print("D6 irreps_out:", emb_D6.irreps_out)   # expected: 2x2e+1x4e+1x6e

#     G_O = gram_at_identity(emb_O)
#     G_D6 = gram_at_identity(emb_D6)

#     print("Gram at identity, O:\n", G_O)
#     print("Gram at identity, D6:\n", G_D6)

#     # sample forward pass
#     R = rot_z(0.3, dtype=dtype, device=device).unsqueeze(0)
#     yO = emb_O(R)
#     yD6 = emb_D6(R)
#     print("O feature shape :", yO.shape)
#     print("D6 feature shape:", yD6.shape)

import math
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor


# ============================================================
# Basic rotations
# ============================================================

def rot_x(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, c,   -s ],
         [0.0, s,    c ]],
        dtype=dtype, device=device
    )


def rot_y(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[ c,  0.0,  s ],
         [0.0, 1.0, 0.0],
         [-s,  0.0,  c ]],
        dtype=dtype, device=device
    )


def rot_z(theta: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c,  -s,  0.0],
         [s,   c,  0.0],
         [0.0, 0.0, 1.0]],
        dtype=dtype, device=device
    )


# ============================================================
# Finite groups
# ============================================================

def cubic_group_O(*, dtype=torch.float64, device=None) -> List[torch.Tensor]:
    mats: List[torch.Tensor] = []
    eye = torch.eye(3, dtype=dtype, device=device)

    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]

    for p in perms:
        P = eye[:, list(p)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    S = torch.diag(torch.tensor([sx, sy, sz], dtype=dtype, device=device))
                    R = S @ P
                    if torch.det(R) > 0:
                        if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in mats):
                            mats.append(R)

    assert len(mats) == 24
    return mats


def dihedral_group_D6_zaxis(*, dtype=torch.float64, device=None) -> List[torch.Tensor]:
    r = rot_z(2.0 * math.pi / 6.0, dtype=dtype, device=device)
    s = rot_x(math.pi, dtype=dtype, device=device)

    mats: List[torch.Tensor] = []

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(Rk.clone())
        Rk = r @ Rk

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(s @ Rk)
        Rk = r @ Rk

    out: List[torch.Tensor] = []
    for R in mats:
        if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in out):
            out.append(R)

    assert len(out) == 12
    return out


def dihedral_group_D6_paper(*, dtype=torch.float64, device=None) -> List[torch.Tensor]:
    r = rot_x(2.0 * math.pi / 6.0, dtype=dtype, device=device)
    s = rot_y(math.pi, dtype=dtype, device=device)

    mats: List[torch.Tensor] = []

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(Rk.clone())
        Rk = r @ Rk

    Rk = torch.eye(3, dtype=dtype, device=device)
    for _ in range(6):
        mats.append(s @ Rk)
        Rk = r @ Rk

    out: List[torch.Tensor] = []
    for R in mats:
        if not any(torch.allclose(R, Q, atol=1e-12, rtol=0.0) for Q in out):
            out.append(R)

    assert len(out) == 12
    return out


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


def tensor_power_flat(v: torch.Tensor, rank: int) -> torch.Tensor:
    out = v
    for _ in range(rank - 1):
        out = torch.einsum("...a,...b->...ab", out, v).reshape(*out.shape[:-1], -1)
    return out


def flat_to_multiindex(x: torch.Tensor, rank: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], *([3] * rank))


def drop_scalar_irreps(irreps: o3.Irreps, y: torch.Tensor) -> Tuple[torch.Tensor, o3.Irreps]:
    keep = []
    out_irreps = []

    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        if ir.l != 0:
            keep.append(y[..., sl])
            out_irreps.append((mul, ir))

    if len(keep) == 0:
        y_out = y[..., :0]
    else:
        y_out = torch.cat(keep, dim=-1)

    return y_out, o3.Irreps(out_irreps).regroup()


# ============================================================
# Block spec
# ============================================================

@dataclass
class RawBlockSpec:
    name: str
    rank: int
    beta: float
    u: torch.Tensor
    formula: str


# ============================================================
# Main embedding
# ============================================================

class LocalIsoCrystalEmbedding(nn.Module):
    def __init__(
        self,
        group_name: str,
        *,
        d6_convention: str = "z_axis",
        dtype: torch.dtype = torch.float64,
        device=None,
    ):
        super().__init__()
        self.group_name = group_name
        self.d6_convention = d6_convention
        self.dtype = dtype
        self.device = device

        e1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        e2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        e3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        specs: List[RawBlockSpec] = []

        if group_name == "O":
            self.group = cubic_group_O(dtype=dtype, device=device)
            specs.append(
                RawBlockSpec(
                    name="rank4",
                    rank=4,
                    beta=3.0 / (2.0 * math.sqrt(2.0)),
                    u=e1,
                    formula=full_symmetry_formula(4),
                )
            )

        elif group_name == "D6":
            if d6_convention == "paper":
                self.group = dihedral_group_D6_paper(dtype=dtype, device=device)
                u2 = e1
                u6 = e2
            elif d6_convention == "z_axis":
                self.group = dihedral_group_D6_zaxis(dtype=dtype, device=device)
                u2 = e3
                u6 = e1
            else:
                raise ValueError("d6_convention must be 'z_axis' or 'paper'")

            specs.extend([
                RawBlockSpec(
                    name="rank2",
                    rank=2,
                    beta=1.0 / math.sqrt(24.0),
                    u=u2,
                    formula=full_symmetry_formula(2),
                ),
                RawBlockSpec(
                    name="rank6",
                    rank=6,
                    beta=2.0 * math.sqrt(2.0) / 3.0,
                    u=u6,
                    formula=full_symmetry_formula(6),
                ),
            ])
        else:
            raise ValueError("group_name must be 'O' or 'D6'")

        self.blocks = nn.ModuleList()
        irreps_out_parts = []

        for spec in specs:
            block = nn.Module()
            block.name = spec.name
            block.rank = spec.rank
            block.beta = spec.beta

            # API FIX:
            block.ct = CartesianTensor(spec.formula)
            block.rtp = block.ct.reduced_tensor_products()

            anchors = torch.stack([g @ spec.u for g in self.group], dim=0)
            block.register_buffer("anchors", anchors)

            dummy = torch.zeros(*([3] * spec.rank), dtype=dtype, device=device)
            y_full = block.ct.from_cartesian(dummy)

            # use block.rtp.irreps_out instead of block.ct.irreps_out
            _, irreps_no_scalar = drop_scalar_irreps(block.rtp.irreps_out, y_full)
            block.irreps = irreps_no_scalar

            self.blocks.append(block)
            irreps_out_parts += list(irreps_no_scalar)

        self.irreps_out = o3.Irreps(irreps_out_parts).regroup()

    def _orbit_average_flat(self, R: torch.Tensor, anchors: torch.Tensor, rank: int) -> torch.Tensor:
        v = torch.einsum("...ij,gj->...gi", R, anchors)
        tp = tensor_power_flat(v, rank)
        return tp.mean(dim=-2)

    def forward_raw(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = R.to(dtype=self.dtype, device=self.device)

        outs = []
        for block in self.blocks:
            x = self._orbit_average_flat(R, block.anchors, block.rank)
            outs.append(block.beta * x)

        return torch.cat(outs, dim=-1)

    def forward_irreps(self, R: torch.Tensor) -> torch.Tensor:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3), got {tuple(R.shape)}")

        R = R.to(dtype=self.dtype, device=self.device)

        outs = []
        for block in self.blocks:
            x_flat = self._orbit_average_flat(R, block.anchors, block.rank)
            x_multi = flat_to_multiindex(x_flat, block.rank)

            y_full = block.ct.from_cartesian(x_multi)
            y_centered, _ = drop_scalar_irreps(block.rtp.irreps_out, y_full)
            outs.append(block.beta * y_centered)

        return torch.cat(outs, dim=-1)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.forward_irreps(R)


# ============================================================
# Tests
# ============================================================

def gram_at_identity(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    I = torch.eye(3, dtype=dtype, device=device)

    s1 = torch.tensor(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0]],
        dtype=dtype, device=device
    )
    s2 = torch.tensor(
        [[0.0, 0.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]],
        dtype=dtype, device=device
    )
    s3 = torch.tensor(
        [[0.0, -1.0, 0.0],
         [1.0,  0.0, 0.0],
         [0.0,  0.0, 0.0]],
        dtype=dtype, device=device
    )
    basis = [s1, s2, s3]

    tangents = []
    for S in basis:
        Rp = torch.matrix_exp(eps * S)
        Rm = torch.matrix_exp(-eps * S)

        Ep = embed_fn(Rp.unsqueeze(0))[0]
        Em = embed_fn(Rm.unsqueeze(0))[0]

        tangents.append((Ep - Em) / (2.0 * eps))

    G = torch.stack(
        [torch.stack([(vi * vj).sum() for vj in tangents], dim=0) for vi in tangents],
        dim=0,
    )
    return G


def max_right_invariance_error(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    group: List[torch.Tensor],
    *,
    n_trials: int = 10,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> float:
    torch.manual_seed(0)
    errs = []

    for _ in range(n_trials):
        axis = torch.randn(3, dtype=dtype, device=device)
        axis = axis / axis.norm()
        angle = 2.0 * math.pi * torch.rand((), dtype=dtype, device=device)

        K = torch.tensor(
            [[0.0,      -axis[2],  axis[1]],
             [axis[2],   0.0,     -axis[0]],
             [-axis[1],  axis[0],  0.0]],
            dtype=dtype, device=device
        )
        R = torch.matrix_exp(angle * K)

        ER = embed_fn(R.unsqueeze(0))[0]
        for g in group:
            ERg = embed_fn((R @ g).unsqueeze(0))[0]
            errs.append((ERg - ER).abs().max())

    return float(torch.stack(errs).max().item())


def _load_repo_local_iso_module():
    """
    Load models/local_iso_embedding.py directly to avoid models/__init__ side effects.
    """
    p = Path(__file__).resolve().parent / "models" / "local_iso_embedding.py"
    spec = importlib.util.spec_from_file_location("repo_local_iso_embedding", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rand_unit_quats(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_repo_local_iso_fcc_auto_prunes_to_4e():
    mod = _load_repo_local_iso_module()
    emb = mod.build_local_iso_fcc_embedding(dtype=torch.float32).eval()

    assert str(emb.irreps_a1) == "1x4e"
    assert emb.irreps_a1.dim == 9
    assert str(emb.irreps_full) == "1x2e+1x4e"
    assert emb.irreps_full.dim == 14

    table = emb.get_a1_multiplicity_table(6)
    assert table == {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1}

    q = _rand_unit_quats(1024, dtype=torch.float32)
    y_full = emb.forward_irreps_passive(q)
    y_in = emb.forward_irreps_passive(q, active_only=True)
    assert tuple(y_full.shape) == (1024, 14)
    assert tuple(y_in.shape) == (1024, 9)

    # FCC full output keeps 2e block for TP bookkeeping, but it should be numerically dead.
    two_e_slice = emb.irreps_full.slices()[0]
    assert float(y_full[:, two_e_slice].abs().max().item()) < 1e-5
    assert float(y_in.abs().max().item()) > 1e-2


def test_repo_local_iso_hcp_irreps_and_multiplicities():
    mod = _load_repo_local_iso_module()
    emb = mod.build_local_iso_hcp_embedding(dtype=torch.float32, d6_convention="z_axis").eval()

    assert str(emb.irreps_a1) == "2x2e+1x4e+1x6e"
    assert emb.irreps_a1.dim == 32
    assert str(emb.irreps_full) == "2x2e+1x4e+1x6e"
    assert emb.irreps_full.dim == 32

    table = emb.get_a1_multiplicity_table(6)
    assert table == {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 2}

    q = _rand_unit_quats(1024, dtype=torch.float32)
    y = emb.forward_irreps_passive(q)
    y_in = emb.forward_irreps_passive(q, active_only=True)
    assert tuple(y.shape) == (1024, 32)
    assert tuple(y_in.shape) == (1024, 32)

    for (_, _), sl in zip(emb.irreps_full, emb.irreps_full.slices()):
        block = y[:, sl]
        assert float(block.abs().max().item()) > 1e-2


def run_tests(dtype=torch.float64, device="cpu", d6_convention="z_axis"):
    emb_O = LocalIsoCrystalEmbedding("O", dtype=dtype, device=device)
    emb_D6 = LocalIsoCrystalEmbedding("D6", dtype=dtype, device=device, d6_convention=d6_convention)

    print("O irreps_out :", emb_O.irreps_out)
    print("D6 irreps_out:", emb_D6.irreps_out)
    print()

    G_O_raw = gram_at_identity(lambda R: emb_O.forward_raw(R), dtype=dtype, device=device)
    G_D6_raw = gram_at_identity(lambda R: emb_D6.forward_raw(R), dtype=dtype, device=device)

    print("=== Raw HL local-isometry test ===")
    print("G_O_raw =")
    print(G_O_raw)
    print()
    print("G_D6_raw =")
    print(G_D6_raw)
    print()

    I3 = torch.eye(3, dtype=dtype, device=device)
    print("Deviation from identity:")
    print("||G_O_raw  - I||_max =", float((G_O_raw - I3).abs().max().item()))
    print("||G_D6_raw - I||_max =", float((G_D6_raw - I3).abs().max().item()))
    print()

    err_O = max_right_invariance_error(lambda R: emb_O.forward_raw(R), emb_O.group, dtype=dtype, device=device)
    err_D6 = max_right_invariance_error(lambda R: emb_D6.forward_raw(R), emb_D6.group, dtype=dtype, device=device)
    print("Right-invariance max errors (raw):")
    print(f"O  : {err_O:.3e}")
    print(f"D6 : {err_D6:.3e}")
    print()

    G_O_ir = gram_at_identity(lambda R: emb_O.forward_irreps(R), dtype=dtype, device=device)
    G_D6_ir = gram_at_identity(lambda R: emb_D6.forward_irreps(R), dtype=dtype, device=device)

    print("=== e3nn irreps-coordinate diagnostic ===")
    print("G_O_irreps =")
    print(G_O_ir)
    print()
    print("G_D6_irreps =")
    print(G_D6_ir)
    print()

    R = rot_z(0.3, dtype=dtype, device=device).unsqueeze(0)
    print("Feature shapes:")
    print("O raw     :", tuple(emb_O.forward_raw(R).shape))
    print("D6 raw    :", tuple(emb_D6.forward_raw(R).shape))
    print("O irreps  :", tuple(emb_O.forward_irreps(R).shape))
    print("D6 irreps :", tuple(emb_D6.forward_irreps(R).shape))


if __name__ == "__main__":
    run_tests(dtype=torch.float64, device="cpu", d6_convention="z_axis")
