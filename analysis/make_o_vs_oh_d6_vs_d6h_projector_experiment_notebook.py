from __future__ import annotations

import json
from pathlib import Path


OUT = Path("analysis/O_vs_Oh_D6_vs_D6h_reynolds_projector_experiment.ipynb")

_CELL_COUNTER = 0


def _next_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"cell-{_CELL_COUNTER:03d}"


def md(source: str) -> dict:
    return {
        "id": _next_id(),
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "id": _next_id(),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# O vs Oh and D6 vs D6h Reynolds-projector experiment

This notebook checks whether the Reynolds projection to invariant irrep subspaces changes
when the proper rotational crystal group is replaced by the corresponding centrosymmetric
Laue group:

$$
O \quad\text{vs.}\quad O_h = O \cup (-I)O,
$$

and

$$
D_6 \quad\text{vs.}\quad D_{6h} = D_6 \cup (-I)D_6.
$$

The notebook uses the active convention throughout.

For a physical crystal orientation \(R\in SO(3)\):

$$
\text{right crystal action:}\qquad R \mapsto R H,\quad H\in G\subset SO(3),
$$

and

$$
\text{left sample-frame action:}\qquad R \mapsto A R,\quad A\in SO(3).
$$

The feature block is

$$
F_l(R)=D^l(R)U_l,
$$

where \(U_l\) spans the fixed subspace of the Reynolds projector. The required behavior is

$$
F_l(RH)=F_l(R),\qquad H\in G,
$$

and

$$
F_l(AR)=D^l(A)F_l(R),\qquad A\in SO(3).
$$

Important distinction: \(O_h\) and \(D_{6h}\) contain improper operations and are not
subgroups of \(SO(3)\). They cannot be the orientation quotient group \(SO(3)/G\). Here they
are used only for an \(O(3)\)-representation projector comparison.
        """
    ),
    md(
        r"""
## What should happen

For a centrosymmetric closure \(G_h = G\cup(-I)G\), an \(O(3)\) irrep carries a parity
\(p\in\{+1,-1\}\). Its Reynolds projector is

$$
P_l^{G_h,p} =
\frac{1}{2|G|}\sum_{H\in G}\left[D_l(H) + D_l^p((-I)H)\right].
$$

Since inversion is central,

$$
D_l^p((-I)H)=p\,D_l(H),
$$

so

$$
P_l^{G_h,p} = \frac{1+p}{2} P_l^G.
$$

Thus:

- even-parity \(O(3)\) irreps (\(p=+1\)) have the same projector as the proper group;
- odd-parity \(O(3)\) irreps (\(p=-1\)) project to zero;
- for ordinary polar spherical harmonics, \(p=(-1)^l\), so even \(l\) survives unchanged
  and odd \(l\) is killed by inversion.

The direct-Reynolds encoder used in the paper keeps \(l=4\) for FCC and \(l=2,4,6\) for HCP.
Those are even degrees, so the projector should be numerically identical under
\(O\) vs. polar-\(O_h\), and under \(D_6\) vs. polar-\(D_{6h}\). Odd-degree proper invariants,
if included, would differ.
        """
    ),
    code(
        r"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from e3nn import o3

ROOT = Path("/data/home/umang/Materials/Reynolds-QSR_paper")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.local_iso_embedding import (  # noqa: E402
    cubic_group_O,
    dihedral_group_D6_zaxis,
)

OUT_DIR = ROOT / "analysis" / "out" / "o_vs_oh_d6_vs_d6h_projector_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
DEVICE = torch.device("cpu")
MAX_L = 12
EIG_TOL = 1.0e-8
SEED = 20260709

print("repo:", ROOT)
print("out:", OUT_DIR)
print("torch:", torch.__version__)
        """
    ),
    code(
        r"""
def det_sign(mats: torch.Tensor) -> torch.Tensor:
    return torch.sign(torch.linalg.det(mats).round(decimals=8))


def centrosymmetric_closure(proper_group: torch.Tensor) -> torch.Tensor:
    # Return G_h = G union (-I)G as active orthogonal matrices.
    return torch.cat([proper_group, -proper_group], dim=0).contiguous()


def irrep_D(mats: torch.Tensor, l: int, parity: int = 1) -> torch.Tensor:
    # Real e3nn O(3) irrep matrices in the active convention.
    return o3.Irrep(int(l), int(parity)).D_from_matrix(mats.to(dtype=DTYPE, device=DEVICE))


def reynolds_projector(mats: torch.Tensor, l: int, parity: int = 1) -> torch.Tensor:
    D = irrep_D(mats, l, parity=parity)
    P = D.mean(dim=0)
    return 0.5 * (P + P.T)


def projector_rank(P: torch.Tensor, tol: float = EIG_TOL) -> int:
    evals = torch.linalg.eigvalsh(P)
    return int((evals > 1.0 - tol).sum().item())


def projector_basis_from_projector(P: torch.Tensor, tol: float = EIG_TOL) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(P)
    keep = evals > 1.0 - tol
    U = evecs[:, keep].contiguous()
    # Deterministic signs for reproducible printed comparisons.
    for j in range(U.shape[1]):
        nz = torch.nonzero(U[:, j].abs() > 1.0e-10, as_tuple=False)
        if nz.numel() and U[int(nz[0, 0]), j] < 0:
            U[:, j] *= -1.0
    return U


def parity_label(parity: int) -> str:
    return "even" if int(parity) == 1 else "odd"


def polar_parity(l: int) -> int:
    return 1 if int(l) % 2 == 0 else -1


G_O = cubic_group_O(dtype=DTYPE, device=DEVICE)
G_D6 = dihedral_group_D6_zaxis(dtype=DTYPE, device=DEVICE)
GH_OH = centrosymmetric_closure(G_O)
GH_D6H = centrosymmetric_closure(G_D6)

print("O size / det signs:", len(G_O), det_sign(G_O).unique(sorted=True).tolist())
print("Oh size / det signs:", len(GH_OH), det_sign(GH_OH).unique(sorted=True).tolist())
print("D6 size / det signs:", len(G_D6), det_sign(G_D6).unique(sorted=True).tolist())
print("D6h size / det signs:", len(GH_D6H), det_sign(GH_D6H).unique(sorted=True).tolist())
        """
    ),
    md(
        r"""
## Projector rank and matrix comparison

The following table compares:

- \(P_l^G\), the proper-group \(SO(3)\) Reynolds projector;
- \(P_l^{G_h,p=+1}\), full centrosymmetric group with even parity;
- \(P_l^{G_h,p=-1}\), full centrosymmetric group with odd parity;
- \(P_l^{G_h,p=(-1)^l}\), the polar spherical-harmonic parity convention.

The key columns are the rank and the max absolute matrix difference from the proper projector.
        """
    ),
    code(
        r"""
def projector_comparison_rows(name_proper: str, name_full: str, G: torch.Tensor, GH: torch.Tensor, max_l: int) -> list[dict]:
    rows = []
    for l in range(max_l + 1):
        P_G = reynolds_projector(G, l, parity=1)
        proper_rank = projector_rank(P_G)
        for label, parity in [
            ("full_even_parity", 1),
            ("full_odd_parity", -1),
            ("full_polar_parity", polar_parity(l)),
        ]:
            P_H = reynolds_projector(GH, l, parity=parity)
            rows.append(
                {
                    "proper_group": name_proper,
                    "full_group": name_full,
                    "l": l,
                    "full_projector_variant": label,
                    "parity": parity_label(parity),
                    "proper_rank": proper_rank,
                    "full_rank": projector_rank(P_H),
                    "max_abs_projector_diff_vs_proper": float((P_H - P_G).abs().max().item()),
                    "fro_projector_diff_vs_proper": float(torch.linalg.norm(P_H - P_G).item()),
                }
            )
    return rows


rows = []
rows.extend(projector_comparison_rows("O", "Oh", G_O, GH_OH, MAX_L))
rows.extend(projector_comparison_rows("D6", "D6h", G_D6, GH_D6H, MAX_L))
projector_df = pd.DataFrame(rows)

csv_path = OUT_DIR / "projector_rank_and_matrix_comparison.csv"
json_path = OUT_DIR / "projector_rank_and_matrix_comparison.json"
projector_df.to_csv(csv_path, index=False)
json_path.write_text(projector_df.to_json(orient="records", indent=2))

display(projector_df)
print("saved:", csv_path)
print("saved:", json_path)
        """
    ),
    md(
        r"""
## Current paper degrees

The current direct-Reynolds encoder uses only even degrees:

- FCC: \(l=4\) (through \(l\le 4\), dropping scalar \(l=0\));
- HCP: \(l=2,4,6\).

This cell extracts only those rows under polar parity \(p=(-1)^l\). These are the rows that
directly answer whether the manuscript encoder changes if the full Laue label is used in
the projector comparison.
        """
    ),
    code(
        r"""
current_degree_mask = (
    ((projector_df["proper_group"] == "O") & (projector_df["l"].isin([4])))
    | ((projector_df["proper_group"] == "D6") & (projector_df["l"].isin([2, 4, 6])))
)
current_df = projector_df[
    current_degree_mask
    & (projector_df["full_projector_variant"] == "full_polar_parity")
].reset_index(drop=True)

display(current_df)
print(
    "max current-degree projector diff:",
    current_df["max_abs_projector_diff_vs_proper"].max(),
)
        """
    ),
    md(
        r"""
## Odd-degree diagnostic

The current encoder does not use these odd blocks, but this diagnostic shows where the full
centrosymmetric group would differ from the proper group under polar parity. This is why
writing the actual quotient group as \(O\) and \(D_6\) is cleaner than calling it \(O_h\) or
\(D_{6h}\).
        """
    ),
    code(
        r"""
odd_diff_df = projector_df[
    (projector_df["full_projector_variant"] == "full_polar_parity")
    & (projector_df["l"] % 2 == 1)
    & (projector_df["proper_rank"] != projector_df["full_rank"])
].reset_index(drop=True)

display(odd_diff_df)
        """
    ),
    md(
        r"""
## Feature-space comparison for random active orientations

For each current paper block, compare the projected feature subspace for random active
orientations:

$$
D^l(R)P_l^G \quad\text{vs.}\quad D^l(R)P_l^{G_h,p=(-1)^l}.
$$

Using the projector rather than an arbitrary basis avoids false differences from rotations
inside a repeated invariant subspace.
        """
    ),
    code(
        r"""
def random_active_rotations(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q = torch.randn((int(n), 4), generator=gen, dtype=DTYPE)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    q = torch.where(q[:, :1] < 0.0, -q, q)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.empty((int(n), 3, 3), dtype=DTYPE)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def feature_projector_comparison(name: str, G: torch.Tensor, GH: torch.Tensor, degrees: list[int]) -> pd.DataFrame:
    R = random_active_rotations(24, seed=SEED + len(name))
    rows = []
    for l in degrees:
        P_G = reynolds_projector(G, l, parity=1)
        P_H = reynolds_projector(GH, l, parity=polar_parity(l))
        D_R = irrep_D(R, l, parity=1)
        F_G = torch.einsum("nij,jk->nik", D_R, P_G)
        F_H = torch.einsum("nij,jk->nik", D_R, P_H)
        rows.append(
            {
                "group_pair": name,
                "l": l,
                "proper_rank": projector_rank(P_G),
                "full_polar_rank": projector_rank(P_H),
                "max_abs_feature_projector_diff": float((F_H - F_G).abs().max().item()),
                "fro_feature_projector_diff": float(torch.linalg.norm(F_H - F_G).item()),
            }
        )
    return pd.DataFrame(rows)


feature_df = pd.concat(
    [
        feature_projector_comparison("O_vs_Oh", G_O, GH_OH, [4]),
        feature_projector_comparison("D6_vs_D6h", G_D6, GH_D6H, [2, 4, 6]),
    ],
    ignore_index=True,
)
feature_df.to_csv(OUT_DIR / "current_degree_feature_projector_comparison.csv", index=False)
display(feature_df)
        """
    ),
    md(
        r"""
## Side-specific invariance and equivariance tests

This test uses the proper rotational group \(G\), because only \(G\subset SO(3)\) acts on the
right while staying inside the physical orientation space.

For active rotations:

$$
R \mapsto R H \quad (H\in G)
$$

must be invariant, and

$$
R \mapsto A R \quad (A\in SO(3))
$$

must be equivariant.

The test is run on the current paper degrees. It also runs the algebraic fixed-subspace
check \(D^{l,p}(K)U=U\) for all \(K\in G_h\), but that is an \(O(3)\)-representation statement,
not a statement that \(RK\) is a valid orientation when \(K\) is improper.
        """
    ),
    code(
        r"""
def block_from_basis(R: torch.Tensor, l: int, U: torch.Tensor) -> torch.Tensor:
    D_R = irrep_D(R, l, parity=1)
    return torch.einsum("nij,jr->nir", D_R, U)


def max_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(x.shape[0], -1), dim=-1).max().item())


def side_tests(name: str, G: torch.Tensor, GH: torch.Tensor, degrees: list[int]) -> pd.DataFrame:
    R = random_active_rotations(16, seed=SEED + 101 + len(name))
    A = random_active_rotations(16, seed=SEED + 202 + len(name))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 303 + len(name))
    H = G[torch.randint(0, G.shape[0], (R.shape[0],), generator=gen)]

    rows = []
    for l in degrees:
        P = reynolds_projector(G, l, parity=1)
        U = projector_basis_from_projector(P)
        if U.numel() == 0:
            continue

        F = block_from_basis(R, l, U)
        F_right = block_from_basis(torch.matmul(R, H), l, U)
        F_left_actual = block_from_basis(torch.matmul(A, R), l, U)
        D_A = irrep_D(A, l, parity=1)
        F_left_expected = torch.einsum("nij,njr->nir", D_A, F)

        # Full Laue fixed-subspace test under polar parity.
        parity = polar_parity(l)
        P_full = reynolds_projector(GH, l, parity=parity)
        U_full = projector_basis_from_projector(P_full)
        if U_full.numel():
            D_K = irrep_D(GH, l, parity=parity)
            fixed_resid = torch.einsum("kij,jr->kir", D_K, U_full) - U_full.unsqueeze(0)
            full_fixed_resid = float(fixed_resid.abs().max().item())
        else:
            full_fixed_resid = 0.0

        rows.append(
            {
                "group": name,
                "l": l,
                "rank": U.shape[1],
                "right_G_invariance_max_norm": max_norm(F_right - F),
                "left_SO3_equivariance_max_norm": max_norm(F_left_actual - F_left_expected),
                "full_Laue_polar_fixed_subspace_resid": full_fixed_resid,
            }
        )
    return pd.DataFrame(rows)


side_df = pd.concat(
    [
        side_tests("O", G_O, GH_OH, [4]),
        side_tests("D6", G_D6, GH_D6H, [2, 4, 6]),
    ],
    ignore_index=True,
)
side_df.to_csv(OUT_DIR / "current_degree_right_invariance_left_equivariance_tests.csv", index=False)
display(side_df)
        """
    ),
    md(
        r"""
## Summary interpretation

For the current paper degrees, the numerical answer should be:

- \(O\) vs. polar-\(O_h\): same projector for \(l=4\);
- \(D_6\) vs. polar-\(D_{6h}\): same projector for \(l=2,4,6\);
- right multiplication by the proper rotational group remains invariant;
- left multiplication by arbitrary \(SO(3)\) rotations remains equivariant.

The difference appears when odd proper-group invariant degrees are included. Under polar
parity, inversion kills those odd blocks in the full Laue group. Therefore the safest paper
language is:

> The encoder/evaluator quotient uses the proper rotational subgroup \(G\subset SO(3)\):
> \(G=O\) for FCC and \(G=D_6\) for HCP. The conventional Laue labels
> \(m\bar{3}m\) / \(O_h\) and \(6/mmm\) / \(D_{6h}\) are material descriptors; they are not
> the \(SO(3)\) quotient groups.
        """
    ),
    code(
        r"""
summary = {
    "current_degree_max_projector_diff": float(current_df["max_abs_projector_diff_vs_proper"].max()),
    "current_degree_max_feature_projector_diff": float(feature_df["max_abs_feature_projector_diff"].max()),
    "current_degree_max_right_invariance_residual": float(side_df["right_G_invariance_max_norm"].max()),
    "current_degree_max_left_equivariance_residual": float(side_df["left_SO3_equivariance_max_norm"].max()),
    "odd_polar_rank_differences": odd_diff_df[
        ["proper_group", "full_group", "l", "proper_rank", "full_rank"]
    ].to_dict(orient="records"),
}
summary_path = OUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("saved:", summary_path)
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (material)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2))
print(f"Wrote {OUT}")
