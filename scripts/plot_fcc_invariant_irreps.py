#!/usr/bin/env python3
"""Plot invariant irreps for FCC cubic symmetry.

This script computes cubic-invariant subspaces U_l (Reynolds projector on Wigner-D)
and plots:
1) invariant multiplicity rank(U_l) vs l
2) each invariant mode as a scalar field on the sphere
3) each invariant mode as a 3D radial lobe surface

Quaternion conventions:
- symmetry quaternions are scalar-first [w, x, y, z]
- built-in table is Bunge/passive inverse symmetries (24 proper cubic rotations)
- internally converted to active by conjugation before e3nn Wigner-D
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from e3nn import o3
from scipy import special


def _build_fcc_syms_inv_wxyz(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return torch.tensor(
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
        dtype=dtype,
    )


def _parse_ls(ls_text: str) -> tuple[int, ...]:
    ls = tuple(int(x.strip()) for x in ls_text.split(",") if x.strip())
    if not ls:
        raise ValueError("Ls cannot be empty.")
    if any(l < 0 for l in ls):
        raise ValueError(f"All l must be >=0, got {ls}.")
    return ls


def _normalize_quat_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def _bunge_to_active_wxyz(q_bunge: torch.Tensor) -> torch.Tensor:
    qa = q_bunge.clone()
    qa[..., 1:] = -qa[..., 1:]
    return qa


def _wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = o3._wigner.so3_generators(l).to(alpha.device)
    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


def _canonicalize_columns(U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    Uc = U.clone()
    for j in range(Uc.shape[-1]):
        col = Uc[:, j]
        k = int(torch.argmax(col.abs()).item())
        pivot = col[k]
        if pivot.abs() > eps:
            phase = torch.exp(-1j * torch.angle(pivot))
            Uc[:, j] = Uc[:, j] * phase
            if Uc[k, j].real < 0:
                Uc[:, j] = -Uc[:, j]
    return Uc


def compute_invariant_basis_Ul(
    l: int,
    sym_quats_wxyz_bunge_inv: torch.Tensor,
    eig_tol: float = 1e-5,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-6,
) -> torch.Tensor:
    sym_bunge = _normalize_quat_wxyz(sym_quats_wxyz_bunge_inv.to(torch.float64))
    sym_active = _bunge_to_active_wxyz(sym_bunge)

    R = o3.quaternion_to_matrix(sym_active)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    D = _wigner_D(l, alpha, beta, gamma)
    P = D.mean(dim=0).to(torch.complex128)
    P_herm = 0.5 * (P + P.conj().transpose(-2, -1))

    evals, evecs = torch.linalg.eigh(P_herm)
    is_inv = evals > (1.0 - eig_tol)
    rank = int(is_inv.sum().item())
    if rank <= 0:
        U, S, _ = torch.linalg.svd(P)
        sigma_tol = max(float(eig_tol), float(rel_tol), float(abs_tol))
        rank = int((S > (1.0 - sigma_tol)).sum().item())
        if rank <= 0:
            raise RuntimeError(f"No invariant directions for l={l}")
        U_l = U[:, :rank].to(torch.complex64)
    else:
        U_l = evecs[:, is_inv].to(torch.complex64)

    U_l, _ = torch.linalg.qr(U_l)
    U_l = _canonicalize_columns(U_l)
    return U_l


def _sph_harm_matrix(l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    # scipy.special.sph_harm signature: (m, l, phi, theta)
    m_vals = np.arange(-l, l + 1)
    cols = [special.sph_harm(int(m), int(l), phi, theta) for m in m_vals]
    return np.stack(cols, axis=-1)


def _mode_field_on_sphere(U_l: np.ndarray, mode_idx: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    Y = _sph_harm_matrix((U_l.shape[0] - 1) // 2, theta, phi)
    coeff = U_l[:, mode_idx]
    f = Y @ coeff

    # Fix global complex phase so the largest-magnitude sample is real-positive.
    idx = np.unravel_index(np.argmax(np.abs(f)), f.shape)
    phase = np.exp(-1j * np.angle(f[idx]))
    f = phase * f
    return np.real(f)


def plot_mode_field(
    field: np.ndarray,
    l: int,
    mode_idx: int,
    out_path: Path,
) -> None:
    vmax = float(np.max(np.abs(field)))
    if vmax <= 0.0:
        vmax = 1.0

    fig = plt.figure(figsize=(8.5, 3.6), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        field,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        extent=[0.0, 360.0, 0.0, 180.0],
    )
    ax.set_title(f"FCC invariant mode: l={l}, basis #{mode_idx}")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("theta (deg)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("amplitude")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mode_lobe_3d(
    field: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    l: int,
    mode_idx: int,
    out_path: Path,
    elev: float,
    azim: float,
) -> None:
    vmax = float(np.max(np.abs(field)))
    if vmax <= 0.0:
        vmax = 1.0

    r = np.abs(field) / vmax
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.coolwarm
    facecolors = cmap(norm(field))

    fig = plt.figure(figsize=(6.0, 6.0), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    lim = float(np.max(np.abs(r)))
    if lim <= 0.0:
        lim = 1.0
    lim *= 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"FCC invariant lobe: l={l}, basis #{mode_idx}", pad=8.0)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("amplitude")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rank_bar(ls: Iterable[int], ranks: Iterable[int], out_path: Path) -> None:
    ls_list = list(ls)
    ranks_list = list(ranks)

    fig = plt.figure(figsize=(6.0, 3.4), dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(ls_list, ranks_list, width=0.8)
    ax.set_xlabel("degree l")
    ax.set_ylabel("rank(U_l)")
    ax.set_title("FCC cubic invariant multiplicity")
    ax.set_xticks(ls_list)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ls", type=str, default="4,6,8,10,12", help="Comma-separated l values")
    p.add_argument("--out-dir", type=Path, default=Path("out/fcc_invariant_irreps_plots"))
    p.add_argument("--n-theta", type=int, default=181, help="Grid size in theta (0..pi)")
    p.add_argument("--n-phi", type=int, default=361, help="Grid size in phi (0..2pi)")
    p.add_argument("--no-heatmap", action="store_true", help="Disable 2D theta/phi heatmap plots")
    p.add_argument("--no-lobe3d", action="store_true", help="Disable 3D radial lobe plots")
    p.add_argument("--lobe-elev", type=float, default=25.0, help="3D lobe camera elevation (deg)")
    p.add_argument("--lobe-azim", type=float, default=35.0, help="3D lobe camera azimuth (deg)")
    p.add_argument("--eig-tol", type=float, default=1e-5)
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--abs-tol", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ls = _parse_ls(args.ls)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    syms_inv = _build_fcc_syms_inv_wxyz(dtype=torch.float64)

    basis_ranks: dict[str, int] = {}
    basis_tables: dict[int, np.ndarray] = {}

    for l in ls:
        U_l = compute_invariant_basis_Ul(
            l=l,
            sym_quats_wxyz_bunge_inv=syms_inv,
            eig_tol=float(args.eig_tol),
            rel_tol=float(args.rel_tol),
            abs_tol=float(args.abs_tol),
        )
        U_np = U_l.detach().cpu().numpy()
        basis_ranks[str(l)] = int(U_np.shape[-1])
        basis_tables[l] = U_np

    plot_rank_bar(ls, [basis_ranks[str(l)] for l in ls], out_dir / "invariant_multiplicity_rank.png")

    theta_1d = np.linspace(0.0, math.pi, int(args.n_theta), dtype=np.float64)
    phi_1d = np.linspace(0.0, 2.0 * math.pi, int(args.n_phi), dtype=np.float64)
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")

    for l in ls:
        U_np = basis_tables[l]
        for mode_idx in range(U_np.shape[-1]):
            field = _mode_field_on_sphere(U_np, mode_idx, theta, phi)
            if not bool(args.no_heatmap):
                heatmap_path = out_dir / f"invariant_mode_l{l:02d}_k{mode_idx:02d}.png"
                plot_mode_field(field, l=l, mode_idx=mode_idx, out_path=heatmap_path)
            if not bool(args.no_lobe3d):
                lobe_path = out_dir / f"invariant_mode_l{l:02d}_k{mode_idx:02d}_lobe3d.png"
                plot_mode_lobe_3d(
                    field=field,
                    theta=theta,
                    phi=phi,
                    l=l,
                    mode_idx=mode_idx,
                    out_path=lobe_path,
                    elev=float(args.lobe_elev),
                    azim=float(args.lobe_azim),
                )

    payload = {
        "ls": list(ls),
        "basis_ranks": basis_ranks,
        "n_theta": int(args.n_theta),
        "n_phi": int(args.n_phi),
        "heatmap_enabled": not bool(args.no_heatmap),
        "lobe3d_enabled": not bool(args.no_lobe3d),
        "lobe_elev": float(args.lobe_elev),
        "lobe_azim": float(args.lobe_azim),
        "symmetry_group": "proper cubic O (24)",
        "quaternion_convention": "Bunge passive inverse symmetries [w,x,y,z]",
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved plots in: {out_dir}")
    print(f"basis ranks: {basis_ranks}")


if __name__ == "__main__":
    main()
