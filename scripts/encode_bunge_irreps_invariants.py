#!/usr/bin/env python3
"""Invariant-subspace Wigner features for Bunge quaternions.

Best-method implementation:
- Step A: Precompute cubic invariant bases U_l from symmetry quaternions.
- Step B: Encode each quaternion q as F_l(q) = D^l(R(q_active)) U_l.
- Flatten + stack over l, optional Re/Im stacking, optional normalization.
- Optional learnable scalar head to reduce collisions further.

Convention:
- Input quaternions are scalar-first [w, x, y, z].
- Input quaternions are Bunge/passive.
- Internally converted to active by conjugation before Wigner-D.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

try:
    from e3nn import o3
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires e3nn. Install with: pip install e3nn torch"
    ) from exc


def wigner_D_cuda(
    l: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """CUDA-friendly Wigner-D from e3nn so3 generators."""
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    X = o3._wigner.so3_generators(l).to(device)
    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


def normalize_quat_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def bunge_to_active_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Bunge passive -> active by quaternion conjugation."""
    qa = q.clone()
    qa[..., 1:] = -qa[..., 1:]
    return qa


def _canonicalize_columns(U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Fix complex column phases to a deterministic convention."""
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
    sym_quats_wxyz_active: torch.Tensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-6,
    eig_tol: float = 1e-5,
    canonicalize: bool = True,
) -> torch.Tensor:
    """Compute invariant basis U_l from Reynolds operator P_l = mean_S D^l(S)."""
    # Build Reynolds operator in float64/complex128 for stable spectral separation.
    sym_q = normalize_quat_wxyz(sym_quats_wxyz_active.to(torch.float64))
    R = o3.quaternion_to_matrix(sym_q)
    a, b, g = o3.matrix_to_angles(R)
    D = wigner_D_cuda(l, a, b, g)
    P = D.mean(dim=0).to(torch.complex128)

    # For a unitary representation and finite group average, P is an orthogonal projector.
    # Numerically enforce Hermitian form and keep eigenmodes with lambda ~= 1.
    P_herm = 0.5 * (P + P.conj().transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(P_herm)
    is_inv = evals > (1.0 - eig_tol)
    rank = int(is_inv.sum().item())
    if rank <= 0:
        # Fallback: keep singular vectors with sigma ~= 1.
        U, S, _ = torch.linalg.svd(P)
        sigma_tol = max(float(eig_tol), float(rel_tol), float(abs_tol))
        rank = int((S > (1.0 - sigma_tol)).sum().item())
        if rank <= 0:
            raise RuntimeError(
                f"No invariant directions found for l={l}. "
                f"eig_tol={eig_tol:.1e}, sigma_tol={sigma_tol:.1e}. "
                "Check quaternion convention/order and symmetry table."
            )
        U_l = U[:, :rank].to(torch.complex64)
    else:
        U_l = evecs[:, is_inv].to(torch.complex64)

    # Orthonormalize basis columns.
    U_l, _ = torch.linalg.qr(U_l)
    if canonicalize:
        U_l = _canonicalize_columns(U_l)
    return U_l


@dataclass
class InvariantEncoderConfig:
    Ls: Tuple[int, ...] = (4, 6, 8, 10, 12)
    stack_re_im: bool = True
    rel_tol: float = 1e-8
    abs_tol: float = 1e-6
    eig_tol: float = 1e-5
    canonicalize_basis: bool = True
    normalize_output: bool = True


class InvariantSubspaceWignerEncoder(nn.Module):
    """Step A+B: cubic-invariant Wigner subspace feature encoder."""

    def __init__(
        self,
        sym_quats_wxyz_bunge: torch.Tensor,
        cfg: InvariantEncoderConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.Ls = cfg.Ls

        sym_active = bunge_to_active_wxyz(normalize_quat_wxyz(sym_quats_wxyz_bunge))
        for l in self.Ls:
            U_l = compute_invariant_basis_Ul(
                l=l,
                sym_quats_wxyz_active=sym_active,
                rel_tol=cfg.rel_tol,
                abs_tol=cfg.abs_tol,
                eig_tol=cfg.eig_tol,
                canonicalize=cfg.canonicalize_basis,
            )
            self.register_buffer(f"U_{l}", U_l)

        self.out_dim = self._compute_out_dim()

    def _compute_out_dim(self) -> int:
        total = 0
        for l in self.Ls:
            U = getattr(self, f"U_{l}")
            block = (2 * l + 1) * U.shape[-1]
            total += (2 * block) if self.cfg.stack_re_im else block
        return total

    def forward(self, quats_wxyz_bunge: torch.Tensor) -> torch.Tensor:
        q_bunge = normalize_quat_wxyz(quats_wxyz_bunge)
        q_active = bunge_to_active_wxyz(q_bunge)

        R = o3.quaternion_to_matrix(q_active)
        a, b, g = o3.matrix_to_angles(R)

        feats: List[torch.Tensor] = []
        for l in self.Ls:
            D = wigner_D_cuda(l, a, b, g)
            U = getattr(self, f"U_{l}")
            D = D.to(U.dtype)
            F = D @ U
            F = F.reshape(*F.shape[:-2], -1)

            if self.cfg.stack_re_im:
                v = torch.cat([F.real, F.imag], dim=-1)
            else:
                v = F.real
            feats.append(v)

        out = torch.cat(feats, dim=-1)
        if self.cfg.normalize_output:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-12)
        return out


class OptionalInvariantHead(nn.Module):
    """Optional learned scalar head for extra collision resistance."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BestInvariantFeatureModel(nn.Module):
    """Encoder plus optional scalar head."""

    def __init__(
        self,
        sym_quats_wxyz_bunge: torch.Tensor,
        encoder_cfg: InvariantEncoderConfig,
        use_head: bool = False,
        head_out_dim: int = 32,
    ):
        super().__init__()
        self.encoder = InvariantSubspaceWignerEncoder(sym_quats_wxyz_bunge, encoder_cfg)
        self.use_head = bool(use_head)
        self.head = (
            OptionalInvariantHead(self.encoder.out_dim, out_dim=head_out_dim)
            if self.use_head
            else None
        )

    def forward(self, q_bunge_wxyz: torch.Tensor) -> torch.Tensor:
        z = self.encoder(q_bunge_wxyz)
        if self.head is None:
            return z
        return torch.cat([z, self.head(z)], dim=-1)


def build_fcc_syms_inv_wxyz_np() -> np.ndarray:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return np.array(
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
        dtype=np.float32,
    )


def _quat_axis_to_last(arr: np.ndarray) -> np.ndarray:
    axes = [i for i, s in enumerate(arr.shape) if s == 4]
    if len(axes) != 1:
        raise ValueError(
            f"Expected exactly one quaternion axis of size 4, got shape {arr.shape}"
        )
    q_axis = axes[0]
    return arr if q_axis == arr.ndim - 1 else np.moveaxis(arr, q_axis, -1)


def _load_input_quats(path: Path, key: str | None) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        return np.load(path), "npy:data"
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        if key is not None:
            if key not in z:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(z.keys())}")
            return z[key], f"npz:{key}"
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"No arrays in {path}")
        return z[keys[0]], f"npz:{keys[0]}"
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _load_symmetry_quats(path: Path | None, device: torch.device) -> torch.Tensor:
    if path is None:
        arr = build_fcc_syms_inv_wxyz_np()
    else:
        arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Symmetry quats must have shape (G,4), got {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _parse_ls(ls_text: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in ls_text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Ls cannot be empty")
    for l in vals:
        if l < 0:
            raise ValueError(f"Invalid l={l}; must be >=0")
    return tuple(vals)


def _verify_invariance(
    model: BestInvariantFeatureModel,
    q_bunge: torch.Tensor,
    sym_quats_bunge: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        base = model(q_bunge)

        # q' = s^{-1} ⊗ q in Bunge convention; if provided syms are already inverse,
        # this still spans the same group orbit due closure.
        all_max = []
        all_mean = []
        for s in sym_quats_bunge:
            sb = s.unsqueeze(0).expand_as(q_bunge)
            q_g = quat_mul_torch(sb, q_bunge)
            z_g = model(q_g)
            d = (z_g - base).abs()
            all_max.append(float(d.max().item()))
            all_mean.append(float(d.mean().item()))

        q_neg = -q_bunge
        d_sign = (model(q_neg) - base).abs()
        return {
            "symmetry_max_abs": max(all_max) if all_max else 0.0,
            "symmetry_mean_abs": float(np.mean(all_mean)) if all_mean else 0.0,
            "sign_max_abs": float(d_sign.max().item()),
            "sign_mean_abs": float(d_sign.mean().item()),
        }


def _compute_l_feature_layout(
    ls: Sequence[int],
    basis_dims: dict[str, int],
    stack_re_im: bool,
) -> list[tuple[int, int, int, int]]:
    """Return [(l, start, end, rank)] contiguous feature slices per degree."""
    layout: list[tuple[int, int, int, int]] = []
    offset = 0
    for l in ls:
        rank = int(basis_dims[str(int(l))])
        complex_block = (2 * int(l) + 1) * rank
        width = (2 * complex_block) if bool(stack_re_im) else complex_block
        layout.append((int(l), offset, offset + width, rank))
        offset += width
    return layout


def _plot_spatial_irrep_maps(
    features: np.ndarray,
    ls: Sequence[int],
    basis_dims: dict[str, int],
    stack_re_im: bool,
    out_dir: Path,
    components_per_l: int,
    robust_percentile: float,
) -> None:
    """Save 2D spatial maps for irreps features.

    Requires feature shape (H, W, F). Produces:
    - per-l magnitude map
    - per-l grid of first K channel maps
    """
    if features.ndim != 3:
        raise ValueError(
            f"--plot-spatial requires 2D spatial input; got feature shape {features.shape} "
            "(expected (H, W, F))."
        )

    # Lazy import so non-plot usage remains lightweight.
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w, _ = features.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = _compute_l_feature_layout(ls, basis_dims, stack_re_im)

    # Combined summary panel: one magnitude map per l.
    n_l = len(layout)
    fig_sum, axes_sum = plt.subplots(
        1,
        n_l,
        figsize=(max(4.0 * n_l, 6.0), 3.8),
        constrained_layout=True,
    )
    if n_l == 1:
        axes_sum = [axes_sum]

    for ax, (l, start, end, rank) in zip(axes_sum, layout):
        block = features[..., start:end]
        mag = np.linalg.norm(block, axis=-1)
        im = ax.imshow(mag, cmap="viridis")
        ax.set_title(f"l={l} | rank={rank}\n||block||")
        ax.axis("off")
        fig_sum.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_sum.suptitle(f"Invariant Irreps Spatial Magnitude Maps ({h}x{w})")
    fig_sum.savefig(out_dir / "irreps_spatial_summary.png", dpi=220)
    plt.close(fig_sum)

    # Per-l channel maps.
    pct = float(np.clip(robust_percentile, 50.0, 100.0))
    for l, start, end, rank in layout:
        block = features[..., start:end]
        n_channels = int(block.shape[-1])
        k = int(max(1, min(int(components_per_l), n_channels)))

        ncols = min(4, k)
        nrows = int(math.ceil(k / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.4 * nrows),
            constrained_layout=True,
        )
        if nrows == 1 and ncols == 1:
            axes_grid = [axes]
        elif nrows == 1:
            axes_grid = list(axes)
        elif ncols == 1:
            axes_grid = list(axes)
        else:
            axes_grid = [a for row in axes for a in row]

        for i, ax in enumerate(axes_grid):
            if i >= k:
                ax.axis("off")
                continue

            v = block[..., i]
            scale = float(np.percentile(np.abs(v), pct))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.max(np.abs(v)))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0

            im = ax.imshow(v, cmap="coolwarm", vmin=-scale, vmax=scale)
            ax.set_title(f"ch {i} (global {start + i})")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title_suffix = "Re/Im-stacked" if stack_re_im else "real-only"
        fig.suptitle(
            f"l={l} channel maps ({k}/{n_channels} shown, rank={rank}, {title_suffix})"
        )
        fig.savefig(out_dir / f"irreps_spatial_l{l:02d}_channels.png", dpi=220)
        plt.close(fig)


def quat_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dim=-1,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help=".npy/.npz quaternion input")
    p.add_argument("--output", required=True, type=Path, help="Output .npz file")
    p.add_argument("--input-key", type=str, default=None, help="Array key if input is .npz")
    p.add_argument(
        "--scalar-last",
        action="store_true",
        help="Interpret input quaternion components as [x,y,z,w] and convert to [w,x,y,z].",
    )
    p.add_argument(
        "--sym-npy",
        type=Path,
        default=None,
        help="Optional symmetry quaternion table (G,4) in Bunge wxyz. Default: built-in FCC 24",
    )
    p.add_argument("--ls", type=str, default="4,6,8,10,12", help="Comma-separated l values")
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--abs-tol", type=float, default=1e-6)
    p.add_argument("--eig-tol", type=float, default=1e-5, help="Tolerance for eigenvalue==1 invariant test")
    p.add_argument("--no-canonicalize-basis", action="store_true")
    p.add_argument("--stack-re-im", action="store_true", default=True)
    p.add_argument("--no-stack-re-im", dest="stack_re_im", action="store_false")
    p.add_argument("--normalize-output", action="store_true", default=True)
    p.add_argument("--no-normalize-output", dest="normalize_output", action="store_false")
    p.add_argument("--use-head", action="store_true", help="Append optional learned scalar head")
    p.add_argument("--head-out-dim", type=int, default=32)
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|cuda:0")
    p.add_argument(
        "--plot-spatial",
        action="store_true",
        help="Save 2D spatial maps of irrep features (requires 2D spatial input).",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Spatial-plot output directory (default: <output_stem>_plots).",
    )
    p.add_argument(
        "--plot-components-per-l",
        type=int,
        default=8,
        help="Number of channels to plot per l (first K channels).",
    )
    p.add_argument(
        "--plot-robust-percentile",
        type=float,
        default=99.0,
        help="Percentile for symmetric channel color scale (50..100).",
    )
    p.add_argument(
        "--verify-invariance",
        action="store_true",
        help="Run symmetry/sign invariance check on encoded outputs",
    )
    p.add_argument(
        "--verify-max-samples",
        type=int,
        default=2048,
        help="Max flattened quaternions for invariance verification",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    q_raw, source = _load_input_quats(args.input, args.input_key)
    q_last = _quat_axis_to_last(np.asarray(q_raw, dtype=np.float32))
    if bool(args.scalar_last):
        q_last = q_last[..., [3, 0, 1, 2]]
    spatial_shape = q_last.shape[:-1]
    q_flat = q_last.reshape(-1, 4)

    q_t = torch.as_tensor(q_flat, dtype=torch.float32, device=device)
    q_t = normalize_quat_wxyz(q_t)

    sym_q = _load_symmetry_quats(args.sym_npy, device=device)

    cfg = InvariantEncoderConfig(
        Ls=_parse_ls(args.ls),
        stack_re_im=bool(args.stack_re_im),
        rel_tol=float(args.rel_tol),
        abs_tol=float(args.abs_tol),
        eig_tol=float(args.eig_tol),
        canonicalize_basis=not bool(args.no_canonicalize_basis),
        normalize_output=bool(args.normalize_output),
    )

    model = BestInvariantFeatureModel(
        sym_quats_wxyz_bunge=sym_q,
        encoder_cfg=cfg,
        use_head=bool(args.use_head),
        head_out_dim=int(args.head_out_dim),
    ).to(device)
    model.eval()

    with torch.no_grad():
        z = model(q_t).detach().cpu().numpy()

    z = z.reshape(*spatial_shape, z.shape[-1]).astype(np.float32, copy=False)

    out_payload = {"features": z}
    basis_dims = {}
    for l in cfg.Ls:
        U_l = getattr(model.encoder, f"U_{l}").detach().cpu().numpy()
        out_payload[f"U_{l}_real"] = U_l.real.astype(np.float32, copy=False)
        out_payload[f"U_{l}_imag"] = U_l.imag.astype(np.float32, copy=False)
        basis_dims[str(l)] = int(U_l.shape[-1])

    verify_stats = None
    if args.verify_invariance:
        n = min(int(args.verify_max_samples), int(q_t.shape[0]))
        q_sub = q_t[:n]
        verify_stats = _verify_invariance(model, q_sub, sym_q)
        print("Invariance check:")
        for k, v in verify_stats.items():
            print(f"  {k}: {v:.6e}")

    meta = {
        "input_path": str(args.input),
        "input_source": source,
        "output_path": str(args.output),
        "input_shape": list(q_raw.shape),
        "feature_shape": list(z.shape),
        "quaternion_convention": "Bunge passive [w,x,y,z]",
        "input_scalar_order": "xyzw" if bool(args.scalar_last) else "wxyz",
        "converted_to_active_inside_encoder": True,
        "Ls": list(cfg.Ls),
        "rel_tol": cfg.rel_tol,
        "abs_tol": cfg.abs_tol,
        "eig_tol": cfg.eig_tol,
        "canonicalize_basis": cfg.canonicalize_basis,
        "stack_re_im": cfg.stack_re_im,
        "normalize_output": cfg.normalize_output,
        "use_head": bool(args.use_head),
        "head_out_dim": int(args.head_out_dim) if args.use_head else 0,
        "device": str(device),
        "symmetry_source": str(args.sym_npy) if args.sym_npy is not None else "built_in_fcc_syms_inv_24",
        "basis_ranks": basis_dims,
        "verification": verify_stats,
    }
    out_payload["meta_json"] = np.asarray(json.dumps(meta, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out_payload)

    plot_dir = None
    if args.plot_spatial:
        if len(spatial_shape) != 2:
            raise ValueError(
                f"--plot-spatial requires 2D spatial quaternion field; got spatial shape {spatial_shape}"
            )
        plot_dir = (
            args.plot_dir
            if args.plot_dir is not None
            else (args.output.parent / f"{args.output.stem}_plots")
        )
        _plot_spatial_irrep_maps(
            features=z,
            ls=cfg.Ls,
            basis_dims=basis_dims,
            stack_re_im=cfg.stack_re_im,
            out_dir=plot_dir,
            components_per_l=int(args.plot_components_per_l),
            robust_percentile=float(args.plot_robust_percentile),
        )

    print(f"saved: {args.output}")
    print(f"input_shape: {tuple(q_raw.shape)}")
    print(f"feature_shape: {tuple(z.shape)}")
    print(f"Ls: {cfg.Ls}")
    print(f"basis ranks: {basis_dims}")
    if plot_dir is not None:
        print(f"spatial plots: {plot_dir}")


if __name__ == "__main__":
    main()
