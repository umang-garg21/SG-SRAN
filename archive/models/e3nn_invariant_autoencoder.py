from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from e3nn import o3


def _build_fcc_syms_inv_wxyz(dtype: torch.dtype = torch.float32) -> torch.Tensor:
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


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    layers: list[nn.Module] = []
    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _wigner_D(
    l: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
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


class E3nnInvariantAutoencoderBunge(nn.Module):
    """Bunge-invariant autoencoder with e3nn Wigner invariant-subspace encoder."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        Ls: tuple[int, ...] = (4, 6, 8, 10, 12),
        stack_re_im: bool = True,
        normalize_wigner_features: bool = True,
        basis_rel_tol: float = 1e-8,
        basis_abs_tol: float = 1e-6,
        basis_eig_tol: float = 1e-5,
        canonicalize_basis: bool = True,
        latent_dim: int = 64,
        encoder_hidden_dim: int = 256,
        encoder_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 3,
        dropout: float = 0.0,
        canonicalize_output: bool = True,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.Ls = tuple(int(l) for l in Ls)
        self.stack_re_im = bool(stack_re_im)
        self.normalize_wigner_features = bool(normalize_wigner_features)
        self.canonicalize_output = bool(canonicalize_output)
        self.quaternion_convention = "bunge_passive_wxyz"
        self.strict_invariant_latent = True

        self.basis_rel_tol = float(basis_rel_tol)
        self.basis_abs_tol = float(basis_abs_tol)
        self.basis_eig_tol = float(basis_eig_tol)
        self.canonicalize_basis = bool(canonicalize_basis)

        syms_inv = _build_fcc_syms_inv_wxyz(dtype=torch.float32)
        self.register_buffer("fcc_syms_inv", syms_inv)
        self.register_buffer("fcc_syms", self._quat_conjugate(syms_inv))

        self._build_invariant_bases()
        self._register_wigner_generators()
        self.wigner_out_dim = self._compute_wigner_out_dim()
        self.latent_dim = int(latent_dim)

        self.encoder_mlp = _build_mlp(
            in_dim=self.wigner_out_dim,
            hidden_dim=int(encoder_hidden_dim),
            out_dim=self.latent_dim,
            num_layers=int(encoder_layers),
            dropout=float(dropout),
        )
        self.decoder = _build_mlp(
            in_dim=self.latent_dim,
            hidden_dim=int(decoder_hidden_dim),
            out_dim=4,
            num_layers=int(decoder_layers),
            dropout=float(dropout),
        )

        self.to(self.device)

    @staticmethod
    def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
        return quats / norm

    @staticmethod
    def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
        out = quats.clone()
        out[..., 1:] = -out[..., 1:]
        return out

    @staticmethod
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

    def _to_active_convention(self, quats_bunge: torch.Tensor) -> torch.Tensor:
        return self._quat_conjugate(quats_bunge)

    def _build_invariant_bases(self) -> None:
        sym_bunge = self._normalize_quaternions(self.fcc_syms_inv.to(torch.float64))
        sym_active = self._to_active_convention(sym_bunge)

        R = o3.quaternion_to_matrix(sym_active)
        a, b, g = o3.matrix_to_angles(R)

        for l in self.Ls:
            D = _wigner_D(l, a, b, g)
            P = D.mean(dim=0).to(torch.complex128)
            P_herm = 0.5 * (P + P.conj().transpose(-2, -1))

            evals, evecs = torch.linalg.eigh(P_herm)
            is_inv = evals > (1.0 - self.basis_eig_tol)
            rank = int(is_inv.sum().item())

            if rank <= 0:
                U, S, _ = torch.linalg.svd(P)
                sigma_tol = max(self.basis_eig_tol, self.basis_rel_tol, self.basis_abs_tol)
                rank = int((S > (1.0 - sigma_tol)).sum().item())
                if rank <= 0:
                    raise RuntimeError(
                        f"No invariant directions found for l={l}. "
                        "Check conventions or loosen tolerances."
                    )
                U_l = U[:, :rank].to(torch.complex64)
            else:
                U_l = evecs[:, is_inv].to(torch.complex64)

            U_l, _ = torch.linalg.qr(U_l)
            if self.canonicalize_basis:
                U_l = _canonicalize_columns(U_l)
            self.register_buffer(f"U_{l}", U_l)

    def _register_wigner_generators(self) -> None:
        # Cache SO(3) generators per degree to avoid rebuilding tensors
        # on every forward pass.
        for l in self.Ls:
            X_l = o3._wigner.so3_generators(l).to(torch.complex64)
            self.register_buffer(f"X_{l}", X_l, persistent=False)

    def _compute_wigner_out_dim(self) -> int:
        total = 0
        for l in self.Ls:
            U = getattr(self, f"U_{l}")
            block = (2 * l + 1) * U.shape[-1]
            total += (2 * block) if self.stack_re_im else block
        return total

    def _wigner_invariant_features(self, quats_bunge: torch.Tensor) -> torch.Tensor:
        # e3nn's matrix_to_angles path relies on ops that are not implemented
        # for float16 on CUDA. Force this block to run in float32.
        q_input = quats_bunge.to(self.device)
        with torch.autocast(device_type=q_input.device.type, enabled=False):
            q = self._normalize_quaternions(q_input.float())
            q_active = self._to_active_convention(q)

            R = o3.quaternion_to_matrix(q_active)
            alpha, beta, gamma = o3.matrix_to_angles(R)

            feats: list[torch.Tensor] = []
            for l in self.Ls:
                X = getattr(self, f"X_{l}")
                a = alpha[..., None, None] % (2 * math.pi)
                b = beta[..., None, None] % (2 * math.pi)
                g = gamma[..., None, None] % (2 * math.pi)
                D = torch.matrix_exp(a * X[1]) @ torch.matrix_exp(b * X[0]) @ torch.matrix_exp(g * X[1])
                U = getattr(self, f"U_{l}")
                D = D.to(U.dtype)
                F = D @ U
                F = F.reshape(*F.shape[:-2], -1)
                if self.stack_re_im:
                    v = torch.cat([F.real, F.imag], dim=-1)
                else:
                    v = F.real
                feats.append(v)

            x = torch.cat(feats, dim=-1)
            if self.normalize_wigner_features:
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        return x

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        q = self._normalize_quaternions(quats.to(self.device))
        n = q.shape[0]
        g = self.fcc_syms_inv.shape[0]
        syms = self.fcc_syms_inv.unsqueeze(0).expand(n, -1, -1)
        q_exp = q.unsqueeze(1).expand(-1, g, -1)
        orbit = self.quat_mul(syms, q_exp)
        orbit = self._normalize_quaternions(orbit)
        best_idx = orbit[..., 0].abs().argmax(dim=1)
        q_fz = orbit[torch.arange(n, device=q.device), best_idx]
        q_fz = torch.where(q_fz[:, :1] < 0.0, -q_fz, q_fz)
        q_fz = self._normalize_quaternions(q_fz)
        if return_op_map:
            return q_fz, best_idx
        return q_fz

    def canonicalize_for_metrics(self, quats: torch.Tensor) -> torch.Tensor:
        return self.reduce_to_fz(quats, return_op_map=False)

    def encode(self, quats_bunge: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        q = quats_bunge.to(self.device)
        if q.ndim != 2 or q.shape[-1] != 4:
            raise ValueError(f"Expected (N,4) quaternions, got {tuple(q.shape)}")
        if normalize_input:
            q = self._normalize_quaternions(q)
        x = self._wigner_invariant_features(q)
        return self.encoder_mlp(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        q = self.decoder(z.to(self.device))
        q = self._normalize_quaternions(q)
        if self.canonicalize_output:
            q = self.reduce_to_fz(q, return_op_map=False)
        return q

    def forward(
        self,
        quats: torch.Tensor,
        normalize_input: bool = True,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(quats, normalize_input=normalize_input)
        q = self.decode(z)
        if return_latent:
            return q, z
        return q


def parse_ls_arg(ls: Iterable[int] | str | None) -> tuple[int, ...]:
    if ls is None:
        return (4, 6, 8, 10, 12)
    if isinstance(ls, str):
        vals = [int(x.strip()) for x in ls.split(",") if x.strip()]
    else:
        vals = [int(x) for x in ls]
    if not vals:
        raise ValueError("Ls cannot be empty")
    return tuple(vals)
