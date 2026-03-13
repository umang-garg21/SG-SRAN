from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_fcc_syms_inv_wxyz(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """24 proper cubic operators as inverse quaternions [w, x, y, z]."""
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
    out_activation: nn.Module | None = None,
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
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class InvariantAutoencoderBunge(nn.Module):
    """Strictly invariant-latent autoencoder for Bunge/passive quaternions [w,x,y,z]."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        latent_dim: int = 32,
        encoder_hidden_dim: int = 128,
        encoder_layers: int = 3,
        decoder_hidden_dim: int = 128,
        decoder_layers: int = 3,
        dropout: float = 0.0,
        canonicalize_output: bool = True,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        self.canonicalize_output = bool(canonicalize_output)
        self.quaternion_convention = "bunge_passive_wxyz"
        self.strict_invariant_latent = True

        syms_inv = _build_fcc_syms_inv_wxyz(dtype=torch.float32)
        self.register_buffer("fcc_syms_inv", syms_inv)
        self.register_buffer("fcc_syms", self._quat_conjugate(syms_inv))

        # Sign-invariant orbit member features via rotation matrices (9 dims).
        self.phi = _build_mlp(
            in_dim=9,
            hidden_dim=int(encoder_hidden_dim),
            out_dim=int(encoder_hidden_dim),
            num_layers=int(encoder_layers),
            dropout=float(dropout),
        )
        self.rho = _build_mlp(
            in_dim=int(encoder_hidden_dim),
            hidden_dim=int(encoder_hidden_dim),
            out_dim=self.latent_dim,
            num_layers=max(1, int(encoder_layers) - 1),
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

    @staticmethod
    def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
        q = InvariantAutoencoderBunge._normalize_quaternions(quats)
        w, x, y, z = q.unbind(dim=-1)
        out = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
        out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        out[..., 0, 1] = 2.0 * (x * y - z * w)
        out[..., 0, 2] = 2.0 * (x * z + y * w)
        out[..., 1, 0] = 2.0 * (x * y + z * w)
        out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        out[..., 1, 2] = 2.0 * (y * z - x * w)
        out[..., 2, 0] = 2.0 * (x * z - y * w)
        out[..., 2, 1] = 2.0 * (y * z + x * w)
        out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return out

    def _left_orbit(self, quats: torch.Tensor) -> torch.Tensor:
        quats = self._normalize_quaternions(quats)
        n = quats.shape[0]
        g = self.fcc_syms_inv.shape[0]
        syms = self.fcc_syms_inv.unsqueeze(0).expand(n, -1, -1)
        q_exp = quats.unsqueeze(1).expand(-1, g, -1)
        orbit = self.quat_mul(syms, q_exp)
        return self._normalize_quaternions(orbit)

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        q = self._normalize_quaternions(quats.to(self.device))
        orbit = self._left_orbit(q)
        best_idx = orbit[..., 0].abs().argmax(dim=1)
        q_fz = orbit[torch.arange(q.shape[0], device=q.device), best_idx]
        q_fz = torch.where(q_fz[:, :1] < 0.0, -q_fz, q_fz)
        q_fz = self._normalize_quaternions(q_fz)
        if return_op_map:
            return q_fz, best_idx
        return q_fz

    def canonicalize_for_metrics(self, quats: torch.Tensor) -> torch.Tensor:
        return self.reduce_to_fz(quats, return_op_map=False)

    def encode(self, quats: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        q = quats.to(self.device)
        if q.ndim != 2 or q.shape[-1] != 4:
            raise ValueError(f"Expected (N,4) quaternions, got {tuple(q.shape)}")
        if normalize_input:
            q = self._normalize_quaternions(q)

        orbit = self._left_orbit(q)  # (N,G,4)
        rot = self._quat_to_rotmat(orbit.reshape(-1, 4)).reshape(orbit.shape[0], orbit.shape[1], 9)
        h = self.phi(rot.reshape(-1, 9)).reshape(orbit.shape[0], orbit.shape[1], -1)
        pooled = h.mean(dim=1)
        return self.rho(pooled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        q = self.decoder(latent.to(self.device))
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
