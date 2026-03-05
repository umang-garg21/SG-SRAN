from typing import Any

import torch
import torch.nn as nn

from models.autoencoder import FCCAutoEncoder, FCCEncoder, FCCPhysics


class LearnableFCCDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = 9 + 13  # concat(f4, f6)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, 4))
        else:
            last_dim = in_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f4, f6], dim=-1)
        q = self.net(x)
        norm = torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        return q / norm

    def forward_with_debug(self, f4: torch.Tensor, f6: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([f4, f6], dim=-1)
        q_raw = self.net(x)
        q_norm = torch.norm(q_raw, dim=-1, keepdim=True).clamp_min(1e-12)
        q_normalized = q_raw / q_norm
        return {
            "q_raw": q_raw,
            "q_norm": q_norm,
            "q_normalized": q_normalized,
        }


class FCCLearnableDecoderAutoEncoder(nn.Module):
    """
    FCC autoencoder with a learnable decoder.

    - Encoder: physics-based invariant extraction (f4, f6)
    - Decoder: MLP predicts quaternion directly from (f4, f6)
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.physics = FCCPhysics(str(self.device))
        self.encoder = FCCEncoder(self.physics)
        self.decoder = LearnableFCCDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
        return quats / norm

    @staticmethod
    def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
        return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)

    def _to_active_convention(self, quats: torch.Tensor) -> torch.Tensor:
        return self._quat_conjugate(quats)

    def _from_active_convention(self, quats: torch.Tensor) -> torch.Tensor:
        return self._quat_conjugate(quats)

    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return FCCAutoEncoder.quat_mul(q1, q2)

    def encode(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quats_active = self._to_active_convention(quats)
        return self.encoder(quats_active)

    def decode(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        q_active = self.decoder(f4, f6)
        return self._from_active_convention(q_active)

    def decode_with_debug(self, f4: torch.Tensor, f6: torch.Tensor) -> dict[str, torch.Tensor]:
        dec = self.decoder.forward_with_debug(f4, f6)
        q_active = dec["q_normalized"]
        dec["q_output"] = self._from_active_convention(q_active)
        return dec

    def forward(self, quats: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(
                f"FCCLearnableDecoderAutoEncoder expects (N,4), got {tuple(quats.shape)}"
            )
        if normalize_input:
            quats = self._normalize_quaternions(quats)
        f4, f6 = self.encode(quats)
        return self.decode(f4, f6)

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        quats = self._normalize_quaternions(quats.to(self.device))
        batch_size = quats.shape[0]

        q_expanded = quats.unsqueeze(1).expand(-1, 24, -1)
        syms = self.physics.fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)

        syms_flat = syms.reshape(-1, 4)
        # Bunge convention: s⁻¹ ⊗ q  (left orbit under crystal symmetry)
        syms_inv_flat = torch.cat([syms_flat[:, :1], -syms_flat[:, 1:]], dim=-1)
        fam = self.quat_mul(
            syms_inv_flat,
            q_expanded.reshape(-1, 4),
        ).view(batch_size, 24, 4)
        fam = self._normalize_quaternions(fam.reshape(-1, 4)).view(batch_size, 24, 4)

        w_abs = fam[..., 0].abs()
        best_idx = torch.argmax(w_abs, dim=1)
        batch_idx = torch.arange(batch_size, device=quats.device)
        q_fz = fam[batch_idx, best_idx]
        q_fz = torch.where(q_fz[:, :1] < 0, -q_fz, q_fz)
        q_fz = self._normalize_quaternions(q_fz)

        if return_op_map:
            return q_fz, best_idx
        return q_fz

    def reconstruct_batch_debug(
        self,
        q_batch: torch.Tensor,
        normalize_input: bool = True,
        return_metrics: bool = True,
    ) -> dict[str, Any]:
        q_batch = q_batch.to(self.device)
        if normalize_input:
            q_batch = self._normalize_quaternions(q_batch)

        f4, f6 = self.encode(q_batch)
        dec = self.decode_with_debug(f4, f6)
        q_decoded = dec["q_output"]
        q_reconstructed, best_indices = self.reduce_to_fz(
            q_decoded,
            return_op_map=True,
        )

        out: dict[str, Any] = {
            "q_input": q_batch,
            "f4": f4,
            "f6": f6,
            "q_raw": dec["q_raw"],
            "q_norm": dec["q_norm"],
            "q_decoded": q_decoded,
            "q_reconstructed": q_reconstructed,
            "symmetry_index": best_indices,
        }

        if return_metrics:
            qA = self._to_active_convention(q_reconstructed)
            qB = self._to_active_convention(q_batch)
            delta = self.quat_mul(qA, self._quat_conjugate(qB))
            delta = self._normalize_quaternions(delta)
            w_abs = delta[:, 0].abs().clamp(max=1.0)
            misorientation_deg = 2.0 * torch.acos(w_abs) * 180.0 / torch.pi
            out["errors"] = misorientation_deg * torch.pi / 180.0
            out["misorientation_deg"] = misorientation_deg

        return out
