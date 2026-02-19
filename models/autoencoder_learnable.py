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
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return FCCAutoEncoder.quat_mul(q1, q2)

    def encode(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(quats)

    def decode(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        return self.decoder(f4, f6)

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

    def match_closest_symmetry(
        self,
        q_decoded: torch.Tensor,
        q_truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_decoded = q_decoded.to(self.device)
        q_truth = q_truth.to(self.device)

        q_rec_expanded = q_decoded.unsqueeze(1).expand(-1, 24, -1)
        fcc_syms_expanded = self.physics.fcc_syms.unsqueeze(0).expand(q_truth.shape[0], -1, -1)

        w1, x1, y1, z1 = (
            q_rec_expanded[..., 0],
            q_rec_expanded[..., 1],
            q_rec_expanded[..., 2],
            q_rec_expanded[..., 3],
        )
        w2, x2, y2, z2 = (
            fcc_syms_expanded[..., 0],
            fcc_syms_expanded[..., 1],
            fcc_syms_expanded[..., 2],
            fcc_syms_expanded[..., 3],
        )

        family = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

        q_truth_expanded = q_truth.unsqueeze(1)
        dist_pos = torch.norm(family - q_truth_expanded, dim=-1)
        dist_neg = torch.norm(family + q_truth_expanded, dim=-1)
        min_dist = torch.minimum(dist_pos, dist_neg)

        errors = torch.min(min_dist, dim=1)[0]
        best_indices = torch.argmin(min_dist, dim=1)

        batch_indices = torch.arange(q_truth.shape[0], device=self.device)
        closest_quats = family[batch_indices, best_indices]
        use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
        closest_quats[use_neg] = -closest_quats[use_neg]

        return closest_quats, errors, best_indices
