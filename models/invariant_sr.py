import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps


def wigner_D_cuda(
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


class FCCPhysics(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        self.s4 = torch.zeros(9, device=device)
        self.s4[4] = 0.7638
        self.s4[8] = 0.6455

        self.s6 = torch.zeros(13, device=device)
        self.s6[6] = 0.3536
        self.s6[10] = -0.9354

        inv_sqrt_2 = 1 / math.sqrt(2)
        half = 0.5
        self.fcc_syms = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [inv_sqrt_2, inv_sqrt_2, 0, 0],
                [inv_sqrt_2, 0, inv_sqrt_2, 0],
                [inv_sqrt_2, 0, 0, inv_sqrt_2],
                [inv_sqrt_2, -inv_sqrt_2, 0, 0],
                [inv_sqrt_2, 0, -inv_sqrt_2, 0],
                [inv_sqrt_2, 0, 0, -inv_sqrt_2],
                [0, inv_sqrt_2, inv_sqrt_2, 0],
                [0, inv_sqrt_2, 0, inv_sqrt_2],
                [0, 0, inv_sqrt_2, inv_sqrt_2],
                [0, inv_sqrt_2, -inv_sqrt_2, 0],
                [0, 0, inv_sqrt_2, -inv_sqrt_2],
                [0, inv_sqrt_2, 0, -inv_sqrt_2],
                [half, half, half, half],
                [half, -half, -half, half],
                [half, -half, half, -half],
                [half, half, -half, -half],
                [half, half, half, -half],
                [half, half, -half, half],
                [half, -half, half, half],
                [half, -half, -half, -half],
            ],
            dtype=torch.float32,
            device=device,
        )


class FCCEncoder(nn.Module):
    def __init__(self, physics: FCCPhysics):
        super().__init__()
        self.physics = physics

    def forward(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)

        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)

        return f4, f6


class SphericalSamplingDecoder(nn.Module):
    def __init__(self, physics: FCCPhysics, grid_res: int = 10_000):
        super().__init__()
        self.n_fib_samples = grid_res
        self.physics = physics

        self.grid_vecs = self._fibonacci_sphere(samples=self.n_fib_samples, device=physics.device)
        self.Y4_grid = o3.spherical_harmonics(4, self.grid_vecs, normalize=True)

    def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        del f6
        batch_size = f4.shape[0]

        signal = torch.einsum("bi,gi->bg", f4, self.Y4_grid)
        _, z_indices = torch.max(signal, dim=1)
        z_axis = self.grid_vecs[z_indices]

        dots = torch.einsum(
            "bij,bij->bi",
            self.grid_vecs.unsqueeze(0).expand(batch_size, -1, -1),
            z_axis.unsqueeze(1).expand(-1, self.n_fib_samples, -1),
        )
        mask = dots.abs() < 0.2

        masked_signal = signal.clone()
        masked_signal[~mask] = -float("inf")

        _, x_indices = torch.max(masked_signal, dim=1)
        x_axis = self.grid_vecs[x_indices]

        z_axis = F.normalize(z_axis, dim=-1)
        proj = torch.sum(x_axis * z_axis, dim=-1, keepdim=True) * z_axis
        x_axis = F.normalize(x_axis - proj, dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)

        R_rec = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return o3.matrix_to_quaternion(R_rec)

    def _fibonacci_sphere(self, samples: int, device: str) -> torch.Tensor:
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])

        return torch.tensor(points, dtype=torch.float32, device=device)


class EquivariantSpatialConv(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.irreps = Irreps("1x4e + 1x6e")

        self.tp = FullyConnectedTensorProduct(
            self.irreps,
            self.irreps,
            self.irreps,
            shared_weights=True,
        )
        self.spatial_weights = nn.Parameter(torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size))

    def forward(self, f4: torch.Tensor, f6: torch.Tensor, img_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = img_shape
        features = torch.cat([f4, f6], dim=-1)  # (H*W, 22)

        feat_img = features.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # (1,22,H,W)
        feat_padded = F.pad(feat_img, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        weights = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neighbor = (patches * weights).sum(dim=(-1, -2))
        neighbor = neighbor.squeeze(0).permute(1, 2, 0).reshape(-1, 22)

        out = self.tp(features, neighbor)

        f4_out = out[:, :9] + f4
        f6_out = out[:, 9:] + f6
        return f4_out, f6_out


class EquivariantUpsampleConv(nn.Module):
    def __init__(self, upsample_factor: int = 4, kernel_size: int = 3):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.irreps = Irreps("1x4e + 1x6e")
        self.tp = FullyConnectedTensorProduct(
            self.irreps,
            self.irreps,
            self.irreps,
            shared_weights=True,
        )

        self.spatial_weights = nn.Parameter(torch.zeros(kernel_size, kernel_size))
        self.spatial_weights.data[kernel_size // 2, kernel_size // 2] = 1.0

        with torch.no_grad():
            self.tp.weight.data.zero_()

    def forward(
        self,
        f4: torch.Tensor,
        f6: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        H, W = img_shape
        C = 22
        r = self.upsample_factor

        features = torch.cat([f4, f6], dim=-1)
        feat_img = features.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

        feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode="nearest")
        Hr, Wr = H * r, W * r

        feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neighbor = (patches * w).sum(dim=(-1, -2))

        feat_flat = feat_hr.squeeze(0).permute(1, 2, 0).reshape(-1, C)
        neighbor_flat = neighbor.squeeze(0).permute(1, 2, 0).reshape(-1, C)

        out = self.tp(feat_flat, neighbor_flat)

        f4_out = out[:, :9] + feat_flat[:, :9]
        f6_out = out[:, 9:] + feat_flat[:, 9:]
        return f4_out, f6_out, (Hr, Wr)


class InvariantSRModel(nn.Module):
    """
    Model-only SR architecture ported from e3nn experimentation modules.

    Pipeline:
      quaternions -> encoder -> LR equivariant conv -> equivariant upsample
      -> HR equivariant conv -> spherical decoder -> output quaternions
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        upsample_factor: int = 4,
        decoder_grid_res: int = 10_000,
        kernel_size: int = 3,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.upsample_factor = int(upsample_factor)

        self.physics = FCCPhysics(str(self.device))
        self.encoder = FCCEncoder(self.physics)
        self.conv_layer = EquivariantSpatialConv(kernel_size=kernel_size)
        self.upsample_layer = EquivariantUpsampleConv(
            upsample_factor=self.upsample_factor,
            kernel_size=kernel_size,
        )
        self.hr_conv_layer = EquivariantSpatialConv(kernel_size=kernel_size)
        self.decoder = SphericalSamplingDecoder(self.physics, grid_res=decoder_grid_res)

    @staticmethod
    def normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return quats / torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=1,
        )

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

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int],
        decode: bool = True,
        match_symmetry_to: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"Expected quaternion tensor shape (N,4), got {tuple(quats.shape)}")

        H, W = img_shape
        if quats.shape[0] != H * W:
            raise ValueError(f"img_shape {img_shape} implies {H*W} quats, got {quats.shape[0]}")

        q_in = self.normalize_quaternions(quats)

        f4, f6 = self.encoder(q_in)
        f4_conv, f6_conv = self.conv_layer(f4, f6, img_shape=(H, W))
        f4_up, f6_up, hr_shape = self.upsample_layer(f4_conv, f6_conv, img_shape=(H, W))
        f4_hr, f6_hr = self.hr_conv_layer(f4_up, f6_up, img_shape=hr_shape)

        out: dict[str, Any] = {
            "input": q_in,
            "encoded": (f4, f6),
            "convolved": (f4_conv, f6_conv),
            "upsampled_irreps": (f4_up, f6_up),
            "hr_convolved_irreps": (f4_hr, f6_hr),
            "hr_shape": hr_shape,
        }

        if decode:
            q_out = self.decoder(f4_hr, f6_hr)
            out["output"] = q_out

            if match_symmetry_to is not None:
                q_ref = self.normalize_quaternions(match_symmetry_to.to(self.device))
                q_match, q_err, sym_idx = self.match_closest_symmetry(q_out, q_ref)
                out["output_matched"] = q_match
                out["match_error"] = q_err
                out["match_symmetry_index"] = sym_idx

        return out

