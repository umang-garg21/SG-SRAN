import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps


# ==============================================================================
# CUDA-Compatible Wigner D Function (Patched)
# ==============================================================================
def wigner_D_cuda(
    l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
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


# ==============================================================================
# FCC Physics
# ==============================================================================
class FCCPhysics(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.register_buffer("s4", torch.zeros(9, device=device))
        self.s4[4] = 0.7638
        self.s4[8] = 0.6455

        self.register_buffer("s6", torch.zeros(13, device=device))
        self.s6[6] = 0.3536
        self.s6[10] = -0.9354

        inv = 1.0 / math.sqrt(2.0)
        half = 0.5
        self.register_buffer(
            "fcc_syms",
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [inv, inv, 0, 0],
                    [inv, 0, inv, 0],
                    [inv, 0, 0, inv],
                    [inv, -inv, 0, 0],
                    [inv, 0, -inv, 0],
                    [inv, 0, 0, -inv],
                    [0, inv, inv, 0],
                    [0, inv, 0, inv],
                    [0, 0, inv, inv],
                    [0, inv, -inv, 0],
                    [0, 0, inv, -inv],
                    [0, inv, 0, -inv],
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
            ),
        )


# ==============================================================================
# Encoder: quats -> (f4,f6)
# Supports quats shaped (N,4) OR (B,4,H,W)
# ==============================================================================
class FCCEncoder(nn.Module):
    def __init__(self, physics: FCCPhysics):
        super().__init__()
        self.physics = physics

    def forward(self, quats):
        """
        quats:
          - (N,4) OR
          - (B,4,H,W)
        returns f4,f6 flattened: (N,9), (N,13)
        """
        if quats.dim() == 4:
            B, C, H, W = quats.shape
            assert C == 4
            q = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)
            quats_flat = q.permute(0, 2, 3, 1).reshape(-1, 4)
        else:
            quats_flat = quats
            quats_flat = quats_flat / (quats_flat.norm(dim=1, keepdim=True) + 1e-8)

        R = o3.quaternion_to_matrix(quats_flat)
        alpha, beta, gamma = o3.matrix_to_angles(R)

        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)

        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)  # (N,9)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)  # (N,13)
        return f4, f6


# ==============================================================================
# Decoder: spherical peak finding operating on feature MAPS
# Input: F (B,22,H,W) -> q_hat (B,4,H,W)
# ==============================================================================
class SphericalSamplingDecoder(nn.Module):
    def __init__(self, physics: FCCPhysics, n_fib_samples=10000, ortho_thresh=0.2):
        super().__init__()
        self.physics = physics
        self.n_fib_samples = int(n_fib_samples)
        self.ortho_thresh = float(ortho_thresh)

        grid = self._fibonacci_sphere(self.n_fib_samples, physics.device)  # (N,3)
        Y4 = o3.spherical_harmonics(4, grid, normalize=True)  # (N,9)

        self.register_buffer("grid_vecs", grid)
        self.register_buffer("Y4_grid", Y4)

    def forward(self, f4, f6=None, img_shape=None):
        """
        Either:
          - f4: (N,9), f6:(N,13), img_shape=(B,H,W) or (H,W)
          - OR f4: (B,H,W,9) etc (not used here)
        Returns:
          q_flat: (N,4) if img_shape is None
          q_img:  (B,4,H,W) if img_shape provided
        """
        # f4: (N,9)
        N = f4.shape[0]
        signal = f4 @ self.Y4_grid.T  # (N, n_grid)

        z_idx = signal.argmax(dim=1)
        z = self.grid_vecs[z_idx]
        z = F.normalize(z, dim=-1)

        dots = z @ self.grid_vecs.T
        mask = dots.abs() < self.ortho_thresh
        masked_signal = torch.where(
            mask, signal, torch.full_like(signal, -float("inf"))
        )
        x_idx = masked_signal.argmax(dim=1)
        x = self.grid_vecs[x_idx]
        x = F.normalize(x, dim=-1)

        proj = (x * z).sum(dim=-1, keepdim=True) * z
        x = F.normalize(x - proj, dim=-1)
        y = F.normalize(torch.cross(z, x, dim=-1), dim=-1)

        R = torch.stack([x, y, z], dim=-1)  # (N,3,3)
        q = o3.matrix_to_quaternion(R)  # (N,4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        if img_shape is None:
            return q

        # img_shape can be (H,W) or (B,H,W)
        if len(img_shape) == 2:
            H, W = img_shape
            B = 1
        else:
            B, H, W = img_shape
        q_img = q.view(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        return q_img

    @staticmethod
    def _fibonacci_sphere(samples, device):
        i = torch.arange(samples, device=device, dtype=torch.float32)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        y = 1.0 - (i / (samples - 1.0)) * 2.0
        r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
        theta = phi * i
        x = torch.cos(theta) * r
        z = torch.sin(theta) * r
        return torch.stack([x, y, z], dim=-1)


# ==============================================================================
# SR on invariant features (simple + stable baseline)
# Input:  (B,22,h,w) -> (B,22,H,W) with PixelShuffle x4
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.a = nn.SiLU()
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(self.a(self.c1(x)))


class FeatureSRNetX4(nn.Module):
    def __init__(self, in_ch=22, hidden=128, num_blocks=12, scale=4):
        super().__init__()
        assert scale == 4
        self.head = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(hidden, in_ch * (scale**2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.ps(self.tail(x))
        return x


# ==============================================================================
# Full symmetry-aware SR wrapper
# ==============================================================================
class EBSDInvariantSR(nn.Module):
    """
    LR quats -> encode (f4,f6) -> SR in invariant feature space -> decode to HR quats
    """

    def __init__(self, device="cpu", scale=4, grid_samples=10000):
        super().__init__()
        self.device = torch.device(device)
        self.scale = scale

        self.physics = FCCPhysics(self.device)
        self.encoder = FCCEncoder(self.physics)
        self.srnet = FeatureSRNetX4(in_ch=22, hidden=128, num_blocks=12, scale=scale)
        self.decoder = SphericalSamplingDecoder(
            self.physics, n_fib_samples=grid_samples
        )

        self.to(self.device)

    def forward(self, q_lr):
        """
        q_lr: (B,4,h,w) or (4,h,w)
        returns dict:
          F_lr, F_hr_pred, q_hr_pred
        """
        if q_lr.dim() == 3:
            q_lr = q_lr.unsqueeze(0)

        B, C, h, w = q_lr.shape
        assert C == 4

        q_lr = q_lr.to(self.device).float()
        q_lr = q_lr / (q_lr.norm(dim=1, keepdim=True) + 1e-8)

        # Encode LR quats -> LR invariant feature map
        f4_lr, f6_lr = self.encoder(q_lr)  # (B*h*w,9),(B*h*w,13)
        F_lr = (
            torch.cat([f4_lr, f6_lr], dim=-1)
            .view(B, h, w, 22)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # SR in invariant space
        F_hr_pred = self.srnet(F_lr)  # (B,22,H,W)
        H, W = F_hr_pred.shape[-2], F_hr_pred.shape[-1]

        # Decode HR features -> HR quats (physics)
        f_hr = F_hr_pred.permute(0, 2, 3, 1).reshape(-1, 22)
        f4_hr = f_hr[:, :9]
        f6_hr = f_hr[:, 9:]
        q_hr_pred = self.decoder(f4_hr, f6_hr, img_shape=(B, H, W))  # (B,4,H,W)

        return {
            "q_lr": q_lr,
            "F_lr": F_lr,
            "F_hr_pred": F_hr_pred,
            "q_hr_pred": q_hr_pred,
        }


if __name__ == "__main__":
    model = EBSDInvariantSR(device="cuda:0", scale=4, grid_samples=10000)
    
    # q_lr: (4,32,32) or (1,4,32,32)
    out = model(q_lr)
    q_hr_pred = out["q_hr_pred"]  # (B,4,128,128)
