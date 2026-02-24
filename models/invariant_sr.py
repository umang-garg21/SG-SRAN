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
        self.weight_l4 = 1.0
        self.weight_l6 = 1.0

        self.grid_vecs = self._fibonacci_sphere(samples=self.n_fib_samples, device=physics.device)
        self.Y4_grid = o3.spherical_harmonics(4, self.grid_vecs, normalize=True)
        self.Y6_grid = o3.spherical_harmonics(6, self.grid_vecs, normalize=True)

    def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        batch_size = f4.shape[0]

        signal4 = torch.einsum("bi,gi->bg", f4, self.Y4_grid)
        signal6 = torch.einsum("bi,gi->bg", f6, self.Y6_grid)
        signal = self.weight_l4 * signal4 + self.weight_l6 * signal6
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


class OptimizingFCCDecoder(nn.Module):
    """
    Decode (f4, f6) -> quaternion by directly minimizing:
      ||D4(q)s4 - f4||^2 + w6 * ||D6(q)s6 - f6||^2
    """

    def __init__(
        self,
        physics: FCCPhysics,
        num_starts: int = 6,
        steps: int = 25,
        lr: float = 0.08,
        w6: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.physics = physics
        self.num_starts = int(num_starts)
        self.steps = int(steps)
        self.lr = float(lr)
        self.w6 = float(w6)
        self.eps = float(eps)

        self.register_buffer("s4", physics.s4.clone())
        self.register_buffer("s6", physics.s6.clone())

    @staticmethod
    def _normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
        norm = q.norm(dim=-1, keepdim=True)
        qn = q / norm.clamp_min(eps)

        bad = norm.squeeze(-1) < eps
        if bad.any():
            qn = qn.clone()
            qn[bad] = qn.new_tensor([1.0, 0.0, 0.0, 0.0])
        return qn

    @staticmethod
    def _fix_sign(q: torch.Tensor) -> torch.Tensor:
        return torch.where(q[..., :1] < 0, -q, q)

    def _loss(self, q: torch.Tensor, f4_tgt: torch.Tensor, f6_tgt: torch.Tensor) -> torch.Tensor:
        R = o3.quaternion_to_matrix(q)
        alpha, beta, gamma = o3.matrix_to_angles(R)

        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)

        f4_pred = torch.einsum("bij,j->bi", D4, self.s4)
        f6_pred = torch.einsum("bij,j->bi", D6, self.s6)

        l4 = (f4_pred - f4_tgt).pow(2).mean(dim=-1)
        l6 = (f6_pred - f6_tgt).pow(2).mean(dim=-1)
        return l4 + self.w6 * l6

    @torch.no_grad()
    def _init_quats(self, bsz: int, device: torch.device) -> torch.Tensor:
        k = self.num_starts
        q0 = torch.zeros((bsz, k, 4), device=device, dtype=torch.float32)
        q0[..., 0] = 1.0
        if k > 1:
            q_rand = torch.randn((bsz, k - 1, 4), device=device, dtype=torch.float32)
            q_rand = self._normalize_quat(q_rand, self.eps)
            q0[:, 1:, :] = q_rand
        q0 = self._normalize_quat(q0, self.eps)
        q0 = self._fix_sign(q0)
        return q0

    def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
        device = f4.device
        bsz = f4.shape[0]
        k = self.num_starts

        f4_tgt = f4.detach().to(torch.float32)
        f6_tgt = f6.detach().to(torch.float32)

        with torch.no_grad():
            q_init = self._init_quats(bsz, device)

        u = nn.Parameter(q_init.clone())
        opt = torch.optim.Adam([u], lr=self.lr)

        f4_rep = f4_tgt[:, None, :].expand(bsz, k, -1).reshape(bsz * k, -1)
        f6_rep = f6_tgt[:, None, :].expand(bsz, k, -1).reshape(bsz * k, -1)

        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            q = self._normalize_quat(u, self.eps)
            q = self._fix_sign(q)
            q_flat = q.reshape(bsz * k, 4)
            loss = self._loss(q_flat, f4_rep, f6_rep).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            q = self._normalize_quat(u, self.eps)
            q = self._fix_sign(q)
            q_flat = q.reshape(bsz * k, 4)
            loss_vec = self._loss(q_flat, f4_rep, f6_rep).view(bsz, k)

            best_k = torch.argmin(loss_vec, dim=1)
            batch_idx = torch.arange(bsz, device=device)
            q_best = q[batch_idx, best_k, :]
            q_best = self._normalize_quat(q_best, self.eps)
            q_best = self._fix_sign(q_best)
            return q_best


class CubochoricOptimizingFCCDecoder(OptimizingFCCDecoder):
    """
    Optimizing decoder initialized from cubochoric samples in the FCC FZ.
    """

    def __init__(
        self,
        physics: FCCPhysics,
        cubochoric_resolution: int = 3,
        **kwargs: Any,
    ):
        super().__init__(physics, **kwargs)
        self.cubochoric_resolution = int(cubochoric_resolution)
        if self.cubochoric_resolution < 1:
            raise ValueError(
                f"cubochoric_resolution must be >= 1, got {cubochoric_resolution}"
            )
        cubochoric_quats = self._build_cubochoric_quat_table(
            resolution=self.cubochoric_resolution,
            device=torch.device(physics.device),
        )
        self.register_buffer("cubochoric_quats", cubochoric_quats)

    @staticmethod
    def _build_cubochoric_quat_table(
        resolution: int,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            import numpy as np
            from orix.quaternion import symmetry
            from orix.sampling import get_sample_fundamental
        except Exception as exc:
            raise ImportError(
                "Cubochoric decoder requires `orix` and `numpy` to be available."
            ) from exc

        rot = get_sample_fundamental(
            int(resolution),
            point_group=symmetry.Oh,
            method="cubochoric",
        )

        raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
        if raw.ndim != 2:
            raw = raw.reshape(-1, 4)
        if raw.shape[-1] != 4 and raw.shape[0] == 4:
            raw = raw.T
        if raw.shape[-1] != 4:
            raise ValueError(f"Unexpected cubochoric quaternion shape: {tuple(raw.shape)}")

        q = torch.as_tensor(raw, dtype=torch.float32, device=device)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q = torch.where(q[..., :1] < 0, -q, q)
        return q

    @torch.no_grad()
    def _init_quats(self, bsz: int, device: torch.device) -> torch.Tensor:
        k = self.num_starts
        q0 = torch.zeros((bsz, k, 4), device=device, dtype=torch.float32)
        q0[..., 0] = 1.0
        if k > 1:
            table = self.cubochoric_quats.to(device=device, dtype=torch.float32)
            table_n = table.shape[0]
            if table_n > 0:
                idx = torch.randint(0, table_n, (bsz, k - 1), device=device)
                q0[:, 1:, :] = table[idx]
            else:
                q_rand = torch.randn((bsz, k - 1, 4), device=device, dtype=torch.float32)
                q0[:, 1:, :] = self._normalize_quat(q_rand, self.eps)
        q0 = self._normalize_quat(q0, self.eps)
        q0 = self._fix_sign(q0)
        return q0


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
        decoder_backend: str = "optimizing",
        decoder_cubochoric_resolution: int = 3,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.08,
        decoder_w6: float = 0.5,
        kernel_size: int = 3,
        learned_decoder_hidden_dim: int = 64,
        train_decode_mode: str = "learnable",
        eval_decode_mode: str = "spherical",
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

        backend = str(decoder_backend).lower()
        if backend == "spherical":
            self.decoder = SphericalSamplingDecoder(self.physics, grid_res=decoder_grid_res)
        elif backend == "optimizing":
            self.decoder = OptimizingFCCDecoder(
                self.physics,
                num_starts=decoder_num_starts,
                steps=decoder_steps,
                lr=decoder_lr,
                w6=decoder_w6,
            )
        elif backend in {"cubochoric", "optimizing_cubochoric"}:
            self.decoder = CubochoricOptimizingFCCDecoder(
                self.physics,
                cubochoric_resolution=decoder_cubochoric_resolution,
                num_starts=decoder_num_starts,
                steps=decoder_steps,
                lr=decoder_lr,
                w6=decoder_w6,
            )
        else:
            raise ValueError(f"Unknown decoder_backend: {decoder_backend}")
        self.learned_decoder = nn.Sequential(
            nn.Linear(22, int(learned_decoder_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(learned_decoder_hidden_dim), 4),
        )
        self.train_decode_mode = str(train_decode_mode).lower()
        self.eval_decode_mode = str(eval_decode_mode).lower()

        if self.train_decode_mode == "spherical" and self.eval_decode_mode == "spherical":
            for param in self.learned_decoder.parameters():
                param.requires_grad = False

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

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        quats = self.normalize_quaternions(quats.to(self.device))
        batch_size = quats.shape[0]

        q_expanded = quats.unsqueeze(1).expand(-1, 24, -1)
        syms = self.physics.fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)

        fam = self.quat_mul(
            q_expanded.reshape(-1, 4),
            syms.reshape(-1, 4),
        ).view(batch_size, 24, 4)
        fam = self.normalize_quaternions(fam.reshape(-1, 4)).view(batch_size, 24, 4)

        w_abs = fam[..., 0].abs()
        best_idx = torch.argmax(w_abs, dim=1)
        batch_idx = torch.arange(batch_size, device=quats.device)
        q_fz = fam[batch_idx, best_idx]
        q_fz = torch.where(q_fz[:, :1] < 0, -q_fz, q_fz)
        q_fz = self.normalize_quaternions(q_fz)

        if return_op_map:
            return q_fz, best_idx
        return q_fz

    def _forward_flat(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int],
        decode: bool = True,
        match_symmetry_to: torch.Tensor | None = None,
        decode_mode: str | None = None,
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
            mode = (decode_mode or (self.train_decode_mode if self.training else self.eval_decode_mode)).lower()
            if mode == "learnable":
                q_logits = self.learned_decoder(torch.cat([f4_hr, f6_hr], dim=-1))
                q_out = self.normalize_quaternions(q_logits)
            elif mode == "spherical":
                q_out = self.decoder(f4_hr, f6_hr)
            else:
                raise ValueError(f"Unknown decode mode: {mode}")
            out["output"] = q_out

            if match_symmetry_to is not None:
                q_match, sym_idx = self.reduce_to_fz(q_out, return_op_map=True)
                out["output_matched"] = q_match
                out["match_symmetry_index"] = sym_idx

        return out

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        decode: bool = True,
        match_symmetry_to: torch.Tensor | None = None,
        decode_mode: str | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        if quats.dim() == 4:
            if quats.shape[1] != 4:
                raise ValueError(f"Expected BCHW quaternion input with 4 channels, got {tuple(quats.shape)}")

            B, _, H, W = quats.shape
            out_quats = []
            for b in range(B):
                q_flat = quats[b].permute(1, 2, 0).reshape(-1, 4)
                out_b = self._forward_flat(
                    q_flat,
                    img_shape=(H, W),
                    decode=decode,
                    match_symmetry_to=None,
                    decode_mode=decode_mode,
                )
                if not decode:
                    raise ValueError("Batched forward requires decode=True")

                q_out_flat = out_b["output"]
                hr_h, hr_w = out_b["hr_shape"]
                q_out_chw = q_out_flat.reshape(hr_h, hr_w, 4).permute(2, 0, 1)
                out_quats.append(q_out_chw)

            return torch.stack(out_quats, dim=0)

        if img_shape is None:
            raise ValueError("img_shape is required for flattened quaternion input")

        return self._forward_flat(
            quats,
            img_shape=img_shape,
            decode=decode,
            match_symmetry_to=match_symmetry_to,
            decode_mode=decode_mode,
        )

