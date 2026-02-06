"""
EBSD symmetry-aware x4 Super-Resolution with VALUE-SPACE equivariance and CRYSTAL O invariance.

Goal:
  LR quaternion map q_lr (B,4,h,w)  --->  HR quaternion map q_sr (B,4,4h,4w)

Key ideas:
  1) Crystal symmetry O acts on the RIGHT: q ~ q ⊗ s, s in O.
  2) We construct per-pixel spherical-tensor features that are:
       - left-SO(3) equivariant (frame change q -> g ⊗ q)
       - right-O invariant (crystal symmetry q -> q ⊗ s leaves features unchanged)
     via invariant-subspace bases v_{l,k} (Reynolds projector) and features:
         x_{l,k}(q) = D^l(q) v_{l,k}   in R^{2l+1}
  3) SR is done on those spherical-tensor feature maps using:
       - translation-equivariant spatial mixing (depthwise Conv2d)
       - value-space equivariant channel mixing (e3nn o3.Linear + NormActivation)
     with PixelShuffle for x4.
  4) Decoding back to quaternions is done by a grid search on SO(3):
       score(q_grid) = sum_{l,k} < x_{l,k},  D^l(q_grid) v_{l,k} >
     and pick argmax.

Training (recommended):
  - Train SRNet to match target HR features (encoder(q_hr)) with SmoothL1/Huber.
  - Decoder is not differentiable (argmax); use it for eval/visualization.

This file provides ALL model code:
  - FCCPhysics
  - wigner_D_cuda
  - Cubic invariant basis computation (Reynolds)
  - Encoder -> spherical-tensor feature maps (value-equivariant, crystal-invariant)
  - Value-equivariant SRNet x4 (hybrid spatial depthwise + e3nn channel mixing)
  - Grid decoder -> quaternions (B,4,H,W)
  - Wrapper module EBSDValueEquivariantSR

Notes:
  - Grid decoding can be expensive. Start with grid_quats ~ 2048-8192 and scale up.
  - This code assumes your physics.fcc_syms are unit quaternions representing cubic O.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn import nn as e3nn_nn


# ==============================================================================
# CUDA-compatible Wigner D (your patched version)
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
# Quaternion utilities (wxyz)
# ==============================================================================
def quat_normalize_last(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_normalize_chfirst(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: (B,4,H,W)
    return q / (q.norm(dim=1, keepdim=True) + eps)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (...,4) x (...,4) -> (...,4)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


# ==============================================================================
# FCC Physics: cubic O symmetry (24) (as you provided)
# ==============================================================================
class FCCPhysics(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        half = 0.5
        syms = torch.tensor(
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
            device=self.device,
        )
        # normalize just in case
        syms = syms / (syms.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("fcc_syms", syms)


# ==============================================================================
# Reynolds projector: invariant basis vectors v_{l,k} in Fix_O(l)
# ==============================================================================
@torch.no_grad()
def cubic_invariant_basis(physics: FCCPhysics, l: int, tol: float = 1e-4):
    """
    Compute basis V (m_l, 2l+1) for the subspace fixed by cubic O under D^l(s).
    P = (1/|O|) sum_s D^l(s). Eigenvectors with eigenvalue ~ 1 span Fix_O(l).
    """
    device = physics.fcc_syms.device
    syms_q = physics.fcc_syms  # (24,4)

    R = o3.quaternion_to_matrix(syms_q)  # (24,3,3)
    a, b, g = o3.matrix_to_angles(R)
    D = wigner_D_cuda(l, a, b, g)  # (24, 2l+1, 2l+1)

    P = D.mean(dim=0)  # (2l+1,2l+1)
    Ps = 0.5 * (P + P.T)  # symmetrize for stable eigendecomp

    evals, evecs = torch.linalg.eigh(Ps)  # ascending
    mask = evals > (1.0 - tol)
    V = evecs[:, mask].T.contiguous()  # (m_l, 2l+1)
    return V, evals


# ==============================================================================
# Encoder: q -> spherical-tensor features x_{l,k} = D^l(q) v_{l,k}
#  - left-SO(3) equivariant, right-O invariant
# ==============================================================================
class CubicEquivariantEncoder(nn.Module):
    """
    Outputs a spherical tensor field with irreps:
        ⊕_l (m_l x l e)
    where m_l = dim Fix_O(l).

    Input:
      q: (B,4,H,W) or (N,4)

    Output:
      x: (B,C,H,W) or (N,C) with C = irreps_out.dim
    """

    def __init__(self, physics: FCCPhysics, Ls=(4, 6), tol=1e-4):
        super().__init__()
        self.physics = physics
        self.Ls = tuple(Ls)

        irreps_list = []
        for l in self.Ls:
            V, _ = cubic_invariant_basis(physics, l, tol=tol)
            self.register_buffer(f"V_l{l}", V)  # (m_l, 2l+1)
            m_l = V.shape[0]
            irreps_list.append((m_l, o3.Irrep(l, 1)))  # parity=+ (even)
        self.irreps_out = o3.Irreps(irreps_list).simplify()

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        is_map = q.dim() == 4
        if is_map:
            B, C, H, W = q.shape
            assert C == 4
            q = quat_normalize_chfirst(q)
            q_flat = q.permute(0, 2, 3, 1).reshape(-1, 4)  # (BHW,4)
        else:
            q_flat = quat_normalize_last(q)

        R = o3.quaternion_to_matrix(q_flat)  # (N,3,3)
        a, b, g = o3.matrix_to_angles(R)  # (N,)

        blocks = []
        for l in self.Ls:
            D = wigner_D_cuda(l, a, b, g)  # (N,2l+1,2l+1)
            V = getattr(self, f"V_l{l}")  # (m_l,2l+1)
            Y = torch.einsum("nij,kj->nik", D, V)  # (N,2l+1,m_l)
            blocks.append(Y.reshape(Y.shape[0], -1))  # (N, m_l*(2l+1))

        x_flat = torch.cat(blocks, dim=1)  # (N,C)

        if not is_map:
            return x_flat

        x = x_flat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


# ==============================================================================
# Value-space equivariant block: spatial depthwise mixing + e3nn channel mixing
# ==============================================================================
class ValueEquivariantBlock(nn.Module):
    """
    x: (B,C,H,W) where C = irreps.dim
    - spatial mixing: depthwise conv (translation equivariant, no channel mixing)
    - value mixing: o3.Linear + BatchNorm + NormActivation + o3.Linear (equivariant)
    """

    def __init__(self, irreps: o3.Irreps, hidden_mul=4):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        C = self.irreps.dim

        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)

        irreps_hid = (hidden_mul * self.irreps).simplify()
        self.lin1 = o3.Linear(self.irreps, irreps_hid)
        self.bn1 = e3nn_nn.BatchNorm(irreps_hid)
        self.act1 = e3nn_nn.NormActivation(irreps_hid, torch.nn.functional.silu)
        self.lin2 = o3.Linear(irreps_hid, self.irreps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = y.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        y = self.lin1(y)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.lin2(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return x + y


# ==============================================================================
# Value-space equivariant SR x4: blocks at LR -> expand equivariantly -> PixelShuffle -> blocks at HR
# ==============================================================================
class ValueEquivariantSRX4(nn.Module):
    """
    x: (B,C,h,w) -> (B,C,4h,4w)
    """

    def __init__(self, irreps: o3.Irreps, blocks_lr=4, blocks_hr=2, hidden_mul=4):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        C = self.irreps.dim

        self.lr_blocks = nn.Sequential(
            *[
                ValueEquivariantBlock(self.irreps, hidden_mul=hidden_mul)
                for _ in range(blocks_lr)
            ]
        )

        self.expand = o3.Linear(self.irreps, (16 * self.irreps).simplify())  # to 16C
        self.ps = nn.PixelShuffle(4)

        self.hr_blocks = nn.Sequential(
            *[
                ValueEquivariantBlock(self.irreps, hidden_mul=hidden_mul)
                for _ in range(blocks_hr)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lr_blocks(x)

        y = x.permute(0, 2, 3, 1).contiguous()  # (B,h,w,C)
        y = self.expand(y)  # (B,h,w,16C)
        y = y.permute(0, 3, 1, 2).contiguous()  # (B,16C,h,w)
        y = self.ps(y)  # (B,C,4h,4w)

        y = self.hr_blocks(y)
        return y


# ==============================================================================
# Grid decoder: x_sr -> quaternion via argmax over precomputed templates
# ==============================================================================
class GridDecoderSO3(nn.Module):
    """
    Decode spherical-tensor features x (B,C,H,W) back to quaternions via grid search.

    For each l and basis vector v_{l,k}, precompute template:
        T_{l,k}(qg) = D^l(qg) v_{l,k}   in R^{2l+1}
    Then score per pixel:
        score(qg) = sum_{l,k} < x_{l,k}, T_{l,k}(qg) >
    Choose argmax score.

    This decoder is NOT differentiable due to argmax, use for inference/eval.
    """

    def __init__(
        self,
        physics: FCCPhysics,
        encoder: CubicEquivariantEncoder,
        grid_quats: torch.Tensor,
    ):
        super().__init__()
        self.physics = physics
        self.encoder = encoder  # provides V_l{l} and Ls and irreps_out structure

        # grid_quats: (G,4) unit quats (wxyz)
        grid_quats = grid_quats.to(physics.fcc_syms.device)
        grid_quats = grid_quats / (grid_quats.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("grid_quats", grid_quats)

        # Precompute templates for each l:
        # templates_l: (G, 2l+1, m_l) => flattened as (G, m_l*(2l+1))
        Rg = o3.quaternion_to_matrix(grid_quats)
        a, b, g = o3.matrix_to_angles(Rg)

        self.templates = nn.ModuleDict()  # store as buffers inside dict-like module
        self.slices = []  # list of (start, end, l, m_l, dim_l)

        idx = 0
        for l in encoder.Ls:
            V = getattr(encoder, f"V_l{l}")  # (m_l, 2l+1)
            m_l = V.shape[0]
            dim_l = 2 * l + 1
            D = wigner_D_cuda(l, a, b, g)  # (G, dim_l, dim_l)
            T = torch.einsum("gij,kj->gik", D, V)  # (G, dim_l, m_l)
            T = T.reshape(T.shape[0], -1).contiguous()  # (G, m_l*dim_l)

            self.register_buffer(f"templ_l{l}", T)
            self.slices.append((idx, idx + m_l * dim_l, l, m_l, dim_l))
            idx += m_l * dim_l

        self.C = idx  # total channels expected

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)  with C == encoder.irreps_out.dim
        Returns:
          q: (B,4,H,W)
        """
        B, C, H, W = x.shape
        assert C == self.C, f"Decoder expected C={self.C}, got {C}"

        # flatten pixels
        N = B * H * W
        x_flat = x.permute(0, 2, 3, 1).reshape(N, C)  # (N,C)

        # score against grid templates: score = x · T^T
        # Build T_all: (G,C) by concatenating templ_l{l}
        templ_list = []
        for s, e, l, m_l, dim_l in self.slices:
            templ_list.append(getattr(self, f"templ_l{l}"))  # (G, blockC)
        T_all = torch.cat(templ_list, dim=1)  # (G,C)

        scores = x_flat @ T_all.T  # (N,G)
        best = scores.argmax(dim=1)  # (N,)

        q = self.grid_quats[best]  # (N,4)
        q = q.view(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        return q


# ==============================================================================
# Full model: LR quats -> encode -> SR -> decode
# ==============================================================================
class EBSDValueEquivariantSR(nn.Module):
    """
    End-to-end inference model.

    Forward:
      q_lr (B,4,h,w) -> x_lr (B,C,h,w) -> x_sr (B,C,4h,4w) -> q_sr (B,4,4h,4w)
    """

    def __init__(
        self,
        device="cuda:0",
        Ls=(4, 6),
        tol=1e-4,
        grid_size=4096,
        blocks_lr=4,
        blocks_hr=2,
        hidden_mul=4,
        grid_seed=0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.physics = FCCPhysics(self.device)

        # Encoder gives irreps and invariant bases
        self.encoder = CubicEquivariantEncoder(self.physics, Ls=Ls, tol=tol)

        # SRNet on spherical tensors
        self.srnet = ValueEquivariantSRX4(
            self.encoder.irreps_out,
            blocks_lr=blocks_lr,
            blocks_hr=blocks_hr,
            hidden_mul=hidden_mul,
        )

        # Build a random grid of quats (replace with better SO(3) sampling later)
        g = torch.Generator(device=self.device)
        g.manual_seed(grid_seed)
        grid_quats = torch.randn(grid_size, 4, device=self.device, generator=g)
        grid_quats = grid_quats / (grid_quats.norm(dim=1, keepdim=True) + 1e-8)

        self.decoder = GridDecoderSO3(self.physics, self.encoder, grid_quats)

        self.to(self.device)

    def forward(self, q_lr: torch.Tensor) -> dict:
        """
        q_lr: (B,4,h,w) or (4,h,w)
        returns dict: x_lr, x_sr, q_sr
        """
        if q_lr.dim() == 3:
            q_lr = q_lr.unsqueeze(0)
        q_lr = q_lr.to(self.device).float()
        q_lr = quat_normalize_chfirst(q_lr)

        # encode -> spherical tensors
        x_lr = self.encoder(q_lr)  # (B,C,h,w)

        # SR in spherical-tensor space
        x_sr = self.srnet(x_lr)  # (B,C,4h,4w)

        # decode -> quats (argmax over grid)
        q_sr = self.decoder(x_sr)  # (B,4,4h,4w)
        q_sr = quat_normalize_chfirst(q_sr)

        return {"q_lr": q_lr, "x_lr": x_lr, "x_sr": x_sr, "q_sr": q_sr}


# ==============================================================================
# Recommended training loss on features (stable, decoder-aligned)
# ==============================================================================
def feature_loss_huber(x_pred: torch.Tensor, x_true: torch.Tensor, beta=0.01):
    """
    x_pred, x_true: (B,C,H,W)
    Use SmoothL1 (Huber). This is stable for EBSD boundary outliers.
    """
    return F.smooth_l1_loss(x_pred, x_true, beta=beta)


# ==============================================================================
# Example usage (inference shape check)
# ==============================================================================
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = EBSDValueEquivariantSR(
        device=device,
        Ls=(4, 6),  # extend later: (4,6,8,10)
        grid_size=4096,  # increase for accuracy
        blocks_lr=4,
        blocks_hr=2,
        hidden_mul=4,
    )

    # dummy LR input
    B, h, w = 1, 32, 32
    q_lr = torch.randn(B, 4, h, w, device=model.device)
    q_lr = quat_normalize_chfirst(q_lr)

    out = model(q_lr)
    print("x_lr:", out["x_lr"].shape)
    print("x_sr:", out["x_sr"].shape)
    print("q_sr:", out["q_sr"].shape)
