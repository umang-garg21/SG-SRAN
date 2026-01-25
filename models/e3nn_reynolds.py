import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet
import numpy as np
import math

# =============================================================================
# 1. THE "CUBIC SEED" GENERATOR
# =============================================================================
def get_cubic_seeds(device):
    """
    Fundamental FCC Invariant vectors (Seeds) for L=4 and L=6.
    These act as the 'Standard Cube' templates.
    """
    s4 = torch.zeros(9, device=device)
    s4[0], s4[4], s4[8] = math.sqrt(7/12), math.sqrt(5/24), math.sqrt(5/24)
    
    s6 = torch.zeros(13, device=device)
    s6[0], s6[4], s6[8] = math.sqrt(1/8), -math.sqrt(7/16), -math.sqrt(7/16)
    return s4, s6


# =============================================================================
# 2. RIGOROUS ENCODER (Lifting Quats to Cubic Harmonics)
# =============================================================================
class RigorousCubicEncoder(nn.Module):
    def __init__(self, n_l4=4, n_l6=2):
        super().__init__()
        # multiplicity of cubic descriptors
        self.n_l4, self.n_l6 = n_l4, n_l6 
        self.irreps_out = o3.Irreps(f"16x0e + {n_l4}x4e + {n_l6}x6e")

    def forward(self, quats):
        B, _, H, W = quats.shape
        q_flat = quats.permute(0, 2, 3, 1).reshape(-1, 4)
        R = o3.quaternion_to_matrix(q_flat)
        s4, s6 = get_cubic_seeds(quats.device)

        # 16x0e Scalars
        scalars = torch.zeros(q_flat.shape[0], 16, device=quats.device)
        scalars[:, 0] = 1.0

        # L=4 & L=6: Rotate the 'Standard Cube' seeds
        # This creates features that are ALREADY cubic-consistent
        f4 = torch.einsum("bij,j->bi", o3.wigner_D(4, R), s4).repeat(1, self.n_l4)
        f6 = torch.einsum("bij,j->bi", o3.wigner_D(6, R), s6).repeat(1, self.n_l6)
        
        out = torch.cat([scalars, f4, f6], dim=-1)
        return out.view(B, H, W, -1).permute(0, 3, 1, 2)


# =============================================================================
# 3. PARALLELIZED MESSAGE PASSING (Injecting L=1 Edge Info)
# =============================================================================
class ParallelCubicMessagePassing(nn.Module):
    def __init__(self, irreps_in, irreps_out, kernel_size=3):
        super().__init__()
        self.K = kernel_size
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.edge_irreps = o3.Irreps("1x1o") # This is where L=1 lives!

        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.edge_irreps, self.irreps_out)
        self.radial_net = FullyConnectedNet([1, 16, self.tp.weight_numel], F.silu)

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.K
        device = x.device
        
        # Geometry: L=1 vectors to the 9 neighbors (3x3)
        r = torch.arange(-(K // 2), (K // 2) + 1, device=device)
        dy, dx = torch.meshgrid(r, r, indexing='ij')
        kernel_vecs = torch.stack([dx, -dy, torch.zeros_like(dx)], dim=-1).float().view(-1, 3)
        dist = kernel_vecs.norm(dim=-1, keepdim=True)
        sh = o3.spherical_harmonics(self.edge_irreps, kernel_vecs, normalize=True)
        
        # Interaction
        weights = self.radial_net(dist)
        x_unfold = F.unfold(x, kernel_size=K, padding=K//2).view(B, C, K*K, H*W).permute(0, 2, 3, 1)
        
        # Batched Tensor Product: Combining L=4/6 with L=1 edges
        sh_exp = sh.view(1, K*K, 1, 3).expand(B, -1, H*W, -1)
        w_exp = weights.view(1, K*K, 1, -1).expand(B, -1, H*W, -1)
        
        out = self.tp(x_unfold.reshape(-1, C), sh_exp.reshape(-1, 3), w_exp.reshape(-1, self.tp.weight_numel))
        return out.view(B, K*K, H*W, -1).sum(dim=1).permute(0, 2, 1).view(B, -1, H, W) / (K*K)


# =============================================================================
# 4. EQUIVARIANT CONTRACTION TAIL
# =============================================================================
class EquivariantTail(nn.Module):
    """
    Physically rigorous decoder.
    Uses a Tensor Product to contract high-order cubic features (L=4, 6)
    down to the Quaternion space (L=0, 1).
    """
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps("1x0e + 1x1o") # w, (i, j, k)

        # Self-interaction: In group theory, (L x L) contains lower L components.
        # This extracts the orientation from the cubic shape manifold.
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_out
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Permute for e3nn: (B, H, W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Perform the contraction
        out = self.tp(x_flat, x_flat)
        
        # Reshape back to (B, 4, H, W)
        return out.view(B, H, W, 4).permute(0, 3, 1, 2)


# =============================================================================
# 5. PARALLELIZED IRREP REYNOLDS WRAPPER
# =============================================================================
class IrrepReynoldsWrap(nn.Module):
    """
    Symmetry-Aware Wrapper.
    Uses Wigner-D matrices to ensure the hidden features are invariant
    under the 48 FCC point group rotations (Oh).
    """
    def __init__(self, fn, irreps, group_rotations):
        super().__init__()
        self.fn = fn
        self.irreps = o3.Irreps(irreps)
        
        # Precompute the Wigner-D matrices for the specific L-levels (0, 4, 6)
        # These act as the symmetry operators for our high-order features.
        Ds = torch.stack([self.irreps.D_from_matrix(R) for R in group_rotations])
        self.register_buffer("group_Ds", Ds.float())

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.group_Ds.shape[0] # Usually 48 for FCC
        
        # 1. Symmetry Lifting: Expand input into 48 equivalent orientations
        # Uses einsum for high-performance tensor contraction
        # (B, C, H, W) -> (B, G, C, H, W)
        x_lifted = torch.einsum("gij,bjhw->bgihw", self.group_Ds, x)
        x_lifted = x_lifted.reshape(B * G, C, H, W)
        
        # 2. Vectorized Processing: Pass all 48 versions through the MP layer
        out = self.fn(x_lifted)
        
        # 3. Reynolds Projection: Average the results to ensure FCC invariance
        out = out.view(B, G, -1, H, W)
        return out.mean(dim=1)


# =============================================================================
# 6. FINAL INTEGRATED MODEL (Physics-Rigorous SR)
# =============================================================================
class Reynolds_TFN_QSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        group_rots = torch.tensor(np.load(cfg.sym_np_path)).float()
        self.irreps = "16x0e + 4x4e + 2x6e"
        
        # Pipeline
        self.encoder = RigorousCubicEncoder()
        self.process = IrrepReynoldsWrap( # (Wrapped as defined previously)
            ParallelCubicMessagePassing(self.irreps, self.irreps), 
            self.irreps, group_rots
        )
        self.tail = EquivariantTail(self.irreps)

    def forward(self, x):
        # normalize to S3 hypersphere
        x = F.normalize(x, p=2, dim=1) 
        x = self.encoder(x)
        x = self.process(x)
        x = self.tail(x)
        return F.normalize(x, p=2, dim=1)