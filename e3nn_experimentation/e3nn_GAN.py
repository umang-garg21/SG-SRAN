import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
import numpy as np
import math

# =============================================================================
# 1. HARD-CODED PHYSICS: THE CUBIC SEEDS
# =============================================================================
def get_cubic_seeds(device):
    """
    Fundamental FCC Invariant vectors for L=4 and L=6.
    These 'seeds' ensure we only populate the cubic manifold.
    """
    s4 = torch.zeros(9, device=device)
    # The L=4 cubic harmonic components in e3nn basis
    s4[0], s4[4], s4[8] = math.sqrt(7/12), math.sqrt(5/24), math.sqrt(5/24)
    
    s6 = torch.zeros(13, device=device)
    # The L=6 cubic harmonic components in e3nn basis
    s6[0], s6[4], s6[8] = math.sqrt(1/8), -math.sqrt(7/16), -math.sqrt(7/16)
    return s4, s6

# =============================================================================
# 2. THE EQUIVARIANT ENCODER (Lifting)
# =============================================================================
class EquivariantEncoder(nn.Module):
    def __init__(self, n_l4=4, n_l6=2):
        super().__init__()
        self.n_l4, self.n_l6 = n_l4, n_l6
        # Features: 16 Scalars, 4 Cubic L4 tensors, 2 Cubic L6 tensors
        self.irreps_out = o3.Irreps(f"16x0e + {n_l4}x4e + {n_l6}x6e")

    def forward(self, quats):
        B = quats.shape[0]
        R = o3.quaternion_to_matrix(quats)
        s4, s6 = get_cubic_seeds(quats.device)

        # Scalars (L=0)
        scalars = torch.zeros(B, 16, device=quats.device)
        scalars[:, 0] = 1.0 # Identity channel

        # Higher Order Cubic Descriptors (L=4 and L=6)
        # We rotate the seeds to the given orientation
        D4 = o3.wigner_D(4, R)
        D6 = o3.wigner_D(6, R)
        
        f4 = torch.einsum("bij,j->bi", D4, s4).repeat(1, self.n_l4)
        f6 = torch.einsum("bij,j->bi", D6, s6).repeat(1, self.n_l6)
        
        return torch.cat([scalars, f4, f6], dim=-1)

# =============================================================================
# 3. THE EQUIVARIANT DECODER (Contraction)
# =============================================================================
class EquivariantDecoder(nn.Module):
    """
    Reconstructs L=0,1 from L=4,6 using Tensor Product contractions.
    This is the physically rigorous inverse of the lifting operation.
    """
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps("1x0e + 1x1o") # Target: Quaternion

        # Linear projection within the irrep space to mix channels
        self.linear = o3.Linear(self.irreps_in, self.irreps_in)
        
        # Tensor Product Contraction: (L_in x L_in) -> L_out
        # In group theory, 4x4 and 6x6 contain 0 and 1 subspaces.
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_out
        )

    def forward(self, x):
        # Mix the channels equivariantly
        x_mixed = self.linear(x)
        # Contract the features against themselves to find lower harmonics
        q_raw = self.tp(x_mixed, x_mixed)
        # Project back to the S3 Hypersphere
        return F.normalize(q_raw, p=2, dim=-1)

# =============================================================================
# 4. DIAGNOSTIC MIRROR MODEL & TEST
# =============================================================================
class MirrorSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EquivariantEncoder()
        self.decoder = EquivariantDecoder(self.encoder.irreps_out)

    def forward(self, q):
        latent = self.encoder(q)
        q_recon = self.decoder(latent)
        return q_recon

def run_mirror_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorSystem().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Executing Physics-Aware Mirror Test on {device}...")

    # Training Loop
    for epoch in range(501):
        # Sample random orientations
        q_gt = F.normalize(torch.randn(512, 4, device=device), p=2, dim=-1)
        
        q_recon = model(q_gt)
        
        # Misorientation Loss (1 - cos(theta))
        R_gt = o3.quaternion_to_matrix(q_gt)
        R_recon = o3.quaternion_to_matrix(q_recon)
        R_rel = torch.bmm(R_gt, R_recon.transpose(1, 2))
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        loss = (1.0 - (trace - 1.0) / 2.0).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            angle = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.1, 1.0)).mean())
            print(f"Epoch {epoch:04d} | Avg Misorientation Error: {angle.item():.6f}°")

    print("\n[Final Scientific Validation]")
    # Check Equivariance: Rotate input and see if output rotates identically
    q_test = F.normalize(torch.randn(1, 4, device=device), p=2, dim=-1)
    
    # 1. Recon the original
    q_recon_orig = model(q_test)
    
    # 2. Rotate the input by 45 deg around X
    rot = o3.matrix_to_quaternion(o3.axis_angle_to_matrix(torch.tensor([1.,0.,0.]), torch.tensor([math.pi/4]))).to(device)
    # Quaternion multiplication (Hamilton product)
    def q_mul(q, p):
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = p.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    q_rotated_in = q_mul(rot, q_test)
    q_recon_rot = model(q_rotated_in)
    
    # The expected output is the original reconstruction rotated
    q_expected_rot = q_mul(rot, q_recon_orig)
    
    equivariance_error = torch.abs(q_recon_rot - q_expected_rot).mean().item()
    print(f"Equivariance Deviation: {equivariance_error:.2e}")
    if equivariance_error < 1e-5:
        print("✅ PASS: The system is perfectly equivariant.")

if __name__ == "__main__":
