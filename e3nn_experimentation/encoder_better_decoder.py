import torch
import torch.nn as nn
import torch.optim as optim
import math
from e3nn import o3
from e3nn.nn import Gate

# ==============================================================================
# CUSTOM ROTATION UTILITIES (fully differentiable, no assertions)
# ==============================================================================
def matrix_to_quaternion_safe(R):
    """
    Convert rotation matrix to quaternion without det=1 assertion.
    Fully differentiable using soft selection (no boolean masks).
    R: (..., 3, 3) -> q: (..., 4) as [w, x, y, z]
    
    This is necessary because:
    1. e3nn's matrix_to_quaternion asserts det(R) = 1 exactly
    2. During training, Gram-Schmidt produces det(R) ≈ 1 but not exactly
    3. We need gradients to flow through this conversion
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Compute all 4 possible quaternion representations
    # Option 1: w is largest
    r1 = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
    q1_w = r1
    q1_x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * r1 + 1e-8)
    q1_y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * r1 + 1e-8)
    q1_z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * r1 + 1e-8)
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)
    
    # Option 2: x is largest
    r2 = torch.sqrt(torch.clamp(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-8)) / 2
    q2_w = (R[:, 2, 1] - R[:, 1, 2]) / (4 * r2 + 1e-8)
    q2_x = r2
    q2_y = (R[:, 0, 1] + R[:, 1, 0]) / (4 * r2 + 1e-8)
    q2_z = (R[:, 0, 2] + R[:, 2, 0]) / (4 * r2 + 1e-8)
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)
    
    # Option 3: y is largest
    r3 = torch.sqrt(torch.clamp(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], min=1e-8)) / 2
    q3_w = (R[:, 0, 2] - R[:, 2, 0]) / (4 * r3 + 1e-8)
    q3_x = (R[:, 0, 1] + R[:, 1, 0]) / (4 * r3 + 1e-8)
    q3_y = r3
    q3_z = (R[:, 1, 2] + R[:, 2, 1]) / (4 * r3 + 1e-8)
    q3 = torch.stack([q3_w, q3_x, q3_y, q3_z], dim=-1)
    
    # Option 4: z is largest
    r4 = torch.sqrt(torch.clamp(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2], min=1e-8)) / 2
    q4_w = (R[:, 1, 0] - R[:, 0, 1]) / (4 * r4 + 1e-8)
    q4_x = (R[:, 0, 2] + R[:, 2, 0]) / (4 * r4 + 1e-8)
    q4_y = (R[:, 1, 2] + R[:, 2, 1]) / (4 * r4 + 1e-8)
    q4_z = r4
    q4 = torch.stack([q4_w, q4_x, q4_y, q4_z], dim=-1)
    
    # Stack all options: (batch, 4 options, 4 quaternion components)
    all_q = torch.stack([q1, q2, q3, q4], dim=1)
    
    # Selection scores (which diagonal element is largest)
    scores = torch.stack([
        1 + trace,
        1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
        1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2],
        1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2],
    ], dim=-1)  # (batch, 4)
    
    # Soft selection (differentiable argmax)
    weights = torch.softmax(scores * 10, dim=-1)
    q = torch.sum(all_q * weights.unsqueeze(-1), dim=1)
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    
    return q.reshape(*batch_shape, 4)

# ==============================================================================
# 1. PHYSICS CONSTANTS
# ==============================================================================
class FCCPhysics(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Seeds
        self.s4 = torch.zeros(9, device=device); self.s4[4] = 0.7638; self.s4[8] = 0.6455
        self.s6 = torch.zeros(13, device=device); self.s6[6] = 0.3536; self.s6[10] = -0.9354
        
        # Symmetry Group (for verification)
        inv_sqrt_2 = 1 / math.sqrt(2); half = 0.5
        self.fcc_syms = torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [inv_sqrt_2, inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, -inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [0, inv_sqrt_2, inv_sqrt_2, 0], [0, inv_sqrt_2, 0, inv_sqrt_2], [0, 0, inv_sqrt_2, inv_sqrt_2],
            [0, inv_sqrt_2, -inv_sqrt_2, 0], [0, 0, inv_sqrt_2, -inv_sqrt_2], [0, inv_sqrt_2, 0, -inv_sqrt_2],
            [half, half, half, half], [half, -half, -half, half], [half, -half, half, -half], [half, half, -half, -half],
            [half, half, half, -half], [half, half, -half, half], [half, -half, half, half], [half, -half, -half, -half],
        ], dtype=torch.float32, device=device)

# ==============================================================================
# 2. ENCODER (Invariant)
# ==============================================================================
class FCCEncoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, quats):
        # Convert Quat -> Rot Matrix -> Euler
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        
        # Generate Features
        D4 = o3.wigner_D(4, alpha, beta, gamma)
        D6 = o3.wigner_D(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)
        
        return torch.cat([f4, f6], dim=-1)

# ==============================================================================
# 3. DECODER (Systematic Geometric Funnel)
# ==============================================================================
class GeometricEulerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. INPUT: Invariant features (L=4 + L=6)
        self.irreps_in = o3.Irreps("1x4e + 1x6e")
        
        # 2. HIDDEN LAYER: The "Mixing Chamber"
        # Include 0e scalars and 1e vectors to enable path to output
        # Gate: scalars_kept + gate_scalars + gated_irreps
        # Need 16 gate scalars for 8+8=16 gated channels (8x1e + 8x2e)
        
        self.gate = Gate(
            "16x0e",                        # Scalars to keep (with activation)
            [torch.nn.functional.silu],     # Activation for kept scalars
            "16x0e",                        # Gate scalars (one per gated channel)
            [torch.sigmoid],                # Gate activation
            "8x1e + 8x2e"                   # Gated: 8+8=16 channels
        )
        # gate.irreps_in = 16x0e + 16x0e + 8x1e + 8x2e (scalars + gates + gated)
        # gate.irreps_out = 16x0e + 8x1e + 8x2e (scalars kept + gated)
        
        # TP1: Expansion (Input -> Hidden + Gates)
        self.tp1 = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_in,  # Self-interaction (Quadratic)
            self.gate.irreps_in
        )
        
        # 3. OUTPUT LAYER: Vector Extraction
        # Distill hidden layer down to 2 Pseudo-Vectors (1e) for X and Z axes
        self.irreps_out = o3.Irreps("2x1e")
        
        # TP2: Self-interaction of gate output
        # gate.irreps_out = 16x0e + 8x1e + 8x2e
        # Paths to 1e: 0e⊗1e→1e, 1e⊗0e→1e, 1e⊗1e→1e, 1e⊗2e→1e, 2e⊗1e→1e
        self.tp2 = o3.FullyConnectedTensorProduct(
            self.gate.irreps_out, 
            self.gate.irreps_out,  # Self-interaction to enable 1e paths
            self.irreps_out
        )

    def forward(self, features):
        """
        Input: Tensor (Batch, 22) -> [f4, f6]
        Output: Euler Angles (alpha, beta, gamma) & Rotation Matrix
        """
        # A. Layer 1: Expansion & Gating
        hidden = self.tp1(features, features)
        hidden = self.gate(hidden)
        
        # B. Layer 2: Contraction to Axes (self-interaction)
        # Output shape: (Batch, 6) -> [Vector1(3), Vector2(3)]
        vectors = self.tp2(hidden, hidden)
        
        v_primary = vectors[:, 0:3]   # Intended Z-axis
        v_secondary = vectors[:, 3:6] # Intended X-axis
        
        # C. Differentiable Gram-Schmidt (6D -> SO(3))
        # Normalize first vector to get z-axis
        z = torch.nn.functional.normalize(v_primary, dim=-1, eps=1e-6)
        
        # Orthogonalize v_secondary against z, then normalize to get x-axis
        dot = (z * v_secondary).sum(dim=-1, keepdim=True)
        x = v_secondary - dot * z
        x = torch.nn.functional.normalize(x, dim=-1, eps=1e-6)
        
        # Cross product gives y-axis (right-handed, guarantees det=+1)
        y = torch.cross(z, x, dim=-1)
        
        # D. Assemble Rotation Matrix (columns are orthonormal basis vectors)
        R = torch.stack([x, y, z], dim=-1)
        
        # E. Convert to Euler Angles (custom implementation without assertions)
        beta = torch.acos(torch.clamp(R[:, 2, 2], -1.0, 1.0))
        safe_mask = (torch.abs(torch.sin(beta)) > 1e-6)
        alpha = torch.where(
            safe_mask,
            torch.atan2(R[:, 1, 2], R[:, 0, 2]),
            torch.zeros_like(beta)
        )
        gamma = torch.where(
            safe_mask,
            torch.atan2(R[:, 2, 1], -R[:, 2, 0]),
            torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        )
        
        return alpha, beta, gamma, R 

# ==============================================================================
# 4. VERIFICATION PIPELINE
# ==============================================================================
def run_geometric_decoder_test():
    print("="*70)
    print("GEOMETRIC EULER DECODER TEST")
    print("Goal: Train systematic network to extract Euler Angles from Irreps")
    print("="*70)
    
    device = "cpu"
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    decoder = GeometricEulerDecoder().to(device)
    
    # We must TRAIN the geometric decoder to find the axes.
    # It starts with random weights, so it doesn't know which direction is "Z" yet.
    optimizer = optim.Adam(decoder.parameters(), lr=0.01)
    
    # --- TRAINING LOOP (To learn the axes mapping) ---
    print("\nTraining Decoder...")
    for i in range(500):
        # 1. Generate Random Rotations
        q_batch = torch.randn(32, 4)
        q_batch = q_batch / torch.norm(q_batch, dim=1, keepdim=True)
        
        # 2. Encode (Get Invariants)
        latents = encoder(q_batch)
        
        # 3. Decode (Get Rotation Matrix)
        _, _, _, R_pred = decoder(latents)
        q_pred = matrix_to_quaternion_safe(R_pred)
        
        # 4. Symmetry-Aware Loss
        # Distance to NEAREST symmetry equivalent
        # Expand q_batch to all 24 equivalents
        # (Batch, 24, 4)
        def quat_mul_batch(q, s):
            # q: (B, 4), s: (24, 4) -> (B, 24, 4)
            q_ex = q.unsqueeze(1).expand(-1, 24, -1)
            s_ex = s.unsqueeze(0).expand(q.shape[0], -1, -1)
            w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
            w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)

        q_targets = quat_mul_batch(q_batch, physics.fcc_syms)
        
        # Compute dot with prediction
        # (Batch, 24)
        dots = torch.sum(q_pred.unsqueeze(1) * q_targets, dim=-1).abs()
        max_dots, _ = torch.max(dots, dim=1)
        
        loss = 1.0 - max_dots.mean() # Minimize distance
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}")

    # --- ONE-SHOT VERIFICATION ---
    print("\nTesting One-Shot Inference...")
    q_test = torch.randn(1, 4)
    q_test = q_test / q_test.norm()
    
    with torch.no_grad():
        latent = encoder(q_test)
        alpha, beta, gamma, R_rec = decoder(latent)
        q_rec = matrix_to_quaternion_safe(R_rec)
        
    print(f"Original: {q_test.numpy()[0]}")
    print(f"Restored: {q_rec.numpy()[0]}")
    print(f"Euler Angles: a={alpha.item():.2f}, b={beta.item():.2f}, g={gamma.item():.2f}")
    
    # Check Error
    q_fam = quat_mul_batch(q_rec, physics.fcc_syms)[0]
    dist = 1.0 - torch.max(torch.sum(q_fam * q_test, dim=-1).abs()).item()
    print(f"Error Distance: {dist:.6e}")
    
    if dist < 0.05:
        print(">> SUCCESS: Geometric Network learned the mapping!")

if __name__ == "__main__":
    run_geometric_decoder_test()