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
# 3. DECODER (Simple MLP - proves encoder features are sufficient)
# ==============================================================================
class MLPDecoder(nn.Module):
    """
    Simple MLP decoder that outputs quaternions directly.
    
    Why not use e3nn TensorProducts?
    - The encoder produces INVARIANT features (22 scalars)
    - TensorProducts expect COVARIANT features with specific transformation properties
    - Declaring invariants as "1x4e + 1x6e" is mathematically incorrect
    - The CG coupling becomes meaningless when inputs don't transform properly
    
    An MLP makes no assumptions about the input structure and can learn
    arbitrary nonlinear mappings from invariant features to quaternions.
    """
    def __init__(self, input_dim=22, hidden_dim=128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)  # Output: 4D quaternion
        )
        
    def forward(self, features):
        """
        Input: Tensor (Batch, 22) - invariant features
        Output: Euler Angles (alpha, beta, gamma) & Rotation Matrix
        """
        # Predict quaternion directly
        q_raw = self.mlp(features)
        
        # Normalize to unit quaternion
        q = torch.nn.functional.normalize(q_raw, dim=-1, eps=1e-8)
        
        # Convert to rotation matrix for consistency
        R = quaternion_to_matrix_safe(q)
        
        # Extract Euler angles
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
        
        return alpha, beta, gamma, R, q


def quaternion_to_matrix_safe(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w,
        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,
        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(-1, 3, 3)
    
    return R

# ==============================================================================
# 4. VERIFICATION PIPELINE
# ==============================================================================
def run_geometric_decoder_test():
    print("="*70)
    print("MLP DECODER TEST")
    print("Goal: Train MLP to extract Quaternions from Invariant Features")
    print("="*70)
    
    device = "cpu"
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    decoder = MLPDecoder(input_dim=22, hidden_dim=256).to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    # Helper: quaternion multiplication for batch
    def quat_mul_batch(q, s):
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
    
    # Helper: find canonical representative (closest to identity [1,0,0,0])
    def to_canonical(q_batch, fcc_syms):
        q_equivs = quat_mul_batch(q_batch, fcc_syms)
        w_components = q_equivs[..., 0].abs()
        best_idx = w_components.argmax(dim=1)
        batch_size = q_batch.shape[0]
        q_canonical = q_equivs[torch.arange(batch_size), best_idx]
        sign = torch.sign(q_canonical[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q_canonical * sign
    
    # --- TRAINING LOOP ---
    print("\nTraining Decoder...")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    for i in range(20000):  # More iterations for MLP
        # 1. Generate Random Rotations
        q_batch = torch.randn(128, 4)  # Larger batch
        q_batch = q_batch / torch.norm(q_batch, dim=1, keepdim=True)
        
        # 2. Encode (Get Invariants)
        latents = encoder(q_batch)
        
        # 3. Decode (Get Quaternion directly)
        _, _, _, R_pred, q_pred = decoder(latents)
        
        # 4. Symmetry-Aware Loss: match ANY of the 24 FCC equivalents
        # Generate all 24 equivalents of the target
        q_targets = quat_mul_batch(q_batch, physics.fcc_syms)  # (B, 24, 4)
        
        # Compute dot product with each equivalent
        # |q1 · q2| = cos(θ/2) where θ is rotation angle between them
        dots = torch.sum(q_pred.unsqueeze(1) * q_targets, dim=-1).abs()  # (B, 24)
        
        # Find the BEST matching equivalent (soft selection)
        max_dots, _ = torch.max(dots, dim=1)  # (B,)
        
        # Loss: minimize angle to nearest equivalent
        # 1 - |dot| ≈ (angle)^2 / 8 for small angles
        loss = (1.0 - max_dots).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 2000 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    # --- 100-TRIAL VERIFICATION ---
    print("\n" + "="*70)
    print("Testing 100 Random Quaternions...")
    print("="*70)
    
    num_tests = 100
    successes = 0
    angle_errors = []
    
    for test_idx in range(num_tests):
        q_test = torch.randn(1, 4)
        q_test = q_test / q_test.norm()
        
        with torch.no_grad():
            latent = encoder(q_test)
            alpha, beta, gamma, R_rec, q_rec = decoder(latent)
            
            # Put in canonical hemisphere
            if q_rec[0, 0] < 0:
                q_rec = -q_rec
        
        # Generate all 24 equivalents of the INPUT quaternion
        q_input_equivs = quat_mul_batch(q_test, physics.fcc_syms)[0]  # (24, 4)
        
        # Find which equivalent the network output matches
        dots = torch.sum(q_rec * q_input_equivs, dim=-1).abs()  # (24,)
        best_match_idx = dots.argmax().item()
        max_dot = dots[best_match_idx].item()
        matched_equiv = q_input_equivs[best_match_idx]
        
        # Compute angle error
        angle_error = 2 * math.acos(min(max_dot, 1.0)) * 180 / math.pi
        angle_errors.append(angle_error)
        
        dist = 1.0 - max_dot
        is_success = dist < 0.01
        if is_success:
            successes += 1
        
        # Print details for first 10 and any failures
        if test_idx < 10 or not is_success:
            status = "✓" if is_success else "✗"
            print(f"\nTest {test_idx+1:3d}: {status}")
            print(f"  Input Quat:    [{q_test[0,0]:.4f}, {q_test[0,1]:.4f}, {q_test[0,2]:.4f}, {q_test[0,3]:.4f}]")
            print(f"  Network Out:   [{q_rec[0,0]:.4f}, {q_rec[0,1]:.4f}, {q_rec[0,2]:.4f}, {q_rec[0,3]:.4f}]")
            print(f"  Matched Equiv: [{matched_equiv[0]:.4f}, {matched_equiv[1]:.4f}, {matched_equiv[2]:.4f}, {matched_equiv[3]:.4f}] (index {best_match_idx})")
            print(f"  Angle Error:   {angle_error:.4f}°")
    
    # Summary statistics
    angle_errors = torch.tensor(angle_errors)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Success Rate:    {successes}/{num_tests} ({100*successes/num_tests:.1f}%)")
    print(f"Angle Error:     mean={angle_errors.mean():.4f}°, std={angle_errors.std():.4f}°, max={angle_errors.max():.4f}°")
    
    if successes == num_tests:
        print("\n>> SUCCESS: MLP Decoder consistently matches one of 24 FCC equivalents!")
    elif successes >= 95:
        print(f"\n>> MOSTLY SUCCESS: {successes}/100 tests passed")
    else:
        print(f"\n>> NEEDS IMPROVEMENT: Only {successes}/100 tests passed")

if __name__ == "__main__":
    run_geometric_decoder_test()