import torch
from e3nn import o3
import math

# ==============================================================================
# 1. PHYSICS LAB SETUP
# ==============================================================================
class FCCPhysicsLab:
    def __init__(self, device='cpu'):
        self.device = device
        
        # A. Define the FCC "Atoms" (The Coefficients)
        self.s4 = torch.zeros(9, device=device); self.s4[4] = 0.7638; self.s4[8] = 0.6455
        self.s6 = torch.zeros(13, device=device); self.s6[6] = 0.3536; self.s6[10] = -0.9354
        
        # B. Define the Interaction Hardware (Tensor Products)
        # We need to calculate 4x4 and 6x6. 
        # (4x6 doesn't produce L=0 or L=1, so we can ignore cross-terms for this experiment)
        self.tp4 = o3.FullTensorProduct("1x4e", "1x4e")
        self.tp6 = o3.FullTensorProduct("1x6e", "1x6e")
        
        # Symmetry group (for verification only)
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
# 2. THE GENERATOR (Unified Feature Creation)
# ==============================================================================
def make_unified_fcc(q, lab):
    """
    Rotates the unified seed (4+6) by quaternion q.
    """
    R = o3.quaternion_to_matrix(q)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    
    # Rotate components
    D4 = o3.wigner_D(4, alpha, beta, gamma)
    D6 = o3.wigner_D(6, alpha, beta, gamma)
    
    f4 = torch.einsum("bij,j->bi", D4, lab.s4)
    f6 = torch.einsum("bij,j->bi", D6, lab.s6)
    
    # Return as single logical block
    return torch.cat([f4, f6], dim=-1)

# ==============================================================================
# 3. THE EXPERIMENT (Deterministic Decoder)
# ==============================================================================
def run_interaction_experiment(feature_in, lab, calibration_table):
    """
    1. Interact Feature with Reference.
    2. Measure Overlap (Scalar) and Torque (Vector).
    3. Look up Angle from Overlap.
    4. Return Reconstructed Quaternion.
    """
    batch_size = feature_in.shape[0]
    
    # A. Split the Unified Feature back into interacting parts
    # (The math is the same as if we fed it into a single TP that zeros out cross-terms)
    f4_in = feature_in[:, :9]
    f6_in = feature_in[:, 9:]
    
    # B. The Reference (Static Upright FCC)
    ref4 = lab.s4.view(1, 9).expand(batch_size, -1)
    ref6 = lab.s6.view(1, 13).expand(batch_size, -1)
    
    # C. Perform Interaction
    out4 = lab.tp4(f4_in, ref4)
    out6 = lab.tp6(f6_in, ref6)
    
    # D. UNIFIED MEASUREMENT
    # We sum the signals. This is "mixing" them with equal weight.
    # L=0 is index 0. L=1 is indices 1,2,3.
    
    total_overlap_score = out4[:, 0] + out6[:, 0]
    total_torque_vector = out4[:, 1:4] + out6[:, 1:4]
    
    # E. DECODING
    
    # 1. Axis Recovery
    # The torque vector points along the rotation axis.
    axis_norm = torch.norm(total_torque_vector, dim=1, keepdim=True)
    axis_pred = total_torque_vector / (axis_norm + 1e-8)
    
    # 2. Angle Recovery (Using Calibration)
    # Find nearest score in table
    # score_in: (B), table_x: (N)
    diff = torch.abs(total_overlap_score.unsqueeze(1) - calibration_table['score'].unsqueeze(0))
    idx = torch.argmin(diff, dim=1)
    
    w_pred = calibration_table['w'][idx].unsqueeze(1)
    
    # 3. Reconstruct
    # q = [w, x*sin, y*sin, z*sin]
    sin_half = torch.sqrt(torch.clamp(1.0 - w_pred**2, min=0))
    xyz_pred = axis_pred * sin_half
    
    q_rec = torch.cat([w_pred, xyz_pred], dim=-1)
    return q_rec

# ==============================================================================
# 4. CALIBRATION (The "Pre-Lab" Setup)
# ==============================================================================
def calibrate_unified_curve(lab, steps=2000):
    print("Calibrating Unified Resonance Curve...")
    
    # Sweep angles from 0 to Pi (0 to 180 degrees)
    angles = torch.linspace(0, torch.pi, steps)
    
    # Create pure rotations (axis doesn't matter for scalar score, pick Z)
    half_angles = angles / 2.0
    w = torch.cos(half_angles)
    z = torch.sin(half_angles)
    
    q_calib = torch.zeros(steps, 4)
    q_calib[:, 0] = w
    q_calib[:, 3] = z
    
    # Generate Features
    feats = make_unified_fcc(q_calib, lab)
    
    # Interact
    f4 = feats[:, :9]; f6 = feats[:, 9:]
    ref4 = lab.s4.view(1, 9).expand(steps, -1)
    ref6 = lab.s6.view(1, 13).expand(steps, -1)
    
    o4 = lab.tp4(f4, ref4)
    o6 = lab.tp6(f6, ref6)
    
    # Sum scores
    scores = o4[:, 0] + o6[:, 0]
    
    return {'score': scores, 'w': w}

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    lab = FCCPhysicsLab()
    
    # 1. Calibrate
    table = calibrate_unified_curve(lab)
    
    print("\n" + "="*60)
    print("RUNNING UNIFIED PHYSICS EXPERIMENT (NO ML)")
    print("="*60)
    
    # 2. Generate Random Test Quaternion
    # (Using a random axis and angle)
    axis = torch.randn(3); axis = axis / axis.norm()
    angle = torch.tensor(45.0 * math.pi / 180.0) # 45 degrees
    
    q_test = torch.cat([torch.cos(angle/2).unsqueeze(0), axis * torch.sin(angle/2)]).unsqueeze(0)
    
    print(f"Input Quaternion (Angle ~45 deg):\n{q_test[0].numpy()}")
    
    # 3. Create Feature
    feat = make_unified_fcc(q_test, lab)
    
    # 4. Run Experiment
    q_rec = run_interaction_experiment(feat, lab, table)
    
    print(f"Reconstructed Quaternion:\n{q_rec[0].numpy()}")
    
    # 5. Verify Symmetry
    # Check if q_rec is equivalent to q_test under FCC symmetry
    def check_symmetry_with_details(q_a, q_b, syms):
        # q_a: (1, 4), q_b: (1, 4)
        # Multiply q_b by all syms
        # q_b (1,4) * syms (24,4) -> (24, 4) targets
        
        # (Standard quaternion multiplication logic)
        w1, x1, y1, z1 = q_b[0,0], q_b[0,1], q_b[0,2], q_b[0,3]
        w2, x2, y2, z2 = syms[:,0], syms[:,1], syms[:,2], syms[:,3]
        
        targets = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
        
        # Compute loss on s,x,y,z components (L2 loss on each component)
        # targets: (24, 4), q_a: (1, 4)
        losses = torch.sum((targets - q_a)**2, dim=-1)  # (24,)
        
        # Find the best match
        min_loss_idx = torch.argmin(losses)
        min_loss = losses[min_loss_idx].item()
        best_symmetry = syms[min_loss_idx]
        best_target = targets[min_loss_idx]
        
        # Also compute dot product for comparison
        dots = torch.sum(targets * q_a, dim=-1).abs()
        max_dot = torch.max(dots).item()
        
        return max_dot, min_loss, min_loss_idx.item(), best_symmetry, best_target

    overlap, min_loss, best_idx, best_sym, best_target = check_symmetry_with_details(q_rec, q_test, lab.fcc_syms)
    
    print(f"\nSymmetry Overlap (1.0 = Perfect): {overlap:.5f}")
    print(f"Minimum Loss (s,x,y,z): {min_loss:.6f}")
    print(f"Best Matching Symmetry Index: {best_idx}")
    print(f"Best Symmetry Quaternion:\n{best_sym.numpy()}")
    print(f"Resulting Quaternion after symmetry:\n{best_target.numpy()}")
    
    if overlap > 0.99:
        print(">> SUCCESS: The physics alone preserves the orientation.")
    else:
        print(">> FAIL: Information lost.")