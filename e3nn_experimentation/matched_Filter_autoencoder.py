import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from datetime import datetime
from e3nn import o3

# ==============================================================================
# KEEP PREVIOUS UTILITIES (matrix_to_quaternion_safe, FCCPhysics, FCCEncoder)
# ==============================================================================
# (Copying relevant parts for self-contained execution)

def matrix_to_quaternion_safe(R):
    """
    Convert rotation matrix to quaternion using Shepperd's method for numerical stability.
    Handles near-singular cases robustly without strict determinant checks.
    
    Args:
        R: Rotation matrix of shape (..., 3, 3)
    
    Returns:
        Quaternion of shape (..., 4) with convention [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    batch_size = R_flat.shape[0]
    
    # Shepperd's method: choose the largest diagonal element to avoid division by small numbers
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # Case 0: trace is largest (w is largest)
    q0_w = torch.sqrt(torch.clamp(1 + trace, min=1e-10)) / 2
    q0_x = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / (4 * q0_w + 1e-10)
    q0_y = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / (4 * q0_w + 1e-10)
    q0_z = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / (4 * q0_w + 1e-10)
    q0 = torch.stack([q0_w, q0_x, q0_y, q0_z], dim=-1)
    
    # Case 1: R[0,0] is largest (x is largest)
    q1_x = torch.sqrt(torch.clamp(1 + R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2], min=1e-10)) / 2
    q1_w = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / (4 * q1_x + 1e-10)
    q1_y = (R_flat[:, 0, 1] + R_flat[:, 1, 0]) / (4 * q1_x + 1e-10)
    q1_z = (R_flat[:, 0, 2] + R_flat[:, 2, 0]) / (4 * q1_x + 1e-10)
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)
    
    # Case 2: R[1,1] is largest (y is largest)
    q2_y = torch.sqrt(torch.clamp(1 - R_flat[:, 0, 0] + R_flat[:, 1, 1] - R_flat[:, 2, 2], min=1e-10)) / 2
    q2_w = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / (4 * q2_y + 1e-10)
    q2_x = (R_flat[:, 0, 1] + R_flat[:, 1, 0]) / (4 * q2_y + 1e-10)
    q2_z = (R_flat[:, 1, 2] + R_flat[:, 2, 1]) / (4 * q2_y + 1e-10)
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)
    
    # Case 3: R[2,2] is largest (z is largest)
    q3_z = torch.sqrt(torch.clamp(1 - R_flat[:, 0, 0] - R_flat[:, 1, 1] + R_flat[:, 2, 2], min=1e-10)) / 2
    q3_w = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / (4 * q3_z + 1e-10)
    q3_x = (R_flat[:, 0, 2] + R_flat[:, 2, 0]) / (4 * q3_z + 1e-10)
    q3_y = (R_flat[:, 1, 2] + R_flat[:, 2, 1]) / (4 * q3_z + 1e-10)
    q3 = torch.stack([q3_w, q3_x, q3_y, q3_z], dim=-1)
    
    # Choose the case with the largest diagonal element for numerical stability
    case0_mask = (trace > R_flat[:, 0, 0]) & (trace > R_flat[:, 1, 1]) & (trace > R_flat[:, 2, 2])
    case1_mask = (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2]) & (~case0_mask)
    case2_mask = (R_flat[:, 1, 1] > R_flat[:, 2, 2]) & (~case0_mask) & (~case1_mask)
    # case3_mask is everything else
    
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    q[case0_mask] = q0[case0_mask]
    q[case1_mask] = q1[case1_mask]
    q[case2_mask] = q2[case2_mask]
    case3_mask = ~(case0_mask | case1_mask | case2_mask)
    q[case3_mask] = q3[case3_mask]
    
    # Normalize the quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-10)
    
    return q.reshape(*batch_shape, 4)

class FCCPhysics(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.s4 = torch.zeros(9, device=device); self.s4[4] = 0.7638; self.s4[8] = 0.6455
        self.s6 = torch.zeros(13, device=device); self.s6[6] = 0.3536; self.s6[10] = -0.9354
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

class FCCEncoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics
    def forward(self, quats):
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        D4 = o3.wigner_D(4, alpha, beta, gamma)
        D6 = o3.wigner_D(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)
        return torch.cat([f4, f6], dim=-1)

# ==============================================================================
#  CORRECTED DECODER: Non-Linear Magnitude Correction
# ==============================================================================
class CorrectedMatchedFilterDecoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        
        # Buffers for Reference Seeds (Both L=4 and L=6)
        self.register_buffer('ref_4', physics.s4.clone().detach())
        self.register_buffer('ref_6', physics.s6.clone().detach())

        # 1. Interaction Layers (Tensor Products)
        # We process L=4 and L=6 separately to preserve distinct geometric info
        self.tp4 = o3.FullyConnectedTensorProduct("1x4e", "1x4e", "1x0e + 1x1e")
        self.tp6 = o3.FullyConnectedTensorProduct("1x6e", "1x6e", "1x0e + 1x1e")
        
        # 2. Radial Correction Network (The "Non-Linear" Fix)
        # Input: 
        #   - Scalar from TP4 (Overlap L4)
        #   - Scalar from TP6 (Overlap L6)
        #   - Norm of Vector TP4 (Torque Strength L4)
        #   - Norm of Vector TP6 (Torque Strength L6)
        # Output:
        #   - Scalar w (Quaternion real part)
        #   - Correction factor alpha (for Quaternion vector part)
        self.radial_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 2) # [q_w, vector_scale]
        )

    def forward(self, features):
        # A. Split Features
        f4_in = features[:, :9]
        f6_in = features[:, 9:]
        batch_size = f4_in.shape[0]
        
        # B. Matched Filter Interactions
        # Expand references
        ref4 = self.ref_4.unsqueeze(0).expand(batch_size, -1)
        ref6 = self.ref_6.unsqueeze(0).expand(batch_size, -1)
        
        out4 = self.tp4(f4_in, ref4) # [scalar(1), vector(3)]
        out6 = self.tp6(f6_in, ref6) # [scalar(1), vector(3)]
        
        # C. Extract Components
        s4, v4 = out4[:, 0:1], out4[:, 1:4]
        s6, v6 = out6[:, 0:1], out6[:, 1:4]
        
        v4_norm = v4.norm(dim=-1, keepdim=True)
        v6_norm = v6.norm(dim=-1, keepdim=True)
        
        # D. Predict Magnitude Correction
        # The scalars tell us "where" we are on the curve.
        # Clamp inputs to prevent extreme values
        mlp_in = torch.cat([s4, s6, v4_norm, v6_norm], dim=-1)
        mlp_in = torch.clamp(mlp_in, -10.0, 10.0)
        corrections = self.radial_mlp(mlp_in)
        
        # Use tanh to bound outputs for stability
        corrections = torch.tanh(corrections)
        
        pred_w = corrections[:, 0:1]      # Predicted Quaternion W
        pred_scale = corrections[:, 1:2]  # Scaling factor for the axis
        
        # E. Combine Vectors for Robust Axis
        # We simply sum the vectors (the TP weights can learn to balance them)
        # We could also use a learned mix, but simple sum + scaling is robust.
        v_combined = v4 + v6 
        
        # F. Reconstruct Quaternion
        # q_vec = (Estimated Axis) * (Estimated Magnitude)
        # Clamp scale to prevent extreme values
        pred_scale = torch.clamp(pred_scale, -3.0, 3.0)
        q_vec = v_combined * pred_scale
        
        q_pred = torch.cat([pred_w, q_vec], dim=-1)
        
        # Normalize (clamping prevents NaN/Inf while maintaining gradients)
        q_norm = torch.norm(q_pred, dim=-1, keepdim=True)
        q_norm = torch.clamp(q_norm, min=1e-6)  # Prevent division by zero
        q_pred = q_pred / q_norm
        
        # Convert to rotation matrix
        R = o3.quaternion_to_matrix(q_pred)
        
        # Differentiable Gram-Schmidt orthogonalization to ensure proper rotation matrix
        # This maintains gradients while ensuring det(R) = 1
        def gram_schmidt_orthogonalize(R):
            """Orthogonalize rotation matrix using Gram-Schmidt"""
            # Extract columns
            v1 = R[:, :, 0]
            v2 = R[:, :, 1]
            v3 = R[:, :, 2]
            
            # Orthogonalize
            u1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
            u2 = v2 - torch.sum(u1 * v2, dim=-1, keepdim=True) * u1
            u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-8)
            u3 = torch.cross(u1, u2, dim=-1)
            u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + 1e-8)  # Normalize u3 for better numerical stability
            
            R_ortho = torch.stack([u1, u2, u3], dim=-1)
            
            # Ensure det(R) = 1 (not -1)
            det = torch.det(R_ortho)
            # If determinant is negative, flip the third column
            u3_corrected = u3 * det.unsqueeze(-1).sign()
            R_ortho = torch.stack([u1, u2, u3_corrected], dim=-1)
            
            return R_ortho
        
        R = gram_schmidt_orthogonalize(R)
        
        # We don't actually need angles for the loss computation (we use quaternions)
        # So just return dummy values to avoid numerical precision issues with matrix_to_angles
        # The important return value is R which we use for converting back to quaternions
        alpha = torch.zeros(R.shape[0], device=R.device)
        beta = torch.zeros(R.shape[0], device=R.device)
        gamma = torch.zeros(R.shape[0], device=R.device)
        
        return alpha, beta, gamma, R

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_sym_max_dot_helper(q_pred, q_gt, fcc_syms):
    """Helper for symmetry check - computes best match across FCC symmetries"""
    q_ex = q_pred.unsqueeze(1).expand(-1, 24, -1)
    s_ex = fcc_syms.unsqueeze(0).expand(q_pred.shape[0], -1, -1)
    # quaternion multiply q_pred * symmetries
    w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
    w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
    q_syms = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)
    
    # Check exact match of all 4 components (s,x,y,z) instead of dot product
    # Consider both q and -q (quaternion double cover)
    q_gt_expanded = q_gt.unsqueeze(1)  # [batch, 1, 4]
    diff_pos = torch.abs(q_syms - q_gt_expanded)  # [batch, 24, 4]
    diff_neg = torch.abs(q_syms + q_gt_expanded)  # [batch, 24, 4]
    
    # Sum absolute differences across all 4 components
    error_pos = torch.sum(diff_pos, dim=-1)  # [batch, 24]
    error_neg = torch.sum(diff_neg, dim=-1)  # [batch, 24]
    error = torch.min(error_pos, error_neg)  # [batch, 24]
    
    # Get minimum error across all symmetries (best match)
    min_error, _ = torch.min(error, dim=1)  # [batch]
    # Convert to similarity metric (1 = perfect match, 0 = maximum difference)
    max_similarity = 1.0 - min_error / 4.0  # Normalize by max possible difference (4.0)
    return max_similarity

# ==============================================================================
# TRAINING
# ==============================================================================
def train():
    """Train the matched filter decoder and save checkpoints"""
    # Create timestamped folder for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/data/home/umang/Materials/e3nn_Reynolds/e3nn_experimentation/matched_filter_pt_files/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved to: {save_dir}\n")
    
    print("="*70)
    print("TRAINING: CORRECTED MATCHED FILTER DECODER")
    print("Goal: Fix linearity flaw using Radial MLP Correction")
    print("="*70)
    
    device = "cpu"
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    decoder = CorrectedMatchedFilterDecoder(physics).to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=0.005)
    
    # helper for symmetry check
    def get_sym_max_dot(q_pred, q_gt):
        return get_sym_max_dot_helper(q_pred, q_gt, physics.fcc_syms)

    print("\nTraining...")
    best_score = 0.0
    save_count = 0
    
    for i in range(10001):
        q_batch = torch.randn(64, 4)
        q_batch = q_batch / torch.norm(q_batch, dim=1, keepdim=True)
        
        latents = encoder(q_batch)
        _, _, _, R_pred = decoder(latents)
        
        # Use simple trace loss for rotation matrices to avoid Quaternion double-cover headaches during training
        # Trace(R_pred^T * R_target) is invariant. 
        # To handle symmetry, we want to maximize the trace with the CLOSEST symmetry.
        # But for simplicity, let's stick to the quaternion dot product which worked okay.
        
        # Need robust quaternion conversion for loss
        # using the one from previous context or e3nn's if stable
        q_pred = matrix_to_quaternion_safe(R_pred) 
        
        # Check for NaN/Inf in predictions and skip this batch if found
        if not torch.isfinite(q_pred).all():
            print(f"Warning: NaN/Inf detected in predictions at step {i}, skipping batch")
            continue
        
        similarity_scores = get_sym_max_dot(q_pred, q_batch)  # [batch_size]
        errors = 1.0 - similarity_scores  # [batch_size]
        
        # Use top-10 worst errors (more stable than single worst)
        # This focuses on worst cases but is less sensitive to outliers
        top_k = min(10, errors.shape[0])
        worst_errors, _ = torch.topk(errors, top_k)
        loss = worst_errors.mean()  # Mean of top-10 worst errors
        avg_loss = errors.mean()  # For logging only
        
        # Sanity check for loss
        if not torch.isfinite(loss):
            print(f"Warning: NaN/Inf loss at step {i}, skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        if i % 200 == 0:
            print(f"Step {i}: Top-10 Worst Loss = {loss.item():.6f}, Avg Loss = {avg_loss.item():.6f}")
        
        # Validation and model saving every 500 steps
        if i > 0 and i % 500 == 0:
            with torch.no_grad():
                # Validate on 10 diverse random quaternions
                val_scores = []
                for _ in range(10):
                    q_val = torch.randn(1, 4)
                    q_val = q_val / torch.norm(q_val, dim=1, keepdim=True)
                    latents_val = encoder(q_val)
                    _, _, _, R_val = decoder(latents_val)
                    q_val_pred = matrix_to_quaternion_safe(R_val)
                    val_scores.append(get_sym_max_dot(q_val_pred, q_val).item())
                
                val_score = sum(val_scores) / len(val_scores)  # Average score over 10 samples
                
                # Save if improved (even if not perfect yet)
                if val_score > best_score:
                    best_score = val_score
                    save_count += 1
                    
                    # Save model
                    save_path = os.path.join(save_dir, f"matched_filter_decoder_step{i}_score{val_score:.4f}.pt")
                    torch.save({
                        'step': i,
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_score': val_score,
                        'best_score': best_score
                    }, save_path)
                    
                    print(f"\n{'='*60}")
                    print(f"CHECKPOINT SAVED: {save_path}")
                    print(f"Validation Score: {val_score:.6f} (avg of 10 samples)")
                    print(f"{'='*60}")
                    
                    # Validation test on single quaternion
                    q_test_val = torch.randn(1, 4)
                    q_test_val = q_test_val / q_test_val.norm()
                    latents_test = encoder(q_test_val)
                    _, _, _, R_test_val = decoder(latents_test)
                    q_test_pred = matrix_to_quaternion_safe(R_test_val)
                    test_score = get_sym_max_dot(q_test_pred, q_test_val).item()
                    
                    # Find closest symmetry equivalent
                    q_ex = q_test_pred.expand(24, -1)
                    s_ex = physics.fcc_syms
                    w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
                    w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
                    q_syms = torch.stack([
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2
                    ], dim=-1)
                    
                    diff_pos = torch.sum(torch.abs(q_syms - q_test_val), dim=-1)
                    diff_neg = torch.sum(torch.abs(q_syms + q_test_val), dim=-1)
                    error = torch.min(diff_pos, diff_neg)
                    best_idx = torch.argmin(error)
                    
                    if diff_pos[best_idx] < diff_neg[best_idx]:
                        closest_match = q_syms[best_idx]
                    else:
                        closest_match = -q_syms[best_idx]
                    
                    print(f"Validation Test Sample:")
                    print(f"  Input:          {q_test_val.squeeze().numpy()}")
                    print(f"  Closest match:  {closest_match.numpy()}")
                    print(f"  Test score:     {test_score:.6f}")
                    print(f"  Error:          {error[best_idx].item():.6f}")
                    print(f"{'='*60}\n")
    
    # Save best model after training completes (if we're past 5000 steps)
    if best_score > 0:
        best_path = os.path.join(save_dir, f"matched_filter_decoder_best_score{best_score:.4f}.pt")
        torch.save({
            'step': 10000,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_score': best_score,
            'best_score': best_score
        }, best_path)
        print(f"\n{'='*60}")
        print(f"BEST MODEL SAVED: {best_path}")
        print(f"Best Score: {best_score:.6f}")
        print(f"{'='*60}\n")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total checkpoints saved: {save_count}")
    print(f"Best validation score: {best_score:.6f}")
    print(f"Best model path: {best_path}")
    print("="*70)
    
    return best_path

# ==============================================================================
# TESTING
# ==============================================================================
def test(model_path, num_samples=50):
    """Test a trained model on random quaternions
    
    Args:
        model_path: Path to the .pt file containing the trained model
        num_samples: Number of random quaternions to test (default: 50)
    """
    print("\n" + "="*70)
    print("TESTING MATCHED FILTER DECODER")
    print(f"Model: {model_path}")
    print(f"Testing on {num_samples} random quaternions")
    print("="*70)
    
    device = "cpu"
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    decoder = CorrectedMatchedFilterDecoder(physics).to(device)
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    print(f"\nLoaded model from step {checkpoint.get('step', 'unknown')}")
    print(f"Model validation score: {checkpoint.get('val_score', 'unknown'):.6f}\n")
    
    # Test with diverse random quaternions
    print("Running tests...")
    
    all_scores = []
    all_errors = []
    all_test_quats = []
    all_closest_matches = []
    
    with torch.no_grad():
        for test_idx in range(num_samples):
            # Generate diverse quaternions by sampling from different regions
            if test_idx < num_samples // 5:
                # High w component (small rotations)
                q_test = torch.randn(1, 4)
                q_test[0, 0] = torch.randn(1).abs() + 1.0
            elif test_idx < 2 * num_samples // 5:
                # High x component
                q_test = torch.randn(1, 4)
                q_test[0, 1] = torch.randn(1).abs() + 1.0
            elif test_idx < 3 * num_samples // 5:
                # High y component
                q_test = torch.randn(1, 4)
                q_test[0, 2] = torch.randn(1).abs() + 1.0
            elif test_idx < 4 * num_samples // 5:
                # High z component
                q_test = torch.randn(1, 4)
                q_test[0, 3] = torch.randn(1).abs() + 1.0
            else:
                # Completely random (general rotations)
                q_test = torch.randn(1, 4)
            
            q_test = q_test / q_test.norm()
            
            _, _, _, R_rec = decoder(encoder(q_test))
            q_rec = matrix_to_quaternion_safe(R_rec)
            
            score = get_sym_max_dot_helper(q_rec, q_test, physics.fcc_syms).item()
            all_scores.append(score)
            
            # Find closest symmetry equivalent
            q_ex = q_rec.expand(24, -1)
            s_ex = physics.fcc_syms
            w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
            w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
            q_syms = torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)
            
            diff_pos = torch.sum(torch.abs(q_syms - q_test), dim=-1)
            diff_neg = torch.sum(torch.abs(q_syms + q_test), dim=-1)
            error = torch.min(diff_pos, diff_neg)
            best_idx = torch.argmin(error)
            
            if diff_pos[best_idx] < diff_neg[best_idx]:
                closest_match = q_syms[best_idx]
            else:
                closest_match = -q_syms[best_idx]
            
            comp_error = error[best_idx].item()
            all_errors.append(comp_error)
            all_test_quats.append(q_test.clone())
            all_closest_matches.append(closest_match.clone())
    
    # Summary statistics
    import numpy as np
    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    avg_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    worst_idx = np.argmin(all_scores)
    best_idx = np.argmax(all_scores)
    
    print("\n" + "="*70)
    print(f"TEST RESULTS ({num_samples} samples)")
    print("="*70)
    print(f"Average Score: {avg_score:.6f} ± {std_score:.6f}")
    print(f"Score Range:   [{min_score:.6f}, {max_score:.6f}]")
    print(f"Average Error: {avg_error:.6f} ± {std_error:.6f}")
    
    # Highlight best reconstruction
    print(f"\nBest Reconstruction (Sample {best_idx + 1}/{num_samples}):")
    best_quat = all_test_quats[best_idx]
    best_match = all_closest_matches[best_idx]
    print(f"  Input:   [{best_quat[0,0].item():7.4f}, {best_quat[0,1].item():7.4f}, {best_quat[0,2].item():7.4f}, {best_quat[0,3].item():7.4f}]")
    print(f"  Closest: [{best_match[0].item():7.4f}, {best_match[1].item():7.4f}, {best_match[2].item():7.4f}, {best_match[3].item():7.4f}]")
    print(f"  Score:   {all_scores[best_idx]:.6f}")
    print(f"  Error:   {all_errors[best_idx]:.6f}")
    
    # Highlight worst reconstruction
    print(f"\nWorst Reconstruction (Sample {worst_idx + 1}/{num_samples}):")
    worst_quat = all_test_quats[worst_idx]
    worst_match = all_closest_matches[worst_idx]
    print(f"  Input:   [{worst_quat[0,0].item():7.4f}, {worst_quat[0,1].item():7.4f}, {worst_quat[0,2].item():7.4f}, {worst_quat[0,3].item():7.4f}]")
    print(f"  Closest: [{worst_match[0].item():7.4f}, {worst_match[1].item():7.4f}, {worst_match[2].item():7.4f}, {worst_match[3].item():7.4f}]")
    print(f"  Score:   {all_scores[worst_idx]:.6f}")
    print(f"  Error:   {all_errors[worst_idx]:.6f}")
    
    print("\n" + "="*70)
    if avg_score > 0.99 and std_score < 0.01:
        print(">> SUCCESS: Model generalizes well across diverse quaternions!")
    elif avg_score > 0.95:
        print(">> PARTIAL SUCCESS: Model works reasonably well but has some variation.")
    else:
        print(">> FAIL: Model does not generalize well across quaternion space.")
    print("="*70)
    
    return {
        'avg_score': avg_score,
        'std_score': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'avg_error': avg_error,
        'all_scores': all_scores,
        'all_errors': all_errors
    }

# ==============================================================================
# TEST WORST CASES
# ==============================================================================
def test_worst_cases(model_path, num_samples=100, show_worst=10):
    """Test a trained model and show the worst performing quaternions
    
    Args:
        model_path: Path to the .pt file containing the trained model
        num_samples: Number of random quaternions to test (default: 100)
        show_worst: Number of worst cases to display (default: 10)
    """
    print("\n" + "="*70)
    print("TESTING WORST CASES")
    print(f"Model: {model_path}")
    print(f"Testing on {num_samples} random quaternions")
    print("="*70)
    
    device = "cpu"
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    decoder = CorrectedMatchedFilterDecoder(physics).to(device)
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    print(f"\nLoaded model from step {checkpoint.get('step', 'unknown')}")
    print(f"Model validation score: {checkpoint.get('val_score', 'unknown'):.6f}\n")
    
    print("Running tests...")
    
    all_scores = []
    all_errors = []
    all_test_quats = []
    all_closest_matches = []
    
    with torch.no_grad():
        for test_idx in range(num_samples):
            # Generate completely random normalized quaternions
            q_test = torch.randn(1, 4)
            q_test = q_test / q_test.norm()
            
            _, _, _, R_rec = decoder(encoder(q_test))
            q_rec = matrix_to_quaternion_safe(R_rec)
            
            score = get_sym_max_dot_helper(q_rec, q_test, physics.fcc_syms).item()
            all_scores.append(score)
            
            # Find closest symmetry equivalent
            q_ex = q_rec.expand(24, -1)
            s_ex = physics.fcc_syms
            w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
            w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
            q_syms = torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)
            
            diff_pos = torch.sum(torch.abs(q_syms - q_test), dim=-1)
            diff_neg = torch.sum(torch.abs(q_syms + q_test), dim=-1)
            error = torch.min(diff_pos, diff_neg)
            best_idx = torch.argmin(error)
            
            if diff_pos[best_idx] < diff_neg[best_idx]:
                closest_match = q_syms[best_idx]
            else:
                closest_match = -q_syms[best_idx]
            
            comp_error = error[best_idx].item()
            all_errors.append(comp_error)
            all_test_quats.append(q_test.clone())
            all_closest_matches.append(closest_match.clone())
    
    # Sort by error (worst first)
    import numpy as np
    error_indices = np.argsort(all_errors)[::-1]  # Descending order
    
    print("\n" + "="*70)
    print(f"TOP {show_worst} WORST PERFORMING QUATERNIONS")
    print("="*70)
    
    for rank in range(min(show_worst, len(error_indices))):
        idx = error_indices[rank]
        q_in = all_test_quats[idx]
        q_out = all_closest_matches[idx]
        err = all_errors[idx]
        score = all_scores[idx]
        
        print(f"\n#{rank+1} - Error: {err:.6f} | Score: {score:.6f}")
        print(f"  Input  (s,x,y,z): [{q_in[0,0].item():8.5f}, {q_in[0,1].item():8.5f}, {q_in[0,2].item():8.5f}, {q_in[0,3].item():8.5f}]")
        print(f"  Output (s,x,y,z): [{q_out[0].item():8.5f}, {q_out[1].item():8.5f}, {q_out[2].item():8.5f}, {q_out[3].item():8.5f}]")
        print(f"  Component errors: s={abs(q_in[0,0].item()-q_out[0].item()):.5f}, "
              f"x={abs(q_in[0,1].item()-q_out[1].item()):.5f}, "
              f"y={abs(q_in[0,2].item()-q_out[2].item()):.5f}, "
              f"z={abs(q_in[0,3].item()-q_out[3].item()):.5f}")
    
    # Overall statistics
    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    avg_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    print("\n" + "="*70)
    print(f"OVERALL STATISTICS ({num_samples} samples)")
    print("="*70)
    print(f"Average Score: {avg_score:.6f} ± {std_score:.6f}")
    print(f"Score Range:   [{min_score:.6f}, {max_score:.6f}]")
    print(f"Average Error: {avg_error:.6f} ± {std_error:.6f}")
    print(f"Worst Error:   {all_errors[error_indices[0]]:.6f}")
    print(f"Best Error:    {all_errors[error_indices[-1]]:.6f}")
    print("="*70)
    
    return {
        'all_scores': all_scores,
        'all_errors': all_errors,
        'all_test_quats': all_test_quats,
        'all_closest_matches': all_closest_matches,
        'worst_indices': error_indices[:show_worst]
    }

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: test <model_path> [num_samples]
        if len(sys.argv) < 3:
            print("Usage: python matched_Filter_autoencoder.py test <model_path> [num_samples]")
            sys.exit(1)
        model_path = sys.argv[2]
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        test(model_path, num_samples)
    elif len(sys.argv) > 1 and sys.argv[1] == "worst":
        # Worst cases mode: worst <model_path> [num_samples] [show_worst]
        if len(sys.argv) < 3:
            print("Usage: python matched_Filter_autoencoder.py worst <model_path> [num_samples] [show_worst]")
            sys.exit(1)
        model_path = sys.argv[2]
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        show_worst = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        test_worst_cases(model_path, num_samples, show_worst)
    else:
        # Train mode (default)
        best_model_path = train()
        print(f"\nTraining complete! Best model saved at: {best_model_path}")
        print(f"\nTo test the model, run:")
        print(f"python matched_Filter_autoencoder.py test {best_model_path} [num_samples]")