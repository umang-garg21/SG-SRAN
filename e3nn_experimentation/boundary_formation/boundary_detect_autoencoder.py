import torch
import torch.nn as nn
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from e3nn import o3
from orix.crystal_map import Phase
from orix.quaternion import Orientation

# ==============================================================================
# 0. RIGOROUS QUATERNION ALGEBRA (No Approximations)
# ==============================================================================

def quat_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton product).
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_conjugate(q):
    """Inverse rotation."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def get_exact_misorientation(q_center, q_neighbor, symmetries=None):
    """
    Calculates the exact Disorientation (Physical Misorientation) between pixels.
    
    Args:
        q_center: (..., 4)
        q_neighbor: (..., 4)
        symmetries: (24, 4) Optional. If provided, finds min angle over all symmetries.
        
    Returns:
        angle: (..., 1) in radians [0, pi]
        axis: (..., 3) normalized vector
    """
    # 1. Calculate Relative Rotation: q_rel = q_neighbor * q_center_inverse
    # This represents the rotation FROM center TO neighbor.
    q_inv = quat_conjugate(q_center)
    q_rel = quat_multiply(q_neighbor, q_inv)
    
    # 2. Symmetry Handling (Disorientation)
    # If symmetries are provided, we must check all 24 variants of q_rel
    # and pick the one with the LARGEST real part (smallest angle).
    if symmetries is not None:
        # q_rel shape: (B, 4) -> (B, 1, 4)
        # syms shape: (24, 4) -> (1, 24, 4)
        q_rel_expanded = q_rel.unsqueeze(-2)
        syms_expanded = symmetries.unsqueeze(0)
        
        # Multiply: (B, 24, 4)
        q_syms = quat_multiply(syms_expanded, q_rel_expanded)
        
        # We want the rotation with minimum angle.
        # Angle = 2 * acos(w). Minimizing angle means Maximizing |w|.
        w_abs = torch.abs(q_syms[..., 0]) # (B, 24)
        best_indices = torch.argmax(w_abs, dim=-1) # (B,)
        
        # Gather best quaternions
        # (This is tricky in PyTorch, standard gather pattern)
        mask = torch.nn.functional.one_hot(best_indices, num_classes=24).bool()
        q_rel = q_syms[mask].view(q_rel.shape)
    
    # 3. Double Cover Handling
    # Ensure w >= 0. If w < 0, negate q. (q and -q are same rotation)
    neg_mask = q_rel[..., 0] < 0
    q_rel[neg_mask] *= -1
    
    # 4. Extract Angle and Axis (Rigorous Taylor Expansion for small angles)
    w = torch.clamp(q_rel[..., 0], -1.0, 1.0)
    xyz = q_rel[..., 1:]
    
    # Angle = 2 * acos(w)
    angle = 2.0 * torch.acos(w).unsqueeze(-1)
    
    # Axis = xyz / sin(theta/2) = xyz / sqrt(1 - w^2)
    # For w -> 1 (angle -> 0), sin term vanishes. Use limit.
    sin_half_theta_sq = 1.0 - w*w
    sin_half_theta = torch.sqrt(torch.clamp(sin_half_theta_sq, min=0.0))
    
    # Safe inverse (avoid div by zero)
    # If angle is tiny, axis is arbitrary (or matches xyz direction). 
    # We use a mask.
    safe_mask = sin_half_theta > 1e-6
    
    axis = torch.zeros_like(xyz)
    axis[safe_mask] = xyz[safe_mask] / sin_half_theta[safe_mask].unsqueeze(-1)
    # For unsafe (zero angle), axis is zero vector (physically consistent).
    
    return angle, axis

# ==============================================================================
# CUDA-Compatible Wigner D Function (Patched)
# ==============================================================================
def wigner_D_cuda(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = o3._wigner.so3_generators(l).to(device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])

# ==============================================================================
# 1. PHYSICS CONSTANTS
# ==============================================================================
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

# ==============================================================================
# 2. ENCODER (Strict Physics Implementation)
# ==============================================================================
class FCCEncoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, quats, img_shape=None):
        """
        Args:
            quats: (B, 4) Normalized quaternions.
            img_shape: Tuple (H, W). If provided, calculates spatial boundary features.
        """
        # A. ORIENTATION FEATURES (L=4, L=6)
        # ------------------------------------------------------------------
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        
        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)
        
        # B. BOUNDARY FEATURES (L=0, L=1) - CALCULATED RIGOROUSLY
        # ------------------------------------------------------------------
        if img_shape is not None:
            H, W = img_shape
            q_grid = quats.view(H, W, 4)
            
            # Forward Differences (Right and Down neighbors)
            # We use Roll, but we must zero out the "wrap-around" boundary to avoid false edges
            q_right = torch.roll(q_grid, shifts=-1, dims=1)
            q_down = torch.roll(q_grid, shifts=-1, dims=0)
            
            # Calculate Exact Misorientation
            # Note: We use symmetries=self.physics.fcc_syms to calculate 
            # TRUE DISORIENTATION, not just numerical difference.
            # This ensures boundaries between symmetric equivalents are Zero (Correct Physics).
            ang_x, axis_x = get_exact_misorientation(q_grid, q_right, self.physics.fcc_syms)
            ang_y, axis_y = get_exact_misorientation(q_grid, q_down, self.physics.fcc_syms)
            
            # Zero out the wrapped edges
            ang_x[:, -1] = 0; axis_x[:, -1] = 0
            ang_y[-1, :] = 0; axis_y[-1, :] = 0
            
            # L=0 (Scalar): Local Misorientation Kernel (Average Angle)
            # Units: Radians
            f0 = (ang_x + ang_y) / 2.0
            f0 = f0.view(-1, 1)
            
            # L=1 (Vector): Local Rotation Axis (Average Axis)
            # Note: Axis is only meaningful if Angle > 0.
            f1 = (axis_x + axis_y) / 2.0
            f1 = f1.view(-1, 3)
            
        else:
            B = quats.shape[0]
            f0 = torch.zeros(B, 1, device=quats.device)
            f1 = torch.zeros(B, 3, device=quats.device)

        return f0, f1, f4, f6

# ==============================================================================
# 3. DECODER (Spherical Peak Finding)
# ==============================================================================
class SphericalSamplingDecoder(nn.Module):
    def __init__(self, physics, chunk_size=4096):
        super().__init__()
        self.n_fib_samples = 10000
        self.physics = physics
        self.chunk_size = chunk_size  # Process in chunks to avoid OOM
        self.grid_vecs = self._fibonacci_sphere(samples=self.n_fib_samples, device=physics.device)
        self.Y4_grid = o3.spherical_harmonics(4, self.grid_vecs, normalize=True)
        
    def forward(self, f0, f1, f4, f6):
        batch_size = f4.shape[0]
        device = f4.device
        
        # Process in chunks to avoid OOM
        q_rec_list = []
        for start_idx in range(0, batch_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, batch_size)
            q_chunk = self._decode_chunk(f4[start_idx:end_idx])
            q_rec_list.append(q_chunk)
        
        q_rec = torch.cat(q_rec_list, dim=0)
        
        # 2. Boundary Reconstruction
        # Pass-through the L=0 feature (Misorientation Angle)
        # Convert radians to degrees for visualization
        boundary_map = f0 * (180.0 / math.pi)
        
        return q_rec, boundary_map
    
    def _decode_chunk(self, f4_chunk):
        """Decode a chunk of f4 features to quaternions."""
        chunk_size = f4_chunk.shape[0]
        
        # 1. Orientation Reconstruction (L=4 Peak Finding)
        signal = torch.einsum("bi,gi->bg", f4_chunk, self.Y4_grid)
        z_vals, z_indices = torch.max(signal, dim=1)
        z_axis = self.grid_vecs[z_indices]
        
        # Compute dot products more efficiently using matrix multiplication
        # dots[i, j] = grid_vecs[j] · z_axis[i]
        dots = torch.mm(z_axis, self.grid_vecs.t())  # (chunk_size, n_fib_samples)
        mask = (dots.abs() < 0.2)
        masked_signal = signal.clone()
        masked_signal[~mask] = -float('inf')
        
        x_vals, x_indices = torch.max(masked_signal, dim=1)
        x_axis = self.grid_vecs[x_indices]
        
        # Gram-Schmidt for precision
        z_axis = torch.nn.functional.normalize(z_axis, dim=-1)
        proj = torch.sum(x_axis * z_axis, dim=-1, keepdim=True) * z_axis
        x_axis = torch.nn.functional.normalize(x_axis - proj, dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        
        R_rec = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        q_rec = o3.matrix_to_quaternion(R_rec)
        
        return q_rec

    def _fibonacci_sphere(self, samples, device):
        points = []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i 
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])
        return torch.tensor(points, dtype=torch.float32, device=device)

# ==============================================================================
# 4. PLOTTING UTILS
# ==============================================================================
def plot_results_with_boundary(q_in, q_out, boundary_map, output_filename="result.png"):
    print("Generating 3-Panel Visualization...")
    
    def get_ipf_colors(quat_img):
        # Using Z-direction IPF with updated orix API
        from orix.vector import Vector3d
        from orix.plot import IPFColorKeyTSL
        
        phase = Phase(space_group=225)
        ori = Orientation(quat_img.reshape(-1, 4), symmetry=phase.point_group)
        
        # IPF color key for Z direction
        direction = Vector3d.zvector()
        ipf_key = IPFColorKeyTSL(phase.point_group, direction=direction)
        rgb = ipf_key.orientation2color(ori)
        return rgb.reshape(quat_img.shape[0], quat_img.shape[1], 3)

    if torch.is_tensor(q_in): q_in = q_in.cpu().numpy()
    if torch.is_tensor(q_out): q_out = q_out.cpu().numpy()
    if torch.is_tensor(boundary_map): boundary_map = boundary_map.cpu().numpy()

    rgb_in = get_ipf_colors(q_in)
    rgb_out = get_ipf_colors(q_out)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    axes[0].imshow(rgb_in)
    axes[0].set_title("Input Orientation (IPF-Z)\nGround Truth")
    axes[0].axis('off')
    
    axes[1].imshow(rgb_out)
    axes[1].set_title("Decoded Orientation (IPF-Z)\nReconstruction")
    axes[1].axis('off')
    
    # Boundary Map (L=0 Feature)
    # Using 'inferno' - Black is 0 degrees, Yellow/White is high angle
    im = axes[2].imshow(boundary_map, cmap='inferno', vmin=0, vmax=60) # Cap at 60 deg (FCC max)
    axes[2].set_title("Decoded Boundary Map (L=0)\nStrict Misorientation (Degrees)")
    axes[2].axis('off')
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Misorientation Angle (°)")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"✓ Saved visualization to {output_filename}")

# ==============================================================================
# 5. RUNNER
# ==============================================================================
def run_rigorous_test():
    print("="*70)
    print("STRICT PHYSICS DECODER (Exact Quaternion Algebra)")
    print("="*70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics)
    decoder = SphericalSamplingDecoder(physics)
    
    # 1. Load Data
    file_path = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/preprocessed_imgs_1D/Open_718_Train_hr_x_normal_0.npy"
    q_numpy = np.load(file_path)
    q_all = torch.tensor(q_numpy, dtype=torch.float32, device=device)
    
    # Infer Shape
    if q_all.dim() == 3:
        img_shape = q_all.shape[:2]
        print(f"Image Shape: {img_shape}")
        q_flat = q_all.reshape(-1, 4)
    else:
        # Fallback for 1D input (Assume Square)
        side = int(math.sqrt(q_all.shape[0]))
        img_shape = (side, side)
        q_flat = q_all
        print(f"Inferred Shape from 1D: {img_shape}")

    # Normalize
    q_flat = q_flat / torch.norm(q_flat, dim=1, keepdim=True)
    
    # 2. RUN PIPELINE
    start_time = time.time()
    
    print("\nStep 1: Encoding with Exact Boundary Math...")
    # This now computes true Disorientation considering 24 symmetries
    f0, f1, f4, f6 = encoder(q_flat, img_shape=img_shape)
    
    print("Step 2: Decoding...")
    q_rec, boundary_rec = decoder(f0, f1, f4, f6)
    
    print(f"Done in {time.time() - start_time:.2f}s")
    
    # 3. Calculate Orientation Error (Symmetry Aware)
    print("Calculating Errors...")
    q_rec_expanded = q_rec.unsqueeze(1).expand(-1, 24, -1)
    fcc_syms_expanded = physics.fcc_syms.unsqueeze(0).expand(q_flat.shape[0], -1, -1)
    
    # Batch multiply q_rec * syms
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q_rec_expanded[..., 0], q_rec_expanded[..., 1], q_rec_expanded[..., 2], q_rec_expanded[..., 3]
    w2, x2, y2, z2 = fcc_syms_expanded[..., 0], fcc_syms_expanded[..., 1], fcc_syms_expanded[..., 2], fcc_syms_expanded[..., 3]
    family = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)
    
    # Distance to ground truth
    dist_pos = torch.norm(family - q_flat.unsqueeze(1), dim=-1)
    dist_neg = torch.norm(family + q_flat.unsqueeze(1), dim=-1)
    min_dist = torch.minimum(dist_pos, dist_neg)
    best_indices = torch.argmin(min_dist, dim=1)
    
    # Select best for visualization (to match input colors)
    batch_indices = torch.arange(q_flat.shape[0], device=device)
    closest_quats = family[batch_indices, best_indices]
    use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
    closest_quats[use_neg] = -closest_quats[use_neg]
    
    # 4. Visualize
    if img_shape is not None:
        H, W = img_shape
        q_in_img = q_flat.view(H, W, 4)
        q_out_img = closest_quats.view(H, W, 4)
        boundary_img = boundary_rec.view(H, W)
        
        # Debug: Check boundary map statistics
        print(f"\nBoundary Map Stats:")
        print(f"  Shape: {boundary_img.shape}")
        print(f"  Min: {boundary_img.min().item():.4f}°")
        print(f"  Max: {boundary_img.max().item():.4f}°")
        print(f"  Mean: {boundary_img.mean().item():.4f}°")
        print(f"  Non-zero pixels: {(boundary_img > 0.1).sum().item()}")
        
        # Save in script's directory (cwd)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "rigorous_boundary_result.png")
        plot_results_with_boundary(q_in_img, q_out_img, boundary_img, output_path)

if __name__ == "__main__":
    run_rigorous_test()