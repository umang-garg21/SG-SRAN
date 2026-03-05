import torch
import torch.nn as nn
import math
import numpy as np
import os
import time
from e3nn import o3
import sys
sys.path.append("/home/warren/projects/Reynolds-QSR/")
sys.path.append("/home/warren/projects/Reynolds-QSR/utils")
# Dataset builder
from training.data_loading import QuaternionDataset
from visualization.visualize_sr_results import render_input_output_side_by_side
from utils.quat_ops import to_spatial_quat
import utils


# ==============================================================================
# CUDA-Compatible Wigner D Function (Patched)
# ==============================================================================
# The e3nn wigner_D doesn't properly handle device placement for the generators.
# This wrapper fixes that by moving the generators to the correct device.

def wigner_D_cuda(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """CUDA-compatible wrapper for e3nn's wigner_D function."""
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    
    # Get generators and move to the correct device
    X = o3._wigner.so3_generators(l)
    X = X.to(device)
    
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])

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
        print(f"Encoding batch of quaternions with shape: {quats.shape}")
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        
        # Generate Features using CUDA-compatible wigner_D
        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)
        
        return f4, f6 # Return separated for the decoder

# ==============================================================================
# 3. DECODER (Spherical Peak Finding)
# ==============================================================================
class SphericalSamplingDecoder(nn.Module):
    def __init__(self, physics, grid_res=100000):
        super().__init__()
        # Reduced to 10k for faster processing
        self.n_fib_samples = grid_res
        self.physics = physics
        
        # A. Precompute a Scanning Grid (Fibonacci Sphere)
        # 1000-2000 points is usually enough for <2 degree accuracy
        self.grid_vecs = self._fibonacci_sphere(samples=self.n_fib_samples, device=physics.device)
        
        # B. Precompute Spherical Harmonics for this grid
        # We only need L=4 because L=4 Peaks ARE the Cubic Axes (Face Centers)
        # Shape: (N_grid, 9)
        self.Y4_grid = o3.spherical_harmonics(4, self.grid_vecs, normalize=True)
        
    def forward(self, f4, f6):
        """
        Input: Invariant Features f4, f6
        Output: Canonical Quaternion
        """
        batch_size = f4.shape[0]
        
        # 1. EVALUATE SHAPE ON SPHERE
        # We calculate the "Amplitude" of the L=4 shape at every grid point.
        # Signal = Dot(f4, Y4)
        # Shape: (Batch, N_grid)
        signal = torch.einsum("bi,gi->bg", f4, self.Y4_grid)
        
        # 2. FIND PRIMARY AXIS (Z)
        # The maximum of the L=4 signal corresponds to the cube faces.
        # We pick the highest peak.
        z_vals, z_indices = torch.max(signal, dim=1)
        z_axis = self.grid_vecs[z_indices] # (Batch, 3)
        
        # 3. FIND SECONDARY AXIS (X)
        # We need a peak that is 90 degrees away from Z.
        # Filter points: Dot(v, z) approx 0
        
        # Compute dot products of all grid points with our found Z-axis
        # (Batch, N_grid)
        dots = torch.einsum("bij,bij->bi", self.grid_vecs.unsqueeze(0).expand(batch_size, -1, -1), z_axis.unsqueeze(1).expand(-1, self.n_fib_samples, -1))
        
        # Mask out points that are not orthogonal (keep points within +/- 10 deg of equator)
        mask = (dots.abs() < 0.2)
        
        # Apply mask to signal (set non-orthogonal points to -infinity)
        masked_signal = signal.clone()
        masked_signal[~mask] = -float('inf')
        
        # Find max on the "Equator"
        x_vals, x_indices = torch.max(masked_signal, dim=1)
        x_axis = self.grid_vecs[x_indices]
        
        # 4. GRAM-SCHMIDT CLEANUP (Precision)
        z_axis = torch.nn.functional.normalize(z_axis, dim=-1)
        
        # Orthogonalize X against Z
        proj = torch.sum(x_axis * z_axis, dim=-1, keepdim=True) * z_axis
        x_axis = torch.nn.functional.normalize(x_axis - proj, dim=-1)
        
        # Y is Cross Product
        y_axis = torch.cross(z_axis, x_axis, dim=-1) # Note: cyclic order Z, X -> Y might be X, Y -> Z. 
        # Let's stick to standard X, Y, Z construction:
        # If we found Z and X, then Y = Z cross X
        
        # Build Matrix: [x, y, z] columns
        R_rec = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        return o3.matrix_to_quaternion(R_rec)

    def _fibonacci_sphere(self, samples, device):
        # Creates evenly distributed points on a sphere
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)
            theta = phi * i 
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])
            
        return torch.tensor(points, dtype=torch.float32, device=device)

# ==============================================================================
# 4. VERIFICATION
# ==============================================================================
def run_physics_decoder_test():
    print("="*70)
    print("PHYSICS-BASED DECODER TEST (Spherical Sampling)")
    print("="*70)
    
    # Now we can use CUDA with our patched wigner_D!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics)
    decoder = SphericalSamplingDecoder(physics)
    
    # Helper function for quaternion multiplication
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
        w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)
    
        # # Example usage
    dataset_out_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4/"

    dataset_dir = os.path.join(dataset_out_root, dataset_name)

    train_ds = QuaternionDataset(
        dataset_root=dataset_dir,
        split="Test",
        preload=True,
        preload_torch=True,  # preload as CPU torch tensors
    )
    print("train_ds[0][1].shape", train_ds[0][1].shape)
    
    q_all= train_ds[0][1]  # Get the first sample (HR quaternions)
    
    # Convert from (C, H, W) to (H, W, C)
    q_all = q_all.permute(1, 2, 0)
    
    q_all =q_all.reshape(-1, 4).to(device)  # Flatten to (N, 4)
    # Normalize all quaternions
    q_all = q_all / torch.norm(q_all, dim=1, keepdim=True)
    num_quats = q_all.shape[0]
    print(f"\nProcessing {num_quats} quaternions on {device}...")
    
    # 2. Batch encode and decode all quaternions
    # Process in batches to avoid memory issues
    # Keep batch size small due to (batch × grid_size) intermediate tensors
    batch_size = 1000 if device.type == 'cuda' else 500
    all_errors = []
    all_misorientation_angles = []
    q_reconstructed_all = []
    
    start_time = time.time()
    
    for batch_start in range(0, num_quats, batch_size):
        batch_end = min(batch_start + batch_size, num_quats)
        q_batch = q_all[batch_start:batch_end]
        
        # Encode
        f4, f6 = encoder(q_batch)
        
        # Decode
        q_rec = decoder(f4, f6)
        
        # 3. Calculate error for the entire batch (vectorized)
        # Generate symmetry family for all quaternions at once
        # Shape: (batch, 24, 4)
        q_rec_expanded = q_rec.unsqueeze(1).expand(-1, 24, -1)  # (batch, 24, 4)
        fcc_syms_expanded = physics.fcc_syms.unsqueeze(0).expand(q_batch.shape[0], -1, -1)  # (batch, 24, 4)
        
        # Batched quaternion multiplication
        w1, x1, y1, z1 = q_rec_expanded[..., 0], q_rec_expanded[..., 1], q_rec_expanded[..., 2], q_rec_expanded[..., 3]
        w2, x2, y2, z2 = fcc_syms_expanded[..., 0], fcc_syms_expanded[..., 1], fcc_syms_expanded[..., 2], fcc_syms_expanded[..., 3]
        family = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)  # (batch, 24, 4)
        
        # Find closest match for all quaternions at once
        q_truth_expanded = q_batch.unsqueeze(1)  # (batch, 1, 4)
        dist_pos = torch.norm(family - q_truth_expanded, dim=-1)  # (batch, 24)
        dist_neg = torch.norm(family + q_truth_expanded, dim=-1)  # (batch, 24)
        min_dist = torch.minimum(dist_pos, dist_neg)  # (batch, 24)
        errors = torch.min(min_dist, dim=1)[0]  # (batch,)
        best_indices = torch.argmin(min_dist, dim=1)  # (batch,)
        
        # Get closest quaternions (correct symmetry variant matching input)
        batch_indices = torch.arange(q_batch.shape[0], device=device)
        closest_quats = family[batch_indices, best_indices]  # (batch, 4)
        use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
        closest_quats[use_neg] = -closest_quats[use_neg]
        
        # IMPORTANT: Save the closest variant, not the raw decoder output
        # This ensures IPF colors match the input
        q_reconstructed_all.append(closest_quats)
        
        # Calculate misorientation angles (vectorized)
        q_conj = torch.stack([-q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]], dim=1)
        error_quats = quat_mul(closest_quats, q_conj)
        w_errors = error_quats[:, 0]
        w_errors_clamped = torch.clamp(torch.abs(w_errors), max=1.0)
        misorientation_angles = 2 * torch.acos(w_errors_clamped) * 180 / math.pi
        
        all_errors.extend(errors.cpu().tolist())
        all_misorientation_angles.extend(misorientation_angles.cpu().tolist())
        
        if (batch_start // batch_size) % 5 == 0 or batch_end == num_quats:
            elapsed = time.time() - start_time
            progress = batch_end / num_quats
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Processed {batch_end}/{num_quats} quaternions ({progress*100:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    # Concatenate all reconstructed quaternions
    q_reconstructed_all = torch.cat(q_reconstructed_all, dim=0)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s ({num_quats/total_time:.0f} quaternions/sec)")
    
    # 4. Render IPF comparison if we have image data
    is_image= True
    if is_image:
        print("\n" + "="*70)
        print("RENDERING IPF COMPARISON")
        print("="*70)
        
        # Import the rendering function
        # import sys
        # sys.path.append('/data/home/umang/Materials/Reynolds-QSR_clean_ipf')  # Ensure we can import from the main project

        img_shape= train_ds[0][1].shape[1:3]  # Assuming (C, H, W) format for the quaternions
        # Reshape back to image format (H, W, 4) and move to CPU
        print(f"Reshaping reconstructed quaternions to image format: {img_shape} with 4 channels")

        q_input_img = q_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        q_output_img = q_reconstructed_all.cpu().reshape(img_shape[0], img_shape[1],4).numpy()
    
        fcc_sym = utils.symmetry_utils.resolve_symmetry(train_ds.symmetry) 

        output_png = "ipf_comparison.png" 
        # Render comparison
        print(f"Rendering IPF comparison to: {output_png}")
        render_input_output_side_by_side(
            q_input_img,
            q_output_img,
            fcc_sym,
            out_png=output_png,
            ref_dir="ALL",
            include_key=True,
            overwrite=True,
            format_input=False,  # Already normalized
            dpi=300
        )
        print(f"✓ Saved comparison image!")
    
    # 5. Report statistics
    all_errors = np.array(all_errors)
    all_misorientation_angles = np.array(all_misorientation_angles)
    
    print("\n" + "="*70)
    print("RECONSTRUCTION ERROR STATISTICS")
    print("="*70)
    print(f"Total quaternions processed: {num_quats}")
    print(f"\nError Distance:")
    print(f"  Maximum: {np.max(all_errors):.6e}")
    print(f"  Mean:    {np.mean(all_errors):.6e}")
    print(f"  Median:  {np.median(all_errors):.6e}")
    print(f"  Std Dev: {np.std(all_errors):.6e}")
    print(f"\nMisorientation Angle:")
    print(f"  Maximum: {np.max(all_misorientation_angles):.4f}°")
    print(f"  Mean:    {np.mean(all_misorientation_angles):.4f}°")
    print(f"  Median:  {np.median(all_misorientation_angles):.4f}°")
    print(f"  Std Dev: {np.std(all_misorientation_angles):.4f}°")
    
    # Find and report worst case
    worst_idx = np.argmax(all_errors)
    print(f"\nWorst Case (index {worst_idx}):")
    print(f"  Original:     {q_all[worst_idx].cpu().numpy()}")
    print(f"  Error:        {all_errors[worst_idx]:.6e}")
    print(f"  Misorientation: {all_misorientation_angles[worst_idx]:.4f}°")
    
    if np.max(all_errors) < 0.05:
        print("\n>> SUCCESS: All quaternions restored within tolerance!")
    else:
        print(f"\n>> WARNING: {np.sum(all_errors >= 0.05)} quaternion(s) exceeded error threshold of 0.05")
    print("   (Note: Error depends on grid size. Increase grid for more precision.)")

if __name__ == "__main__":
    run_physics_decoder_test()