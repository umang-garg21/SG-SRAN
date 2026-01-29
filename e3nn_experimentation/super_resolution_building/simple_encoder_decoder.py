import torch
import torch.nn as nn
import math
import numpy as np
import os
import time
from e3nn import o3
from orix.crystal_map import Phase

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
    def __init__(self, physics, grid_res=50):
        super().__init__()
        # Reduced to 10k for faster processing
        self.n_fib_samples = 10000
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
# 4. UNIFIED SUPER-RESOLUTION MODULE
# ==============================================================================
class EBSDSuper(nn.Module):
    """
    Unified EBSD Super-Resolution Module
    
    This class combines the FCC encoder and decoder into a single module
    for easy inference and training on EBSD quaternion data.
    
    Args:
        device: Device to run computations on ('cpu', 'cuda', 'cuda:0', etc.)
        grid_samples: Number of Fibonacci sphere samples for decoder (default: 10000)
        batch_size: Batch size for processing large datasets (default: 1000)
        
    Usage:
        model = EBSDSuper(device='cuda:0')
        output_quats = model(input_quats)  # Simple forward pass
        
        # Or for images with automatic batching:
        output_img = model.process_image(input_img_path, output_path='result.png')
    """
    
    def __init__(self, device='cpu', grid_samples=10000, batch_size=1000):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Initialize physics constants and modules
        self.physics = FCCPhysics(self.device)
        self.encoder = FCCEncoder(self.physics)
        self.decoder = SphericalSamplingDecoder(self.physics, grid_res=grid_samples)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, quaternions):
        """
        Forward pass: quaternions -> latent features -> reconstructed quaternions
        
        Args:
            quaternions: Input quaternions of shape (N, 4) or (H, W, 4)
            
        Returns:
            Reconstructed quaternions of same shape as input
        """
        # Store original shape
        original_shape = quaternions.shape
        is_image = len(original_shape) == 3
        
        # Flatten if image format
        if is_image:
            quaternions = quaternions.reshape(-1, 4)
        
        # Ensure 2D (N, 4)
        if quaternions.dim() == 1:
            quaternions = quaternions.unsqueeze(0)
        
        # Normalize
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        
        # Encode
        f4, f6 = self.encoder(quaternions)
        
        # Decode
        q_reconstructed = self.decoder(f4, f6)
        
        # Match to closest symmetry variant
        q_reconstructed = self._match_symmetry(quaternions, q_reconstructed)
        
        # Restore original shape
        if is_image:
            q_reconstructed = q_reconstructed.reshape(original_shape)
        
        return q_reconstructed
    
    def _match_symmetry(self, q_truth, q_reconstructed):
        """
        Find the closest symmetry variant of reconstructed quaternions
        to match the input quaternions.
        
        This ensures consistent IPF coloring and minimal error.
        """
        batch_size = q_truth.shape[0]
        
        # Generate symmetry family for all quaternions
        q_rec_expanded = q_reconstructed.unsqueeze(1).expand(-1, 24, -1)  # (batch, 24, 4)
        fcc_syms_expanded = self.physics.fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 24, 4)
        
        # Batched quaternion multiplication
        w1, x1, y1, z1 = q_rec_expanded[..., 0], q_rec_expanded[..., 1], q_rec_expanded[..., 2], q_rec_expanded[..., 3]
        w2, x2, y2, z2 = fcc_syms_expanded[..., 0], fcc_syms_expanded[..., 1], fcc_syms_expanded[..., 2], fcc_syms_expanded[..., 3]
        family = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)  # (batch, 24, 4)
        
        # Find closest match
        q_truth_expanded = q_truth.unsqueeze(1)  # (batch, 1, 4)
        dist_pos = torch.norm(family - q_truth_expanded, dim=-1)  # (batch, 24)
        dist_neg = torch.norm(family + q_truth_expanded, dim=-1)  # (batch, 24)
        min_dist = torch.minimum(dist_pos, dist_neg)  # (batch, 24)
        best_indices = torch.argmin(min_dist, dim=1)  # (batch,)
        
        # Get closest quaternions
        batch_indices = torch.arange(batch_size, device=self.device)
        closest_quats = family[batch_indices, best_indices]  # (batch, 4)
        use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
        closest_quats[use_neg] = -closest_quats[use_neg]
        
        return closest_quats
    
    def process_batch(self, quaternions, return_stats=False):
        """
        Process a batch of quaternions with automatic batching for memory efficiency.
        
        Args:
            quaternions: Input quaternions of shape (N, 4)
            return_stats: If True, also return reconstruction statistics
            
        Returns:
            Reconstructed quaternions, and optionally statistics dict
        """
        num_quats = quaternions.shape[0]
        q_reconstructed_all = []
        stats = {'errors': [], 'misorientation_angles': []} if return_stats else None
        
        # Process in batches
        for batch_start in range(0, num_quats, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_quats)
            q_batch = quaternions[batch_start:batch_end]
            
            # Forward pass
            q_rec_batch = self.forward(q_batch)
            q_reconstructed_all.append(q_rec_batch)
            
            # Calculate statistics if requested
            if return_stats:
                errors, misorientation = self._calculate_errors(q_batch, q_rec_batch)
                stats['errors'].extend(errors.cpu().tolist())
                stats['misorientation_angles'].extend(misorientation.cpu().tolist())
        
        # Concatenate results
        q_reconstructed = torch.cat(q_reconstructed_all, dim=0)
        
        if return_stats:
            # Convert to numpy arrays and calculate summary statistics
            stats['errors'] = np.array(stats['errors'])
            stats['misorientation_angles'] = np.array(stats['misorientation_angles'])
            stats['summary'] = {
                'error_max': np.max(stats['errors']),
                'error_mean': np.mean(stats['errors']),
                'error_median': np.median(stats['errors']),
                'error_std': np.std(stats['errors']),
                'misorientation_max': np.max(stats['misorientation_angles']),
                'misorientation_mean': np.mean(stats['misorientation_angles']),
                'misorientation_median': np.median(stats['misorientation_angles']),
                'misorientation_std': np.std(stats['misorientation_angles']),
            }
            return q_reconstructed, stats
        
        return q_reconstructed
    
    def _calculate_errors(self, q_truth, q_reconstructed):
        """Calculate reconstruction errors and misorientation angles."""
        # Error distance
        errors = torch.norm(q_truth - q_reconstructed, dim=-1)
        
        # Misorientation angle
        q_conj = torch.stack([q_truth[:, 0], -q_truth[:, 1], -q_truth[:, 2], -q_truth[:, 3]], dim=1)
        error_quats = self._quat_mul(q_reconstructed, q_conj)
        w_errors = torch.clamp(torch.abs(error_quats[:, 0]), max=1.0)
        misorientation_angles = 2 * torch.acos(w_errors) * 180 / math.pi
        
        return errors, misorientation_angles
    
    def _quat_mul(self, q1, q2):
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)
    
    def process_image(self, input_path, output_path=None, render_comparison=True, dpi=300):
        """
        Process an EBSD quaternion image from file.
        
        Args:
            input_path: Path to .npy file containing quaternion data
            output_path: Path to save output image (default: 'ebsd_super_output.png')
            render_comparison: Whether to render IPF comparison (default: True)
            dpi: DPI for output image (default: 300)
            
        Returns:
            Reconstructed quaternion array, statistics dict
        """
        if output_path is None:
            output_path = 'ebsd_super_output.png'
        
        print("="*70)
        print("EBSD SUPER-RESOLUTION - IMAGE PROCESSING")
        print("="*70)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Device: {self.device}")
        
        # Load data
        q_numpy = np.load(input_path)
        print(f"Loaded data shape: {q_numpy.shape}")
        
        # Convert to torch tensor
        q_input = torch.tensor(q_numpy, dtype=torch.float32, device=self.device)
        
        # Handle different input formats
        is_image = False
        img_shape = None
        
        if q_input.dim() == 3:
            is_image = True
            original_shape = q_input.shape
            
            # Detect format: (H, W, 4) or (4, H, W)
            if q_input.shape[-1] == 4:
                img_shape = q_input.shape[:2]
                q_input = q_input.reshape(-1, 4)
            elif q_input.shape[0] == 4:
                img_shape = q_input.shape[1:]
                q_input = q_input.permute(1, 2, 0).reshape(-1, 4)
            else:
                raise ValueError(f"Cannot determine quaternion dimension in shape {original_shape}")
            
            print(f"Image shape: {img_shape}, Total quaternions: {q_input.shape[0]}")
        
        # Process with batching and statistics
        start_time = time.time()
        q_output, stats = self.process_batch(q_input, return_stats=True)
        elapsed_time = time.time() - start_time
        
        print(f"\nProcessing complete in {elapsed_time:.2f}s ({q_input.shape[0]/elapsed_time:.0f} quats/sec)")
        
        # Print statistics
        self._print_statistics(stats, q_input.shape[0])
        
        # Render comparison if requested and is image
        if render_comparison and is_image:
            self._render_comparison(q_input, q_output, img_shape, output_path, dpi)
        
        # Reshape output to original format if image
        if is_image:
            q_output = q_output.reshape(img_shape[0], img_shape[1], 4)
        
        return q_output.cpu().numpy(), stats
    
    def _print_statistics(self, stats, num_quats):
        """Print reconstruction statistics."""
        print("\n" + "="*70)
        print("RECONSTRUCTION STATISTICS")
        print("="*70)
        print(f"Total quaternions: {num_quats}")
        print(f"\nError Distance:")
        print(f"  Maximum: {stats['summary']['error_max']:.6e}")
        print(f"  Mean:    {stats['summary']['error_mean']:.6e}")
        print(f"  Median:  {stats['summary']['error_median']:.6e}")
        print(f"  Std Dev: {stats['summary']['error_std']:.6e}")
        print(f"\nMisorientation Angle:")
        print(f"  Maximum: {stats['summary']['misorientation_max']:.4f}°")
        print(f"  Mean:    {stats['summary']['misorientation_mean']:.4f}°")
        print(f"  Median:  {stats['summary']['misorientation_median']:.4f}°")
        print(f"  Std Dev: {stats['summary']['misorientation_std']:.4f}°")
        
        if stats['summary']['error_max'] < 0.05:
            print("\n✓ SUCCESS: All quaternions reconstructed within tolerance!")
        else:
            n_failed = np.sum(stats['errors'] >= 0.05)
            print(f"\n⚠ WARNING: {n_failed} quaternion(s) exceeded error threshold")
    
    def _render_comparison(self, q_input, q_output, img_shape, output_path, dpi):
        """Render IPF comparison image."""
        print("\n" + "="*70)
        print("RENDERING IPF COMPARISON")
        print("="*70)
        
        try:
            import sys
            sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')
            from visualization.ipf_render import render_input_output_comparison
            from orix.crystal_map import Phase
            
            # Reshape to image format (H, W, 4)
            q_input_img = q_input.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
            q_output_img = q_output.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
            
            # Define FCC symmetry
            fcc_sym = Phase(space_group=225).point_group
            
            # Render
            render_input_output_comparison(
                q_input_img,
                q_output_img,
                fcc_sym,
                out_png=output_path,
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
                format_input=False,
                dpi=dpi
            )
            print(f"✓ Saved comparison to: {output_path}")
        except Exception as e:
            print(f"⚠ Could not render comparison: {e}")

# ==============================================================================
# 5. VERIFICATION
# ==============================================================================
def run_physics_decoder_test():
    print("="*70)
    print("PHYSICS-BASED DECODER TEST (Spherical Sampling)")
    print("="*70)
    
    # Now we can use CUDA with our patched wigner_D!
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    
    # 1. Load ALL quaternions from file
    file_path = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data/Open718_QSR_x4_train_hr_x_block_0.npy"
    q_numpy = np.load(file_path)
    print(f"Loaded quaternion data with shape: {q_numpy.shape}")
    
    # Save output image in current working directory
    output_png = "input_output_comparison.png"
    
    # Convert to torch tensor and normalize
    q_all = torch.tensor(q_numpy, dtype=torch.float32, device=device)
    # Handle different input shapes
    is_image = False
    img_shape = None
    if q_all.dim() == 1:
        q_all = q_all.unsqueeze(0)  # Single quaternion
    elif q_all.dim() == 3:
        # Flatten spatial dimensions if image format
        # Could be (H, W, 4) or (4, H, W)
        is_image = True
        original_shape = q_all.shape
        
        # Check which dimension is the quaternion dimension (should be size 4)
        if q_all.shape[-1] == 4:
            # Format: (H, W, 4)
            img_shape = q_all.shape[:2]  # Save (H, W)
            q_all = q_all.reshape(-1, 4)  # (H*W, 4)
        elif q_all.shape[0] == 4:
            # Format: (4, H, W) - need to transpose
            img_shape = q_all.shape[1:]  # Save (H, W)
            q_all = q_all.permute(1, 2, 0).reshape(-1, 4)  # (H*W, 4)
        else:
            raise ValueError(f"Cannot determine quaternion dimension in shape {original_shape}")
        
        print(f"Reshaped from {original_shape} to {q_all.shape}, image shape: {img_shape}")
    
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
    if is_image:
        print("\n" + "="*70)
        print("RENDERING IPF COMPARISON")
        print("="*70)
        
        # Import the rendering function
        import sys
        sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')
        from visualization.ipf_render import render_input_output_comparison
        
        # Reshape back to image format (H, W, 4) and move to CPU
        # Note: q_all has shape (num_quats, 4) where num_quats = H*W
        # We need to reconstruct (H, W, 4)
        # img_shape contains (H, W) from the original data
        print(f"Reconstructing image shape: {img_shape} from {num_quats} quaternions")
        q_input_img = q_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        q_output_img = q_reconstructed_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        
        # Define FCC symmetry (m-3m, space group 225)
        fcc_sym = Phase(space_group=225).point_group
        
        # Render comparison
        print(f"Rendering IPF comparison to: {output_png}")
        render_input_output_comparison(
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