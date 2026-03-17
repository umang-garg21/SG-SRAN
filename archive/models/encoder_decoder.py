import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import time
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
import sys
sys.path.append("/data/home/umang/Materials/Reynolds-QSR_clean_ipf")
sys.path.append("/data/home/umang/Materials/Reynolds-QSR_clean_ipf/utils")
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
# 3.5 EQUIVARIANT SPATIAL CONVOLUTION LAYER (TRUE IRREPS)
# ==============================================================================
class EquivariantSpatialConv(nn.Module):
    """
    Equivariant spatial convolution layer that mixes features from nearby pixels
    while preserving O(3) symmetry.
    
    PHYSICS OF TRUE IRREPS:
    -----------------------
    When treating f4 and f6 as TRUE l=4 and l=6 irreps (not scalars), the 
    tensor product decomposition follows Clebsch-Gordan rules:
    
    4e ⊗ 4e = 0e + 2e + 4e + 6e + 8e
    6e ⊗ 6e = 0e + 2e + 4e + 6e + 8e + 10e + 12e
    4e ⊗ 6e = 2e + 4e + 6e + 8e + 10e
    
    To output only l=4 and l=6, we project onto these subspaces.
    This is physically meaningful: it represents allowed angular momentum
    coupling between orientation fields at neighboring pixels.
    """
    def __init__(self, kernel_size=3, upsample_factor=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.upsample_factor = upsample_factor
        self._upsample_copies = self.upsample_factor ** 2
        
        # Define irreps as TRUE l=4 and l=6 representations
        # 1x4e = one copy of l=4 (9 components)
        # 1x6e = one copy of l=6 (13 components)
        self.irreps_in = Irreps("1x4e + 1x6e")  # TRUE irreps, not scalars!
        # Output multiple copies for pixelshuffle upsampling (r^2 copies)
        self.irreps_out = Irreps(f"{self._upsample_copies}x4e + {self._upsample_copies}x6e")
        
        # One tensor product per SR copy to keep weights independent
        self.tp_per_copy = nn.ModuleList([
            FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_in,  # Interaction with neighbors
                Irreps("1x4e + 1x6e"),
                shared_weights=True
            )
            for _ in range(self._upsample_copies)
        ])
        
        # Print the tensor product structure for understanding
        print("\n" + "="*70)
        print("EQUIVARIANT CONVOLUTION LAYER - TRUE IRREPS PHYSICS")
        print("="*70)
        print(f"Input irreps:  {self.irreps_in}")
        print(f"Output irreps: {self.irreps_out}")
        print(f"\nTensor Product Paths (Clebsch-Gordan allowed couplings):")
        for ins in self.tp_per_copy[0].instructions:
            l1 = self.irreps_in[ins.i_in1].ir.l
            l2 = self.irreps_in[ins.i_in2].ir.l
            lo = self.tp_per_copy[0].irreps_out[ins.i_out].ir.l
            print(f"  {l1} ⊗ {l2} → {lo}  (weight shape: {ins.path_shape})")
        print("="*70 + "\n")
        
        # Spatial convolution weights (learnable)
        # 3x3 kernel for gathering neighbor information
        self.spatial_weights = nn.Parameter(torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size))
        
    def forward(self, f4, f6, img_shape):
        """
        Args:
            f4: (H*W, 9) L=4 features (true l=4 irrep)
            f6: (H*W, 13) L=6 features (true l=6 irrep)
            img_shape: (H, W) tuple
            
        Returns:
            f4_out: (H*W, 9) Convolved L=4 features
            f6_out: (H*W, 13) Convolved L=6 features
        """
        H, W = img_shape
        device = f4.device
        
        # Concatenate features: (H*W, 22) - ordered as [l=4 (9), l=6 (13)]
        features = torch.cat([f4, f6], dim=-1)
        
        # Reshape to image format: (1, 22, H, W)
        features_img = features.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        
        # Apply spatial convolution to gather neighbor information
        C = features_img.shape[1]  # 22 channels
        
        # Pad the image
        features_padded = F.pad(features_img, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        
        # Unfold to get patches: (1, C, H, W, k, k)
        patches = features_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        
        # Apply spatial weights and sum: (1, C, H, W)
        weights = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neighbor_features = (patches * weights).sum(dim=(-1, -2))
        
        # Reshape back: (H*W, 22)
        neighbor_features = neighbor_features.squeeze(0).permute(1, 2, 0).reshape(-1, C)
        
        # Apply tensor product for equivariant mixing (independent per copy)
        # This performs Clebsch-Gordan coupling between self and neighbor features
        # and projects onto l=4 and l=6 output subspaces
        out_per_copy = [tp(features, neighbor_features) for tp in self.tp_per_copy]
        out_features = torch.cat(out_per_copy, dim=-1)

        # Split back to f4 and f6 (with r^2 copies)
        f4_size = 9 * self._upsample_copies
        f4_out = out_features[:, :f4_size]
        f6_out = out_features[:, f4_size:]

        # --- BEGIN CHANGE: residual connection (fix dying-TP / zero-gradient) ---
        # Residual connection: ensures nonzero output and gradient flow even
        # when TP weights initialise to a cancelling configuration (dying TP).
        # For upsample_factor=1 the input and output shapes match directly.
        # For upsample_factor>1 (r^2 copies), tile the input r^2 times to match.
        if self._upsample_copies == 1:
            f4_out = f4_out + f4
            f6_out = f6_out + f6
        else:
            f4_out = f4_out + f4.repeat(1, self._upsample_copies)
            f6_out = f6_out + f6.repeat(1, self._upsample_copies)
        # --- END CHANGE ---

        return f4_out, f6_out

# ==============================================================================
# 3.6 EQUIVARIANT UPSAMPLE CONVOLUTION
# ==============================================================================
class EquivariantUpsampleConv(nn.Module):
    """
    Equivariant upsample convolution.

    Pipeline:
        1. Nearest-neighbour upsample: copy each LR pixel's irreps r×r times.
           F.interpolate with mode='nearest' applies the same op to every
           channel — equivariant by construction.

        2. SH-informed equivariant 2×2 aggregation at HR:
           For each HR pixel, gather its 2×2 neighbourhood (4 neighbours).
           Each neighbour j contributes via tp_aggregate(feat_j, SH_j), where
           SH_j are the spherical harmonics evaluated at the fixed unit-vector
           direction from the kernel centre to position j.  Sum over 4
           neighbours gives an equivariant context vector.

        3. tp(feat_self, context) → output.

        4. Residual from step-1 output.

    Parity constraint:
        Features are 1x4e + 1x6e (EVEN parity).  Coupling with odd-l SH
        (1o) produces odd intermediate irreps that cannot project onto 4e/6e.
        Only even-l SH contribute: sh_irreps = "1x0e + 1x2e" (6 components).

        The 4 kernel directions are fixed physical constants stored as a
        buffer.  The TPs learn how to use them; the directions are not learned.

    Init: both TP weights = 0  =>  output = residual = clean NN upsample.
    """
    def __init__(self, upsample_factor=4):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.irreps_feat = Irreps("1x4e + 1x6e")
        self.sh_irreps   = Irreps("1x0e + 1x2e")  # even-only SH: 1 + 5 = 6 components

        # Precompute SH at the 4 fixed 2×2 kernel positions (z=0 plane).
        # 2×2 kernel centred at (0.5, 0.5) in (row, col) space.
        # Convention: x = col direction, y = row direction, z = 0.
        # Unfold order (kH, kW): (0,0), (0,1), (1,0), (1,1)
        #   (0,0)  col=-0.5, row=-0.5  =>  unit vec (-1/√2, -1/√2, 0)
        #   (0,1)  col=+0.5, row=-0.5  =>  unit vec (+1/√2, -1/√2, 0)
        #   (1,0)  col=-0.5, row=+0.5  =>  unit vec (-1/√2, +1/√2, 0)
        #   (1,1)  col=+0.5, row=+0.5  =>  unit vec (+1/√2, +1/√2, 0)
        s = 1.0 / math.sqrt(2)
        kernel_dirs = torch.tensor([
            [-s, -s, 0.0],
            [+s, -s, 0.0],
            [-s, +s, 0.0],
            [+s, +s, 0.0],
        ], dtype=torch.float32)                                        # (4, 3) unit vecs

        sh_kernel = o3.spherical_harmonics(
            self.sh_irreps, kernel_dirs, normalize=False)              # (4, 6) — fixed
        self.register_buffer('sh_kernel', sh_kernel)

        # TP aggregation: feat_j ⊗ SH_j  →  feat
        # Active CG paths: 4e⊗0e→4e, 4e⊗2e→4e, 4e⊗2e→6e,
        #                  6e⊗0e→6e, 6e⊗2e→4e, 6e⊗2e→6e
        self.tp_aggregate = FullyConnectedTensorProduct(
            self.irreps_feat,
            self.sh_irreps,
            self.irreps_feat,
            shared_weights=True,
        )
        with torch.no_grad():
            self.tp_aggregate.weight.data.zero_()

        # TP mixing: feat_self ⊗ context  →  output
        self.tp = FullyConnectedTensorProduct(
            self.irreps_feat,
            self.irreps_feat,
            self.irreps_feat,
            shared_weights=True,
        )
        with torch.no_grad():
            self.tp.weight.data.zero_()

    def forward(self, f4, f6, img_shape):
        """
        Args:
            f4: (H*W, 9)  l=4 features at LR
            f6: (H*W, 13) l=6 features at LR
            img_shape: (H, W)

        Returns:
            f4_out:   (rH*rW, 9)
            f6_out:   (rH*rW, 13)
            hr_shape: (rH, rW)
        """
        H, W = img_shape
        r = self.upsample_factor
        C = 22

        # Pack to image: (1, 22, H, W)
        features = torch.cat([f4, f6], dim=-1)
        feat_img = features.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

        # 1. Nearest-neighbour upsample — copies each LR pixel r×r times
        feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode='nearest')  # (1, 22, rH, rW)

        Hr, Wr = H * r, W * r
        N = Hr * Wr

        # 2. SH-informed 2×2 neighbourhood aggregation
        # Pad right+bottom by 1 so every HR pixel (h,w) has a full 2×2 patch
        feat_padded = F.pad(feat_hr, [0, 1, 0, 1], mode='replicate')  # (1, 22, Hr+1, Wr+1)
        patches = feat_padded.unfold(2, 2, 1).unfold(3, 2, 1)         # (1, 22, Hr, Wr, kH, kW)
        patches = patches.reshape(1, C, Hr, Wr, 4)                    # (1, 22, Hr, Wr, 4)

        # -> (N, 4, C): pixel-major, neighbour, feature
        patches_flat = patches.squeeze(0).permute(1, 2, 3, 0).reshape(N, 4, C)

        # Expand fixed SH to every pixel: (4, 6) -> (N*4, 6)
        sh_exp = self.sh_kernel.unsqueeze(0).expand(N, -1, -1).reshape(N * 4, -1)

        # tp_aggregate(feat_j, SH_j) for every pixel × every neighbour
        agg_flat = self.tp_aggregate(
            patches_flat.reshape(N * 4, C),   # (N*4, 22)
            sh_exp,                            # (N*4,  6)
        )                                      # (N*4, 22)

        # Sum the 4 neighbour contributions -> context per pixel
        context = agg_flat.reshape(N, 4, C).sum(dim=1)                # (N, 22)

        # Self features at HR
        feat_flat = feat_hr.squeeze(0).permute(1, 2, 0).reshape(N, C) # (N, 22)

        # 3. Equivariant TP: self ⊗ context -> output
        out_features = self.tp(feat_flat, context)

        # 4. Residual from NN upsample
        f4_out = out_features[:, :9] + feat_flat[:, :9]
        f6_out = out_features[:, 9:] + feat_flat[:, 9:]

        return f4_out, f6_out, (Hr, Wr)

    def forward_with_internals(self, f4, f6, img_shape):
        """Run forward pass and return each intermediate state for visualization.

        Returns a dict with keys:
            'nn_upsample'  : (f4, f6) after F.interpolate, before any TP
            'sh_context'   : (f4, f6) of the summed SH-aggregated context
            'tp_out'       : (f4, f6) after tp(self, context), before residual
            'final'        : (f4, f6) after adding residual  (== forward output)
            'hr_shape'     : (Hr, Wr)
        """
        H, W = img_shape
        r = self.upsample_factor
        C = 22

        features = torch.cat([f4, f6], dim=-1)
        feat_img = features.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

        # Op 1: NN upsample
        feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode='nearest')
        Hr, Wr = H * r, W * r
        N = Hr * Wr
        feat_flat = feat_hr.squeeze(0).permute(1, 2, 0).reshape(N, C)

        # Op 2: SH context
        feat_padded = F.pad(feat_hr, [0, 1, 0, 1], mode='replicate')
        patches = feat_padded.unfold(2, 2, 1).unfold(3, 2, 1).reshape(1, C, Hr, Wr, 4)
        patches_flat = patches.squeeze(0).permute(1, 2, 3, 0).reshape(N, 4, C)
        sh_exp = self.sh_kernel.unsqueeze(0).expand(N, -1, -1).reshape(N * 4, -1)
        agg_flat = self.tp_aggregate(patches_flat.reshape(N * 4, C), sh_exp)
        context = agg_flat.reshape(N, 4, C).sum(dim=1)

        # Op 3: TP output (before residual)
        tp_out = self.tp(feat_flat, context)

        # Op 4: Final (after residual)
        f4_out = tp_out[:, :9] + feat_flat[:, :9]
        f6_out = tp_out[:, 9:] + feat_flat[:, 9:]

        return {
            'nn_upsample': (feat_flat[:, :9], feat_flat[:, 9:]),
            'sh_context':  (context[:, :9],   context[:, 9:]),
            'tp_out':      (tp_out[:, :9],     tp_out[:, 9:]),
            'final':       (f4_out,            f6_out),
            'hr_shape':    (Hr, Wr),
        }

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
    
    def __init__(self, device='cpu', grid_samples=10000, batch_size=1000, upsample_factor=4):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.upsample_factor = upsample_factor
        
        # Initialize physics constants and modules
        self.physics = FCCPhysics(self.device)
        self.encoder = FCCEncoder(self.physics)
        self.decoder = SphericalSamplingDecoder(self.physics, grid_res=grid_samples)

        # LR spatial convolution (no upsampling, just neighbor mixing)
        self.conv_layer = EquivariantSpatialConv(kernel_size=3, upsample_factor=1)
        # Equivariant transpose conv for learned upsampling (replaces pixelshuffle + hr_conv)
        self.upsample_layer = EquivariantUpsampleConv(upsample_factor=self.upsample_factor)
        # HR spatial convolution (refinement at SR resolution after transpose conv)
        self.hr_conv_layer = EquivariantSpatialConv(kernel_size=3, upsample_factor=1)

        # Move to device
        self.to(self.device)

    def forward(self, quaternions, img_shape=None, decode=True):
        """
        Forward pass: quaternions -> latent features -> convolved features -> reconstructed quaternions

        Args:
            quaternions: Input quaternions of shape (N, 4) or (H, W, 4)
            img_shape: Optional (H, W) tuple. Required for convolution stage.
            decode: If False, skip the SphericalSamplingDecoder and return None for
                    'output'. Set to False during training to avoid the costly
                    (N_hr_pixels × 10 000) intermediate tensor when the decoder
                    output is not needed for the loss.

        Returns:
            Dict with intermediate stages for visualization
        """
        # Store original shape
        original_shape = quaternions.shape
        is_image = len(original_shape) == 3
        
        # Flatten if image format
        if is_image:
            if img_shape is None:
                img_shape = original_shape[:2]
            quaternions = quaternions.reshape(-1, 4)
        
        # Ensure 2D (N, 4)
        if quaternions.dim() == 1:
            quaternions = quaternions.unsqueeze(0)
        
        # Normalize
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        
        # Stage 1: Encode
        f4, f6 = self.encoder(quaternions)
        
        #import pdb; pdb.set_trace()
        # Stage 2: LR spatial convolution (trainable, no upsampling)
        if img_shape is not None:
            f4_conv, f6_conv = self.conv_layer(f4, f6, img_shape)
        else:
            f4_conv, f6_conv = f4, f6

        #import pdb; pdb.set_trace()

        # Stage 3: Equivariant transpose-conv upsample (trainable)
        f4_up, f6_up, hr_shape = f4_conv, f6_conv, img_shape
        if img_shape is not None and self.upsample_factor > 1:
            f4_up, f6_up, hr_shape = self.upsample_layer(f4_conv, f6_conv, img_shape)

        #import pdb; pdb.set_trace()

        # Stage 4: HR spatial convolution — refinement at SR resolution (trainable)
        f4_hr, f6_hr = f4_up, f6_up
        if hr_shape is not None:
            f4_hr, f6_hr = self.hr_conv_layer(f4_up, f6_up, hr_shape)

        #import pdb; pdb.set_trace()

        # Stage 5: Decode (frozen) — skipped when decode=False (e.g. during training)
        # The decoder creates a (N_pixels × 10 000) intermediate tensor which is
        # ~10 GB at HR resolution; skip it when only the irreps are needed.
        q_reconstructed = self.decoder(f4_hr, f6_hr) if decode else None

        #import pdb; pdb.set_trace()

        return {
            "input": quaternions,
            "encoded": (f4, f6),
            "convolved": (f4_conv, f6_conv),
            "upsampled_irreps": (f4_up, f6_up),
            "hr_convolved_irreps": (f4_hr, f6_hr),
            "hr_shape": hr_shape,
            "output": q_reconstructed,
        }

        # --- OLD PIPELINE (PixelShuffle + hr_conv_layer) ---
        # # Stage 2: Apply equivariant spatial convolution (if img_shape provided)
        # if img_shape is not None:
        #     f4_conv, f6_conv = self.conv_layer(f4, f6, img_shape)
        # else:
        #     # Skip convolution if no spatial info
        #     f4_conv, f6_conv = f4, f6
        #
        # # Stage 3: PixelShuffle upsample (requires img_shape)
        # q_pixelshuffled = None
        # f4_ps = f4_conv
        # f6_ps = f6_conv
        # hr_shape = None
        # if img_shape is not None and self.upsample_factor > 1:
        #     H, W = img_shape
        #     r = self.upsample_factor
        #     copies = r * r
        #
        #     # Combine features and reshape to image
        #     features = torch.cat([f4_conv, f6_conv], dim=-1)  # (H*W, copies*22)
        #     features_img = features.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        #
        #     # PixelShuffle: (1, copies*22, H, W) -> (1, 22, H*r, W*r)
        #     features_up = F.pixel_shuffle(features_img, upscale_factor=r)
        #
        #     # Split back to f4/f6 at HR resolution
        #     features_up = features_up.squeeze(0)  # (22, H*r, W*r)
        #     f4_up = features_up[:9, :, :]
        #     f6_up = features_up[9:, :, :]
        #
        #     f4_ps = f4_up.permute(1, 2, 0).reshape(-1, 9)
        #     f6_ps = f6_up.permute(1, 2, 0).reshape(-1, 13)
        #
        #     hr_shape = (H * r, W * r)
        #
        #     # Decode pixelshuffled features (optional visualization)
        #     q_pixelshuffled = self.decoder(f4_ps, f6_ps)
        #
        # # Stage 4: HR spatial convolution on irreps
        # f4_hr_conv = f4_ps
        # f6_hr_conv = f6_ps
        # if hr_shape is not None:
        #     f4_hr_conv, f6_hr_conv = self.hr_conv_layer(f4_ps, f6_ps, hr_shape)
        #
        # # Stage 4: Decode final output (HR if pixelshuffle used)
        # if q_pixelshuffled is not None:
        #     q_reconstructed = self.decoder(f4_hr_conv, f6_hr_conv)
        # else:
        #     q_reconstructed = self.decoder(f4_hr_conv, f6_hr_conv)
        #
        # # Return intermediate stages for visualization
        # return {
        #     "input": quaternions,
        #     "encoded": (f4, f6),
        #     "convolved": (f4_conv, f6_conv),
        #     "pixelshuffled_irreps": (f4_ps, f6_ps),
        #     "hr_convolved_irreps": (f4_hr_conv, f6_hr_conv),
        #     "pixelshuffled": q_pixelshuffled,
        #     "output": q_reconstructed
        # }
    
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
# 5. BATCHED VARIANTS (process B images per forward call)
# ==============================================================================

class BatchedEquivariantSpatialConv(EquivariantSpatialConv):
    """
    Batched variant of EquivariantSpatialConv.

    Overrides forward to accept an explicit batch_size so that B images
    (each of shape H×W) are processed together as a single (B, C, H, W)
    tensor through the spatial stage, then flattened to (B*H*W, C) for the
    FullyConnectedTensorProduct.  All parameters and __init__ logic are
    inherited unchanged from EquivariantSpatialConv.
    """

    def forward(self, f4, f6, img_shape, batch_size=1):
        H, W = img_shape
        B = batch_size

        features = torch.cat([f4, f6], dim=-1)          # (B*H*W, 22)
        C = features.shape[1]

        # Reshape to (B, C, H, W) for batched spatial ops
        features_img = features.view(B, H, W, C).permute(0, 3, 1, 2)

        features_padded = F.pad(features_img, [self.padding] * 4, mode='replicate')
        patches = features_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        weights = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neighbor_features = (patches * weights).sum(dim=(-1, -2))  # (B, C, H, W)

        # Flatten to (B*H*W, C) for TP
        feat_flat     = features_img.permute(0, 2, 3, 1).reshape(-1, C)
        neighbor_flat = neighbor_features.permute(0, 2, 3, 1).reshape(-1, C)

        out_per_copy = [tp(feat_flat, neighbor_flat) for tp in self.tp_per_copy]
        out_features = torch.cat(out_per_copy, dim=-1)

        f4_size = 9 * self._upsample_copies
        f4_out = out_features[:, :f4_size]
        f6_out = out_features[:, f4_size:]

        # Residual (same logic as parent)
        if self._upsample_copies == 1:
            f4_out = f4_out + f4
            f6_out = f6_out + f6
        else:
            f4_out = f4_out + f4.repeat(1, self._upsample_copies)
            f6_out = f6_out + f6.repeat(1, self._upsample_copies)

        return f4_out, f6_out


class BatchedEquivariantUpsampleConv(EquivariantUpsampleConv):
    """
    Batched variant of EquivariantUpsampleConv.

    Reshapes flat (B*H*W, C) input to (B, C, H, W) before interpolation,
    then runs the SH-informed 2×2 aggregation and TP over the full batch.
    """

    def forward(self, f4, f6, img_shape, batch_size=1):
        H, W = img_shape
        B = batch_size
        r = self.upsample_factor
        C = 22

        features = torch.cat([f4, f6], dim=-1)                       # (B*H*W, 22)
        feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)    # (B, 22, H, W)

        # 1. Nearest-neighbour upsample (batch-native)
        feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode='nearest')  # (B, 22, rH, rW)

        Hr, Wr = H * r, W * r
        N = Hr * Wr

        # 2. SH-informed 2×2 neighbourhood aggregation
        feat_padded = F.pad(feat_hr, [0, 1, 0, 1], mode='replicate') # (B, 22, Hr+1, Wr+1)
        patches = feat_padded.unfold(2, 2, 1).unfold(3, 2, 1)        # (B, 22, Hr, Wr, kH, kW)
        patches = patches.reshape(B, C, Hr, Wr, 4)                   # (B, 22, Hr, Wr, 4)

        # -> (B*N, 4, C): batch×pixel-major, neighbour, feature
        patches_flat = patches.permute(0, 2, 3, 4, 1).contiguous().reshape(B * N, 4, C)

        # Expand fixed SH: (4, 6) -> (B*N*4, 6)
        sh_exp = self.sh_kernel.unsqueeze(0).expand(B * N, -1, -1).reshape(B * N * 4, -1)

        # tp_aggregate(feat_j, SH_j) for every batch × pixel × neighbour
        agg_flat = self.tp_aggregate(
            patches_flat.reshape(B * N * 4, C),   # (B*N*4, 22)
            sh_exp,                                # (B*N*4,  6)
        )                                          # (B*N*4, 22)

        # Sum the 4 neighbour contributions -> context per pixel
        context = agg_flat.reshape(B * N, 4, C).sum(dim=1)           # (B*N, 22)

        # Self features at HR
        feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)   # (B*N, 22)

        # 3. Equivariant TP: self ⊗ context -> output
        out_features = self.tp(feat_flat, context)

        # 4. Residual from NN upsample
        f4_out = out_features[:, :9] + feat_flat[:, :9]
        f6_out = out_features[:, 9:] + feat_flat[:, 9:]

        return f4_out, f6_out, (Hr, Wr)


class BatchedEBSDSuper(nn.Module):
    """
    Batched variant of EBSDSuper.

    Processes B LR images per forward call by stacking their flattened pixels
    into a single (B*H*W, 4) tensor and routing them through
    BatchedEquivariantSpatialConv / BatchedEquivariantUpsampleConv so that
    all spatial ops run as a single (B, C, H, W) kernel instead of B separate
    (1, C, H, W) calls.

    The existing EBSDSuper class is completely unchanged.

    Args:
        device:          Device string ('cpu', 'cuda', 'cuda:0', …)
        grid_samples:    Fibonacci sphere samples for the decoder (default 10000)
        upsample_factor: Spatial upsampling ratio (default 4)
    """

    def __init__(self, device='cpu', grid_samples=10000, upsample_factor=4):
        super().__init__()
        self.device = torch.device(device)
        self.upsample_factor = upsample_factor

        self.physics      = FCCPhysics(self.device)
        self.encoder      = FCCEncoder(self.physics)
        self.decoder      = SphericalSamplingDecoder(self.physics, grid_res=grid_samples)
        self.conv_layer   = BatchedEquivariantSpatialConv(kernel_size=3, upsample_factor=1)
        self.upsample_layer = BatchedEquivariantUpsampleConv(upsample_factor=upsample_factor)
        self.hr_conv_layer  = BatchedEquivariantSpatialConv(kernel_size=3, upsample_factor=1)

        self.to(self.device)

    def forward(self, quaternions, img_shape, batch_size=1, decode=False):
        """
        Args:
            quaternions: (B*H*W, 4) — B LR images stacked pixel-wise
            img_shape:   (H, W)     — LR spatial shape (identical for all B images)
            batch_size:  B          — number of images stacked in quaternions
            decode:      If False, skip SphericalSamplingDecoder (saves ~10 GB at HR)

        Returns:
            Same dict as EBSDSuper.forward; 'output' is None when decode=False.
        """
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

        f4, f6 = self.encoder(quaternions)

        f4_conv, f6_conv = self.conv_layer(f4, f6, img_shape, batch_size=batch_size)

        f4_up, f6_up, hr_shape = self.upsample_layer(f4_conv, f6_conv, img_shape, batch_size=batch_size)

        f4_hr, f6_hr = self.hr_conv_layer(f4_up, f6_up, hr_shape, batch_size=batch_size)

        q_reconstructed = self.decoder(f4_hr, f6_hr) if decode else None

        return {
            "input":              quaternions,
            "encoded":            (f4, f6),
            "convolved":          (f4_conv, f6_conv),
            "upsampled_irreps":   (f4_up, f6_up),
            "hr_convolved_irreps":(f4_hr, f6_hr),
            "hr_shape":           hr_shape,
            "output":             q_reconstructed,
        }


# ==============================================================================
# 6. VERIFICATION
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

    # Load data using QuaternionDataset
    dataset_out_root = "/data/home/umang/Materials/Materials_data_mount/EBSD//"
    dataset_name = "IN718_FZ_2D_SR_x4/Open718_QSR_x4/"

    dataset_dir = os.path.join(dataset_out_root, dataset_name)

    train_ds = QuaternionDataset(
        dataset_root=dataset_dir,
        split="Train",
        preload=True,
        preload_torch=True,  # preload as CPU torch tensors
    )
    print("train_ds[0][1].shape", train_ds[0][1].shape)

    q_all = train_ds[0][1]  # Get the first sample (HR quaternions)

    # Convert from (C, H, W) to (H, W, C)
    q_all = q_all.permute(1, 2, 0)

    q_all = q_all.reshape(-1, 4).to(device)  # Flatten to (N, 4)
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
    is_image = True
    if is_image:
        print("\n" + "="*70)
        print("RENDERING IPF COMPARISON")
        print("="*70)

        img_shape = train_ds[0][1].shape[1:3]  # Assuming (C, H, W) format for the quaternions
        # Reshape back to image format (H, W, 4) and move to CPU
        print(f"Reshaping reconstructed quaternions to image format: {img_shape} with 4 channels")

        q_input_img = q_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        q_output_img = q_reconstructed_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()

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