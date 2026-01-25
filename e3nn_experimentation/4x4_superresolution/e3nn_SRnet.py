import torch
import torch.nn as nn
import math
from e3nn import o3, nn as e3nn_nn

# ==============================================================================
# 0. HELPER: NormActivation wrapper for image tensors
# ==============================================================================
class NormActivation2D(nn.Module):
    """Wraps e3nn NormActivation to work with (B, C, H, W) image tensors."""
    def __init__(self, irreps, scalar_nonlinearity):
        super().__init__()
        self.act = e3nn_nn.NormActivation(irreps, scalar_nonlinearity)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) for e3nn
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

# ==============================================================================
# 0.5 HELPER: Equivariant Pixel Shuffle Upsampling
# ==============================================================================
class EquivariantPixelShuffle(nn.Module):
    """
    Equivariant upsampling using pixel shuffle that PRESERVES crystal symmetry.
    
    Key insight: Standard Conv2d mixes irrep channels and BREAKS equivariance.
    We must expand each irrep type separately using o3.Linear.
    
    For scale_factor r, we create r^2 copies of each irrep, then spatially rearrange.
    This preserves equivariance because:
    - Each copy of an irrep transforms identically under SO(3)
    - Pixel shuffle is a purely spatial operation (doesn't touch channel values)
    """
    def __init__(self, irreps_in, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.irreps_in = o3.Irreps(irreps_in)
        self.r2 = scale_factor ** 2
        
        # Output irreps are the same as input (we're just rearranging spatially)
        self.irreps_out = self.irreps_in
        
        # Build expanded irreps: multiply each mul by r^2
        # e.g., "8x0e + 8x1e" with r=2 -> "32x0e + 32x1e" (intermediate)
        expanded_irreps_list = []
        for mul, ir in self.irreps_in:
            expanded_irreps_list.append((mul * self.r2, ir))
        self.irreps_expanded = o3.Irreps(expanded_irreps_list)
        
        # EQUIVARIANT channel expansion using o3.Linear
        # This respects irrep structure: only mixes within same (l, p) types
        self.expansion = o3.Linear(self.irreps_in, self.irreps_expanded)
        
        # Track channel structure for proper pixel shuffle
        # We need to shuffle each irrep block separately
        self.irrep_slices = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = (2 * ir.l + 1)  # dimension of this irrep type
            expanded_dim = mul * self.r2 * dim
            self.irrep_slices.append((idx, idx + expanded_dim, mul, dim))
            idx += expanded_dim
        
    def forward(self, x):
        # x: (B, C, H, W) - channels are irreps
        B, C, H, W = x.shape
        
        # 1. Permute for e3nn: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # 2. Equivariant channel expansion: (B, H, W, C) -> (B, H, W, C*r^2)
        x = self.expansion(x)
        
        # 3. Permute back: (B, H, W, C*r^2) -> (B, C*r^2, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 4. Pixel shuffle each irrep block separately to maintain structure
        # This ensures the irrep ordering is preserved after upsampling
        outputs = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = 2 * ir.l + 1
            # Each irrep block: (B, mul*r^2*dim, H, W)
            block_channels = mul * self.r2 * dim
            block = x[:, idx:idx + block_channels, :, :]
            
            # Reshape to group by irrep instance: (B, mul*r^2, dim, H, W)
            block = block.view(B, mul * self.r2, dim, H, W)
            
            # Reorder to (B, mul, r^2, dim, H, W) for proper pixel shuffle
            block = block.view(B, mul, self.r2, dim, H, W)
            
            # Permute to (B, mul, dim, r^2, H, W) then merge for shuffle
            block = block.permute(0, 1, 3, 2, 4, 5)  # (B, mul, dim, r^2, H, W)
            block = block.reshape(B, mul * dim * self.r2, H, W)
            
            # Now reshape for pixel shuffle: need (B, C_out * r^2, H, W)
            # where C_out = mul * dim
            # Actually we need to interleave properly...
            
            # Simpler approach: reshape to (B, mul*dim, r^2, H, W)
            block = block.view(B, mul * dim, self.r2, H, W)
            # Permute to (B, r^2, mul*dim, H, W) 
            block = block.permute(0, 2, 1, 3, 4)
            # Merge: (B, r^2 * mul * dim, H, W)
            block = block.reshape(B, self.r2 * mul * dim, H, W)
            
            # Standard pixel shuffle on this block
            block = nn.functional.pixel_shuffle(block, self.scale_factor)
            # Result: (B, mul*dim, H*r, W*r)
            
            outputs.append(block)
            idx += block_channels
        
        # 5. Concatenate all irrep blocks
        x = torch.cat(outputs, dim=1)
        
        return x


# ==============================================================================
# 0.6 ALTERNATIVE: Static Bilinear Upsampling (Fully Equivariant, No Learning)
# ==============================================================================
class StaticEquivariantUpsample(nn.Module):
    """
    Static (non-learned) equivariant upsampling using bilinear interpolation.
    
    This is GUARANTEED equivariant because:
    - Bilinear interpolation is a purely spatial operation
    - It applies the same interpolation to each channel independently
    - No mixing between irrep channels
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # x: (B, C, H, W)
        # Bilinear upsampling - each channel is interpolated independently
        return nn.functional.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        )

# ==============================================================================
# 1. THE GEOMETRIC KERNEL (Grid Convolution Logic)
# ==============================================================================

class ScalarSpatialConv(nn.Module):
    """
    A spatial convolution that ONLY operates on the scalar (l=0) component.
    This is used for spatial mixing while preserving equivariance.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=padding, stride=stride, bias=False)
    
    def forward(self, x):
        return self.conv(x)


class TrueEquivariantConv(nn.Module):
    """
    A TRULY equivariant convolution for 2D grids with SO(3) feature channels.
    
    FIXED DESIGN: Uses unfold to gather neighbor features, then o3.Linear.
    
    1. SPATIAL GATHERING: Use unfold to get 3x3 neighborhood
    2. SCALAR AGGREGATION: Compute attention weights from norms (equivariant scalars)
    3. GEOMETRIC MIXING: Apply o3.Linear to weighted-sum of neighbor features
    
    This ensures f(D·x) = D·f(x) for any Wigner-D rotation D.
    """
    def __init__(self, irreps_in, irreps_out, kernel_size=3, padding=1, stride=1):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.K2 = kernel_size * kernel_size  # 9 for 3x3
        
        # Count number of irrep blocks for scalar extraction
        self.num_irreps_in = len(self.irreps_in)
        self.num_irreps_out = len(self.irreps_out)
        self.in_dim = self.irreps_in.dim
        self.out_dim = self.irreps_out.dim
        
        # A. SCALAR PATHWAY: Learn spatial aggregation weights from norms
        # Each neighbor contributes a scalar (norm of features) → learn how to weight neighbors
        # Input: K2 scalars per irrep block → Output: K2 weights
        self.neighbor_weights = nn.Sequential(
            nn.Conv2d(self.num_irreps_in * self.K2, 64, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(64, self.K2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  # Weights sum to 1 over neighbors
        )
        
        # B. GEOMETRIC PATHWAY: Equivariant linear transformation
        # Applied to the aggregated neighbor features
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        
        # Store irrep structure for norm extraction
        self.irrep_slices_in = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = (2 * ir.l + 1) * mul
            self.irrep_slices_in.append((idx, idx + dim, mul, ir.l))
            idx += dim
    
    def extract_norms_per_neighbor(self, x_unfolded):
        """
        Extract L2 norm of each irrep block for each neighbor.
        x_unfolded: (B, C, K2, H*W)
        Returns: (B, num_irreps * K2, H*W)
        """
        B, C, K2, L = x_unfolded.shape
        norms = []
        for start, end, mul, l in self.irrep_slices_in:
            block = x_unfolded[:, start:end, :, :]  # (B, mul*(2l+1), K2, L)
            dim = 2 * l + 1
            # Reshape to (B, mul, dim, K2, L)
            block = block.view(B, mul, dim, K2, L)
            # Compute norm over the irrep dimension
            block_norm = block.norm(dim=2)  # (B, mul, K2, L)
            # Average over multiplicity
            block_norm = block_norm.mean(dim=1)  # (B, K2, L)
            norms.append(block_norm)
        # Stack along a new dimension then flatten
        norms = torch.stack(norms, dim=1)  # (B, num_irreps, K2, L)
        norms = norms.view(B, self.num_irreps_in * self.K2, L)  # (B, num_irreps*K2, L)
        return norms
    
    def forward(self, x):
        """
        x: (B, C, H, W) where C = irreps_in.dim
        """
        B, C, H, W = x.shape
        
        # 1. Unfold: extract K2 neighbors for each pixel
        # Output: (B, C * K2, H_out * W_out)
        x_unfolded = torch.nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride
        )
        L = x_unfolded.shape[-1]  # H_out * W_out
        
        # Reshape to (B, C, K2, L)
        x_unfolded = x_unfolded.view(B, C, self.K2, L)
        
        # 2. Extract scalar norms for each neighbor
        scalar_norms = self.extract_norms_per_neighbor(x_unfolded)  # (B, num_irreps*K2, L)
        
        # Reshape to (B, num_irreps*K2, H, W) for conv
        H_out = W_out = int(math.sqrt(L))
        scalar_norms_2d = scalar_norms.view(B, -1, H_out, W_out)
        
        # 3. Compute aggregation weights from scalars
        weights = self.neighbor_weights(scalar_norms_2d)  # (B, K2, H, W)
        weights = weights.view(B, self.K2, L)  # (B, K2, L)
        
        # 4. Weighted sum of neighbor features (equivariant aggregation)
        # x_unfolded: (B, C, K2, L), weights: (B, K2, L)
        # We want: sum over K2 of x_unfolded * weights
        weights_expanded = weights.unsqueeze(1)  # (B, 1, K2, L)
        x_aggregated = (x_unfolded * weights_expanded).sum(dim=2)  # (B, C, L)
        
        # 5. Reshape for e3nn: (B, C, L) -> (B*L, C)
        x_flat = x_aggregated.permute(0, 2, 1).reshape(B * L, C)
        
        # 6. Apply equivariant linear transformation
        y_flat = self.linear(x_flat)  # (B*L, C_out)
        
        # 7. Reshape back to image: (B*L, C_out) -> (B, C_out, H, W)
        y = y_flat.view(B, L, self.out_dim).permute(0, 2, 1)  # (B, C_out, L)
        y = y.view(B, self.out_dim, H_out, W_out)
        
        return y


class EquivariantGridConv(nn.Module):
    """
    ORIGINAL IMPLEMENTATION (NOT STRICTLY EQUIVARIANT)
    
    Uses tensor products with spatial spherical harmonics.
    This breaks SO(3) equivariance because 2D spatial directions
    don't transform under 3D feature rotations.
    
    Kept for reference and comparison.
    """
    def __init__(self, irreps_in, irreps_out, kernel_size=3, padding=1, stride=1):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # A. Define the Geometry of the Grid Kernel (e.g. 3x3)
        # We calculate the relative (x, y) vectors for every pixel in the kernel.
        # We assume z=0 for the 2D grid.
        # This defines the "Spatial Geometry" we are convolving against.
        
        # Generate grid offsets
        k = kernel_size // 2
        r_range = torch.arange(-k, k + 1)
        grid_x, grid_y = torch.meshgrid(r_range, r_range, indexing='ij')
        
        # Shape: (K*K, 3) -> Vectors pointing to neighbors
        self.grid_vectors = torch.stack([
            grid_x.flatten(), 
            grid_y.flatten(), 
            torch.zeros_like(grid_x.flatten())
        ], dim=1).float()
        
        # Remove the center pixel (0,0,0) from interaction geometry if strictly looking at neighbors?
        # Actually, TFN usually includes self-interaction. We keep it.
        
        # B. Spherical Harmonics of the Kernel
        # The network needs to "see" the direction of the neighbors.
        # We choose L_max for the spatial filter. L=2 or 3 is usually enough for 3x3 grids.
        self.l_max_spatial = 3 
        self.irreps_spatial = o3.Irreps.spherical_harmonics(self.l_max_spatial)
        
        # Precompute Y(r) for the kernel (Fixed Geometry)
        # We verify vectors are not zero before normalizing to avoid NaN at center
        non_zero_mask = self.grid_vectors.norm(dim=1) > 1e-6
        grid_dirs = torch.zeros_like(self.grid_vectors)
        grid_dirs[non_zero_mask] = self.grid_vectors[non_zero_mask] / self.grid_vectors[non_zero_mask].norm(dim=1, keepdim=True)
        
        # Register as buffer (fixed constants)
        self.register_buffer('kernel_Y', o3.spherical_harmonics(self.irreps_spatial, grid_dirs, normalize=True))
        
        # C. The Interaction (Tensor Product)
        # Input: Feature (Irreps)
        # Geometry: Kernel Space (Spherical Harmonics)
        # Output: Next Feature (Irreps)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_spatial,
            self.irreps_out,
            shared_weights=False # We want distinct weights for every spatial offset
        )
        
        # D. The Radial Weights (The "Learning" Part)
        # In TFN, weights are a function of radius. Here, since it's a fixed grid,
        # we learn a unique weight matrix for every pixel in the 3x3 kernel.
        # TP expects weights of shape (N_paths,)
        # We need (K*K, N_paths)
        num_weights = self.tp.weight_numel
        self.kernel_weights = nn.Parameter(torch.randn(self.grid_vectors.shape[0], num_weights) / math.sqrt(num_weights))

    def forward(self, x):
        """
        x: (Batch, Channels_In, H, W) -> Irreps are in the Channel dimension
        """
        B, C, H, W = x.shape
        
        # 1. Unfold (Extract 3x3 patches)
        # Converts image to patches: (B, C*K*K, L) where L = H_out*W_out
        x_unfolded = torch.nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride
        )
        # Shape: (B, C, K*K, L)
        L = x_unfolded.shape[-1]
        K2 = self.grid_vectors.shape[0] # K*K
        
        x_unfolded = x_unfolded.view(B, C, K2, L).permute(0, 3, 2, 1) 
        # Now: (B, L, K*K, C) -> (Batch*Pixels, Neighbors, Features)
        
        # Flatten Batch and Pixels for e3nn processing
        x_flat = x_unfolded.reshape(-1, K2, C) 
        
        # 2. The Interaction
        # We want to compute: sum_neighbors TP(x_neighbor, Y_neighbor, w_neighbor)
        
        # Expand Kernel Geometry to match batch
        # kernel_Y: (K*K, Y_dim)
        # kernel_weights: (K*K, W_dim)
        
        # We iterate over neighbors (K*K) and sum the results
        # This is effectively the convolution sum
        
        out_accum = 0
        
        # NOTE: Optimized e3nn implementation would batch this differently, 
        # but iterating K*K is clear and correct for this demonstration.
        for k in range(K2):
            # Features of neighbor k across all images/pixels
            feat_k = x_flat[:, k, :] # (TotalPixels, C_in)
            
            # Geometry of neighbor k (Broadcasting)
            geom_k = self.kernel_Y[k].unsqueeze(0).expand(feat_k.shape[0], -1)
            
            # Weights for neighbor k
            w_k = self.kernel_weights[k].unsqueeze(0).expand(feat_k.shape[0], -1)
            
            # Tensor Product interaction
            out_k = self.tp(feat_k, geom_k, w_k)
            
            out_accum = out_accum + out_k
            
        # 3. Reshape back to Image
        # out_accum: (B*L, C_out)
        out = out_accum.view(B, L, -1).permute(0, 2, 1) # (B, C_out, L)
        
        H_out = int(math.sqrt(L)) # Assuming square for simple logic
        out = out.view(B, -1, H_out, H_out)
        
        return out

# ==============================================================================
# 2. THE DEEP INTERACTION NETWORK (4x4 Super-Resolution)
# ==============================================================================
class FCCInteractionNet(nn.Module):
    def __init__(self, physics, depth=4, scale_factor=4, static_upsample=False, 
                 strict_equivariance=True):
        """
        Args:
            physics: Crystal physics configuration (unused for now)
            depth: Number of interaction layers
            scale_factor: Super-resolution factor (default 4x)
            static_upsample: If True, use bilinear interpolation (no learnable params).
                           If False, use learned equivariant pixel shuffle.
            strict_equivariance: If True, use TrueEquivariantConv (guaranteed equivariant).
                                If False, use EquivariantGridConv (breaks equivariance).
        """
        super().__init__()
        self.physics = physics
        self.scale_factor = scale_factor
        self.static_upsample = static_upsample
        self.strict_equivariance = strict_equivariance
        
        # Choose conv layer type
        ConvLayer = TrueEquivariantConv if strict_equivariance else EquivariantGridConv
        
        # The Language of the Crystal
        self.fcc_irreps = o3.Irreps("1x4e + 1x6e")  # 9 + 13 = 22 channels
        
        # FIX 1: Hidden irreps MUST stay in FCC manifold (4e + 6e only)
        # o3.Linear can only mix within the same L value, not create new L values.
        # Using L=0,1,2,3 in hidden would be zeros since they can't be created from L=4,6.
        # 
        # We increase multiplicity instead to add capacity:
        # 8x4e = 8*9 = 72 channels, 8x6e = 8*13 = 104 channels → 176 total
        self.hidden_irreps = o3.Irreps("8x4e + 8x6e")
        
        # A. Input Projection (FCC -> Hidden)
        self.entry_conv = ConvLayer(self.fcc_irreps, self.hidden_irreps)
        
        # B. Deep Interaction Layers (The "Thinking" Part) - BEFORE upsampling
        self.pre_upsample_layers = nn.ModuleList([
            ConvLayer(self.hidden_irreps, self.hidden_irreps)
            for _ in range(depth // 2)
        ])
        
        # C. Upsampling Stage (4x = 2x + 2x)
        # Two-stage upsampling for 4x: first 2x, then another 2x
        if static_upsample:
            # Static bilinear - guaranteed equivariant, no learning in upsample
            self.upsample1 = StaticEquivariantUpsample(scale_factor=2)
            self.upsample2 = StaticEquivariantUpsample(scale_factor=2)
        else:
            # Learned equivariant pixel shuffle
            self.upsample1 = EquivariantPixelShuffle(self.hidden_irreps, scale_factor=2)
            self.upsample2 = EquivariantPixelShuffle(self.hidden_irreps, scale_factor=2)
        
        self.post_upsample1_conv = ConvLayer(self.hidden_irreps, self.hidden_irreps)
        self.post_upsample2_conv = ConvLayer(self.hidden_irreps, self.hidden_irreps)
        
        # D. Deep Interaction Layers - AFTER upsampling (at high resolution)
        self.post_upsample_layers = nn.ModuleList([
            ConvLayer(self.hidden_irreps, self.hidden_irreps)
            for _ in range(depth - depth // 2)
        ])
        
        # E. Output Projection (Hidden -> FCC)
        # We force the output to be a valid FCC state (L=4, L=6)
        self.exit_conv = ConvLayer(self.hidden_irreps, self.fcc_irreps)
        
        # F. Non-Linearities (Gated)
        # Standard ReLU breaks geometry. We use Gate or Norm-Activation.
        # For simplicity in this demo, we assume linearity in the geometric path
        # or use Norm-based activation.
        self.act = NormActivation2D(self.hidden_irreps, torch.nn.functional.silu)

    def forward(self, f4, f6):
        # Input: Separate tensors (B, 9, H, W) and (B, 13, H, W)
        # Concatenate channel-wise to form the unified irrep tensor
        x_in = torch.cat([f4, f6], dim=1) # (B, 22, H, W)
        
        # GLOBAL SKIP: Upsample input for residual connection
        # This is critical for SR - network only needs to learn high-freq details
        x_skip = nn.functional.interpolate(
            x_in, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )
        
        # 1. Entry
        x = self.entry_conv(x_in)
        x = self.act(x)
        
        # 2. Pre-upsample processing (at low resolution - more efficient)
        for layer in self.pre_upsample_layers:
            x_res = x
            x = layer(x)
            x = self.act(x)
            x = x + x_res # ResNet connection
        
        # 3. First 2x Upsample: (B, C, H, W) -> (B, C, 2H, 2W)
        x = self.upsample1(x)
        x = self.post_upsample1_conv(x)
        x = self.act(x)
        
        # 4. Second 2x Upsample: (B, C, 2H, 2W) -> (B, C, 4H, 4W)
        x = self.upsample2(x)
        x = self.post_upsample2_conv(x)
        x = self.act(x)
        
        # 5. Post-upsample processing (at high resolution - refines details)
        for layer in self.post_upsample_layers:
            x_res = x
            x = layer(x)
            x = self.act(x)
            x = x + x_res # ResNet connection
            
        # 6. Exit (Project back to Crystal Manifold)
        x = self.exit_conv(x)
        
        # 7. GLOBAL RESIDUAL: Add upsampled input
        x = x + x_skip
        
        # Split back to f4, f6
        f4_out = x[:, :9, :, :]
        f6_out = x[:, 9:, :, :]
        
        return f4_out, f6_out

# ==============================================================================
# 3. USAGE EXAMPLE & EQUIVARIANCE TEST
# ==============================================================================
def test_equivariance(model, irreps_in, rtol=1e-3, atol=1e-4):
    """
    Test SO(3) equivariance of the network in FEATURE SPACE.
    
    IMPORTANT NOTE ON 2D SPATIAL vs 3D FEATURE EQUIVARIANCE:
    --------------------------------------------------------
    This network operates on 2D spatial grids but with 3D rotation-equivariant
    feature channels (spherical harmonics). The equivariance is in the 
    FEATURE CHANNEL dimension, not the spatial dimension.
    
    For FCC crystal super-resolution:
    - Features (f4, f6) are spherical harmonic coefficients describing orientations
    - Rotating a crystal orientation = applying Wigner-D to feature channels
    - The spatial layout of the 2D image is fixed (pixel grid)
    
    We test: f(D·x) ≈ D·f(x) where D is Wigner-D rotation in feature space.
    """
    print("\n" + "="*60)
    print("EQUIVARIANCE TEST (Feature-Space Rotation)")
    print("="*60)
    
    # Random rotation
    angles = torch.randn(3) * 2 * math.pi
    R = o3.angles_to_matrix(*angles)
    
    # Wigner D-matrices for rotating irreps
    D4 = o3.Irrep("4e").D_from_matrix(R)  # 9x9 for l=4
    D6 = o3.Irrep("6e").D_from_matrix(R)  # 13x13 for l=6
    
    # Test input (smaller for faster test)
    f4_in = torch.randn(1, 9, 8, 8)
    f6_in = torch.randn(1, 13, 8, 8)
    
    # Forward pass on original input
    model.eval()
    with torch.no_grad():
        f4_out, f6_out = model(f4_in, f6_in)
    
    # Rotate input features
    f4_rot = torch.einsum('ij, bjhw -> bihw', D4, f4_in)
    f6_rot = torch.einsum('ij, bjhw -> bihw', D6, f6_in)
    
    # Forward pass on rotated input
    with torch.no_grad():
        f4_out_from_rot, f6_out_from_rot = model(f4_rot, f6_rot)
    
    # Rotate original output
    f4_out_rotated = torch.einsum('ij, bjhw -> bihw', D4, f4_out)
    f6_out_rotated = torch.einsum('ij, bjhw -> bihw', D6, f6_out)
    
    # Check equivariance: f(D·x) ≈ D·f(x)
    f4_error = (f4_out_from_rot - f4_out_rotated).abs().max().item()
    f6_error = (f6_out_from_rot - f6_out_rotated).abs().max().item()
    
    # Use scale-invariant relative error
    f4_scale = max(f4_out_rotated.abs().max().item(), f4_out_from_rot.abs().max().item(), 1e-6)
    f6_scale = max(f6_out_rotated.abs().max().item(), f6_out_from_rot.abs().max().item(), 1e-6)
    
    f4_rel_error = f4_error / f4_scale
    f6_rel_error = f6_error / f6_scale
    
    print(f"L=4 (f4) Max Absolute Error: {f4_error:.2e}")
    print(f"L=4 (f4) Relative Error: {f4_rel_error:.2e}")
    print(f"L=6 (f6) Max Absolute Error: {f6_error:.2e}")
    print(f"L=6 (f6) Relative Error: {f6_rel_error:.2e}")
    
    is_equivariant = f4_rel_error < rtol and f6_rel_error < rtol
    
    if is_equivariant:
        print("✓ PASSED: Network is SO(3) equivariant in feature space!")
    else:
        print("✗ FAILED: Network breaks equivariance!")
        print("\n  NOTE: This is expected with the current EquivariantGridConv design.")
        print("  The spatial kernel uses fixed 2D directions which don't transform")
        print("  under 3D feature rotations. For strict equivariance, use:")
        print("  1. Purely pointwise operations (no spatial mixing), OR")
        print("  2. Scalar-only spatial convolutions + equivariant pointwise, OR")
        print("  3. Full 3D voxel representations")
    
    print("="*60 + "\n")
    return is_equivariant


def test_simple_equivariance():
    """
    Test equivariance of a minimal network with ONLY equivariant layers.
    This isolates the issue to help debug.
    """
    print("\n" + "="*60)
    print("SIMPLE EQUIVARIANCE TEST (Isolated Components)")
    print("="*60)
    
    from e3nn import o3, nn as e3nn_nn
    
    # Test just o3.Linear (should be perfectly equivariant)
    irreps = o3.Irreps("1x4e + 1x6e")
    hidden = o3.Irreps("4x0e + 4x2e + 2x4e + 2x6e")
    
    linear1 = o3.Linear(irreps, hidden)
    linear2 = o3.Linear(hidden, irreps)
    
    # Random rotation
    angles = torch.randn(3) * 2 * math.pi
    R = o3.angles_to_matrix(*angles)
    D4 = o3.Irrep("4e").D_from_matrix(R)
    D6 = o3.Irrep("6e").D_from_matrix(R)
    
    # Block diagonal D for full irreps
    D_in = torch.block_diag(D4, D6)
    
    # Test input
    x = torch.randn(10, irreps.dim)
    
    # Forward then rotate
    with torch.no_grad():
        y = linear2(linear1(x))
    y_rotated = y @ D_in.T
    
    # Rotate then forward
    x_rot = x @ D_in.T
    with torch.no_grad():
        y_from_rot = linear2(linear1(x_rot))
    
    error = (y_from_rot - y_rotated).abs().max().item()
    print(f"o3.Linear equivariance error: {error:.2e}")
    print("✓ o3.Linear is equivariant" if error < 1e-5 else "✗ o3.Linear has issues")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # First, verify e3nn components work correctly
    test_simple_equivariance()
    
    # Define FCC physics
    fcc_irreps = o3.Irreps("1x4e + 1x6e")
    
    # Dummy Input
    f4_in = torch.randn(1, 9, 16, 16)
    f6_in = torch.randn(1, 13, 16, 16)
    
    print("="*60)
    print("Testing STRICT EQUIVARIANT Network (TrueEquivariantConv)")
    print("  - Uses scalar-only spatial mixing")
    print("  - Guaranteed SO(3) equivariant in feature space")
    print("="*60)
    
    # Instantiate with STRICT equivariance
    model_strict = FCCInteractionNet(
        physics=None, depth=4, scale_factor=4, 
        static_upsample=True, strict_equivariance=True
    )
    
    print("Input Shapes:", f4_in.shape, f6_in.shape)
    print("Expected Output: 16x16 -> 64x64 (4x super-resolution)")
    
    f4_out, f6_out = model_strict(f4_in, f6_in)
    print("Output Shapes:", f4_out.shape, f6_out.shape)
    
    # Test equivariance
    test_equivariance(model_strict, fcc_irreps)
    
    print("="*60)
    print("Testing NON-STRICT Network (EquivariantGridConv)")
    print("  - Uses tensor product with spatial spherical harmonics")
    print("  - More expressive but breaks strict equivariance")
    print("="*60)
    
    # Instantiate WITHOUT strict equivariance (original design)
    model_original = FCCInteractionNet(
        physics=None, depth=4, scale_factor=4, 
        static_upsample=True, strict_equivariance=False
    )
    
    f4_out_o, f6_out_o = model_original(f4_in, f6_in)
    print("Output Shapes:", f4_out_o.shape, f6_out_o.shape)
    
    # Test equivariance
    test_equivariance(model_original, fcc_irreps)
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print("strict_equivariance=True:  Guaranteed SO(3) equivariant")
    print("strict_equivariance=False: More expressive, approx. equivariant")
    print("\n4x Super-Resolution Complete.")