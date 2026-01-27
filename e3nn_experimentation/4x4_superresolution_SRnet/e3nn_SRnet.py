import torch
import torch.nn as nn
import math
from e3nn import o3, nn as e3nn_nn

# ==============================================================================
# 0. RIGOROUS QUATERNION ALGEBRA (From boundary_detect_autoencoder.py)
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
    q_inv = quat_conjugate(q_center)
    q_rel = quat_multiply(q_neighbor, q_inv)
    
    # 2. Symmetry Handling (Disorientation)
    if symmetries is not None:
        q_rel_expanded = q_rel.unsqueeze(-2)
        syms_expanded = symmetries.unsqueeze(0)
        q_syms = quat_multiply(syms_expanded, q_rel_expanded)
        w_abs = torch.abs(q_syms[..., 0])
        best_indices = torch.argmax(w_abs, dim=-1)
        mask = torch.nn.functional.one_hot(best_indices, num_classes=24).bool()
        q_rel = q_syms[mask].view(q_rel.shape)
    
    # 3. Double Cover Handling
    neg_mask = q_rel[..., 0] < 0
    q_rel[neg_mask] *= -1
    
    # 4. Extract Angle and Axis
    w = torch.clamp(q_rel[..., 0], -1.0, 1.0)
    xyz = q_rel[..., 1:]
    angle = 2.0 * torch.acos(w).unsqueeze(-1)
    sin_half_theta_sq = 1.0 - w*w
    sin_half_theta = torch.sqrt(torch.clamp(sin_half_theta_sq, min=0.0))
    safe_mask = sin_half_theta > 1e-6
    axis = torch.zeros_like(xyz)
    axis[safe_mask] = xyz[safe_mask] / sin_half_theta[safe_mask].unsqueeze(-1)
    
    return angle, axis

# ==============================================================================
# 0.1 CUDA-Compatible Wigner D Function
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
# 0.2 PHYSICS CONSTANTS (FCC Crystal)
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
# 0.3 ENCODER: Quaternions -> (f0, f1, f4, f6) with Boundary Features
# ==============================================================================
class FCCEncoder(nn.Module):
    """
    Encodes quaternions into spherical harmonic features.
    - L=0, L=1: Boundary features (misorientation angle and axis)
    - L=4, L=6: Orientation features (FCC invariant descriptors)
    """
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, quats, img_shape=None):
        """
        Args:
            quats: (B, 4) or (B, H, W, 4) Normalized quaternions.
            img_shape: Tuple (H, W). If provided, calculates spatial boundary features.
        """
        # Handle image-batched inputs by flattening spatial dims
        is_image = (quats.dim() == 4)
        if is_image:
            B, H, W, C = quats.shape
            quats_flat = quats.reshape(-1, 4)
        else:
            quats_flat = quats

        # A. ORIENTATION FEATURES (L=4, L=6)
        R = o3.quaternion_to_matrix(quats_flat)
        alpha, beta, gamma = o3.matrix_to_angles(R)

        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)

        # B. BOUNDARY FEATURES (L=0, L=1) - CALCULATED RIGOROUSLY
        if img_shape is not None:
            H_img, W_img = img_shape
            if is_image:
                f0_list = []
                f1_list = []
                # compute per-image boundary features and append
                for b in range(B):
                    q_grid = quats[b]
                    q_right = torch.roll(q_grid, shifts=-1, dims=1)
                    q_down = torch.roll(q_grid, shifts=-1, dims=0)
                    ang_x, axis_x = get_exact_misorientation(q_grid, q_right, self.physics.fcc_syms)
                    ang_y, axis_y = get_exact_misorientation(q_grid, q_down, self.physics.fcc_syms)
                    ang_x[:, -1] = 0; axis_x[:, -1] = 0
                    ang_y[-1, :] = 0; axis_y[-1, :] = 0
                    f0 = ((ang_x + ang_y) / 2.0).view(-1, 1)
                    f1 = ((axis_x + axis_y) / 2.0).view(-1, 3)
                    f0_list.append(f0)
                    f1_list.append(f1)
                f0 = torch.cat(f0_list, dim=0)
                f1 = torch.cat(f1_list, dim=0)
            else:
                q_grid = quats_flat.view(H_img, W_img, 4)
                q_right = torch.roll(q_grid, shifts=-1, dims=1)
                q_down = torch.roll(q_grid, shifts=-1, dims=0)
                ang_x, axis_x = get_exact_misorientation(q_grid, q_right, self.physics.fcc_syms)
                ang_y, axis_y = get_exact_misorientation(q_grid, q_down, self.physics.fcc_syms)
                ang_x[:, -1] = 0; axis_x[:, -1] = 0
                ang_y[-1, :] = 0; axis_y[-1, :] = 0
                f0 = ((ang_x + ang_y) / 2.0).view(-1, 1)
                f1 = ((axis_x + axis_y) / 2.0).view(-1, 3)
        else:
            if is_image:
                f0 = torch.zeros(B * H * W, 1, device=quats.device)
                f1 = torch.zeros(B * H * W, 3, device=quats.device)
            else:
                B_flat = quats_flat.shape[0]
                f0 = torch.zeros(B_flat, 1, device=quats.device)
                f1 = torch.zeros(B_flat, 3, device=quats.device)

        return f0, f1, f4, f6

# ==============================================================================
# 0.4 HELPER: NormActivation wrapper for image tensors
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
# 0.5 SIMPLE BILINEAR UPSAMPLING (No skip connections)
# ==============================================================================
class EquivariantIrrepUpsample(nn.Module):
    """
    Equivariant upsampling for irreps using o3.Linear to expand channels, then pixel shuffle.
    This preserves equivariance by only mixing within irrep types and rearranging spatially.
    """
    def __init__(self, irreps_in, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.irreps_in = o3.Irreps(irreps_in)
        self.r2 = scale_factor ** 2
        # Output irreps are the same as input (just rearranged spatially)
        self.irreps_out = self.irreps_in
        # Expand each irrep multiplicity by r^2
        expanded_irreps_list = []
        for mul, ir in self.irreps_in:
            expanded_irreps_list.append((mul * self.r2, ir))
        self.irreps_expanded = o3.Irreps(expanded_irreps_list)
        self.expansion = o3.Linear(self.irreps_in, self.irreps_expanded)
        # Track channel structure for pixel shuffle
        self.irrep_slices = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = (2 * ir.l + 1)
            expanded_dim = mul * self.r2 * dim
            self.irrep_slices.append((idx, idx + expanded_dim, mul, dim))
            idx += expanded_dim

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 1. Permute for e3nn: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # 2. Equivariant channel expansion: (B, H, W, C) -> (B, H, W, C*r^2)
        x = self.expansion(x)
        # 3. Permute back: (B, H, W, C*r^2) -> (B, C*r^2, H, W)
        x = x.permute(0, 3, 1, 2)
        # 4. Pixel shuffle each irrep block separately
        outputs = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = 2 * ir.l + 1
            block_channels = mul * self.r2 * dim
            block = x[:, idx:idx + block_channels, :, :]
            block = block.view(B, mul * self.r2, dim, H, W)
            block = block.view(B, mul, self.r2, dim, H, W)
            block = block.permute(0, 1, 3, 2, 4, 5)
            block = block.reshape(B, mul * dim * self.r2, H, W)
            block = block.view(B, mul * dim, self.r2, H, W)
            block = block.permute(0, 2, 1, 3, 4)
            block = block.reshape(B, self.r2 * mul * dim, H, W)
            block = nn.functional.pixel_shuffle(block, self.scale_factor)
            outputs.append(block)
            idx += block_channels
        x = torch.cat(outputs, dim=1)
        return x

# ==============================================================================
# 1. BOUNDARY-SUPPRESSED EQUIVARIANT CONVOLUTION
# ==============================================================================
class BoundarySuppressedConv(nn.Module):
    """
    Equivariant 3x3 convolution with learned weights that suppresses 
    interaction at grain boundaries.
    
    Uses L=0 (scalar misorientation) to suppress mixing at boundaries.
    Learns weights for 3x3 kernel interactions.
    """
    def __init__(self, irreps_in, irreps_out, kernel_size=3, padding=1, stride=1):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.K2 = kernel_size * kernel_size  # 9 for 3x3
        
        self.in_dim = self.irreps_in.dim
        self.out_dim = self.irreps_out.dim
        
        # Learnable kernel weights for 3x3 spatial interactions
        # Each neighbor position gets a weight that scales its contribution
        self.kernel_weights = nn.Parameter(torch.ones(self.K2) / self.K2)
        
        # Equivariant linear transformation applied after spatial aggregation
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        
        # Boundary suppression threshold (radians) - ~5 degrees
        self.boundary_threshold = nn.Parameter(torch.tensor(0.087), requires_grad=False)
        
    def forward(self, x, boundary_map=None):
        """
        x: (B, C, H, W) where C = irreps_in.dim
        boundary_map: (B, 1, H, W) L=0 misorientation angles in radians
        """
        B, C, H, W = x.shape
        
        # 1. Unfold: extract K2 neighbors for each pixel
        x_unfolded = torch.nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride
        )
        L = x_unfolded.shape[-1]  # H_out * W_out
        
        # Reshape to (B, C, K2, L)
        x_unfolded = x_unfolded.view(B, C, self.K2, L)
        
        # 2. Compute spatial weights
        weights = torch.softmax(self.kernel_weights, dim=0)  # (K2,)
        weights = weights.view(1, 1, self.K2, 1)  # (1, 1, K2, 1) for broadcasting
        
        # 3. If boundary map provided, suppress interactions at boundaries
        if boundary_map is not None:
            # Unfold boundary map same way
            boundary_unfolded = torch.nn.functional.unfold(
                boundary_map,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride
            )  # (B, K2, L)
            boundary_unfolded = boundary_unfolded.view(B, 1, self.K2, L)
            
            # Compute suppression factor: 1 at low misorientation, 0 at high
            # Using smooth sigmoid-like suppression
            suppression = torch.sigmoid(
                -10.0 * (boundary_unfolded - self.boundary_threshold)
            )  # (B, 1, K2, L)
            
            # Apply suppression to weights
            weights = weights * suppression
            
            # Renormalize weights so they sum to 1 at each pixel
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        # 4. Weighted sum of neighbor features
        x_aggregated = (x_unfolded * weights).sum(dim=2)  # (B, C, L)
        
        # 5. Reshape for e3nn: (B, C, L) -> (B*L, C)
        x_flat = x_aggregated.permute(0, 2, 1).reshape(B * L, C)
        
        # 6. Apply equivariant linear transformation
        y_flat = self.linear(x_flat)  # (B*L, C_out)
        
        # 7. Reshape back to image
        H_out = W_out = int(math.sqrt(L))
        y = y_flat.view(B, L, self.out_dim).permute(0, 2, 1)  # (B, C_out, L)
        y = y.view(B, self.out_dim, H_out, W_out)
        
        return y


# ==============================================================================
# 2. SIMPLE SUPER-RESOLUTION NETWORK (Depth=1, No Skip Connections)
# ==============================================================================
class SimpleSRNet(nn.Module):
    """
    Simple equivariant super-resolution network.
    
    Design:
    - L=0/L=1 features for boundary detection (from encoder)
    - L=4/L=6 features for orientation (main learning)
    - 1 hidden layer with 3x3 kernel, boundary-suppressed interactions
    - Simple bilinear upsampling (no skip connections)
    - Loss computed in L=4/L=6 space over entire image
    """
    def __init__(self, physics, scale_factor=4):
        """
        Args:
            physics: FCCPhysics instance
            scale_factor: Super-resolution factor (default 4x)
        """
        super().__init__()
        self.physics = physics
        self.scale_factor = scale_factor
        
        # Irreps: only L=4 and L=6 for orientation learning
        # L=0, L=1 are used for boundary detection (passed through encoder)
        self.orientation_irreps = o3.Irreps("1x4e + 1x6e")  # 9 + 13 = 22 channels
        
        # Hidden irreps: increased multiplicity for capacity
        self.hidden_irreps = o3.Irreps("4x4e + 4x6e")  # 36 + 52 = 88 channels
        
        # Equivariant upsampling (learned channel expansion, spatial rearrangement)
        self.upsample = EquivariantIrrepUpsample(self.hidden_irreps, scale_factor=scale_factor)
        
        # Single hidden layer with boundary-suppressed 3x3 convolution
        self.entry_conv = BoundarySuppressedConv(
            self.orientation_irreps, 
            self.hidden_irreps, 
            kernel_size=3
        )
        
        self.exit_conv = BoundarySuppressedConv(
            self.hidden_irreps, 
            self.orientation_irreps, 
            kernel_size=3
        )
        
        # Activation (norm-based, preserves equivariance)
        self.act = NormActivation2D(self.hidden_irreps, torch.nn.functional.silu)
        
    def forward(self, f0, f1, f4, f6, img_shape):
        """
        Args:
            f0: (B, 1) L=0 boundary scalar (misorientation angle)
            f1: (B, 3) L=1 boundary vector (misorientation axis)  
            f4: (B, 9) L=4 orientation features
            f6: (B, 13) L=6 orientation features
            img_shape: (H, W) spatial dimensions
            
        Returns:
            f4_out: (B_out, 9) upsampled L=4 features
            f6_out: (B_out, 13) upsampled L=6 features
            f0_out: (B_out, 1) upsampled boundary features
            f1_out: (B_out, 3) upsampled boundary axis
        """
        H, W = img_shape
        B = f4.shape[0]
        
        # Reshape to image format: (1, C, H, W)
        f4_img = f4.view(1, H, W, 9).permute(0, 3, 1, 2)  # (1, 9, H, W)
        f6_img = f6.view(1, H, W, 13).permute(0, 3, 1, 2)  # (1, 13, H, W)
        f0_img = f0.view(1, H, W, 1).permute(0, 3, 1, 2)  # (1, 1, H, W)
        f1_img = f1.view(1, H, W, 3).permute(0, 3, 1, 2)  # (1, 3, H, W)
        
        # Concatenate orientation features
        x = torch.cat([f4_img, f6_img], dim=1)  # (1, 22, H, W)
        
        # 1. Entry conv with boundary suppression (at low resolution)
        x = self.entry_conv(x, boundary_map=f0_img)
        x = self.act(x)
        
        # 2. Equivariant upsampling for orientation features
        x = self.upsample(x)

        # 3. Exit conv (at high resolution)
        # Upsample boundary map for high-res suppression (use bilinear for scalar/vector fields)
        scale = self.scale_factor
        f0_hr = torch.nn.functional.interpolate(f0_img, scale_factor=scale, mode='bilinear', align_corners=False)
        x = self.exit_conv(x, boundary_map=f0_hr)

        # Split output back to f4, f6
        f4_out = x[:, :9, :, :]
        f6_out = x[:, 9:, :, :]

        # Upsample boundary features (simple bilinear, no learning)
        f0_out = torch.nn.functional.interpolate(f0_img, scale_factor=scale, mode='bilinear', align_corners=False)
        f1_out = torch.nn.functional.interpolate(f1_img, scale_factor=scale, mode='bilinear', align_corners=False)
        
        # Reshape to flat format for decoder
        H_out, W_out = H * self.scale_factor, W * self.scale_factor
        f4_out = f4_out.permute(0, 2, 3, 1).reshape(-1, 9)
        f6_out = f6_out.permute(0, 2, 3, 1).reshape(-1, 13)
        f0_out = f0_out.permute(0, 2, 3, 1).reshape(-1, 1)
        f1_out = f1_out.permute(0, 2, 3, 1).reshape(-1, 3)
        
        return f4_out, f6_out, f0_out, f1_out


# ==============================================================================
# 3. L4/L6 SPACE LOSS FUNCTION
# ==============================================================================
class SphericalHarmonicLoss(nn.Module):
    """
    Loss computed in L=4/L=6 spherical harmonic space.
    
    Simple MSE loss between predicted and target features in the 
    invariant descriptor space, computed over the entire image.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, f4_pred, f6_pred, f4_target, f6_target):
        """
        Args:
            f4_pred: (B, 9) predicted L=4 features
            f6_pred: (B, 13) predicted L=6 features
            f4_target: (B, 9) target L=4 features
            f6_target: (B, 13) target L=6 features
            
        Returns:
            loss: scalar loss value
        """
        # L4 loss
        l4_loss = torch.mean((f4_pred - f4_target) ** 2)
        
        # L6 loss  
        l6_loss = torch.mean((f6_pred - f6_target) ** 2)
        
        # Combined loss (can weight differently if needed)
        total_loss = l4_loss + l6_loss
        
        return total_loss, {'l4_loss': l4_loss.item(), 'l6_loss': l6_loss.item()}
    
    # Commented out misorientation loss for now
    # def misorientation_loss(self, q_pred, q_target, symmetries):
    #     """
    #     Loss based on misorientation angle between predicted and target quaternions.
    #     Takes into account crystal symmetry.
    #     """
    #     # TODO: Implement if needed
    #     pass


# ==============================================================================
# 4. EQUIVARIANCE TEST
# ==============================================================================
def test_simple_srnet_equivariance(model, physics, rtol=1e-3):
    """
    Test SO(3) equivariance of the SimpleSRNet in FEATURE SPACE.
    """
    print("\n" + "="*60)
    print("EQUIVARIANCE TEST (SimpleSRNet)")
    print("="*60)
    
    # Random rotation
    angles = torch.randn(3) * 2 * math.pi
    R = o3.angles_to_matrix(*angles)
    
    # Wigner D-matrices for rotating irreps
    D4 = o3.Irrep("4e").D_from_matrix(R)  # 9x9 for l=4
    D6 = o3.Irrep("6e").D_from_matrix(R)  # 13x13 for l=6
    
    # Test input (flat format)
    H, W = 8, 8
    B = H * W
    f0 = torch.randn(B, 1)
    f1 = torch.randn(B, 3)
    f4 = torch.randn(B, 9)
    f6 = torch.randn(B, 13)
    
    # Forward pass on original input
    model.eval()
    with torch.no_grad():
        f4_out, f6_out, _, _ = model(f0, f1, f4, f6, (H, W))
    
    # Rotate input features (only L=4 and L=6)
    f4_rot = f4 @ D4.T
    f6_rot = f6 @ D6.T
    
    # Forward pass on rotated input
    with torch.no_grad():
        f4_out_from_rot, f6_out_from_rot, _, _ = model(f0, f1, f4_rot, f6_rot, (H, W))
    
    # Rotate original output
    f4_out_rotated = f4_out @ D4.T
    f6_out_rotated = f6_out @ D6.T
    
    # Check equivariance: f(D·x) ≈ D·f(x)
    f4_error = (f4_out_from_rot - f4_out_rotated).abs().max().item()
    f6_error = (f6_out_from_rot - f6_out_rotated).abs().max().item()
    
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
    
    print("="*60 + "\n")
    return is_equivariant


# ==============================================================================
# 5. MAIN RUNNER
# ==============================================================================
if __name__ == "__main__":
    import numpy as np
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize physics
    physics = FCCPhysics(device)
    
    # Initialize encoder
    encoder = FCCEncoder(physics)
    
    # Initialize SimpleSRNet
    model = SimpleSRNet(physics, scale_factor=4)
    model = model.to(device)
    
    # Initialize loss function
    loss_fn = SphericalHarmonicLoss()
    
    print("="*60)
    print("SimpleSRNet Architecture Summary")
    print("="*60)
    print("- L=0/L=1: Boundary features (misorientation angle/axis)")
    print("- L=4/L=6: Orientation features (FCC invariant descriptors)")
    print("- 1 hidden layer with boundary-suppressed 3x3 convolution")
    print("- Simple bilinear 4x upsampling (no skip connections)")
    print("- Loss computed in L=4/L=6 space over entire image")
    print("="*60)
    
    # Test with dummy data
    H, W = 16, 16
    B = H * W
    
    # Dummy quaternions (random normalized)
    q_dummy = torch.randn(B, 4, device=device)
    q_dummy = q_dummy / q_dummy.norm(dim=1, keepdim=True)
    
    print(f"\nTest Input: {H}x{W} quaternion image")
    
    # Encode to spherical harmonics
    with torch.no_grad():
        f0, f1, f4, f6 = encoder(q_dummy, img_shape=(H, W))
    
    print(f"Encoded features:")
    print(f"  f0 (L=0 boundary): {f0.shape}")
    print(f"  f1 (L=1 boundary): {f1.shape}")
    print(f"  f4 (L=4 orient):   {f4.shape}")
    print(f"  f6 (L=6 orient):   {f6.shape}")
    
    # Forward pass through SR network
    with torch.no_grad():
        f4_out, f6_out, f0_out, f1_out = model(f0, f1, f4, f6, (H, W))
    
    H_out, W_out = H * 4, W * 4
    print(f"\nOutput: {H_out}x{W_out} super-resolved features")
    print(f"  f4_out: {f4_out.shape}")
    print(f"  f6_out: {f6_out.shape}")
    print(f"  f0_out: {f0_out.shape}")
    print(f"  f1_out: {f1_out.shape}")
    
    # Dummy target for loss computation
    f4_target = torch.randn_like(f4_out)
    f6_target = torch.randn_like(f6_out)
    
    # Compute loss in L=4/L=6 space
    loss, loss_dict = loss_fn(f4_out, f6_out, f4_target, f6_target)
    print(f"\nSpherical Harmonic Loss:")
    print(f"  L4 loss: {loss_dict['l4_loss']:.6f}")
    print(f"  L6 loss: {loss_dict['l6_loss']:.6f}")
    print(f"  Total:   {loss.item():.6f}")
    
    # Test equivariance
    model_cpu = SimpleSRNet(physics, scale_factor=4)
    test_simple_srnet_equivariance(model_cpu, physics)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("SimpleSRNet Ready for Training!")
    print("="*60)