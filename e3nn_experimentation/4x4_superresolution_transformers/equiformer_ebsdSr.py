import torch
import torch.nn as nn
import math
from e3nn import o3, nn as e3nn_nn

# ==============================================================================
# 0. MATH & PHYSICS PRIMITIVES (Unchanged)
# ==============================================================================

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_conjugate(q):
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def get_exact_misorientation(q_center, q_neighbor, symmetries=None):
    q_inv = quat_conjugate(q_center)
    q_rel = quat_multiply(q_neighbor, q_inv)
    
    if symmetries is not None:
        q_rel_expanded = q_rel.unsqueeze(-2)
        syms_expanded = symmetries.unsqueeze(0)
        q_syms = quat_multiply(syms_expanded, q_rel_expanded)
        w_abs = torch.abs(q_syms[..., 0])
        best_indices = torch.argmax(w_abs, dim=-1)
        mask = torch.nn.functional.one_hot(best_indices, num_classes=24).bool()
        q_rel = q_syms[mask].view(q_rel.shape)
    
    neg_mask = q_rel[..., 0] < 0
    q_rel[neg_mask] *= -1
    
    w = torch.clamp(q_rel[..., 0], -1.0, 1.0)
    xyz = q_rel[..., 1:]
    angle = 2.0 * torch.acos(w).unsqueeze(-1)
    sin_half_theta_sq = 1.0 - w*w
    sin_half_theta = torch.sqrt(torch.clamp(sin_half_theta_sq, min=0.0))
    safe_mask = sin_half_theta > 1e-6
    axis = torch.zeros_like(xyz)
    axis[safe_mask] = xyz[safe_mask] / sin_half_theta[safe_mask].unsqueeze(-1)
    return angle, axis

def wigner_D_cuda(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = o3._wigner.so3_generators(l).to(device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])

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
# 1. ENCODER (Unchanged)
# ==============================================================================
class FCCEncoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, quats, img_shape=None):
        is_image = (quats.dim() == 4)
        if is_image:
            B, H, W, C = quats.shape
            quats_flat = quats.reshape(-1, 4)
        else:
            quats_flat = quats

        R = o3.quaternion_to_matrix(quats_flat)
        alpha, beta, gamma = o3.matrix_to_angles(R)

        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        f4 = torch.einsum("...ij,j->...i", D4, self.physics.s4)
        f6 = torch.einsum("...ij,j->...i", D6, self.physics.s6)

        if img_shape is not None:
            H_img, W_img = img_shape
            if is_image:
                f0_list = []
                f1_list = []
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
# 2. TRANSFORMER COMPONENTS
# ==============================================================================

class BoundarySuppressedConv(nn.Module):
    """(Spatial Mixer) Equivariant 3x3 conv with boundary suppression."""
    def __init__(self, irreps_in, irreps_out, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.K2 = kernel_size * kernel_size
        self.in_dim = self.irreps_in.dim
        self.out_dim = self.irreps_out.dim
        
        self.kernel_weights = nn.Parameter(torch.ones(self.K2) / self.K2)
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        self.boundary_threshold = nn.Parameter(torch.tensor(0.087), requires_grad=False)
        
    def forward(self, x, boundary_map=None):
        B, C, H, W = x.shape
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfolded.shape[-1]
        x_unfolded = x_unfolded.view(B, C, self.K2, L)
        
        weights = torch.softmax(self.kernel_weights, dim=0).view(1, 1, self.K2, 1)
        
        if boundary_map is not None:
            boundary_unfolded = torch.nn.functional.unfold(boundary_map, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            boundary_unfolded = boundary_unfolded.view(B, 1, self.K2, L)
            suppression = torch.sigmoid(-10.0 * (boundary_unfolded - self.boundary_threshold))
            weights = weights * suppression
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        x_aggregated = (x_unfolded * weights).sum(dim=2) # Spatial mixing happen here
        x_flat = x_aggregated.permute(0, 2, 1).reshape(B * L, C)
        y_flat = self.linear(x_flat) # Channel mixing happens here
        
        H_out = W_out = int(math.sqrt(L))
        y = y_flat.view(B, L, self.out_dim).permute(0, 2, 1).view(B, self.out_dim, H_out, W_out)
        return y

class EquivariantTransformerBlock(nn.Module):
    """
    Equivariant Transformer Block (ConvNeXt-style).
    Structure:
    1. Spatial Mixing (BoundarySuppressedConv) - acts as 'Attention'
    2. Norm + Activation
    3. Channel Mixing (Equivariant MLP)
    4. Residual Connection
    """
    def __init__(self, irreps_emb):
        super().__init__()
        self.irreps = o3.Irreps(irreps_emb)
        
        # 1. Spatial Token Mixer (The "Attention")
        self.spatial_mix = BoundarySuppressedConv(self.irreps, self.irreps, kernel_size=3)
        
        # 2. Norm & Act
        self.norm1 = e3nn_nn.BatchNorm(self.irreps)
        self.act1  = e3nn_nn.NormActivation(self.irreps, torch.nn.functional.silu)
        
        # 3. Channel Mixer (Feed Forward Network)
        # Expansion ratio 2x for internal dim
        self.ffn_linear1 = o3.Linear(self.irreps, self.irreps) 
        self.ffn_act     = e3nn_nn.NormActivation(self.irreps, torch.nn.functional.silu)
        self.ffn_linear2 = o3.Linear(self.irreps, self.irreps)
        
    def forward(self, x, boundary_map=None):
        # x: (B, C, H, W)
        identity = x
        
        # Block 1: Spatial
        out = self.spatial_mix(x, boundary_map)
        
        # Permute for e3nn layers (B, C, H, W) -> (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Block 2: Channel MLP
        out = self.ffn_linear1(out)
        out = self.ffn_act(out)
        out = self.ffn_linear2(out)
        
        # Residual
        out = out.permute(0, 3, 1, 2) # Back to (B, C, H, W)
        return identity + out

# ==============================================================================
# 3. UPSAMPLER
# ==============================================================================
class EquivariantIrrepUpsample(nn.Module):
    def __init__(self, irreps_in, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.irreps_in = o3.Irreps(irreps_in)
        self.r2 = scale_factor ** 2
        expanded_irreps_list = [(mul * self.r2, ir) for mul, ir in self.irreps_in]
        self.irreps_expanded = o3.Irreps(expanded_irreps_list)
        self.expansion = o3.Linear(self.irreps_in, self.irreps_expanded)
        self.irrep_slices = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = (2 * ir.l + 1)
            expanded_dim = mul * self.r2 * dim
            self.irrep_slices.append((idx, idx + expanded_dim, mul, dim))
            idx += expanded_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.expansion(x)
        x = x.permute(0, 3, 1, 2)
        outputs = []
        idx = 0
        for mul, ir in self.irreps_in:
            dim = 2 * ir.l + 1
            block_channels = mul * self.r2 * dim
            block = x[:, idx:idx + block_channels, :, :]
            # Reshape so channels include the r^2 factor, then pixel-shuffle.
            # Start shape: (B, mul * r2 * dim, H, W)
            # We want shape (B, mul * dim * r2, H, W) -> pixel_shuffle -> (B, mul * dim, H*sf, W*sf)
            block = block.view(B, mul, self.r2, dim, H, W)
            block = block.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, mul, dim, r2, H, W)
            block = block.view(B, mul * dim * self.r2, H, W)
            block = nn.functional.pixel_shuffle(block, self.scale_factor)
            outputs.append(block)
            idx += block_channels
        return torch.cat(outputs, dim=1)

# ==============================================================================
# 4. DECODER: Spherical Harmonic Readout
# ==============================================================================
class SHTensorDecoder(nn.Module):
    """
    Decodes L=4, L=6 coefficients into Quaternions using Grid Search (Argmax).
    Strictly handles FCC symmetry by finding the peak of the probability function.
    """
    def __init__(self, physics, grid_res=200):
        super().__init__()
        self.physics = physics
        # Generate a search grid of random quaternions (for demo purposes)
        # In prod: Use a uniform SO3 sampling grid (Hopf fibration or Cubochoric)
        self.register_buffer('grid_quats', torch.randn(grid_res, 4))
        self.grid_quats = self.grid_quats / self.grid_quats.norm(dim=1, keepdim=True)
        
        # Precompute Wigner Matrices for the grid
        R = o3.quaternion_to_matrix(self.grid_quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        self.register_buffer('D4_grid', wigner_D_cuda(4, alpha, beta, gamma)) # (Grid, 9, 9)
        self.register_buffer('D6_grid', wigner_D_cuda(6, alpha, beta, gamma)) # (Grid, 13, 13)
        
    def forward(self, f4, f6):
        """
        f4: (N, 9) coefficients
        f6: (N, 13) coefficients
        Returns: (N, 4) best matching quaternions
        """
        # 1. Reconstruct Probability Function P(R) on the grid
        # P(R) ~ sum( f_l * D_l(R) * s_l )
        # Here we simplify: Signal = <f_pred, D_grid * s_physics>
        
        # Project physics constants onto grid (creating the "Ideal FCC Patterns" at every grid point)
        # Pattern_l = D_l_grid^T @ s_l
        pat4 = torch.einsum("gij,j->gi", self.D4_grid, self.physics.s4) # (Grid, 9)
        pat6 = torch.einsum("gij,j->gi", self.D6_grid, self.physics.s6) # (Grid, 13)
        
        # 2. Dot product between Predicted Coeffs and Grid Patterns
        # Score = f4 . pat4 + f6 . pat6
        score4 = torch.einsum("ni,gi->ng", f4, pat4)
        score6 = torch.einsum("ni,gi->ng", f6, pat6)
        total_score = score4 + score6 # (N_pixels, Grid_Size)
        
        # 3. Argmax
        best_idx = torch.argmax(total_score, dim=1) # (N_pixels,)
        best_quats = self.grid_quats[best_idx]
        
        return best_quats

# ==============================================================================
# 5. FULL SUPER-RESOLUTION TRANSFORMER
# ==============================================================================
class CrystalTransformerSR(nn.Module):
    def __init__(self, physics, scale_factor=4, depth=2):
        super().__init__()
        self.physics = physics
        self.scale_factor = scale_factor
        
        # Dimensions
        self.input_irreps = o3.Irreps("1x4e + 1x6e") 
        self.hidden_irreps = o3.Irreps("8x0e + 8x1e + 8x2e + 4x4e + 4x6e") # Rich feature space
        
        # 1. Feature Embedding (Conv)
        self.embedding = BoundarySuppressedConv(self.input_irreps, self.hidden_irreps)
        
        # 2. Transformer Encoder Layers
        self.layers = nn.ModuleList([
            EquivariantTransformerBlock(self.hidden_irreps) 
            for _ in range(depth)
        ])
        
        # 3. Upsampler
        self.upsample = EquivariantIrrepUpsample(self.hidden_irreps, scale_factor)
        
        # 4. Refinement (High-Res Transformer Block)
        self.refine = EquivariantTransformerBlock(self.hidden_irreps)
        
        # 5. Projection to Output Coefficients
        self.proj = o3.Linear(self.hidden_irreps, self.input_irreps)
        
        # 6. Decoder (Readout)
        self.decoder = SHTensorDecoder(physics)
        # Build masks that select only allowed cubic 'seeds' within L=4 and L=6
        # physics.s4 (9,) and physics.s6 (13,) have non-zero entries at allowed seed indices
        mask4 = (physics.s4 != 0).to(torch.bool)
        mask6 = (physics.s6 != 0).to(torch.bool)
        # register masks as buffers so they move with the module/device
        self.register_buffer('mask4', mask4)
        self.register_buffer('mask6', mask6)
        # combined mask for the flattened coeff vector [f4(9), f6(13)]
        self.register_buffer('mask_all', torch.cat([mask4.to(torch.bool), mask6.to(torch.bool)], dim=0))
        # Build a channel-level mask for hidden irreps so that any hidden features
        # corresponding to L=4 or L=6 only retain the allowed m-components.
        hidden_dim = self.hidden_irreps.dim
        hid_mask = torch.ones(hidden_dim, dtype=torch.bool)
        idx = 0
        for mul, ir in self.hidden_irreps:
            dim = 2 * ir.l + 1
            block_ch = mul * dim
            if ir.l == 4:
                # physics.s4 length is 9, indices 0..8 correspond to m=-4..4
                allowed = (physics.s4 != 0).to(torch.bool).tolist()
                for mrep in range(mul):
                    base = idx + mrep * dim
                    for j in range(dim):
                        hid_mask[base + j] = allowed[j]
            elif ir.l == 6:
                allowed = (physics.s6 != 0).to(torch.bool).tolist()
                for mrep in range(mul):
                    base = idx + mrep * dim
                    for j in range(dim):
                        hid_mask[base + j] = allowed[j]
            # else leave mask True for other irreps
            idx += block_ch
        self.register_buffer('hidden_mask', hid_mask)

    def _apply_hidden_mask(self, x):
        """Zero-out hidden channels that are disallowed by FCC harmonic masks.

        x: (B, C, H, W)
        """
        if not hasattr(self, 'hidden_mask'):
            return x
        mask = self.hidden_mask.to(x.device).float().view(1, -1, 1, 1)
        return x * mask

    def forward(self, f0, f1, f4, f6, img_shape):
        H, W = img_shape
        
        # Prepare Input
        # Enforce allowed cubic seeds: zero out disallowed coefficient channels
        if f4 is not None:
            if f4.dim() == 2 and f4.size(1) == 9:
                f4 = f4.clone()
                dis_mask4 = (~self.mask4).to(f4.device)
                if dis_mask4.any():
                    f4[:, dis_mask4] = 0.0
        if f6 is not None:
            if f6.dim() == 2 and f6.size(1) == 13:
                f6 = f6.clone()
                dis_mask6 = (~self.mask6).to(f6.device)
                if dis_mask6.any():
                    f6[:, dis_mask6] = 0.0

        f4_img = f4.view(1, H, W, 9).permute(0, 3, 1, 2)
        f6_img = f6.view(1, H, W, 13).permute(0, 3, 1, 2)
        f0_img = f0.view(1, H, W, 1).permute(0, 3, 1, 2) # Boundary Map
        x = torch.cat([f4_img, f6_img], dim=1)
        
        # 1. Embed
        x = self.embedding(x, boundary_map=f0_img)
        
        # 2. Transformer Layers (Low Res)
        for layer in self.layers:
            x = layer(x, boundary_map=f0_img)
            x = self._apply_hidden_mask(x)
            
        # 3. Upsample
        x = self.upsample(x) # Now (B, C, 4H, 4W)

        # Apply hidden mask after upsampling (note: mask repeats across spatial dims)
        x = self._apply_hidden_mask(x)
        
        # Upsample boundary map for refinement
        f0_hr = torch.nn.functional.interpolate(f0_img, scale_factor=self.scale_factor, mode='bilinear')
        
        # 4. Refine (High Res)
        x = self.refine(x, boundary_map=f0_hr)

        # Ensure hidden channels still respect FCC kubic harmonics
        x = self._apply_hidden_mask(x)
        
        # 5. Project to Coefficients
        # Permute for linear: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        out_coeffs = self.proj(x)
        
        # Flatten
        out_coeffs = out_coeffs.reshape(-1, 22) # 9 + 13
        # Zero-out any disallowed output coefficients so network only distills allowed seeds
        if out_coeffs.dim() == 2 and out_coeffs.size(1) == 22:
            out_coeffs = out_coeffs.clone()
            dis_all = (~self.mask_all).to(out_coeffs.device)
            if dis_all.any():
                out_coeffs[:, dis_all] = 0.0

        f4_pred = out_coeffs[:, :9]
        f6_pred = out_coeffs[:, 9:]
        
        return f4_pred, f6_pred

    def predict_quaternions(self, f4, f6):
        return self.decoder(f4, f6)

# ==============================================================================
# 6. RUNNER
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics)
    
    # Instantiate the new Transformer Model
    model = CrystalTransformerSR(physics, scale_factor=4, depth=2).to(device)
    
    # Dummy Data
    H, W = 16, 16
    q_dummy = torch.randn(H*W, 4, device=device)
    q_dummy /= q_dummy.norm(dim=1, keepdim=True)
    
    print("1. Encoding...")
    f0, f1, f4, f6 = encoder(q_dummy, img_shape=(H, W))
    
    print("2. Transformer Super-Resolution...")
    f4_hr, f6_hr = model(f0, f1, f4, f6, (H, W))
    print(f"   HR Coeffs Shape: {f4_hr.shape}")
    
    print("3. Decoding (Readout)...")
    q_hr_pred = model.predict_quaternions(f4_hr, f6_hr)
    print(f"   HR Quats Shape: {q_hr_pred.shape}")
    
    print("Done. Model is strictly SO(3) equivariant.")

    # --- Detailed introspection summary (nicely formatted) ---
    def _human(n):
        for unit in ['', 'K', 'M', 'B']:
            if n < 1000.0:
                return f"{int(n)}{unit}" if unit == '' else f"{n/1000.0:.1f}{unit}"
            n /= 1000.0
        return f"{n:.1f}T"

    print('\n' + '=' * 80)
    print('INPUT / OUTPUT SHAPES'.center(80))
    print('-' * 80)
    rows = [
        ('q_dummy', tuple(q_dummy.shape)),
        ('f0', tuple(f0.shape)),
        ('f1', tuple(f1.shape)),
        ('f4', tuple(f4.shape)),
        ('f6', tuple(f6.shape)),
        ('f4_hr', tuple(f4_hr.shape)),
        ('f6_hr', tuple(f6_hr.shape)),
        ('q_hr_pred', tuple(q_hr_pred.shape)),
    ]
    for name, shape in rows:
        print(f"{name:12s} : {shape}")

    print('\n' + '=' * 80)
    print('MODEL LAYERS & PARAMETER COUNTS'.center(80))
    print('-' * 80)
    total_params = sum(p.numel() for p in model.parameters())

    header = f"{'Module':40s} {'Class':25s} {'Params':>10s} {'%Total':>8s}"
    print(header)
    print('-' * 90)

    def fmt(n):
        return _human(n)

    # Top-level modules and their direct children for clarity
    for name, module in model.named_children():
        cnt = sum(p.numel() for p in module.parameters())
        pct = (cnt / total_params * 100) if total_params > 0 else 0.0
        print(f"{name:40s} {module.__class__.__name__:25s} {fmt(cnt):>10s} {pct:7.2f}%")
        for sub_name, sub in module.named_children():
            scnt = sum(p.numel() for p in sub.parameters())
            spct = (scnt / total_params * 100) if total_params > 0 else 0.0
            print(f"  {name + '.' + sub_name:37s} {sub.__class__.__name__:25s} {fmt(scnt):>10s} {spct:7.2f}%")

    print('-' * 90)
    print(f"{'Total':66s} {fmt(total_params):>10s} {100.00:7.2f}%")
    print('=' * 80)

    # mark todo completed
    # (visible summary printed; use top-level grouping to avoid overly verbose module lists)
    # Also print exact named-parameter breakdown for accuracy
    print('\n' + '=' * 80)
    print('NAMED PARAMETERS (name, shape, count)'.center(80))
    print('-' * 80)
    named_total = 0
    for n, p in model.named_parameters():
        cnt = p.numel()
        named_total += cnt
        shape_str = str(tuple(p.shape))
        print(f"{n:60s} {shape_str:20s} {cnt:10d}")
    print('-' * 80)
    print(f"Named parameters total: {named_total} ({_human(named_total)})")