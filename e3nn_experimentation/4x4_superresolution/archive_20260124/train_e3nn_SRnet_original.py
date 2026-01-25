"""
Training script for e3nn-based 4x Super-Resolution Network
Trains both strict equivariant and non-strict configurations
Saves IPF maps comparing input (subsampled) and output quaternion maps

ORIGINAL VERSION - Restored from outputs_20260124_163036
Uses HR subsampling to create LR input (not separate LR_Data folder)
Uses 2-row Input vs Output visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent paths for imports
sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')

from e3nn import o3
from orix.crystal_map import Phase

# Import the network
from e3nn_SRnet import FCCInteractionNet, TrueEquivariantConv, EquivariantGridConv

# ==============================================================================
# CUDA-Compatible Wigner D Function
# ==============================================================================
def wigner_D_cuda(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """CUDA-compatible wrapper for e3nn's wigner_D function."""
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    
    X = o3._wigner.so3_generators(l).to(device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])


# ==============================================================================
# PHYSICS CONSTANTS
# ==============================================================================
class FCCPhysics(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # FCC Seeds for encoding
        self.s4 = torch.zeros(9, device=device)
        self.s4[4] = 0.7638
        self.s4[8] = 0.6455
        
        self.s6 = torch.zeros(13, device=device)
        self.s6[6] = 0.3536
        self.s6[10] = -0.9354
        
        # FCC Symmetry Group (24 elements)
        inv_sqrt_2 = 1 / math.sqrt(2)
        half = 0.5
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
# ENCODER: Quaternion -> Spherical Harmonic Features
# ==============================================================================
class FCCEncoder(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, quats):
        """
        quats: (B, H, W, 4) or (B, 4) quaternions
        Returns: f4 (B, H, W, 9) or (B, 9), f6 (B, H, W, 13) or (B, 13)
        """
        original_shape = quats.shape
        is_image = quats.dim() == 4
        
        if is_image:
            B, H, W, _ = quats.shape
            quats = quats.reshape(-1, 4)
        
        R = o3.quaternion_to_matrix(quats)
        alpha, beta, gamma = o3.matrix_to_angles(R)
        
        D4 = wigner_D_cuda(4, alpha, beta, gamma)
        D6 = wigner_D_cuda(6, alpha, beta, gamma)
        
        f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
        f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)
        
        if is_image:
            f4 = f4.reshape(B, H, W, 9)
            f6 = f6.reshape(B, H, W, 13)
        
        return f4, f6


# ==============================================================================
# DECODER: Spherical Harmonic Features -> Quaternion
# ==============================================================================
class SphericalSamplingDecoder(nn.Module):
    def __init__(self, physics, n_samples=1000):
        super().__init__()
        self.n_samples = n_samples
        self.physics = physics
        
        # Fibonacci sphere sampling
        self.register_buffer('grid_vecs', self._fibonacci_sphere(n_samples, physics.device))
        self.register_buffer('Y4_grid', o3.spherical_harmonics(4, self.grid_vecs, normalize=True))
    
    def forward(self, f4, f6):
        """
        f4: (B, H, W, 9) or (B, 9)
        f6: (B, H, W, 13) or (B, 13)
        Returns: quaternions matching input shape
        """
        original_shape = f4.shape
        is_image = f4.dim() == 4
        
        if is_image:
            B, H, W, _ = f4.shape
            f4 = f4.reshape(-1, 9)
            f6 = f6.reshape(-1, 13)
        
        batch_size = f4.shape[0]
        
        # Evaluate signal on sphere
        signal = torch.einsum("bi,gi->bg", f4, self.Y4_grid)
        
        # Find Z-axis (primary peak)
        z_vals, z_indices = torch.max(signal, dim=1)
        z_axis = self.grid_vecs[z_indices]
        
        # Find X-axis (orthogonal peak)
        dots = torch.einsum("bij,bij->bi", 
                          self.grid_vecs.unsqueeze(0).expand(batch_size, -1, -1),
                          z_axis.unsqueeze(1).expand(-1, self.n_samples, -1))
        mask = (dots.abs() < 0.2)
        masked_signal = signal.clone()
        masked_signal[~mask] = -float('inf')
        x_vals, x_indices = torch.max(masked_signal, dim=1)
        x_axis = self.grid_vecs[x_indices]
        
        # Gram-Schmidt orthogonalization
        z_axis = torch.nn.functional.normalize(z_axis, dim=-1)
        proj = torch.sum(x_axis * z_axis, dim=-1, keepdim=True) * z_axis
        x_axis = torch.nn.functional.normalize(x_axis - proj, dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        
        R_rec = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        quats = o3.matrix_to_quaternion(R_rec)
        
        if is_image:
            quats = quats.reshape(B, H, W, 4)
        
        return quats
    
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
# DATASET
# ==============================================================================
class QuaternionSRDataset(Dataset):
    """Dataset for quaternion super-resolution"""
    def __init__(self, data_dir, scale_factor=4, device='cpu'):
        self.scale_factor = scale_factor
        self.device = device
        
        # Load all HR quaternion maps
        self.hr_files = []
        for f in os.listdir(data_dir):
            if f.endswith('.npy'):
                self.hr_files.append(os.path.join(data_dir, f))
        
        print(f"Found {len(self.hr_files)} quaternion maps")
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load HR quaternion map
        hr_quats = np.load(self.hr_files[idx])  # (H, W, 4)
        hr_quats = torch.tensor(hr_quats, dtype=torch.float32)
        
        # Normalize quaternions
        hr_quats = hr_quats / torch.norm(hr_quats, dim=-1, keepdim=True)
        
        # Create LR by subsampling
        H, W = hr_quats.shape[:2]
        lr_quats = hr_quats[::self.scale_factor, ::self.scale_factor, :]  # (H/4, W/4, 4)
        
        return lr_quats, hr_quats


def create_synthetic_data(num_samples=100, hr_size=64, device='cpu'):
    """Create synthetic quaternion data for testing"""
    data = []
    for _ in range(num_samples):
        # Create random orientation field with some spatial coherence
        hr_quats = torch.randn(hr_size, hr_size, 4, device=device)
        # Add spatial smoothing for coherence
        hr_quats = torch.nn.functional.avg_pool2d(
            hr_quats.permute(2, 0, 1).unsqueeze(0), 
            kernel_size=3, stride=1, padding=1
        ).squeeze(0).permute(1, 2, 0)
        hr_quats = hr_quats / torch.norm(hr_quats, dim=-1, keepdim=True)
        data.append(hr_quats)
    return data


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================
class SphericalHarmonicLoss(nn.Module):
    """Loss in spherical harmonic feature space"""
    def __init__(self, alpha_f4=1.0, alpha_f6=1.0):
        super().__init__()
        self.alpha_f4 = alpha_f4
        self.alpha_f6 = alpha_f6
    
    def forward(self, f4_pred, f6_pred, f4_target, f6_target):
        loss_f4 = torch.mean((f4_pred - f4_target) ** 2)
        loss_f6 = torch.mean((f6_pred - f6_target) ** 2)
        return self.alpha_f4 * loss_f4 + self.alpha_f6 * loss_f6


class MisorientationLoss(nn.Module):
    """
    Misorientation-based loss for quaternions.
    Computes the angular distance between predicted and target quaternions,
    accounting for FCC crystal symmetry (24 equivalent orientations).
    """
    def __init__(self, physics, eps=1e-8):
        super().__init__()
        self.physics = physics
        self.eps = eps
        # Register symmetry group as buffer
        self.register_buffer('fcc_syms', physics.fcc_syms.clone())
    
    def quat_multiply(self, q1, q2):
        """Quaternion multiplication: q1 * q2"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
    
    def quat_conjugate(self, q):
        """Quaternion conjugate"""
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
    
    def forward(self, q_pred, q_target):
        """
        q_pred, q_target: (B, H, W, 4) quaternions
        Returns: scalar loss (mean misorientation angle in radians)
        """
        # Normalize
        q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + self.eps)
        q_target = q_target / (torch.norm(q_target, dim=-1, keepdim=True) + self.eps)
        
        # Flatten spatial dimensions
        B, H, W, _ = q_pred.shape
        q_pred_flat = q_pred.reshape(-1, 4)  # (N, 4)
        q_target_flat = q_target.reshape(-1, 4)  # (N, 4)
        N = q_pred_flat.shape[0]
        
        # Compute q_diff = q_target * conj(q_pred) for all symmetry variants
        # This gives the rotation from pred to target
        q_pred_conj = self.quat_conjugate(q_pred_flat)  # (N, 4)
        
        # Apply all 24 symmetries to q_pred: (24, N, 4)
        # sym * q_pred_conj
        syms = self.fcc_syms.unsqueeze(1).expand(-1, N, -1)  # (24, N, 4)
        q_pred_conj_exp = q_pred_conj.unsqueeze(0).expand(24, -1, -1)  # (24, N, 4)
        
        # Multiply: sym * conj(q_pred)
        q_pred_sym = self.quat_multiply(syms, q_pred_conj_exp)  # (24, N, 4)
        
        # Compute misorientation: q_target * (sym * conj(q_pred))
        q_target_exp = q_target_flat.unsqueeze(0).expand(24, -1, -1)  # (24, N, 4)
        q_diff = self.quat_multiply(q_target_exp, q_pred_sym)  # (24, N, 4)
        
        # Misorientation angle = 2 * acos(|w|)
        # We want the minimum across all symmetry variants
        w_abs = torch.abs(q_diff[..., 0])  # (24, N)
        w_max, _ = torch.max(w_abs, dim=0)  # (N,) - max |w| = min angle
        w_max = torch.clamp(w_max, max=1.0)  # Numerical safety
        
        # Angle = 2 * acos(|w|)
        angles = 2.0 * torch.acos(w_max)  # (N,) in radians
        
        return angles.mean()


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================
def train_model(model, physics, encoder, decoder, train_data, config, output_dir, device):
    """
    Train the super-resolution model with combined loss:
    - SphericalHarmonicLoss: MSE on f4, f6 features (ensures feature-level accuracy)
    - MisorientationLoss: Angular distance in quaternion space (ensures physical accuracy)
    
    ORIGINAL: Uses HR subsampling to create LR
    """
    
    model = model.to(device)
    decoder = decoder.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    
    # Combined loss: feature-level + quaternion-level
    sh_criterion = SphericalHarmonicLoss()
    mis_criterion = MisorientationLoss(physics)
    mis_criterion = mis_criterion.to(device)
    
    # Loss weighting (feature loss + misorientation loss)
    alpha_sh = config.get('alpha_sh', 1.0)
    alpha_mis = config.get('alpha_mis', 0.0)  # Set to 0 to skip slow misorientation loss
    mis_eval_freq = config.get('mis_eval_freq', 10)  # Compute misorientation every N epochs for monitoring
    
    # Batch size
    batch_size = config.get('batch_size', 5)
    
    losses = []
    
    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"Config: strict_equivariance={config['strict_equivariance']}, "
          f"static_upsample={config['static_upsample']}, batch_size={batch_size}")
    print(f"Loss weights: alpha_sh={alpha_sh}, alpha_mis={alpha_mis}")
    
    # Create batches
    n_samples = len(train_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Store first sample for periodic visualization
    test_hr_quats = train_data[0]
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_sh_loss = 0.0
        epoch_mis_loss = 0.0
        
        # Shuffle data each epoch
        indices = torch.randperm(n_samples).tolist()
        
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Stack batch of HR quaternions
            hr_batch = torch.stack([train_data[i] for i in batch_indices]).to(device)  # (B, H, W, 4)
            B, H, W, _ = hr_batch.shape
            
            # Create LR by subsampling HR (ORIGINAL - no separate LR folder)
            scale_factor = config['scale_factor']
            lr_batch = hr_batch[:, ::scale_factor, ::scale_factor, :]  # (B, H/4, W/4, 4)
            
            # Encode to features (batched)
            with torch.no_grad():
                f4_lr, f6_lr = encoder(lr_batch)  # (B, H/4, W/4, 9), (B, H/4, W/4, 13)
                f4_hr, f6_hr = encoder(hr_batch)  # (B, H, W, 9), (B, H, W, 13)
            
            # Convert to (B, C, H, W) format for network
            f4_lr = f4_lr.permute(0, 3, 1, 2)  # (B, 9, H/4, W/4)
            f6_lr = f6_lr.permute(0, 3, 1, 2)  # (B, 13, H/4, W/4)
            f4_hr = f4_hr.permute(0, 3, 1, 2)
            f6_hr = f6_hr.permute(0, 3, 1, 2)
            
            # Forward pass
            optimizer.zero_grad()
            f4_pred, f6_pred = model(f4_lr, f6_lr)
            
            # Loss 1: Spherical harmonic feature loss (fast)
            sh_loss = sh_criterion(f4_pred, f6_pred, f4_hr, f6_hr)
            
            # Loss 2: Misorientation loss (slow - only compute if enabled)
            if alpha_mis > 0:
                # Convert from (B, C, H, W) to (B, H, W, C) for decoder
                f4_pred_hwc = f4_pred.permute(0, 2, 3, 1)  # (B, H, W, 9)
                f6_pred_hwc = f6_pred.permute(0, 2, 3, 1)  # (B, H, W, 13)
                
                # Decode to quaternions
                q_pred = decoder(f4_pred_hwc, f6_pred_hwc)  # (B, H, W, 4)
                
                # Target quaternions
                q_target = hr_batch  # (B, H, W, 4)
                
                # Compute misorientation loss
                mis_loss = mis_criterion(q_pred, q_target)
            else:
                mis_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = alpha_sh * sh_loss + alpha_mis * mis_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * B
            epoch_sh_loss += sh_loss.item() * B
            epoch_mis_loss += mis_loss.item() * B
        
        scheduler.step()
        avg_loss = epoch_loss / n_samples
        avg_sh = epoch_sh_loss / n_samples
        avg_mis = epoch_mis_loss / n_samples
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            mis_deg = avg_mis * 180.0 / math.pi  # Convert to degrees
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Total={avg_loss:.6f}, SH={avg_sh:.6f}, "
                  f"Misori={mis_deg:.2f}°")
    
    return losses


# ==============================================================================
# EVALUATION AND IPF RENDERING (ORIGINAL 2-ROW VERSION)
# ==============================================================================
def render_input_output_comparison(
    input_q_arr: np.ndarray,
    output_q_arr: np.ndarray,
    sym_class,
    out_png: str = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    format_input: bool = True,
    dpi: int = 300,
):
    """
    ORIGINAL VERSION: 2-row layout showing Input (subsampled HR, upsampled to HR size) vs Output
    
    Parameters
    ----------
    input_q_arr : ndarray
        LR quaternion array - will be upsampled via bilinear to HR size for display.
    output_q_arr : ndarray
        SR output quaternion array of shape (H, W, 4).
    """
    from utils.quat_ops import format_quaternions
    
    # Early exit if file already exists
    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    # Format quaternions (reduce to FZ, normalize, hemisphere, etc.)
    if format_input:
        input_q_arr = format_quaternions(
            input_q_arr, normalize=True, hemisphere=True, reduce_fz=True,
            sym=sym_class, quat_first=False,
        )
        output_q_arr = format_quaternions(
            output_q_arr, normalize=True, hemisphere=True, reduce_fz=True,
            sym=sym_class, quat_first=False,
        )

    # Upsample input to match output size for comparison
    out_h, out_w = output_q_arr.shape[:2]
    in_h, in_w = input_q_arr.shape[:2]
    if in_h != out_h or in_w != out_w:
        # Bilinear upsampling for display
        input_tensor = torch.tensor(input_q_arr).permute(2, 0, 1).unsqueeze(0).float()
        input_upsampled = torch.nn.functional.interpolate(
            input_tensor, size=(out_h, out_w), mode='bilinear', align_corners=False
        )
        input_q_arr = input_upsampled.squeeze(0).permute(1, 2, 0).numpy()
        # Re-normalize after interpolation
        input_q_arr = input_q_arr / (np.linalg.norm(input_q_arr, axis=-1, keepdims=True) + 1e-8)

    # Convert to IPF RGB maps
    from visualization.ipf_render import render_ipf_rgb
    in_rgb = render_ipf_rgb(input_q_arr, sym_class, ref_dir=ref_dir)
    out_rgb = render_ipf_rgb(output_q_arr, sym_class, ref_dir=ref_dir)

    multi_ref = isinstance(in_rgb, list)
    ncols = 3 if multi_ref else 1
    key_cols = 1 if include_key else 0
    total_cols = ncols + key_cols
    total_rows = 2  # Input, Output

    # Figure setup
    base_w = 5.0
    key_w = 2.6 if include_key else 0
    fig_w = base_w * ncols + key_w
    fig_h = 2 * 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_rows,
        total_cols,
        width_ratios=[1] * ncols + ([0.9] if include_key else []),
        height_ratios=[1, 1],
        hspace=0.25,
        wspace=0.25,
    )

    def _imshow(ax, img, title):
        ax.imshow(img)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Plot Input (row 0), Output (row 1)
    if multi_ref:
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), in_rgb)):
            _imshow(fig.add_subplot(gs[0, j]), img, f"Input IPF-{name}")
        for j, (name, img) in enumerate(zip(("X", "Y", "Z"), out_rgb)):
            _imshow(fig.add_subplot(gs[1, j]), img, f"Output IPF-{name}")
    else:
        _imshow(fig.add_subplot(gs[0, 0]), in_rgb, f"Input IPF-{ref_dir.upper()}")
        _imshow(fig.add_subplot(gs[1, 0]), out_rgb, f"Output IPF-{ref_dir.upper()}")

    # IPF color key
    if include_key:
        from orix import plot as orix_plot
        ax_key = fig.add_subplot(gs[:, -1], projection="ipf", symmetry=sym_class.laue)
        ax_key.plot_ipf_color_key()
        ax_key.set_title("")

    # Save figure
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return out_png
    else:
        plt.close(fig)
        return None


def evaluate_and_render(model, physics, encoder, decoder, test_hr_quats, config, output_dir, device):
    """
    ORIGINAL VERSION: Evaluate model and render IPF comparison showing Input vs Output
    Uses HR subsampling to create LR input
    """
    
    model.eval()
    model = model.to(device)
    
    # Prepare test data
    hr_quats = test_hr_quats.to(device)  # (H, W, 4)
    H, W = hr_quats.shape[:2]
    
    # Create LR by subsampling (ORIGINAL - no separate LR folder)
    scale_factor = config['scale_factor']
    lr_quats = hr_quats[::scale_factor, ::scale_factor, :]  # (H/4, W/4, 4)
    
    H_lr, W_lr = lr_quats.shape[:2]
    
    print(f"\nEvaluating: LR {H_lr}x{W_lr} -> HR {H}x{W}")
    
    with torch.no_grad():
        # Encode LR
        f4_lr, f6_lr = encoder(lr_quats.unsqueeze(0))
        f4_lr = f4_lr.permute(0, 3, 1, 2)
        f6_lr = f6_lr.permute(0, 3, 1, 2)
        
        # Super-resolve features
        f4_sr, f6_sr = model(f4_lr, f6_lr)
        
        # Convert back to (B, H, W, C)
        f4_sr = f4_sr.permute(0, 2, 3, 1).squeeze(0)
        f6_sr = f6_sr.permute(0, 2, 3, 1).squeeze(0)
        
        # Decode to quaternions
        sr_quats = decoder(f4_sr.unsqueeze(0), f6_sr.unsqueeze(0)).squeeze(0)
    
    # Move to CPU and convert to numpy
    hr_quats_np = hr_quats.cpu().numpy()
    lr_quats_np = lr_quats.cpu().numpy()
    sr_quats_np = sr_quats.cpu().numpy()
    
    # Define FCC symmetry
    fcc_sym = Phase(space_group=225).point_group
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate config name
    config_name = "strict" if config['strict_equivariance'] else "nonstrict"
    
    # Render Input vs Output comparison (ORIGINAL 2-row format)
    out_file = os.path.join(output_dir, f"ipf_comparison_{config_name}.png")
    print(f"Rendering IPF comparison (Input vs Output) to: {out_file}")
    
    render_input_output_comparison(
        lr_quats_np,  # Will be upsampled for display
        sr_quats_np,
        fcc_sym,
        out_png=out_file,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=300
    )
    
    # Also render ground truth comparison
    gt_file = os.path.join(output_dir, f"ipf_ground_truth_{config_name}.png")
    print(f"Rendering ground truth comparison to: {gt_file}")
    
    render_input_output_comparison(
        lr_quats_np,  # Input (subsampled HR)
        hr_quats_np,  # Ground truth HR
        fcc_sym,
        out_png=gt_file,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=300
    )
    
    # Compute metrics
    with torch.no_grad():
        f4_hr, f6_hr = encoder(hr_quats.unsqueeze(0))
        f4_sr_eval, f6_sr_eval = encoder(sr_quats.unsqueeze(0))
        
        mse_f4 = torch.mean((f4_sr_eval - f4_hr) ** 2).item()
        mse_f6 = torch.mean((f6_sr_eval - f6_hr) ** 2).item()
    
    print(f"  MSE (f4): {mse_f4:.6f}")
    print(f"  MSE (f6): {mse_f6:.6f}")
    
    return {
        'mse_f4': mse_f4,
        'mse_f6': mse_f6,
        'sr_quats': sr_quats_np,
        'hr_quats': hr_quats_np,
        'lr_quats': lr_quats_np
    }


# ==============================================================================
# MAIN (ORIGINAL - HR folder only, no LR_Data)
# ==============================================================================
def main():
    print("="*70)
    print("E3NN SUPER-RESOLUTION TRAINING (ORIGINAL VERSION)")
    print("Training both strict and non-strict equivariance configurations")
    print("Uses HR subsampling to create LR input")
    print("="*70)
    
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Real EBSD training data directory (HR only - ORIGINAL)
    hr_dir = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data"
    
    if os.path.exists(hr_dir):
        # Load all HR quaternion files
        hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.npy')])
        print(f"Found {len(hr_files)} HR quaternion files")
        
        # Load training samples (limit to first N for faster training)
        max_samples = 200  # Use subset for training
        train_data = []
        for i, hr_f in enumerate(hr_files[:max_samples]):
            hr_filepath = os.path.join(hr_dir, hr_f)
            
            hr_quats_np = np.load(hr_filepath)  # Shape: (4, 128, 128)
            
            # Convert from (C, H, W) to (H, W, C)
            hr_quats = torch.tensor(hr_quats_np, dtype=torch.float32).permute(1, 2, 0)
            hr_quats = hr_quats / torch.norm(hr_quats, dim=-1, keepdim=True)
            train_data.append(hr_quats)
        
        # Use first sample for testing
        test_hr_quats = train_data[0]
        print(f"Loaded {len(train_data)} training samples - HR: {train_data[0].shape}")
    else:
        print(f"HR directory not found: {hr_dir}")
        print("Using synthetic data as fallback...")
        train_data = create_synthetic_data(num_samples=50, hr_size=64, device=device)
        test_hr_quats = train_data[0]
    
    # Initialize physics and encoder/decoder
    physics = FCCPhysics(device=device)
    encoder = FCCEncoder(physics).to(device)
    decoder = SphericalSamplingDecoder(physics).to(device)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training configurations
    configs = [
        {
            'name': 'strict_equivariance',
            'strict_equivariance': True,
            'static_upsample': True,
            'scale_factor': 4,
            'depth': 4,
            'epochs': 100,
            'lr': 1e-3,
            'lr_step': 30,
            'batch_size': 5,
            'alpha_sh': 1.0,
            'alpha_mis': 10.0,  # Enable misorientation loss
        },
        {
            'name': 'non_strict_equivariance', 
            'strict_equivariance': False,
            'static_upsample': True,
            'scale_factor': 4,
            'depth': 4,
            'epochs': 100,
            'lr': 1e-3,
            'lr_step': 30,
            'batch_size': 5,
            'alpha_sh': 1.0,
            'alpha_mis': 10.0,  # Enable misorientation loss
        }
    ]
    
    results = {}
    
    for config in configs:
        print("\n" + "="*70)
        print(f"CONFIGURATION: {config['name']}")
        print("="*70)
        
        # Create model
        model = FCCInteractionNet(
            physics=physics,
            depth=config['depth'],
            scale_factor=config['scale_factor'],
            static_upsample=config['static_upsample'],
            strict_equivariance=config['strict_equivariance']
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Train (pass decoder for misorientation loss) - uses HR subsampling for LR
        losses = train_model(model, physics, encoder, decoder, train_data, config, output_dir, device)
        
        # Evaluate and render IPF (final)
        eval_results = evaluate_and_render(
            model, physics, encoder, decoder, 
            test_hr_quats, config, output_dir, device
        )
        
        results[config['name']] = {
            'losses': losses,
            'eval': eval_results
        }
        
        # Save model
        model_path = os.path.join(output_dir, f"model_{config['name']}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Final Loss: {res['losses'][-1]:.6f}")
        print(f"  MSE (f4): {res['eval']['mse_f4']:.6f}")
        print(f"  MSE (f6): {res['eval']['mse_f6']:.6f}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print("  - ipf_comparison_strict.png")
    print("  - ipf_comparison_nonstrict.png")
    print("  - ipf_ground_truth_strict.png")
    print("  - ipf_ground_truth_nonstrict.png")
    print("  - model_strict_equivariance.pt")
    print("  - model_non_strict_equivariance.pt")


if __name__ == "__main__":
    main()
