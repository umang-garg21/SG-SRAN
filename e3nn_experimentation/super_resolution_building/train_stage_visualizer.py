#!/usr/bin/env python3
"""
Training script with stage-by-stage IPF visualization.

This script:
1. Loads quaternion data from disk
2. Trains the convolution layer using Irrep Representation Loss
3. Generates IPF maps at each processing stage:
   - Stage 0: Input quaternions
   - Stage 1: Encoded features decoded to quaternions
   - Stage 2: Convolved features decoded to quaternions
   - Stage 3: Final output
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import time
from pathlib import Path
import cv2  # OpenCV for image processing

# Add project root to path
sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')
sys.path.append('/data/home/umang/Materials/e3nn_Reynolds/e3nn_experimentation')

from e3nn import o3
from orix.crystal_map import Phase
from visualization.ipf_render import render_ipf_rgb
from simple_encoder_decoder import (
    FCCPhysics, 
    FCCEncoder, 
    SphericalSamplingDecoder,
    EBSDSuper,
    wigner_D_cuda
)


# ==============================================================================
# IRREP REPRESENTATION LOSS (Loss 2)
# ==============================================================================
class IrrepLoss(nn.Module):
    """
    Loss function that compares l=4 and l=6 irrep representations.
    
    L = ||f4_conv - f4_target||² + ||f6_conv - f6_target||²
    
    This trains the convolution layer to preserve/reconstruct the irrep features.
    """
    def __init__(self, lambda_f4=1.0, lambda_f6=1.0):
        super().__init__()
        self.lambda_f4 = lambda_f4
        self.lambda_f6 = lambda_f6
        self.mse = nn.MSELoss()
    
    def forward(self, f4_pred, f6_pred, f4_target, f6_target):
        """
        Args:
            f4_pred: Predicted l=4 features (N, 9)
            f6_pred: Predicted l=6 features (N, 13)
            f4_target: Target l=4 features (N, 9)
            f6_target: Target l=6 features (N, 13)
        
        Returns:
            loss: Combined MSE loss
            loss_dict: Dictionary with individual losses for logging
        """
        loss_f4 = self.mse(f4_pred, f4_target)
        loss_f6 = self.mse(f6_pred, f6_target)
        
        total_loss = self.lambda_f4 * loss_f4 + self.lambda_f6 * loss_f6
        
        loss_dict = {
            'loss_f4': loss_f4.item(),
            'loss_f6': loss_f6.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_convolution_layer(model, q_tensor, img_shape, num_epochs=100, lr=1e-3, 
                            log_interval=10, output_dir='./training_outputs'):
    """
    Train the convolution layer using Irrep Representation Loss.
    
    The target is reconstruction: conv output should match encoder output.
    This teaches the conv layer to not destroy the signal initially.
    
    Args:
        model: EBSDSuper model with conv_layer
        q_tensor: Input quaternions (N, 4)
        img_shape: (H, W) tuple
        num_epochs: Number of training epochs
        lr: Learning rate
        log_interval: Print loss every N epochs
        output_dir: Directory to save training outputs
    """
    print("\n" + "="*70)
    print("TRAINING CONVOLUTION LAYER (Irrep Representation Loss)")
    print("="*70)
    
    device = q_tensor.device
    
    # Get target features (encoder output - these are the "ground truth" irreps)
    model.eval()
    with torch.no_grad():
        f4_target, f6_target = model.encoder(q_tensor)
    
    print(f"Target f4 shape: {f4_target.shape}")
    print(f"Target f6 shape: {f6_target.shape}")
    
    # Setup optimizer (only optimize conv_layer parameters)
    optimizer = optim.Adam(model.conv_layer.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Loss function
    criterion = IrrepLoss(lambda_f4=1.0, lambda_f6=1.0)
    
    # Count trainable parameters
    num_params = sum(p.numel() for p in model.conv_layer.parameters() if p.requires_grad)
    print(f"Trainable parameters in conv_layer: {num_params}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {num_epochs}")
    print("-"*70)
    
    # Training history
    history = {
        'loss_total': [],
        'loss_f4': [],
        'loss_f6': [],
        'lr': []
    }
    
    # Training loop
    model.conv_layer.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass through encoder (no grad needed, it's fixed)
        with torch.no_grad():
            f4_encoded, f6_encoded = model.encoder(q_tensor)
        
        # Forward pass through conv_layer (grad needed)
        f4_conv, f6_conv = model.conv_layer(f4_encoded, f6_encoded, img_shape)
        
        # Compute loss (target = encoder output, i.e., reconstruction)
        loss, loss_dict = criterion(f4_conv, f6_conv, f4_target, f6_target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.conv_layer.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update scheduler
        scheduler.step(loss)
        
        # Record history
        history['loss_total'].append(loss_dict['loss_total'])
        history['loss_f4'].append(loss_dict['loss_f4'])
        history['loss_f6'].append(loss_dict['loss_f6'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                  f"Loss: {loss_dict['loss_total']:.6f} | "
                  f"L_f4: {loss_dict['loss_f4']:.6f} | "
                  f"L_f6: {loss_dict['loss_f6']:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print("-"*70)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    return history


def plot_training_curves(history, output_dir):
    """Plot and save training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['loss_total'], 'b-', linewidth=1)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss (L_f4 + L_f6)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Individual losses
    axes[1].plot(history['loss_f4'], 'r-', label='L_f4 (l=4)', linewidth=1)
    axes[1].plot(history['loss_f6'], 'g-', label='L_f6 (l=6)', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Individual Irrep Losses')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(history['lr'], 'k-', linewidth=1)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to: {save_path}")


# ==============================================================================
# DATASET LOADER
# ==============================================================================
class QuaternionDataset(torch.utils.data.Dataset):
    """Simple dataset for loading quaternion patches from .npy files."""
    
    def __init__(self, data_dir, max_files=None):
        """
        Args:
            data_dir: Directory containing .npy files with quaternion data
            max_files: Maximum number of files to load (None = all)
        """
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.npy")))
        
        if max_files:
            self.files = self.files[:max_files]
        
        print(f"Found {len(self.files)} files in {data_dir}")
        
        # Load all data into memory for simplicity
        self.data = []
        for f in self.files:
            q_data = np.load(f)
            # Handle different formats: (H, W, 4) or (4, H, W)
            if q_data.shape[-1] == 4:
                q_data = q_data.reshape(-1, 4)
            elif q_data.shape[0] == 4:
                q_data = q_data.reshape(4, -1).T
            
            self.data.append(q_data)
        
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        print(f"Loaded {len(self.data)} quaternions total")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q = self.data[idx]
        # Normalize
        q = q / np.linalg.norm(q)
        return torch.from_numpy(q)


# ==============================================================================
# STAGE-BY-STAGE DECODER (for visualization)
# ==============================================================================
class StageDecoder(nn.Module):
    """Decoder that can decode individual feature stages for visualization."""
    
    def __init__(self, physics, decoder):
        super().__init__()
        self.physics = physics
        self.decoder = decoder
    
    def decode_f4_only(self, f4):
        """Decode using only f4 features (set f6 to zero)."""
        batch_size = f4.shape[0]
        f6_zero = torch.zeros(batch_size, 13, device=f4.device)
        return self.decoder(f4, f6_zero)
    
    def decode_final(self, f4, f6):
        """Decode using the model's final output (currently only uses f4)."""
        return self.decoder(f4, f6)


# ==============================================================================
# STAGE VISUALIZATION (FLEXIBLE)
# ==============================================================================
# BOUNDARY MAP GENERATION (Exact mechanism from boundary_formation)
# ==============================================================================
def quat_multiply(q1, q2):
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)

def quat_conjugate(q):
    """Inverse rotation."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

def get_fcc_symmetries():
    """Get FCC symmetry group quaternions."""
    import math
    inv_sqrt_2 = 1 / math.sqrt(2)
    half = 0.5
    return np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [inv_sqrt_2, inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, inv_sqrt_2],
        [inv_sqrt_2, -inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, -inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, -inv_sqrt_2],
        [0, inv_sqrt_2, inv_sqrt_2, 0], [0, inv_sqrt_2, 0, inv_sqrt_2], [0, 0, inv_sqrt_2, inv_sqrt_2],
        [0, inv_sqrt_2, -inv_sqrt_2, 0], [0, 0, inv_sqrt_2, -inv_sqrt_2], [0, inv_sqrt_2, 0, -inv_sqrt_2],
        [half, half, half, half], [half, -half, -half, half], [half, -half, half, -half], [half, half, -half, -half],
        [half, half, half, -half], [half, half, -half, half], [half, -half, half, half], [half, -half, -half, -half],
    ], dtype=np.float32)

def get_misorientation_angle(q_center, q_neighbor, symmetries):
    """
    Calculate exact disorientation angle between two quaternions.
    Returns angle in degrees.
    """
    # Calculate relative rotation: q_rel = q_neighbor * q_center_inverse
    q_inv = quat_conjugate(q_center)
    q_rel = quat_multiply(q_neighbor, q_inv)
    
    # Check all 24 symmetry variants and find minimum angle
    # q_rel shape: (..., 4), syms shape: (24, 4)
    q_rel_expanded = q_rel[..., np.newaxis, :]  # (..., 1, 4)
    syms_expanded = symmetries  # (24, 4)
    
    # Multiply: (..., 24, 4)
    q_syms = quat_multiply(syms_expanded, q_rel_expanded)
    
    # Find rotation with minimum angle (maximize |w|)
    w_abs = np.abs(q_syms[..., 0])  # (..., 24)
    best_w = np.max(w_abs, axis=-1)  # (...)
    
    # Clamp and compute angle
    best_w = np.clip(best_w, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(best_w)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def generate_boundary_map(quaternions):
    """
    Generate boundary map using the exact mechanism from boundary_formation.
    Uses misorientation angles with FCC symmetry handling.

    Args:
        quaternions: Quaternion array (H, W, 4) as numpy array

    Returns:
        boundary_map: Colored boundary map using inferno colormap (H, W, 3)
    """
    H, W, _ = quaternions.shape
    symmetries = get_fcc_symmetries()
    
    # Forward differences (right and down neighbors)
    q_right = np.roll(quaternions, shift=-1, axis=1)
    q_down = np.roll(quaternions, shift=-1, axis=0)
    
    # Calculate misorientation angles
    ang_x = get_misorientation_angle(quaternions, q_right, symmetries)
    ang_y = get_misorientation_angle(quaternions, q_down, symmetries)
    
    # Zero out wrapped edges
    ang_x[:, -1] = 0
    ang_y[-1, :] = 0
    
    # Average misorientation (L=0 scalar feature)
    misorientation = (ang_x + ang_y) / 2.0
    
    # Normalize to [0, 1] with max at 60 degrees (FCC max disorientation)
    norm_misorientation = np.clip(misorientation / 60.0, 0, 1)
    
    # Apply colormap
    cmap = plt.cm.inferno
    boundary_map = cmap(norm_misorientation)[:, :, :3]  # Remove alpha channel
    boundary_map = (boundary_map * 255).astype(np.uint8)
    
    return boundary_map, misorientation

# Modify render_stage_ipf_maps to save boundary maps separately
def render_stage_ipf_maps(stages_config, img_shape, output_path, fcc_sym):
    """
    Render IPF maps for each processing stage and save boundary maps separately.

    Args:
        stages_config: List of dicts, each containing:
            - 'name': Stage name (e.g., "Stage 0: Input")
            - 'quaternions': Quaternion array (N, 4) as numpy or torch tensor
        img_shape: Tuple (H, W) for reshaping quaternions to image
        output_path: Path to save the output figure
        fcc_sym: FCC symmetry for IPF rendering
    """
    print("\n" + "="*70)
    print("RENDERING STAGE-BY-STAGE IPF AND BOUNDARY MAPS")
    print("="*70)

    num_stages = len(stages_config)
    print(f"Number of stages: {num_stages}")

    # Convert all quaternions to numpy and reshape to images
    stage_images = []
    stage_names = []
    boundary_maps = []

    for i, stage in enumerate(stages_config):
        print(f"Processing {stage['name']}...")

        # Convert to numpy if needed
        q = stage['quaternions']
        if torch.is_tensor(q):
            q = q.cpu().numpy()

        # Reshape to image
        H, W = img_shape
        q_img = q.reshape(H, W, 4)
        stage_images.append(q_img)
        stage_names.append(stage['name'])

        # Generate boundary map using the exact mechanism from boundary_formation
        boundary_map, misorientation = generate_boundary_map(q_img)
        boundary_maps.append(boundary_map)

    # Render IPF maps (X, Y, Z directions) for all stages
    print("Rendering IPF RGB maps...")
    rgb_stages = []
    for q_img in stage_images:
        rgb = render_ipf_rgb(q_img, fcc_sym, ref_dir="ALL")
        rgb_stages.append(rgb)

    # Create flexible figure layout based on number of stages
    fig_height = 4 + num_stages * 4.5  # Adjust height based on number of stages
    fig = plt.figure(figsize=(17, fig_height), facecolor='white')
    gs = GridSpec(num_stages, 4, figure=fig, width_ratios=[1, 1, 1, 0.35], 
                  hspace=0.25, wspace=0.05, left=0.12, right=0.95, top=0.95, bottom=0.05)

    directions = ['X', 'Y', 'Z']

    # Plot each stage
    for row, (stage_name, rgb_list) in enumerate(zip(stage_names, rgb_stages)):
        for col, (direction, rgb) in enumerate(zip(directions, rgb_list)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(rgb)
            ax.set_aspect('equal')

            # Column headers (only on first row)
            if row == 0:
                ax.set_title(f"IPF-{direction}", fontsize=14, fontweight='bold', pad=10)

            # Row labels (only on first column)
            if col == 0:
                ax.text(-0.25, 0.5, stage_name, 
                       transform=ax.transAxes,
                       fontsize=13,
                       fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=0)

            ax.axis('off')

    # Add IPF color key (spans all rows)
    ax_key = fig.add_subplot(gs[:, 3], projection='ipf', symmetry=fcc_sym.laue)
    ax_key.plot_ipf_color_key()
    ax_key.set_title("IPF Color Key", fontsize=12, fontweight='bold', pad=10)

    # Save figure
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved stage visualization to: {output_path}")

    # Save boundary maps as a separate image
    boundary_output_path = output_path.replace('.png', '_boundary_maps.png')
    fig_boundary = plt.figure(figsize=(10, num_stages * 3), facecolor='white')
    gs_boundary = GridSpec(num_stages, 1, figure=fig_boundary, hspace=0.3)

    for row, (stage_name, boundary_map) in enumerate(zip(stage_names, boundary_maps)):
        ax = fig_boundary.add_subplot(gs_boundary[row, 0])
        ax.imshow(boundary_map)
        ax.set_title(stage_name, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    fig_boundary.savefig(boundary_output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_boundary)
    print(f"✓ Saved boundary maps to: {boundary_output_path}")


def match_symmetry_batch(q_truth, q_reconstructed, physics):
    """Match reconstructed quaternions to closest symmetry variant."""
    batch_size = q_truth.shape[0]
    device = q_truth.device
    
    # Generate symmetry family
    q_rec_expanded = q_reconstructed.unsqueeze(1).expand(-1, 24, -1)
    fcc_syms_expanded = physics.fcc_syms.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Quaternion multiplication
    w1, x1, y1, z1 = q_rec_expanded[..., 0], q_rec_expanded[..., 1], q_rec_expanded[..., 2], q_rec_expanded[..., 3]
    w2, x2, y2, z2 = fcc_syms_expanded[..., 0], fcc_syms_expanded[..., 1], fcc_syms_expanded[..., 2], fcc_syms_expanded[..., 3]
    family = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)
    
    # Find closest match
    q_truth_expanded = q_truth.unsqueeze(1)
    dist_pos = torch.norm(family - q_truth_expanded, dim=-1)
    dist_neg = torch.norm(family + q_truth_expanded, dim=-1)
    min_dist = torch.minimum(dist_pos, dist_neg)
    best_indices = torch.argmin(min_dist, dim=1)
    
    # Get closest quaternions
    batch_indices = torch.arange(batch_size, device=device)
    closest_quats = family[batch_indices, best_indices]
    use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
    closest_quats[use_neg] = -closest_quats[use_neg]
    
    return closest_quats


# ==============================================================================
# MAIN TRAINER
# ==============================================================================
def main():
    """Main training and visualization pipeline."""
    
    print("="*70)
    print("QUATERNION ENCODER-DECODER WITH TRAINING")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Training hyperparameters
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-3
    LOG_INTERVAL = 50
    
    data_dir = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data"
    output_dir = "./training_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # ==============================================================================
    # 1. INITIALIZE MODEL
    # ==============================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = EBSDSuper(device=device, grid_samples=10000, batch_size=1000)
    print(f"Model initialized on {device}")
    
    # ==============================================================================
    # 2. LOAD SAMPLE DATA
    # ==============================================================================
    print("\n" + "="*70)
    print("LOADING SAMPLE DATA")
    print("="*70)
    
    # Load a sample image
    sample_file = Path(data_dir) / "Open718_QSR_x4_train_hr_x_block_0.npy"
    q_sample = np.load(sample_file)
    print(f"Loaded sample image: {sample_file.name}, shape: {q_sample.shape}")
    
    # Handle format
    if q_sample.shape[-1] == 4:
        img_shape = q_sample.shape[:2]
    elif q_sample.shape[0] == 4:
        img_shape = q_sample.shape[1:]
    else:
        raise ValueError(f"Cannot determine format from shape {q_sample.shape}")
    
    # Take a smaller crop for faster training (256x256)
    crop_size = min(256, img_shape[0], img_shape[1])
    q_crop = q_sample[:crop_size, :crop_size, :] if q_sample.shape[-1] == 4 else q_sample[:, :crop_size, :crop_size]
    
    if q_crop.shape[-1] == 4:
        img_shape = (crop_size, crop_size)
        q_flat = q_crop.reshape(-1, 4)
    else:
        img_shape = (crop_size, crop_size)
        q_flat = q_crop.reshape(4, -1).T
    
    # Convert to torch and normalize
    q_tensor = torch.from_numpy(q_flat).float().to(device)
    q_tensor = q_tensor / torch.norm(q_tensor, dim=1, keepdim=True)
    
    print(f"Training data shape: {q_tensor.shape}")
    print(f"Image shape: {img_shape}")
    
    # ==============================================================================
    # 3. VISUALIZE BEFORE TRAINING (untrained conv layer)
    # ==============================================================================
    print("\n" + "="*70)
    print("VISUALIZATION BEFORE TRAINING")
    print("="*70)
    
    fcc_sym = Phase(space_group=225).point_group
    decoder_stage = StageDecoder(model.physics, model.decoder)
    decoder_stage.to(device)
    
    # Run forward pass before training
    model.eval()
    with torch.no_grad():
        outputs_before = model.forward(q_tensor, img_shape=img_shape)
        
        # Decode stages
        f4_enc, f6_enc = outputs_before['encoded']
        q_encoded = decoder_stage.decode_final(f4_enc, f6_enc)
        q_encoded = match_symmetry_batch(q_tensor, q_encoded, model.physics)
        
        f4_conv, f6_conv = outputs_before['convolved']
        q_convolved_before = decoder_stage.decode_final(f4_conv, f6_conv)
        q_convolved_before = match_symmetry_batch(q_tensor, q_convolved_before, model.physics)
        
        q_output_before = match_symmetry_batch(q_tensor, outputs_before['output'], model.physics)
    
    stages_before = [
        {"name": "Stage 0: Input", "quaternions": outputs_before['input']},
        {"name": "Stage 1: Encoded", "quaternions": q_encoded},
        {"name": "Stage 2: Convolved (BEFORE)", "quaternions": q_convolved_before},
        {"name": "Stage 3: Output (BEFORE)", "quaternions": q_output_before}
    ]
    
    render_stage_ipf_maps(stages_before, img_shape, 
                          os.path.join(output_dir, 'stage_ipf_maps_BEFORE_training.png'), fcc_sym)
    
    # ==============================================================================
    # 4. TRAIN CONVOLUTION LAYER
    # ==============================================================================
    history = train_convolution_layer(
        model=model,
        q_tensor=q_tensor,
        img_shape=img_shape,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        log_interval=LOG_INTERVAL,
        output_dir=output_dir
    )
    
    # ==============================================================================
    # 5. VISUALIZE AFTER TRAINING
    # ==============================================================================
    print("\n" + "="*70)
    print("VISUALIZATION AFTER TRAINING")
    print("="*70)
    
    # Run forward pass after training
    model.eval()
    with torch.no_grad():
        outputs_after = model.forward(q_tensor, img_shape=img_shape)
        
        # Decode stages
        f4_conv, f6_conv = outputs_after['convolved']
        q_convolved_after = decoder_stage.decode_final(f4_conv, f6_conv)
        q_convolved_after = match_symmetry_batch(q_tensor, q_convolved_after, model.physics)
        
        q_output_after = match_symmetry_batch(q_tensor, outputs_after['output'], model.physics)
    
    stages_after = [
        {"name": "Stage 0: Input", "quaternions": outputs_after['input']},
        {"name": "Stage 1: Encoded", "quaternions": q_encoded},
        {"name": "Stage 2: Convolved (AFTER)", "quaternions": q_convolved_after},
        {"name": "Stage 3: Output (AFTER)", "quaternions": q_output_after}
    ]
    
    render_stage_ipf_maps(stages_after, img_shape,
                          os.path.join(output_dir, 'stage_ipf_maps_AFTER_training.png'), fcc_sym)
    
    # ==============================================================================
    # 6. SUMMARY
    # ==============================================================================
    print("\n" + "="*70)
    print("TRAINING AND VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput saved to: {output_dir}/")
    print(f"  - stage_ipf_maps_BEFORE_training.png: IPF maps before training")
    print(f"  - stage_ipf_maps_AFTER_training.png: IPF maps after training")
    print(f"  - training_curves.png: Loss curves during training")
    print(f"\nTraining Summary:")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Initial Loss: {history['loss_total'][0]:.6f}")
    print(f"  - Final Loss: {history['loss_total'][-1]:.6f}")
    print(f"  - Loss Reduction: {(1 - history['loss_total'][-1]/history['loss_total'][0])*100:.1f}%")


if __name__ == "__main__":
    main()
