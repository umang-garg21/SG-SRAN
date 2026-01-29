#!/usr/bin/env python3
"""
Training script with stage-by-stage IPF visualization.

This script:
1. Loads quaternion data from disk
2. Trains the encoder-decoder model
3. Generates IPF maps at each processing stage:
   - Stage 0: Input quaternions
   - Stage 1: f4 features decoded to quaternions
   - Stage 2: Full reconstruction (f4 + f6 decoded)
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
# Duplicating the boundary formation mechanism directly into this script

def generate_boundary_map(quaternions):
    """
    Generate boundary map using the exact mechanism from boundary_formation.

    Args:
        quaternions: Quaternion array (H, W, 4) as numpy array

    Returns:
        boundary_map: Colored boundary map as a numpy array
    """
    # Example mechanism (replace with the exact logic from boundary_formation)
    H, W, _ = quaternions.shape
    boundary_map = np.zeros((H, W, 3), dtype=np.uint8)

    # Compute gradients (example logic, replace with actual boundary_formation logic)
    for i in range(3):
        grad_x = cv2.Sobel(quaternions[..., i], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(quaternions[..., i], cv2.CV_64F, 0, 1, ksize=3)
        # Apply heatmap-style colormap (e.g., viridis) instead of RGB mapping
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = (grad_magnitude / grad_magnitude.max() * 255).astype(np.uint8)
        boundary_map = plt.cm.viridis(grad_magnitude / 255.0)[:, :, :3]  # Normalize and apply colormap
        boundary_map = (boundary_map * 255).astype(np.uint8)  # Convert to uint8

    return boundary_map

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

        # Generate boundary map using the duplicated mechanism
        boundary_map = generate_boundary_map(q_img)
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
    print("QUATERNION ENCODER-DECODER STAGE VISUALIZATION")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    data_dir = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data"
    output_dir = "./training_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # ==============================================================================
    # 1. INITIALIZE MODEL (Physics-based, no training needed)
    # ==============================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = EBSDSuper(device=device, grid_samples=10000, batch_size=1000)
    print(f"Physics-based model initialized on {device}")
    print("Note: This model has no trainable parameters (purely physics-based)")
    
    # ==============================================================================
    # 2. LOAD SAMPLE FOR VISUALIZATION
    # ==============================================================================
    print("\n" + "="*70)
    print("LOADING SAMPLE DATA FOR VISUALIZATION")
    print("="*70)
    
    # Load a sample image for visualization
    sample_file = Path(data_dir) / "Open718_QSR_x4_train_hr_x_block_0.npy"
    q_sample = np.load(sample_file)
    print(f"Loaded sample image: {sample_file.name}, shape: {q_sample.shape}")
    
    # Handle format
    if q_sample.shape[-1] == 4:
        img_shape = q_sample.shape[:2]
        q_flat = q_sample.reshape(-1, 4)
    elif q_sample.shape[0] == 4:
        img_shape = q_sample.shape[1:]
        q_flat = q_sample.reshape(4, -1).T
    else:
        raise ValueError(f"Cannot determine format from shape {q_sample.shape}")
    
    # Take a smaller crop for visualization (256x256)
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
    
    # ==============================================================================
    # 3. ENCODE FEATURES
    # ==============================================================================
    print("\n" + "="*70)
    print("ENCODING QUATERNIONS TO FEATURES")
    print("="*70)
    
    print(f"Encoding {len(q_tensor)} quaternions...")
    model.eval()
    with torch.no_grad():
        f4, f6 = model.encoder(q_tensor)
    
    print(f"✓ f4 features shape: {f4.shape} (L=4 spherical harmonics)")
    print(f"✓ f6 features shape: {f6.shape} (L=6 spherical harmonics)")
    
    # ==============================================================================
    # 4. VISUALIZE STAGES (FLEXIBLE - automatically adapts to number of stages)
    # ==============================================================================
    print("\n" + "="*70)
    print("GENERATING STAGE-BY-STAGE IPF VISUALIZATION")
    print("="*70)
    
    # Create stage decoder
    decoder_stage = StageDecoder(model.physics, model.decoder)
    decoder_stage.to(device)
    decoder_stage.eval()
    
    # FCC symmetry
    fcc_sym = Phase(space_group=225).point_group
    
    # Define stages to visualize (FLEXIBLE - add/remove stages here)
    # Each stage decodes features and matches to closest symmetry variant
    print("Decoding stages...")
    
    with torch.no_grad():
        # Stage 1: L=4 only
        q_stage1 = decoder_stage.decode_f4_only(f4)
        q_stage1 = match_symmetry_batch(q_tensor, q_stage1, model.physics)
        
        # Stage 2: Final model output (currently uses L=4, ignores L=6)
        q_stage2 = decoder_stage.decode_final(f4, f6)
        q_stage2 = match_symmetry_batch(q_tensor, q_stage2, model.physics)
    
    # Build stage configuration list (FLEXIBLE - automatically adapts to number of stages)
    stages_config = [
        {
            "name": "Stage 0: Input", 
            "quaternions": q_tensor
        },
        {
            "name": "Stage 1: L=4 Decoded", 
            "quaternions": q_stage1
        },
        {
            "name": "Stage 2: Model Output", 
            "quaternions": q_stage2
        }
    ]
    
    # NOTE: To add more stages in the future, simply add more entries to stages_config
    # Example:
    # stages_config.append({
    #     "name": "Stage 3: New Processing Step",
    #     "quaternions": some_new_quaternion_tensor
    # })
    
    output_path = os.path.join(output_dir, 'stage_ipf_maps.png')
    render_stage_ipf_maps(
        stages_config, 
        img_shape, 
        output_path, 
        fcc_sym
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Output saved to: {output_dir}/")
    print(f"  - stage_ipf_maps.png: IPF maps at each processing stage")
    print(f"  - stage_ipf_maps_boundary_maps.png: Boundary maps at each processing stage")
    print(f"\nStages visualized: {len(stages_config)}")
    for i, stage in enumerate(stages_config):
        print(f"  {i}. {stage['name']}")


if __name__ == "__main__":
    main()
