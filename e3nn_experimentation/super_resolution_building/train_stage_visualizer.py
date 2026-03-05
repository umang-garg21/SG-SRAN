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
import random

# Reproducible runs: set global seeds
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# Add project root to path
sys.path.append("/data/home/umang/Materials/Reynolds-QSR_clean_ipf")
sys.path.append("/data/home/umang/Materials/Reynolds-QSR_clean_ipf/utils")

from e3nn import o3
from training.data_loading import QuaternionDataset
from visualization.ipf_render import render_ipf_rgb
from visualization.visualize_sr_results import render_input_output_side_by_side
from utils.quat_ops import to_spatial_quat
import utils
from encoder_decoder import (
    FCCPhysics,
    FCCEncoder,
    SphericalSamplingDecoder,
    EquivariantUpsampleConv,
    EBSDSuper,
    BatchedEBSDSuper,
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
def train_convolution_layer(model, train_ds, crop_size, num_epochs=100, lr=1e-3,
                            log_interval=10, viz_interval=50, viz_context=None,
                            output_dir='./training_outputs'):
    """
    Train the convolution layer using Irrep Representation Loss.

    The target is reconstruction: conv output should match encoder output.
    This teaches the conv layer to not destroy the signal initially.

    Args:
        model: EBSDSuper model with conv_layer
        train_ds: QuaternionDataset providing (LR, HR) pairs in (C, H, W) format
        crop_size: Crop size for training patches
        num_epochs: Number of training epochs
        lr: Learning rate
        log_interval: Print loss every N epochs
        viz_interval: Render stage IPF maps every N epochs (0 to disable)
        viz_context: Dict with keys: q_tensor, q_hr_tensor, img_shape, hr_img_shape,
                     fcc_sym, decoder_stage — needed for periodic visualization
        output_dir: Directory to save training outputs
    """
    print("\n" + "="*70)
    print("TRAINING CONVOLUTION LAYER (Irrep Representation Loss)")
    print("="*70)

    device = model.device
    num_samples = len(train_ds)
    print(f"Number of LR/HR pairs: {num_samples}")

    # Setup optimizer (optimize conv_layer, upsample_layer, and hr_conv_layer parameters)
    trainable_params = (
        list(model.conv_layer.parameters())
        + list(model.upsample_layer.parameters())
        + list(model.hr_conv_layer.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=lr)
    # # OLD: optimizer only had conv_layer + upsample_layer (no hr_conv_layer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Loss function (Irrep loss between SR output irreps and HR irreps)
    criterion = IrrepLoss(lambda_f4=1.0, lambda_f6=1.0)

    # Count trainable parameters
    num_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    # # OLD: only counted conv_layer + upsample_layer
    print(f"Trainable parameters in conv_layer + upsample_layer + hr_conv_layer: {num_params}")
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
    model.upsample_layer.train()
    model.hr_conv_layer.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        total_f4 = 0.0
        total_f6 = 0.0
        epoch_start = time.time()

        for idx in range(num_samples):
            if idx % 50 == 0 or idx == num_samples - 1:
                print(f"  Epoch {epoch+1}/{num_epochs} - Sample {idx+1}/{num_samples}", end='\r', flush=True)
            q_lr_chw, q_hr_chw = train_ds[idx]

            # Convert from (C, H, W) to (H, W, C)
            q_lr_hwc = q_lr_chw.permute(1, 2, 0) if torch.is_tensor(q_lr_chw) else torch.from_numpy(q_lr_chw).permute(1, 2, 0)
            q_hr_hwc = q_hr_chw.permute(1, 2, 0) if torch.is_tensor(q_hr_chw) else torch.from_numpy(q_hr_chw).permute(1, 2, 0)

            # Determine LR shape
            img_shape = q_lr_hwc.shape[:2]

            # Crop LR/HR
            crop = min(crop_size, img_shape[0], img_shape[1])
            q_lr_crop = q_lr_hwc[:crop, :crop, :]
            hr_crop = crop * model.upsample_factor
            q_hr_crop = q_hr_hwc[:hr_crop, :hr_crop, :]

            # Flatten
            q_lr_flat = q_lr_crop.reshape(-1, 4)
            q_hr_flat = q_hr_crop.reshape(-1, 4)

            q_lr_tensor = q_lr_flat.float().to(device)
            q_lr_tensor = q_lr_tensor / torch.norm(q_lr_tensor, dim=1, keepdim=True)

            q_hr_tensor = q_hr_flat.float().to(device)
            q_hr_tensor = q_hr_tensor / torch.norm(q_hr_tensor, dim=1, keepdim=True)

            # Forward pass through full model (LR -> SR)
            # decode=False skips the SphericalSamplingDecoder: its output is not
            # used in the loss and allocating the (N_hr_pixels × 10 000) tensor
            # wastes ~10 GB of memory every step.
            outputs = model.forward(q_lr_tensor, img_shape=(crop, crop), decode=False)

            # HR target irreps (no grad)
            with torch.no_grad():
                f4_hr, f6_hr = model.encoder(q_hr_tensor)

            # SR output irreps (after HR conv refinement, grad flows to all trainable layers)
            f4_sr, f6_sr = outputs['hr_convolved_irreps']
            # # OLD: f4_sr, f6_sr = outputs['upsampled_irreps']

            # Compute Irrep loss (SR irreps vs HR irreps)
            loss, loss_dict = criterion(f4_sr, f6_sr, f4_hr, f6_hr)
            loss.backward()

            total_loss += loss_dict['loss_total']
            total_f4 += loss_dict['loss_f4']
            total_f6 += loss_dict['loss_f6']
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        # # OLD: only clipped conv_layer + upsample_layer
        
        # Update weights
        optimizer.step()
        
        # Update scheduler
        scheduler.step(total_loss / max(1, num_samples))
        
        # Record history
        avg_loss = total_loss / max(1, num_samples)
        avg_f4 = total_f4 / max(1, num_samples)
        avg_f6 = total_f6 / max(1, num_samples)

        history['loss_total'].append(avg_loss)
        history['loss_f4'].append(avg_f4)
        history['loss_f6'].append(avg_f6)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                f"Loss: {avg_loss:.6f} | "
                f"L_f4: {avg_f4:.6f} | "
                f"L_f6: {avg_f6:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s")

        # Save training curves every 10 epochs, overwriting the same file
        if (epoch + 1) % 10 == 0:
            print()  # flush the \r sample-progress line before printing
            plot_training_curves(history, output_dir)

        # Periodic stage visualization
        if viz_interval > 0 and viz_context is not None and (epoch + 1) % viz_interval == 0:
            print(f"\n  Rendering stage IPF maps at epoch {epoch+1}...")
            _viz = viz_context
            model.eval()
            with torch.no_grad():
                outputs_viz = model.forward(_viz['q_tensor'], img_shape=_viz['img_shape'])
                f4_enc, f6_enc = outputs_viz['encoded']
                f4_conv, f6_conv = outputs_viz['convolved']
                f4_up, f6_up = outputs_viz['upsampled_irreps']
                f4_hrc, f6_hrc = outputs_viz['hr_convolved_irreps']
                viz_hr_shape = outputs_viz['hr_shape']
                q_encoded = _viz['decoder_stage'].decode_final(f4_enc, f6_enc)
                q_encoded = match_symmetry_batch(_viz['q_tensor'], q_encoded, model.physics)
                q_convolved = _viz['decoder_stage'].decode_final(f4_conv, f6_conv)
                q_convolved = match_symmetry_batch(_viz['q_tensor'], q_convolved, model.physics)
                q_upsampled = _viz['decoder_stage'].decode_final(f4_up, f6_up)
                q_upsampled = match_symmetry_batch(_viz['q_hr_tensor'], q_upsampled, model.physics)
                q_hr_conv = _viz['decoder_stage'].decode_final(f4_hrc, f6_hrc)
                q_hr_conv = match_symmetry_batch(_viz['q_hr_tensor'], q_hr_conv, model.physics)
                q_out = match_symmetry_batch(_viz['q_hr_tensor'], outputs_viz['output'], model.physics)
            stages_viz = [
                {"name": "Stage 0: Input (LR)", "quaternions": outputs_viz['input'], "img_shape": _viz['img_shape']},
                {"name": "Stage 1: Encoded (LR)", "quaternions": q_encoded, "img_shape": _viz['img_shape']},
                {"name": f"Stage 2: Convolved (Epoch {epoch+1})", "quaternions": q_convolved, "img_shape": _viz['img_shape']},
                {"name": f"Stage 3: TransposeConv Upsampled (Epoch {epoch+1})", "quaternions": q_upsampled, "img_shape": viz_hr_shape},
                {"name": f"Stage 4: HR Conv (Epoch {epoch+1})", "quaternions": q_hr_conv, "img_shape": viz_hr_shape},
                {"name": f"Stage 5: Output (Epoch {epoch+1})", "quaternions": q_out, "img_shape": viz_hr_shape}
            ]
            viz_path = os.path.join(output_dir, f'stage_ipf_maps_epoch_{epoch+1:04d}.png')
            render_stage_ipf_maps(stages_viz, _viz['img_shape'], viz_path, _viz['fcc_sym'])
            model.conv_layer.train()
            model.upsample_layer.train()
            model.hr_conv_layer.train()

    total_time = time.time() - start_time
    print("-"*70)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    return history


def train_batched_convolution_layer(model, train_ds, crop_size, batch_size=100,
                                    min_batch_size=4, batch_size_halflife=50,
                                    num_epochs=100, lr=1e-3,
                                    log_interval=10, viz_interval=50,
                                    viz_context=None,
                                    output_dir='./training_outputs'):
    """
    Batched version of train_convolution_layer.

    Stacks `batch_size` LR/HR samples per forward pass so all spatial ops
    run as a single (B, C, H, W) kernel instead of B sequential (1, C, H, W)
    calls.  The existing train_convolution_layer is completely unchanged.

    Batch size is annealed: it starts at `batch_size`, halves every
    `batch_size_halflife` epochs, and stabilises at `min_batch_size`.
    The DataLoader is recreated whenever the batch size changes.

    Requires model to be a BatchedEBSDSuper instance.

    Args:
        model:                BatchedEBSDSuper instance
        train_ds:             QuaternionDataset (provides (4,H,W) tensors)
        crop_size:            LR crop size applied to each sample
        batch_size:           Initial number of samples stacked per forward pass
        min_batch_size:       Floor for the batch size schedule
        batch_size_halflife:  Epochs between each /2 reduction
        num_epochs, lr, log_interval, viz_interval, viz_context, output_dir:
            same semantics as train_convolution_layer
    """
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("BATCHED TRAINING (Irrep Representation Loss)")
    print("="*70)

    device = model.device
    print(f"Initial batch size: {batch_size}  |  min: {min_batch_size}  |  halflife: {batch_size_halflife} epochs")

    # DataLoader is created/recreated inside the epoch loop when the size changes
    dl = None
    prev_batch_size = None

    trainable_params = (
        list(model.conv_layer.parameters())
        + list(model.upsample_layer.parameters())
        + list(model.hr_conv_layer.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    criterion = IrrepLoss(lambda_f4=1.0, lambda_f6=1.0)

    num_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"Trainable parameters: {num_params}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {num_epochs}")
    print("-"*70)

    history = {'loss_total': [], 'loss_f4': [], 'loss_f6': [], 'lr': []}

    model.conv_layer.train()
    model.upsample_layer.train()
    model.hr_conv_layer.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        # Batch size schedule: halve every batch_size_halflife epochs, floor at min_batch_size
        halvings = epoch // batch_size_halflife
        cur_batch_size = max(min_batch_size, batch_size >> halvings)
        if cur_batch_size != prev_batch_size:
            dl = DataLoader(train_ds, batch_size=cur_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=False)
            if prev_batch_size is not None:
                print(f"\n  [Batch schedule] batch_size {prev_batch_size} → {cur_batch_size}")
            prev_batch_size = cur_batch_size

        optimizer.zero_grad()
        total_loss = 0.0
        total_f4   = 0.0
        total_f6   = 0.0
        num_batches = 0

        for batch_idx, (q_lr_chw_b, q_hr_chw_b) in enumerate(dl):
            # q_lr_chw_b: (B, 4, H, W),  q_hr_chw_b: (B, 4, H', W')
            B = q_lr_chw_b.shape[0]

            # (B, 4, H, W) → (B, H, W, 4)
            q_lr_hwc = q_lr_chw_b.permute(0, 2, 3, 1)
            q_hr_hwc = q_hr_chw_b.permute(0, 2, 3, 1)

            H_lr  = q_lr_hwc.shape[1]
            crop  = min(crop_size, H_lr)
            hr_crop = crop * model.upsample_factor

            # (B, crop, crop, 4) → (B*crop*crop, 4)
            q_lr_flat = q_lr_hwc[:, :crop,    :crop,    :].reshape(-1, 4).float().to(device)
            q_hr_flat = q_hr_hwc[:, :hr_crop, :hr_crop, :].reshape(-1, 4).float().to(device)

            q_lr_flat = q_lr_flat / torch.norm(q_lr_flat, dim=1, keepdim=True)
            q_hr_flat = q_hr_flat / torch.norm(q_hr_flat, dim=1, keepdim=True)

            outputs = model.forward(q_lr_flat, img_shape=(crop, crop),
                                    batch_size=B, decode=False)

            with torch.no_grad():
                f4_hr_tgt, f6_hr_tgt = model.encoder(q_hr_flat)

            f4_sr, f6_sr = outputs['hr_convolved_irreps']
            loss, loss_dict = criterion(f4_sr, f6_sr, f4_hr_tgt, f6_hr_tgt)
            loss.backward()

            total_loss += loss_dict['loss_total']
            total_f4   += loss_dict['loss_f4']
            total_f6   += loss_dict['loss_f6']
            num_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(dl)}",
                      end='\r', flush=True)

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss / max(1, num_batches))

        avg_loss = total_loss / max(1, num_batches)
        avg_f4   = total_f4   / max(1, num_batches)
        avg_f6   = total_f6   / max(1, num_batches)

        history['loss_total'].append(avg_loss)
        history['loss_f4'].append(avg_f4)
        history['loss_f6'].append(avg_f6)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                  f"BS: {cur_batch_size:3d} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"L_f4: {avg_f4:.6f} | "
                  f"L_f6: {avg_f6:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if (epoch + 1) % 10 == 0:
            print()
            plot_training_curves(history, output_dir)

        if viz_interval > 0 and viz_context is not None and (epoch + 1) % viz_interval == 0:
            print(f"\n  Rendering stage IPF maps at epoch {epoch+1}...")
            _viz = viz_context
            model.eval()
            with torch.no_grad():
                outputs_viz = model.forward(_viz['q_tensor'], img_shape=_viz['img_shape'],
                                            batch_size=1, decode=True)
                f4_enc, f6_enc   = outputs_viz['encoded']
                f4_conv, f6_conv = outputs_viz['convolved']
                f4_up, f6_up     = outputs_viz['upsampled_irreps']
                f4_hrc, f6_hrc   = outputs_viz['hr_convolved_irreps']
                viz_hr_shape     = outputs_viz['hr_shape']
                q_encoded   = _viz['decoder_stage'].decode_final(f4_enc, f6_enc)
                q_encoded   = match_symmetry_batch(_viz['q_tensor'],    q_encoded,   model.physics)
                q_convolved = _viz['decoder_stage'].decode_final(f4_conv, f6_conv)
                q_convolved = match_symmetry_batch(_viz['q_tensor'],    q_convolved, model.physics)
                q_upsampled = _viz['decoder_stage'].decode_final(f4_up, f6_up)
                q_upsampled = match_symmetry_batch(_viz['q_hr_tensor'], q_upsampled, model.physics)
                q_hr_conv   = _viz['decoder_stage'].decode_final(f4_hrc, f6_hrc)
                q_hr_conv   = match_symmetry_batch(_viz['q_hr_tensor'], q_hr_conv,   model.physics)
                q_out       = match_symmetry_batch(_viz['q_hr_tensor'], outputs_viz['output'], model.physics)
            stages_viz = [
                {"name": "Stage 0: Input (LR)",                                "quaternions": outputs_viz['input'], "img_shape": _viz['img_shape']},
                {"name": "Stage 1: Encoded (LR)",                              "quaternions": q_encoded,           "img_shape": _viz['img_shape']},
                {"name": f"Stage 2: Convolved (Epoch {epoch+1})",              "quaternions": q_convolved,         "img_shape": _viz['img_shape']},
                {"name": f"Stage 3: TransposeConv Upsampled (Epoch {epoch+1})","quaternions": q_upsampled,         "img_shape": viz_hr_shape},
                {"name": f"Stage 4: HR Conv (Epoch {epoch+1})",                "quaternions": q_hr_conv,           "img_shape": viz_hr_shape},
                {"name": f"Stage 5: Output (Epoch {epoch+1})",                 "quaternions": q_out,               "img_shape": viz_hr_shape},
            ]
            viz_path = os.path.join(output_dir, f'stage_ipf_maps_epoch_{epoch+1:04d}.png')
            render_stage_ipf_maps(stages_viz, _viz['img_shape'], viz_path, _viz['fcc_sym'])
            model.conv_layer.train()
            model.upsample_layer.train()
            model.hr_conv_layer.train()

    total_time = time.time() - start_time
    print("-"*70)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
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
        # If f4/f6 contain multiple irrep copies (e.g., SR r^2 copies),
        # pick a single copy for decoding (index 0).
        if f4.shape[1] % 9 == 0 and f4.shape[1] != 9:
            copies = f4.shape[1] // 9
            f4 = f4.view(-1, copies, 9)[:, 0, :]
        if f6.shape[1] % 13 == 0 and f6.shape[1] != 13:
            copies = f6.shape[1] // 13
            f6 = f6.view(-1, copies, 13)[:, 0, :]

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

        # Reshape to image (allow per-stage override)
        stage_shape = stage.get('img_shape', img_shape)
        H, W = stage_shape
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

    dataset_out_root = "/data/home/umang/Materials/Materials_data_mount/EBSD//"
    dataset_name = "IN718_FZ_2D_SR_x4/Open718_QSR_x4/"
    dataset_dir = os.path.join(dataset_out_root, dataset_name)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ==============================================================================
    # 1. INITIALIZE MODEL
    # ==============================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = BatchedEBSDSuper(device=device, grid_samples=10000, upsample_factor=4)
    print(f"Model initialized on {device}")

    # ==============================================================================
    # 2. LOAD SAMPLE DATA
    # ==============================================================================
    print("\n" + "="*70)
    print("LOADING SAMPLE DATA")
    print("="*70)

    train_ds = QuaternionDataset(
        dataset_root=dataset_dir,
        split="Train",
        preload=True,
        preload_torch=True,  # preload as CPU torch tensors
    )

    test_ds = QuaternionDataset(
        dataset_root=dataset_dir,
        split="Test",
        preload=True,
        preload_torch=True,
    )
    print(f"Train samples: {len(train_ds)}  |  Test samples: {len(test_ds)}")

    # Get first test sample for visualization: (LR, HR) pair in (C, H, W) format
    q_lr_chw = test_ds[0][0]  # LR quaternions (C, H, W)
    q_hr_chw = test_ds[0][1]  # HR quaternions (C, H, W)
    print(f"Viz LR sample shape (C,H,W): {q_lr_chw.shape}")
    print(f"Viz HR sample shape (C,H,W): {q_hr_chw.shape}")

    # Convert from (C, H, W) to (H, W, C)
    q_lr_hwc = q_lr_chw.permute(1, 2, 0)
    q_hr_hwc = q_hr_chw.permute(1, 2, 0)

    # Take a smaller crop for faster training (256x256)
    full_lr_shape = q_lr_hwc.shape[:2]
    crop_size = min(256, full_lr_shape[0], full_lr_shape[1])
    hr_crop_size = crop_size * model.upsample_factor

    q_lr_crop = q_lr_hwc[:crop_size, :crop_size, :]
    q_hr_crop = q_hr_hwc[:hr_crop_size, :hr_crop_size, :]

    img_shape = (crop_size, crop_size)
    hr_img_shape = (hr_crop_size, hr_crop_size)

    # Flatten and move to device
    q_tensor = q_lr_crop.reshape(-1, 4).to(device)
    q_tensor = q_tensor / torch.norm(q_tensor, dim=1, keepdim=True)

    q_hr_tensor = q_hr_crop.reshape(-1, 4).to(device)
    q_hr_tensor = q_hr_tensor / torch.norm(q_hr_tensor, dim=1, keepdim=True)

    print(f"Viz LR tensor shape: {q_tensor.shape}")
    print(f"Viz HR tensor shape: {q_hr_tensor.shape}")
    print(f"LR image shape: {img_shape}")
    print(f"HR image shape: {hr_img_shape}")
    
    # ==============================================================================
    # 3. VISUALIZE BEFORE TRAINING (untrained conv layer)
    # ==============================================================================
    print("\n" + "="*70)
    print("VISUALIZATION BEFORE TRAINING")
    print("="*70)
    
    fcc_sym = utils.symmetry_utils.resolve_symmetry(train_ds.symmetry)
    decoder_stage = StageDecoder(model.physics, model.decoder)
    decoder_stage.to(device)
    
    # Run forward pass before training
    model.eval()
    with torch.no_grad():
        outputs_before = model.forward(q_tensor, img_shape=img_shape, batch_size=1, decode=True)

        # Decode stages
        f4_enc, f6_enc = outputs_before['encoded']
        q_encoded = decoder_stage.decode_final(f4_enc, f6_enc)
        q_encoded = match_symmetry_batch(q_tensor, q_encoded, model.physics)

        f4_conv, f6_conv = outputs_before['convolved']
        q_convolved_before = decoder_stage.decode_final(f4_conv, f6_conv)
        q_convolved_before = match_symmetry_batch(q_tensor, q_convolved_before, model.physics)

        f4_up, f6_up = outputs_before['upsampled_irreps']
        before_hr_shape = outputs_before['hr_shape']
        q_upsampled_before = decoder_stage.decode_final(f4_up, f6_up)
        q_upsampled_before = match_symmetry_batch(q_hr_tensor, q_upsampled_before, model.physics)

        f4_hrc, f6_hrc = outputs_before['hr_convolved_irreps']
        q_hr_conv_before = decoder_stage.decode_final(f4_hrc, f6_hrc)
        q_hr_conv_before = match_symmetry_batch(q_hr_tensor, q_hr_conv_before, model.physics)

        q_output_before = match_symmetry_batch(q_hr_tensor, outputs_before['output'], model.physics)

    stages_before = [
        {"name": "Stage 0: Input (LR)", "quaternions": outputs_before['input'], "img_shape": img_shape},
        {"name": "Stage 1: Encoded (LR)", "quaternions": q_encoded, "img_shape": img_shape},
        {"name": "Stage 2: Convolved (LR, BEFORE)", "quaternions": q_convolved_before, "img_shape": img_shape},
        {"name": "Stage 3: TransposeConv Upsampled (HR, BEFORE)", "quaternions": q_upsampled_before, "img_shape": before_hr_shape},
        {"name": "Stage 4: HR Conv (HR, BEFORE)", "quaternions": q_hr_conv_before, "img_shape": before_hr_shape},
        {"name": "Stage 5: Output (HR, BEFORE)", "quaternions": q_output_before, "img_shape": before_hr_shape}
    ]
    
    render_stage_ipf_maps(stages_before, img_shape, 
                          os.path.join(output_dir, 'stage_ipf_maps_BEFORE_training.png'), fcc_sym)
    
    # ==============================================================================
    # 4. TRAIN CONVOLUTION LAYER
    # ==============================================================================
    VIZ_INTERVAL = 50  # Render stage IPF maps every N epochs

    viz_context = {
        'q_tensor': q_tensor,
        'q_hr_tensor': q_hr_tensor,
        'img_shape': img_shape,
        'hr_img_shape': hr_img_shape,
        'fcc_sym': fcc_sym,
        'decoder_stage': decoder_stage,
    }

    history = train_batched_convolution_layer(
        model=model,
        train_ds=train_ds,
        crop_size=crop_size,
        batch_size=100,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        log_interval=LOG_INTERVAL,
        viz_interval=VIZ_INTERVAL,
        viz_context=viz_context,
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
        outputs_after = model.forward(q_tensor, img_shape=img_shape, batch_size=1, decode=True)

        # Decode stages
        f4_enc_after, f6_enc_after = outputs_after['encoded']
        f4_conv, f6_conv = outputs_after['convolved']
        q_encoded_after = decoder_stage.decode_final(f4_enc_after, f6_enc_after)
        q_encoded_after = match_symmetry_batch(q_tensor, q_encoded_after, model.physics)
        q_convolved_after = decoder_stage.decode_final(f4_conv, f6_conv)
        q_convolved_after = match_symmetry_batch(q_tensor, q_convolved_after, model.physics)

        f4_up_after, f6_up_after = outputs_after['upsampled_irreps']
        after_hr_shape = outputs_after['hr_shape']
        q_upsampled_after = decoder_stage.decode_final(f4_up_after, f6_up_after)
        q_upsampled_after = match_symmetry_batch(q_hr_tensor, q_upsampled_after, model.physics)

        f4_hrc_after, f6_hrc_after = outputs_after['hr_convolved_irreps']
        q_hr_conv_after = decoder_stage.decode_final(f4_hrc_after, f6_hrc_after)
        q_hr_conv_after = match_symmetry_batch(q_hr_tensor, q_hr_conv_after, model.physics)

        q_output_after = match_symmetry_batch(q_hr_tensor, outputs_after['output'], model.physics)

    stages_after = [
        {"name": "Stage 0: Input (LR)", "quaternions": outputs_after['input'], "img_shape": img_shape},
        {"name": "Stage 1: Encoded (LR)", "quaternions": q_encoded_after, "img_shape": img_shape},
        {"name": "Stage 2: Convolved (LR, AFTER)", "quaternions": q_convolved_after, "img_shape": img_shape},
        {"name": "Stage 3: TransposeConv Upsampled (HR, AFTER)", "quaternions": q_upsampled_after, "img_shape": after_hr_shape},
        {"name": "Stage 4: HR Conv (HR, AFTER)", "quaternions": q_hr_conv_after, "img_shape": after_hr_shape},
        {"name": "Stage 5: Output (HR, AFTER)", "quaternions": q_output_after, "img_shape": after_hr_shape}
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