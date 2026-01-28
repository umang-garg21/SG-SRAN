import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model and encoder from local module
# Ensure these files are in your python path
from equiformer_ebsdSr import CrystalTransformerSR, FCCPhysics, FCCEncoder, smooth_boundary_map, clean_boundary_map


# Helpers: convert boundary tensors to PIL / RGB and save with repeated columns
def tensor_to_pil_image(tensor):
    """Convert a PyTorch tensor (B,1,H,W) or (1,1,H,W) or (1,H,W) to a PIL RGB image."""
    if Image is None:
        raise RuntimeError("Pillow is required to create images")
    import numpy as _np

    if hasattr(tensor, 'cpu'):
        arr = tensor.squeeze().cpu().numpy()
    else:
        arr = _np.array(tensor).squeeze()

    # Ensure 2D
    if arr.ndim != 2:
        # If flattened (Npix,), try to reshape to square-ish
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else arr

    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        # Constant array: map relative to absolute max if positive, otherwise produce zeros
        if mx > 0:
            norm = ((_np.clip(arr, 0.0, mx) / mx) * 255.0).astype(_np.uint8)
        else:
            norm = _np.zeros_like(arr, dtype=_np.uint8)
    else:
        norm = ((_np.clip(arr, mn, mx) - mn) / (mx - mn) * 255.0).astype(_np.uint8)

    rgb = _np.stack([norm, norm, norm], axis=2)
    return Image.fromarray(rgb)


def tensor_to_rgb_png(tensor, out_path, repeat_cols=1, gutter_px=0):
    """Save `tensor` as an RGB PNG at `out_path`.

    If `repeat_cols>1`, repeat the image horizontally and insert `gutter_px` pixels between tiles.
    """
    pil = tensor_to_pil_image(tensor)
    if repeat_cols is None or repeat_cols <= 1:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        pil.save(out_path)
        return out_path

    w, h = pil.size
    total_w = w * repeat_cols + max(0, gutter_px) * (repeat_cols - 1)
    out = Image.new('RGB', (total_w, h), (255, 255, 255))
    x = 0
    for i in range(repeat_cols):
        out.paste(pil, (x, 0))
        x += w + gutter_px

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.save(out_path)
    return out_path


def tensor_to_rgb_png_match_ref(tensor, ref_png_path, out_path, cols=3):
    """Create an RGB PNG from `tensor` and tile it `cols` times to match the width/height of `ref_png_path`.

    This resizes the tensor-derived image to (ref_width//cols, ref_height) and pastes it across.
    """
    if Image is None:
        raise RuntimeError("Pillow is required to create images")

    pil = tensor_to_pil_image(tensor)
    try:
        ref = Image.open(ref_png_path).convert('RGB')
    except Exception as e:
        # Fallback to simple repeat if ref cannot be opened
        return tensor_to_rgb_png(tensor, out_path, repeat_cols=cols)

    ref_w, ref_h = ref.size
    tile_w = max(1, ref_w // cols)
    tile_h = ref_h

    # Resize tile to exactly the target tile size while preserving aspect ratio via fit
    try:
        tile = pil.resize((tile_w, tile_h), resample=Image.BILINEAR)
    except Exception:
        tile = pil

    out_im = Image.new('RGB', (tile_w * cols, tile_h), (255, 255, 255))
    for i in range(cols):
        out_im.paste(tile, (i * tile_w, 0))

    # If ref width and computed width differ due to integer division, pad to match ref width
    if out_im.size[0] != ref_w:
        new_out = Image.new('RGB', (ref_w, ref_h), (255, 255, 255))
        x = (ref_w - out_im.size[0]) // 2
        new_out.paste(out_im, (x, 0))
        out_im = new_out

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out_im.save(out_path)
    try:
        ref.close()
    except Exception:
        pass
    return out_path

class PairedQuaternionDataset(Dataset):
    """Loads paired LR/HR quaternion maps. Supports .npy or image files with 4 channels.

    Files are paired by stem (filename without suffix)."""
    def __init__(self, lr_dir, hr_dir, exts=None):
        self.lr_dir = Path(os.path.expanduser(lr_dir))
        self.hr_dir = Path(os.path.expanduser(hr_dir))
        self.exts = exts or ['.npy', '.png', '.jpg', '.tif']

        # Call helper method to index files
        self.lr_files = self._index_files(self.lr_dir)
        self.hr_files = self._index_files(self.hr_dir)

        # Prefer pairing by direct stem replacement: replace 'lr' with 'hr' in stem names.
        hr_stems = set(self.hr_files.keys())
        pairs = []
        for lr_stem, lr_path in self.lr_files.items():
            # Try common patterns to convert lr->hr
            candidates = []
            candidates.append(lr_stem)
            candidates.append(lr_stem.replace('_lr_', '_hr_'))
            candidates.append(lr_stem.replace(' lr ', ' hr '))
            candidates.append(lr_stem.replace('lr', 'hr'))
            # also try replacing first occurrence only
            if 'lr' in lr_stem:
                candidates.append(lr_stem.replace('lr', 'hr', 1))

            found = None
            for cand in candidates:
                if cand in self.hr_files:
                    found = cand
                    break

            if found is not None:
                pairs.append((lr_path, self.hr_files[found]))

        # If no pairs found by replacement, fall back to block_### indexing
        if len(pairs) == 0:
            def build_index_map(dmap):
                idx_map = {}
                for stem, path in dmap.items():
                    m = re.search(r'block_(\d+)', stem)
                    if m:
                        key = int(m.group(1))
                    else:
                        key = stem
                    idx_map[key] = path
                return idx_map

            lr_index = build_index_map(self.lr_files)
            hr_index = build_index_map(self.hr_files)
            common_keys = sorted(set(lr_index.keys()) & set(hr_index.keys()))
            if len(common_keys) == 0:
                raise RuntimeError(f'No matching LR/HR pairs found between {lr_dir} and {hr_dir}')
            pairs = [(lr_index[k], hr_index[k]) for k in common_keys]

        # store pairs and keys
        self.pairs = pairs
        self.keys = list(range(len(self.pairs)))

    def _index_files(self, d: Path):
        files = {}
        for ext in self.exts:
            for p in d.rglob(f'*{ext}'):
                files[p.stem] = p
        return files

    def _load_quat(self, p: Path):
        if p.suffix == '.npy':
            arr = np.load(p)
        else:
            img = Image.open(p)
            arr = np.array(img)
        
        if arr.ndim == 2 and arr.shape[1] != 4 and arr.shape[0] != 4:
             # Just a sanity check for single channel 2D images
             raise RuntimeError(f'Found single-channel file {p}; expected 4-channel quaternion map or .npy')

        arr = arr.astype(np.float32)

        # Handle various shapes: (H, W, 4), (4, H, W), or flattened
        if arr.ndim == 3:
            # either (H,W,4) or (4,H,W)
            if arr.shape[-1] == 4:
                hw4 = arr
            elif arr.shape[0] == 4:
                # (4, H, W) -> transpose to (H, W, 4)
                hw4 = np.transpose(arr, (1, 2, 0))
            else:
                raise RuntimeError(f'Unexpected 3D array shape {arr.shape} in {p}; cannot interpret as quaternions')
        elif arr.ndim == 1:
            if arr.size % 4 != 0:
                raise RuntimeError(f'Flat array length {arr.size} not divisible by 4 in {p}')
            hw4 = arr.reshape(-1, 4)
        elif arr.ndim == 2 and arr.shape[1] == 4:
            # (Npix, 4)
            hw4 = arr
        else:
            raise RuntimeError(f'Unexpected array shape {arr.shape} in {p}')

        # Normalize quaternions
        if hw4.ndim == 3:
            flat = hw4.reshape(-1, 4)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            flat = flat / norms
            hw4 = flat.reshape(hw4.shape)
        else:
            norms = np.linalg.norm(hw4, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            hw4 = hw4 / norms

        return hw4

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        p_lr = lr_path
        p_hr = hr_path
        q_lr = self._load_quat(p_lr)
        q_hr = self._load_quat(p_hr)

        if q_lr.shape[0] * q_lr.shape[1] != q_hr.shape[0] * q_hr.shape[1]:
            # Check for rough divisibility
            if q_hr.shape[0] % q_lr.shape[0] != 0 or q_hr.shape[1] % q_lr.shape[1] != 0:
                # Warning: strict divisibility check might fail for odd sizes, 
                # but generally expected for SR tasks
                pass
        
        # derive LR and HR shapes (expecting (H,W,4))
        if q_lr.ndim == 3:
            H_lr, W_lr = int(q_lr.shape[0]), int(q_lr.shape[1])
        else:
            H_lr, W_lr = None, None

        if q_hr.ndim == 3:
            H_hr, W_hr = int(q_hr.shape[0]), int(q_hr.shape[1])
        else:
            H_hr, W_hr = None, None

        # flatten per-pixel
        q_lr_flat = torch.tensor(q_lr.reshape(-1, 4), dtype=torch.float32)
        q_hr_flat = torch.tensor(q_hr.reshape(-1, 4), dtype=torch.float32)
        
        # return LR stem as key for traceability
        key = p_lr.stem
        return q_lr_flat, q_hr_flat, (H_lr, W_lr), (H_hr, W_hr), key


def quat_loss(pred, target):
    """Quaternion L2 loss accounting for sign ambiguity."""
    # pred, target: (Npix,4)
    diff1 = (pred - target).pow(2).sum(dim=1)
    diff2 = (pred + target).pow(2).sum(dim=1)
    return torch.mean(torch.min(diff1, diff2))


def normalize_shape(s):
    """Normalize a collated shape entry into (H, W) where H and W are ints or None.

    Handles types: None, tuple/list of ints, tuple/list of tensors, torch.Tensor of various shapes.
    If a batch is present, the first item's dimensions are used.
    """
    if s is None:
        return (None, None)
    # Torch tensor: could be shape (2,), (batch,2), or flat
    if isinstance(s, torch.Tensor):
        try:
            if s.dim() == 2 and s.size(1) >= 2:
                # (batch, 2) -> take first row
                row = s[0]
                return (int(row[0].item()), int(row[1].item()))
            flat = s.view(-1)
            if flat.numel() >= 2:
                return (int(flat[0].item()), int(flat[1].item()))
        except Exception:
            return (None, None)

    # list or tuple: elements may be ints, None, or tensors (batched)
    if isinstance(s, (list, tuple)):
        if len(s) >= 2:
            a0, a1 = s[0], s[1]
            # if elements are tensors, try extracting scalar
            if isinstance(a0, torch.Tensor):
                try:
                    return (int(a0.item()), int(a1.item()))
                except Exception:
                    # a0 may be a batch tensor
                    try:
                        return (int(a0[0].item()), int(a1[0].item()))
                    except Exception:
                        return (None, None)
            else:
                try:
                    return (int(a0) if a0 is not None else None, int(a1) if a1 is not None else None)
                except Exception:
                    return (None, None)

    return (None, None)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    physics = FCCPhysics(device)
    encoder = FCCEncoder(physics).to(device)
    model = CrystalTransformerSR(physics, scale_factor=args.scale, depth=args.depth).to(device)

    dataset = PairedQuaternionDataset(args.lr_dir, args.hr_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # optimizer
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    start_epoch = 1
    best_val = float('inf')
    epochs_no_improve = 0

    # Force checkpoint directory to repository transformer folder (user requested absolute path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_ckpt_dir = '/data/home/umang/Materials/e3nn_Reynolds/e3nn_experimentation/4x4_superresolution_transformers/checkpoints'
    ckpt_dir_ts = os.path.join(base_ckpt_dir, timestamp)
    os.makedirs(ckpt_dir_ts, exist_ok=True)

    # Log file location
    print(f"Logging to: {ckpt_dir_ts}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        for q_lr_flat, q_hr_flat, lr_shape, hr_shape, key in pbar:
            q_lr_flat = q_lr_flat.to(device)
            q_hr_flat = q_hr_flat.to(device)
            # Normalize collated shapes coming from DataLoader into (H, W) ints or (None, None)
            lr_shape = normalize_shape(lr_shape)
            hr_shape = normalize_shape(hr_shape)

            # Encode LR -> predicted HR coeffs
            f0, f1, f4, f6 = encoder(q_lr_flat, img_shape=lr_shape)
            f4_pred, f6_pred = model(f0, f1, f4, f6, lr_shape)

            # Encode HR ground truth to get target coefficients (differentiable)
            with torch.no_grad():
                _, _, f4_gt, f6_gt = encoder(q_hr_flat, img_shape=hr_shape)

            # Ensure coefficient tensors have matching batch dims
            if f4_pred.dim() == 2 and f4_gt.dim() == 3:
                f4_pred = f4_pred.unsqueeze(0)
            if f6_pred.dim() == 2 and f6_gt.dim() == 3:
                f6_pred = f6_pred.unsqueeze(0)
            if f4_pred.dim() == 3 and f4_gt.dim() == 2:
                f4_gt = f4_gt.unsqueeze(0)
            if f6_pred.dim() == 3 and f6_gt.dim() == 2:
                f6_gt = f6_gt.unsqueeze(0)

            # Coefficient MSE loss (differentiable)
            loss4 = nn.functional.mse_loss(f4_pred, f4_gt)
            loss6 = nn.functional.mse_loss(f6_pred, f6_gt)
            loss = loss4 + loss6

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / (pbar.n+1):.6f}'})

        scheduler.step()

        # Validation every args.val_every epochs
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss = validate(model, encoder, dataset, device)
            print(f'Epoch {epoch}: train_loss={running_loss/len(loader):.6f} val_loss={val_loss:.6f}')
            
            # create per-epoch subfolder under the timestamped run folder
            epoch_dir = os.path.join(ckpt_dir_ts, f'epoch_{epoch:04d}')
            os.makedirs(epoch_dir, exist_ok=True)

            ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'val_loss': val_loss}

            # Early stopping & best model saving
            best_path = os.path.join(ckpt_dir_ts, 'best.pt')
            if (best_val - val_loss) > getattr(args, 'min_delta', 1e-4):
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(ckpt, best_path)
                print(f'  -> New best val {best_val:.6f}, saved {best_path}')
            else:
                epochs_no_improve += 1
                print(f'  -> No improvement for {epochs_no_improve} validations (patience {getattr(args, "patience", 5)})')

            if epochs_no_improve >= getattr(args, 'patience', 5):
                print(f'Early stopping: no improvement in last {getattr(args, "patience", 5)} validations.')
                break

            # Save IPF comparison visuals
            try:
                save_ipf_comparisons(dataset, encoder, model, epoch_dir, epoch, n_samples=5, scale=args.scale)
            except Exception as e:
                import traceback
                print('Warning: failed to save IPF comparisons; full traceback below:')
                traceback.print_exc()

    print('Training complete')


def validate(model, encoder, dataset, device, max_samples=50):
    model.eval()
    with torch.no_grad():
        total = 0.0
        count = 0
        for i in range(min(len(dataset), max_samples)):
            q_lr_flat, q_hr_flat, lr_shape, hr_shape, key = dataset[i]
            q_lr_flat = q_lr_flat.to(device)
            q_hr_flat = q_hr_flat.to(device)

            # Normalize collated shapes (handle batched tensors coming from DataLoader)
            lr_shape = normalize_shape(lr_shape)
            hr_shape = normalize_shape(hr_shape)

            f0, f1, f4, f6 = encoder(q_lr_flat, img_shape=lr_shape)
            f4_pred, f6_pred = model(f0, f1, f4, f6, lr_shape)

            _, _, f4_gt, f6_gt = encoder(q_hr_flat, img_shape=hr_shape)

            # Align dimensions if encoder/model return different batch axes
            if f4_pred.dim() == 2 and f4_gt.dim() == 3:
                f4_pred = f4_pred.unsqueeze(0)
            if f6_pred.dim() == 2 and f6_gt.dim() == 3:
                f6_pred = f6_pred.unsqueeze(0)
            if f4_pred.dim() == 3 and f4_gt.dim() == 2:
                f4_gt = f4_gt.unsqueeze(0)
            if f6_pred.dim() == 3 and f6_gt.dim() == 2:
                f6_gt = f6_gt.unsqueeze(0)

            loss = nn.functional.mse_loss(f4_pred, f4_gt) + nn.functional.mse_loss(f6_pred, f6_gt)
            total += loss.item()
            count += 1
        return total / max(1, count)


def save_ipf_comparisons(dataset, encoder, model, out_dir, epoch, n_samples=5, scale=4):
    """Render IPF comparisons (HR, LR upsampled, SR) for first n_samples and save PNGs into out_dir."""
    try:
        from visualization.ipf_render import render_ipf_image
    except ImportError:
        try:
            # Fallback path if module is not in root
            sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')
            from visualization.ipf_render import render_ipf_image
        except ImportError:
            print("Could not import render_input_output_comparison. Skipping visualization.")
            return

    from orix.crystal_map import Phase

    os.makedirs(out_dir, exist_ok=True)
    # 225 is Fm-3m (Face Centered Cubic)
    fcc_sym = Phase(space_group=225).point_group

    # If PIL not available, skip visualization (we still allow training)
    if Image is None:
        print('PIL not installed: skipping IPF visualizations.')
        return

    for i in range(min(n_samples, len(dataset))):
        lr_path, hr_path = dataset.pairs[i]
        
        # Helper to load array directly from path again for visualization logic
        def load_safe(p):
            if p.suffix == '.npy':
                return np.load(p)
            else:
                return np.array(Image.open(p))

        q_lr = load_safe(lr_path)
        q_hr = load_safe(hr_path)

        # normalize and ensure shape (H,W,4)
        def to_hw4(arr):
            arr = arr.astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 4:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 1:
                pass 
            return arr

        q_lr_img = to_hw4(q_lr)
        q_hr_img = to_hw4(q_hr)

        # Compute SR quaternions from model
        q_lr_flat = torch.tensor(q_lr_img.reshape(-1, 4), dtype=torch.float32)
        
        # Need device for inference
        device = next(model.parameters()).device
        q_lr_flat = q_lr_flat.to(device)

        with torch.no_grad():
            f0_lr, f1_lr, f4_lr, f6_lr = encoder(q_lr_flat, img_shape=(q_lr_img.shape[0], q_lr_img.shape[1]))
            f4_pred, f6_pred, q_inter = model(f0_lr, f1_lr, f4_lr, f6_lr, (q_lr_img.shape[0], q_lr_img.shape[1]), return_intermediate=True)
            q_stages = model(f0_lr, f1_lr, f4_lr, f6_lr, (q_lr_img.shape[0], q_lr_img.shape[1]), return_all_intermediates=True)
            q_sr_flat = model.predict_quaternions(f4_pred, f6_pred)

        q_sr = q_sr_flat.cpu().numpy()
        
        # Compute the upsampled LR boundary as used in the model forward
        f0_lr_img_for_interp = f0_lr.view(1, q_lr_img.shape[0], q_lr_img.shape[1], 1).permute(0, 3, 1, 2)
        f0_hr_interpolated = torch.nn.functional.interpolate(f0_lr_img_for_interp, scale_factor=scale, mode='bilinear')
        
        # reshape to HR image dimensions
        H_hr, W_hr = q_hr_img.shape[0], q_hr_img.shape[1]
        
        # Ensure SR output matches HR dimensions
        if q_sr.shape[0] == H_hr * W_hr:
            q_sr_img = q_sr.reshape(H_hr, W_hr, 4)
        else:
            # If dimensions don't match, visualization will fail
            print(f"Skipping sample {i} vis: SR shape {q_sr.shape} != HR shape {H_hr}x{W_hr}")
            continue

        # Compute boundary from q_inter (post-upsample)
        q_inter_np = q_inter.cpu().numpy()
        q_inter_img = q_inter_np.reshape(H_hr, W_hr, 4)
        q_inter_flat = torch.tensor(q_inter_img.reshape(-1, 4), dtype=torch.float32).to(device)
        with torch.no_grad():
            f0_inter, _, _, _ = encoder(q_inter_flat, img_shape=(H_hr, W_hr))
        f0_inter_img = f0_inter.view(1, H_hr, W_hr, 1).permute(0, 3, 1, 2)

        # Compute boundaries for HR and SR
        q_hr_flat = torch.tensor(q_hr_img.reshape(-1, 4), dtype=torch.float32).to(device)
        q_sr_flat_tensor = torch.tensor(q_sr_img.reshape(-1, 4), dtype=torch.float32).to(device)
        with torch.no_grad():
            f0_hr, _, _, _ = encoder(q_hr_flat, img_shape=(H_hr, W_hr))
            f0_sr, _, _, _ = encoder(q_sr_flat_tensor, img_shape=(H_hr, W_hr))

        f0_hr_img = f0_hr.view(1, H_hr, W_hr, 1).permute(0, 3, 1, 2)  # (1,1,H,W)
        f0_sr_img = f0_sr.view(1, H_hr, W_hr, 1).permute(0, 3, 1, 2)

        # Upsample LR to HR using nearest neighbor for comparison
        h_ratio = H_hr // q_lr_img.shape[0]
        w_ratio = W_hr // q_lr_img.shape[1]
        
        if h_ratio > 0 and w_ratio > 0:
            q_lr_up = np.repeat(np.repeat(q_lr_img, h_ratio, axis=0), w_ratio, axis=1)
            # Handle potential edge pixels if division wasn't perfect
            q_lr_up = q_lr_up[:H_hr, :W_hr, :]
        else:
            q_lr_up = q_lr_img # Fallback if sizes are weird

        # Compute boundary for upsampled LR quaternions
        q_lr_up_flat = torch.tensor(q_lr_up.reshape(-1, 4), dtype=torch.float32).to(device)
        with torch.no_grad():
            f0_lr_up, _, _, _ = encoder(q_lr_up_flat, img_shape=(H_hr, W_hr))
        f0_lr_up_img = f0_lr_up.view(1, H_hr, W_hr, 1).permute(0, 3, 1, 2)

        # Render three IPF images (each will be 3 columns: IPF-X, IPF-Y, IPF-Z)
        hr_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_hr.png')
        lr_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_lr.png')
        sr_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_sr.png')

        # add a small gutter so the three IPF columns are visually separated
        IPF_GUTTER = 8
        render_ipf_image(
            q_hr_img,
            fcc_sym,
            out_png=hr_png,
            ref_dir='ALL',
            include_key=False,
            overwrite=True,
            format_input=False,
            gutter_px=IPF_GUTTER,
        )
        render_ipf_image(
            q_lr_up,
            fcc_sym,
            out_png=lr_png,
            ref_dir='ALL',
            include_key=False,
            overwrite=True,
            format_input=False,
            gutter_px=IPF_GUTTER,
        )
        render_ipf_image(
            q_sr_img,
            fcc_sym,
            out_png=sr_png,
            ref_dir='ALL',
            include_key=False,
            overwrite=True,
            format_input=False,
            gutter_px=IPF_GUTTER,
        )

        # --- Build smoothed boundary maps for visualization ---
        try:
            # Compute LR f0 using encoder (was already computed above but ensure shapes)
            # f0 is returned as (Npix,1)
            f0_lr = f0_lr
            H_lr, W_lr = q_lr_img.shape[0], q_lr_img.shape[1]
            f0_img = f0_lr.view(1, H_lr, W_lr, 1).permute(0, 3, 1, 2)  # (1,1,H,W)
            # Smooth LR boundary
            # Clean speckles (morphological) then smooth for visualization consistency
            # try:
            #     f0_img_clean = clean_boundary_map(f0_img, thresh=0.087, open_kernel=3, close_kernel=3)
            # except Exception:
            #     f0_img_clean = f0_img
            f0_img_clean = f0_img
            # NOTE: do not smooth the LR boundary map; use the cleaned LR map directly
            # f0_img_smooth = smooth_boundary_map(f0_img_clean, kernel_size=5, sigma=1.0)
            # For SR, use the computed f0_sr_img
            f0_sr = f0_sr_img
            # keep SR smoothing if desired for visualization (leave enabled)
            # f0_sr = smooth_boundary_map(f0_sr, kernel_size=7, sigma=1.5)

            # For HR, use f0_hr_img
            # f0_hr_clean = clean_boundary_map(f0_hr_img, thresh=0.087, open_kernel=3, close_kernel=3)
            f0_hr_clean = f0_hr_img
            # f0_hr_vis = smooth_boundary_map(f0_hr_clean, kernel_size=7, sigma=1.5)
            f0_hr_vis = f0_hr_clean

            # Also create an image for the LR boundary for comparison (unsmoothed)
            f0_lr_vis = f0_img_clean

            sr_bnd_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_sr_boundary_smoothed.png')
            lr_bnd_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_lr_boundary_smoothed.png')
            hr_bnd_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_hr_boundary_smoothed.png')
            # Create PIL images preserving actual SR boundary size and repeat across 3 columns
            try:
                # Create boundary images sized to match the IPF output width so they span all 3 IPF columns
                # Add gutter to boundary tiles to match IPF gutters
                BOUNDARY_GUTTER = 8
                try:
                    # Convert to degrees for visualization (radians -> degrees)
                    f0_up_lr_vis = f0_hr_interpolated * (180.0 / math.pi)
                    tensor_to_rgb_png_match_ref(f0_up_lr_vis, hr_png, sr_bnd_png, cols=3)
                    # ensure gutter inside simple repeat fallback too
                    if os.path.exists(sr_bnd_png):
                        from PIL import Image as _Image
                        im = _Image.open(sr_bnd_png)
                        # if the saved boundary has no gutter (width divisible by 3), add padding
                        if (im.width % 3) == 0:
                            im.close()
                            tensor_to_rgb_png(f0_hr_interpolated, sr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
                except Exception:
                    f0_up_lr_vis = f0_hr_interpolated * (180.0 / math.pi)
                    tensor_to_rgb_png(f0_up_lr_vis, sr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
                try:
                    f0_lr_vis_deg = f0_lr_vis * (180.0 / math.pi)
                    tensor_to_rgb_png_match_ref(f0_lr_vis_deg, hr_png, lr_bnd_png, cols=3)
                    if os.path.exists(lr_bnd_png):
                        from PIL import Image as _Image
                        im = _Image.open(lr_bnd_png)
                        if (im.width % 3) == 0:
                            im.close()
                            tensor_to_rgb_png(f0_lr_vis, lr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
                except Exception:
                    f0_lr_vis_deg = f0_lr_vis * (180.0 / math.pi)
                    tensor_to_rgb_png(f0_lr_vis_deg, lr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
                try:
                    f0_hr_vis_deg = f0_hr_vis * (180.0 / math.pi)
                    tensor_to_rgb_png_match_ref(f0_hr_vis_deg, hr_png, hr_bnd_png, cols=3)
                    if os.path.exists(hr_bnd_png):
                        from PIL import Image as _Image
                        im = _Image.open(hr_bnd_png)
                        if (im.width % 3) == 0:
                            im.close()
                            tensor_to_rgb_png(f0_hr_vis, hr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
                except Exception:
                    f0_hr_vis_deg = f0_hr_vis * (180.0 / math.pi)
                    tensor_to_rgb_png(f0_hr_vis_deg, hr_bnd_png, repeat_cols=3, gutter_px=BOUNDARY_GUTTER)
            except Exception as e:
                print(f"Failed to create boundary PIL images: {e}")
                sr_bnd_png = None
                lr_bnd_png = None
                hr_bnd_png = None

            # Create boundary comparison plot
            try:
                import matplotlib.pyplot as plt
                # Do not upsample LR boundary; show at original resolution
                # f0_lr_up = F.interpolate(f0_lr_vis, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Row 0: LR, HR, SR
                lr_deg = (f0_lr_vis.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_lr, W_lr)
                im0 = axes[0,0].imshow(lr_deg, cmap='inferno', vmin=0, vmax=60)
                axes[0,0].set_title("LR Boundary")
                axes[0,0].axis('off')
                
                hr_deg = (f0_hr_vis.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_hr, W_hr)
                im1 = axes[0,1].imshow(hr_deg, cmap='inferno', vmin=0, vmax=60)
                axes[0,1].set_title("HR Boundary")
                axes[0,1].axis('off')
                
                sr_deg = (f0_sr.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_hr, W_hr)
                im2 = axes[0,2].imshow(sr_deg, cmap='inferno', vmin=0, vmax=60)
                axes[0,2].set_title("SR Boundary")
                axes[0,2].axis('off')
                
                # Row 1: LR Upsampled, Post-Upsample, f0_hr
                lr_up_deg = (f0_lr_up.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_hr, W_hr)
                im3 = axes[1,0].imshow(lr_up_deg, cmap='inferno', vmin=0, vmax=60)
                axes[1,0].set_title("LR Upsampled Boundary")
                axes[1,0].axis('off')
                
                inter_deg = (f0_inter.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_hr, W_hr)
                im4 = axes[1,1].imshow(inter_deg, cmap='inferno', vmin=0, vmax=60)
                axes[1,1].set_title("Post-Upsample Boundary")
                axes[1,1].axis('off')
                
                f0_hr_deg = (f0_hr.squeeze().cpu().numpy() * (180.0 / math.pi)).reshape(H_hr, W_hr)
                im5 = axes[1,2].imshow(f0_hr_deg, cmap='inferno', vmin=0, vmax=60)
                axes[1,2].set_title("f0_hr (Upsampled LR Boundary)")
                axes[1,2].axis('off')
                
                # Add colorbar
                cbar = fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)
                cbar.set_label("Misorientation Angle (°)")
                
                plt.tight_layout()
                comp_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_boundary_comparison.png')
                plt.savefig(comp_png, dpi=150)
                plt.close()
                print(f"Saved boundary comparison plot: {comp_png}")
            except Exception as e:
                print(f"Failed to create boundary comparison plot: {e}")

            # --- Create a separate boundary comparison figure (LR, SR, HR) ---
            try:
                b_paths = [lr_bnd_png, sr_bnd_png, hr_bnd_png]
                # Filter out missing
                b_imgs = [Image.open(p).convert('RGB') for p in b_paths if p is not None]
                if len(b_imgs) == 3:
                    widths_b, heights_b = zip(*(im.size for im in b_imgs))
                    total_w = sum(widths_b)
                    max_h = max(heights_b)
                    comp = Image.new('RGB', (total_w, max_h), (255, 255, 255))
                    xoff = 0
                    for im in b_imgs:
                        comp.paste(im, (xoff, 0))
                        xoff += im.size[0]
                    compare_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_boundary_LR_SR_HR.png')
                    comp.save(compare_png)
                    for im in b_imgs:
                        im.close()
                else:
                    compare_png = None
            except Exception as e:
                print(f"Could not create boundary comparison figure: {e}")
                compare_png = None
        except Exception as e:
            print(f"Could not create smoothed boundary images: {e}")
            sr_bnd_png = None
            lr_bnd_png = None

            # Stack the three 3-column images vertically to form 3 rows x 3 cols
        try:
            # Order rows now: HR, LR upsampled, SR boundary (smoothed) repeated across 3 columns, SR final
            ordered_paths = [hr_png, lr_png]
            # third row: use HR proxy boundary (preferred) instead of the upsampled SR boundary
            if hr_bnd_png is not None:
                ordered_paths.append(hr_bnd_png)
            elif sr_bnd_png is not None:
                ordered_paths.append(sr_bnd_png)
            else:
                ordered_paths.append(sr_png)  # fallback
            # Do not include LR or HR boundary maps in the main stacked image per request
            ordered_paths.append(sr_png)
            imgs = [Image.open(p).convert('RGB') for p in ordered_paths]
            widths, heights = zip(*(im.size for im in imgs))
            max_width = max(widths)
            total_height = sum(heights)
            new_im = Image.new('RGB', (max_width, total_height + 40 * len(imgs)), (255, 255, 255))
            y_offset = 0
            draw = ImageDraw.Draw(new_im)
            # Try to load a default font; fallback if unavailable
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            row_labels = ['HR', 'LR upsampled', 'Upsampled LR Boundary', 'SR (final)']
            for idx, im in enumerate(imgs):
                x = (max_width - im.size[0]) // 2
                # Place image with some padding for label above
                label_y = y_offset + 4
                new_im.paste(im, (x, y_offset + 20))
                # Draw centered label for the row (robust text size computation)
                label = row_labels[idx] if idx < len(row_labels) else ''
                try:
                    if font is not None:
                        # Try a few ways to compute text size depending on Pillow version
                        try:
                            w, h = font.getsize(label)
                        except Exception:
                            try:
                                w, h = draw.textsize(label, font=font)
                            except Exception:
                                bbox = draw.textbbox((0, 0), label, font=font)
                                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        try:
                            bbox = draw.textbbox((0, 0), label)
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        except Exception:
                            w = len(label) * 6
                            h = 10
                except Exception:
                    # Last-resort fallback
                    w = len(label) * 6
                    h = 10

                try:
                    draw.text(((max_width - w) // 2, label_y), label, fill=(0, 0, 0), font=font)
                except Exception:
                    try:
                        # Try drawing without a font if font drawing fails
                        draw.text(((max_width - w) // 2, label_y), label, fill=(0, 0, 0))
                    except Exception:
                        # If labeling fails completely, continue — we will still save the image
                        pass

                y_offset += im.size[1] + 20

            out_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_HR_LR_SR.png')
            new_im.save(out_png)

            # Create a 2x3 subplot for boundaries
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Row 1: LR upsampled, HR, SR
            axes[0,0].imshow(f0_lr_up_img[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[0,0].set_title('LR Boundary (upsampled)')

            axes[0,1].imshow(f0_hr_img[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[0,1].set_title('HR Boundary')

            axes[0,2].imshow(f0_sr_img[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[0,2].set_title('SR Boundary')

            # Row 2: Post-upsample, Upsampled LR Boundary, SR
            axes[1,0].imshow(f0_inter_img[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[1,0].set_title('Post-Upsample Boundary')

            axes[1,1].imshow(f0_hr_interpolated[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[1,1].set_title('Upsampled LR Boundary')

            axes[1,2].imshow(f0_sr_img[0,0].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
            axes[1,2].set_title('SR Boundary')

            plt.tight_layout()
            boundary_plot_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_boundary_comparison.png')
            plt.savefig(boundary_plot_png, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Create a plot for IPF stages with projections
            fig, axes = plt.subplots(7, 3, figsize=(18, 42))
            stage_pngs = []
            H_lr = q_lr_img.shape[0]
            W_lr = q_lr_img.shape[1]
            hr_ipf_size = None
            projections = ['X', 'Y', 'Z']
            stage_names = ['Input (LR upsampled)', 'Stage 1 (after embed)', 'Stage 2 (after layers)', 'Stage 3 (after upsample)', 'Stage 4 (after refine)', 'Stage 5 (final)', 'Output (SR)']
            stage_quats = [q_lr_up] + q_stages + [q_sr_img.reshape(-1, 4)]
            for stage_idx, (q_stage, name) in enumerate(zip(stage_quats, stage_names)):
                q_stage_np = q_stage.cpu().numpy() if hasattr(q_stage, 'cpu') else q_stage
                if stage_idx == 0 or stage_idx == 6:
                    H_s, W_s = H_hr, W_hr
                elif stage_idx < 3:
                    H_s, W_s = H_lr, W_lr
                else:
                    H_s, W_s = H_hr, W_hr
                q_stage_img = q_stage_np.reshape(H_s, W_s, 4)
                for proj_idx, proj in enumerate(projections):
                    stage_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_stage{stage_idx}_{proj}_ipf.png')
                    render_ipf_image(
                        q_stage_img,
                        fcc_sym,
                        out_png=stage_png,
                        ref_dir=proj,
                        include_key=False,
                        overwrite=True,
                        format_input=False,
                        gutter_px=0,  # No gutter for single projection
                    )
                    stage_img = Image.open(stage_png)
                    if stage_idx == 2:  # Use stage 3 (first HR) as reference
                        hr_ipf_size = stage_img.size
                    if (stage_idx == 0 or stage_idx == 6 or stage_idx >= 3) and hr_ipf_size is not None:
                        pass  # Already HR
                    elif hr_ipf_size is not None:
                        stage_img = stage_img.resize(hr_ipf_size, Image.NEAREST)
                    axes[stage_idx, proj_idx].imshow(np.array(stage_img))
                    if proj_idx == 0:
                        axes[stage_idx, proj_idx].set_ylabel(name, fontsize=12, rotation=0, labelpad=50, va='center')
                    if stage_idx == 0:
                        axes[stage_idx, proj_idx].set_title(f'{proj}')
                    stage_pngs.append(stage_png)
            plt.tight_layout()
            ipf_stages_png = os.path.join(out_dir, f'epoch{epoch:04d}_sample{i}_ipf_stages_projections.png')
            plt.savefig(ipf_stages_png, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Cleanup stage PNGs
            for sp in stage_pngs:
                try:
                    if os.path.exists(sp):
                        os.remove(sp)
                except Exception:
                    pass

            for im in imgs:
                im.close()

            # cleanup intermediate files
            for p in (hr_png, lr_png, sr_png, sr_bnd_png, lr_bnd_png, hr_bnd_png, compare_png):
                try:
                    if p is not None and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        except Exception as e:
            print(f"Error combining images: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lr_dir', type=str, default='~/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/LR_Data', help='LR maps folder')
    p.add_argument('--hr_dir', type=str, default='~/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data', help='HR maps folder')
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--val_every', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    p.add_argument('--no_cuda', action='store_true')
    # Optional arguments for early stopping customization
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min_delta', type=float, default=1e-4)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)