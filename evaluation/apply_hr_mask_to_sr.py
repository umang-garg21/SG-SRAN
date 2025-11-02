#!/usr/bin/env python3
"""
Apply an HR segmentation mask (from CSV) to an SR image and save the masked SR in the evaluation folder.

Usage:
    python evaluation/apply_hr_mask_to_sr.py <csv_path> <sr_image_path> [-o output_path]

Default output: saved into the `evaluation` folder with suffix `_masked_by_hr_seg.png`.
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image


def build_mask_from_csv(csv_path):
    """Build a boolean mask from the CSV file. Returns mask (HxW bool) and (height,width).
    CSV is expected to have columns: segment_id,y,x,r,g,b,area (y,x are pixel coords)
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    max_y = int(df['y'].max())
    max_x = int(df['x'].max())
    height = max_y + 1
    width = max_x + 1

    print(f"CSV-derived mask size: {height} x {width}")

    mask = np.zeros((height, width), dtype=np.uint8)
    ys = df['y'].to_numpy(dtype=np.int64)
    xs = df['x'].to_numpy(dtype=np.int64)
    mask[ys, xs] = 1

    return mask.astype(bool), (height, width)


def apply_mask_to_sr(mask_bool, sr_img: Image.Image):
    """Resize mask to match SR image if needed and apply it (black-out masked-out pixels).
    Returns the masked PIL image and mask applied as boolean array matching SR size.
    """
    sr_w, sr_h = sr_img.size
    mask_h, mask_w = mask_bool.shape

    if (mask_w, mask_h) != (sr_w, sr_h):
        print(f"Resizing mask from ({mask_h},{mask_w}) to SR image size ({sr_h},{sr_w})")
        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255).astype(np.uint8))
        mask_resized = mask_img.resize((sr_w, sr_h), resample=Image.NEAREST)
        mask_applied = np.array(mask_resized) > 0
    else:
        mask_applied = mask_bool

    sr_arr = np.array(sr_img.convert('RGB'))

    # Ensure mask_applied has shape (h,w)
    if mask_applied.shape != (sr_h, sr_w):
        # transpose if necessary
        mask_applied = mask_applied.reshape((sr_h, sr_w))

    # Apply mask: keep pixels where mask==True, black-out where False
    masked = sr_arr.copy()
    masked[~mask_applied] = 0

    masked_img = Image.fromarray(masked)
    return masked_img, mask_applied


def main():
    parser = argparse.ArgumentParser(description="Apply HR segmentation CSV mask to SR image")
    parser.add_argument('csv_path', type=str, help='Path to HR segmentation CSV')
    parser.add_argument('sr_image', type=str, help='Path to SR image (PNG/JPEG)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output path (default: evaluation/<sr_basename>_masked_by_hr_seg.png)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"CSV not found: {args.csv_path}")
        return
    if not os.path.exists(args.sr_image):
        print(f"SR image not found: {args.sr_image}")
        return

    mask_bool, (mask_h, mask_w) = build_mask_from_csv(args.csv_path)

    sr_img = Image.open(args.sr_image)
    sr_w, sr_h = sr_img.size
    print(f"SR image size: {sr_h} x {sr_w}")

    masked_img, mask_applied = apply_mask_to_sr(mask_bool, sr_img)

    # Default output into evaluation folder
    if args.output is None:
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        sr_base = os.path.splitext(os.path.basename(args.sr_image))[0]
        out_name = sr_base + '_masked_by_hr_seg.png'
        output_path = os.path.join(eval_dir, out_name)
    else:
        output_path = args.output

    masked_img.save(output_path)
    print(f"Masked SR image saved to: {output_path}")

    # Print some stats
    total_sr_pixels = sr_w * sr_h
    kept_pixels = int(np.count_nonzero(mask_applied))
    print(f"Mask applied coverage on SR image: {kept_pixels}/{total_sr_pixels} = {kept_pixels/total_sr_pixels*100:.2f}%")


if __name__ == '__main__':
    main()
