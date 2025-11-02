#!/usr/bin/env python3
"""
Script to apply colored segmentation map to SR image.
Each segment gets a random distinct color overlaid on the SR image.
Black lines represent grain boundaries (pixels not in the CSV).
"""

import pandas as pd
import numpy as np
from PIL import Image
import argparse
import os


def generate_distinct_colors(n_colors, seed=42):
    """
    Generate n distinct random colors.
    
    Args:
        n_colors: Number of colors to generate
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping segment_id to RGB color
    """
    np.random.seed(seed)
    colors = {}
    
    # Use HSV color space for better distribution
    for i in range(n_colors):
        # Spread hues evenly, randomize saturation and value
        hue = (i * 360 / n_colors) % 360
        saturation = np.random.uniform(0.5, 1.0)
        value = np.random.uniform(0.5, 1.0)
        
        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors[i] = (r, g, b)
    
    return colors


def apply_colored_segmentation(csv_path, sr_image_path, output_path=None, alpha=0.5, seed=42):
    """
    Apply colored segmentation map to SR image.
    
    Args:
        csv_path: Path to CSV file with segmentation data
        sr_image_path: Path to SR image
        output_path: Path to save output (if None, saves to evaluation folder)
        alpha: Transparency of colored overlay (0=invisible, 1=opaque)
        seed: Random seed for color generation
    
    Returns:
        The composite image as numpy array
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loading SR image: {sr_image_path}")
    sr_img = Image.open(sr_image_path)
    sr_array = np.array(sr_img)
    
    print(f"SR image shape: {sr_array.shape}")
    print(f"Number of segments: {df['segment_id'].nunique()}")
    
    # Get image dimensions from CSV
    max_y = df['y'].max()
    max_x = df['x'].max()
    seg_height = max_y + 1
    seg_width = max_x + 1
    
    print(f"Segmentation map dimensions: {seg_height} x {seg_width}")
    
    # Check dimension compatibility
    if sr_array.shape[0] != seg_height or sr_array.shape[1] != seg_width:
        print(f"WARNING: SR image dimensions {sr_array.shape[:2]} don't match segmentation {seg_height}x{seg_width}")
        print("Will try to proceed anyway...")
    
    # Generate distinct colors for each segment
    segment_ids = sorted(df['segment_id'].unique())
    colors = generate_distinct_colors(len(segment_ids), seed=seed)
    segment_to_color = {seg_id: colors[i] for i, seg_id in enumerate(segment_ids)}
    
    # Create segmentation mask with random colors
    print("Creating colored segmentation mask...")
    seg_mask = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)
    grain_boundary_mask = np.ones((seg_height, seg_width), dtype=bool)  # Track which pixels are grain boundaries
    
    for idx, row in df.iterrows():
        y = int(row['y'])
        x = int(row['x'])
        seg_id = int(row['segment_id'])
        color = segment_to_color[seg_id]
        
        seg_mask[y, x] = color
        grain_boundary_mask[y, x] = False
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} pixels...")
    
    print(f"Grain boundary pixels (black lines): {np.sum(grain_boundary_mask)}")
    
    # Create composite image
    print(f"Creating composite with alpha={alpha}...")
    
    # Ensure SR image has the right dimensions
    if sr_array.shape[:2] != (seg_height, seg_width):
        sr_array = np.array(sr_img.resize((seg_width, seg_height), Image.LANCZOS))
    
    # Blend SR image with colored segmentation
    composite = sr_array.copy()
    
    # Apply colored overlay only where we have segmentation data (not on grain boundaries)
    mask = ~grain_boundary_mask[:, :, np.newaxis]  # True where we have segment data
    composite = np.where(
        mask,
        (alpha * seg_mask + (1 - alpha) * sr_array).astype(np.uint8),
        sr_array  # Keep original SR on grain boundaries
    )
    
    # Determine output path
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sr_basename = os.path.basename(sr_image_path)
        sr_name = os.path.splitext(sr_basename)[0]
        output_path = os.path.join(script_dir, f"{sr_name}_colored_segmentation.png")
    
    # Save the result
    print(f"Saving result to: {output_path}")
    result_img = Image.fromarray(composite)
    result_img.save(output_path)
    
    # Also save just the colored segmentation map
    seg_only_path = output_path.replace('.png', '_segments_only.png')
    print(f"Saving segmentation-only map to: {seg_only_path}")
    seg_img = Image.fromarray(seg_mask)
    seg_img.save(seg_only_path)
    
    print(f"\nStatistics:")
    print(f"  Total segments: {len(segment_ids)}")
    print(f"  Grain boundary pixels (black): {np.sum(grain_boundary_mask)}")
    print(f"  Segment pixels (colored): {len(df)}")
    print(f"  Coverage: {len(df) / (seg_height * seg_width) * 100:.2f}%")
    
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Apply colored segmentation map to SR image. "
                    "Each segment gets a random distinct color. "
                    "Black lines are grain boundaries (pixels not in CSV)."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with segmentation data"
    )
    parser.add_argument(
        "sr_image_path",
        type=str,
        help="Path to SR image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for composite image (default: evaluation folder)"
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (0=invisible, 1=opaque, default=0.5)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for color generation (default=42)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    if not os.path.exists(args.sr_image_path):
        print(f"Error: SR image not found: {args.sr_image_path}")
        return
    
    # Apply colored segmentation
    apply_colored_segmentation(
        args.csv_path,
        args.sr_image_path,
        args.output,
        args.alpha,
        args.seed
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
