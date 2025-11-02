#!/usr/bin/env python3
"""
Script to recreate segmentation map PNG from CSV file.
This verifies that the CSV was constructed properly by recreating the visualization.
"""

import pandas as pd
import numpy as np
from PIL import Image
import argparse
import os


def recreate_segmentation_map(csv_path, output_path=None):
    """
    Recreate a segmentation map PNG from a CSV file.
    
    Args:
        csv_path: Path to the CSV file with columns [segment_id, y, x, r, g, b, area]
        output_path: Path to save the output PNG. If None, will save in evaluation folder
    
    Returns:
        numpy array of the recreated image
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of unique segments: {df['segment_id'].nunique()}")
    
    # Get image dimensions from the data
    max_y = df['y'].max()
    max_x = df['x'].max()
    height = max_y + 1
    width = max_x + 1
    
    print(f"Image dimensions: {height} x {width}")
    
    # Initialize empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill in the pixels from CSV
    print("Reconstructing image...")
    for idx, row in df.iterrows():
        y = int(row['y'])
        x = int(row['x'])
        r = int(row['r'])
        g = int(row['g'])
        b = int(row['b'])
        
        image[y, x] = [r, g, b]
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} pixels...")
    
    print(f"Reconstruction complete!")
    
    # Determine output path
    if output_path is None:
        # Save in evaluation folder with basename from CSV
        csv_basename = os.path.basename(csv_path).replace('.csv', '_reconstructed.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, csv_basename)
    
    # Save the image
    print(f"Saving image to: {output_path}")
    img_pil = Image.fromarray(image)
    img_pil.save(output_path)
    
    # Print statistics
    non_zero_pixels = np.count_nonzero(np.any(image != 0, axis=2))
    total_pixels = height * width
    print(f"\nStatistics:")
    print(f"  Total pixels in image: {total_pixels}")
    print(f"  Non-zero pixels: {non_zero_pixels}")
    print(f"  Zero pixels: {total_pixels - non_zero_pixels}")
    print(f"  Coverage: {non_zero_pixels / total_pixels * 100:.2f}%")
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Recreate segmentation map PNG from CSV file"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file containing pixel data"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the PNG file (default: evaluation folder with _reconstructed.png suffix)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return
    
    # Recreate the segmentation map
    recreate_segmentation_map(args.csv_path, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
