#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File:        create_HR_ipf.py
Created at:  2025/10/26
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Generate IPF maps from HR quaternion data
"""

import os
import sys
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from visualization.ipf_render import render_ipf_image, render_ipf_rgb
from utils.symmetry_utils import resolve_symmetry
from PIL import Image

# ==========================
# USER CONFIGURATION
# ==========================
BASE_DIR = os.path.expanduser("~/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4")
SPLITS = ["Train", "Val", "Test"]  # Process all splits
SYMMETRY = "Oh"  # Cubic symmetry for IN718
REF_DIR = "Z"  # Reference direction (X, Y, Z, or ALL)
INCLUDE_KEY = False  # Include IPF color key in output
OVERWRITE = False  # Skip existing files
NUM_WORKERS = 4  # For potential parallel processing (not used yet, but reserved)

# ==========================
# MAIN SCRIPT
# ==========================
def main():
    print("=" * 70)
    print("IPF Map Generation Script")
    print("=" * 70)
    print(f"Base Directory:    {BASE_DIR}")
    print(f"Splits to Process: {', '.join(SPLITS)}")
    print(f"Symmetry:          {SYMMETRY}")
    print(f"Reference Dir:     {REF_DIR}")
    print(f"Include Key:       {INCLUDE_KEY}")
    print(f"Overwrite:         {OVERWRITE}")
    print("=" * 70)
    
    # Resolve symmetry
    sym_class = resolve_symmetry(SYMMETRY)
    print(f"Resolved symmetry class: {sym_class}")
    
    # Process each split
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}
    all_errors = []
    
    for split in SPLITS:
        print("\n" + "=" * 70)
        print(f"Processing {split} split")
        print("=" * 70)
        
        HR_DATA_DIR = os.path.join(BASE_DIR, split, "HR_Data")
        OUTPUT_DIR = os.path.join(BASE_DIR, split, "HR_IPF_Maps")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"HR Data Directory: {HR_DATA_DIR}")
        print(f"Output Directory:  {OUTPUT_DIR}")
        
        # Find all .npy files
        npy_pattern = os.path.join(HR_DATA_DIR, "*.npy")
        npy_files = sorted(glob.glob(npy_pattern))
        
        if not npy_files:
            print(f"WARNING: No .npy files found in {HR_DATA_DIR}")
            continue
        
        print(f"Found {len(npy_files)} .npy files to process")
        
        # Process each file
        errors = []
        skipped = 0
        processed = 0
        
        for npy_path in tqdm(npy_files, desc=f"Rendering {split} IPF maps", unit="file"):
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(npy_path))[0]
                out_png = os.path.join(OUTPUT_DIR, f"{base_name}_ipf.png")
                
                # Skip if file exists and overwrite is False
                if not OVERWRITE and os.path.exists(out_png):
                    skipped += 1
                    continue
                
                # Load quaternion data
                q_arr = np.load(npy_path)
                
                # Format quaternions
                from utils.quat_ops import format_quaternions
                q_formatted = format_quaternions(
                    q_arr,
                    normalize=True,
                    hemisphere=True,
                    reduce_fz=True,
                    sym=SYMMETRY,
                    scalar_first=True,
                    quat_first=False,
                )
                
                # Render IPF RGB (just the raw RGB array)
                rgb_array = render_ipf_rgb(
                    q_arr=q_formatted,
                    sym_class=sym_class,
                    ref_dir=REF_DIR,
                )
                
                # Convert to uint8 and save as image (no matplotlib, no axes, no labels)
                rgb_uint8 = (rgb_array * 255).astype(np.uint8)
                img = Image.fromarray(rgb_uint8)
                img.save(out_png, dpi=(300, 300))
                
                processed += 1
                
            except Exception as e:
                errors.append((npy_path, str(e)))
                tqdm.write(f"ERROR processing {os.path.basename(npy_path)}: {e}")
        
        # Split summary
        print(f"\n{split} Summary:")
        print(f"  Processed: {processed}")
        print(f"  Skipped:   {skipped}")
        print(f"  Errors:    {len(errors)}")
        
        total_stats["processed"] += processed
        total_stats["skipped"] += skipped
        total_stats["errors"] += len(errors)
        all_errors.extend(errors)
    
    # Overall Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total Processed:  {total_stats['processed']}")
    print(f"Total Skipped:    {total_stats['skipped']}")
    print(f"Total Errors:     {total_stats['errors']}")
    
    if all_errors:
        print("\nErrors encountered:")
        for fpath, err in all_errors[:10]:  # Show first 10 errors
            print(f"  - {os.path.basename(fpath)}: {err}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
    
    print(f"\nIPF maps saved to: {BASE_DIR}/[Train|Val|Test]/HR_IPF_Maps/")
    print("=" * 70)

if __name__ == "__main__":
    main()
