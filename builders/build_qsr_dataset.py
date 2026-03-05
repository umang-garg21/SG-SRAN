# -*-coding:utf-8 -*-
"""
File:        build_qsr_dataset.py
Created at:  2025/10/22 10:10:58
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Runner script for building Quaternion Super-Resolution datasets
             using build_quaternion_sr_dataset() from dataset_builder.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports from utils and visualization
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from dataset_builder import build_quaternion_sr_dataset
from visualization.save_dataset_ipfs import save_dataset_ipfs
import os

# ==========================
# USER CONFIGURATION
# ==========================
DATASET_NAME = "Open718_QSR_x4"
DATASET_OUT_ROOT = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/"
SCALE = 4
SYMMETRY = "Oh"
NORMALIZE = True
HEMISPHERE = True
REDUCE_TO_FZ = True
CREATOR = "Warren Zamudio"
CONTACT = "wzamudio@ucsb.edu"

HR_TRAIN = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718/Train/HR_Images/*.npy"
HR_VAL = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718/Val/HR_Images/preprocessed_imgs_all_Blocks/*.npy"
HR_TEST = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718/Test/HR_Images/*.npy"

# ==========================
# BUILD DATASET
# ==========================
if __name__ == "__main__":
    print("Starting Quaternion SR Dataset build...")
    print(f"Output root: {DATASET_OUT_ROOT}")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Scale: {SCALE}, Symmetry: {SYMMETRY}")

    dataset_info = build_quaternion_sr_dataset(
        hr_dirs={
            "Train": HR_TRAIN,
            "Val": HR_VAL,
            "Test": HR_TEST,
        },
        out_root=DATASET_OUT_ROOT,
        dataset_name=DATASET_NAME,
        scale=SCALE,
        symmetry=SYMMETRY,
        normalize=NORMALIZE,
        hemisphere=HEMISPHERE,
        reduce_to_fz=REDUCE_TO_FZ,
        creator=CREATOR,
        contact=CONTACT,
    )

    print("\nDataset build completed successfully!")
    print(f"Dataset saved to: {DATASET_OUT_ROOT}/{DATASET_NAME}")
    print(f"Metadata: {dataset_info.get('splits', {})}")

    # save_dataset_ipfs(
    #     dataset_root=os.path.join(DATASET_OUT_ROOT, DATASET_NAME),
    #     splits=("Train", "Val", "Test"),
    #     which_list=("HR", "LR", "Original"),
    #     ref_dir="ALL",
    #     include_key=True,
    #     overwrite=False,
    #     num_workers=4,
    # )
