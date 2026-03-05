# -*-coding:utf-8 -*-
"""
File:        test_save_dataset_ipfs.py
Created at:  2025/10/17 18:34:55
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Unit tests for save_dataset_ipfs pipeline.
"""


import os
import json
import tempfile
import numpy as np
from pathlib import Path

import pytest

from visualization.save_dataset_ipfs import (
    _normalize_data_folder_name,
    _ipf_output_folder_name,
    _build_ipf_tasks,
    _ipf_dir_exists_and_populated,
    save_dataset_ipfs,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset structure with fake quaternion .npy files."""
    dataset_root = tmp_path / "Fake_Dataset"
    dataset_root.mkdir()

    # Create dataset_info.json
    info = {"symmetry": "Oh"}
    with open(dataset_root / "dataset_info.json", "w") as f:
        json.dump(info, f)

    # Splits and folders
    splits = ["Train", "Val", "Test"]
    which_list = ["HR", "LR", "Original"]

    for split in splits:
        for which in which_list:
            if which == "Original":
                data_folder = dataset_root / split / "Original_Data"
            else:
                data_folder = dataset_root / split / f"{which}_Data"
            data_folder.mkdir(parents=True)

            # Create dummy .npy file
            arr = np.random.rand(8, 8, 4).astype(np.float32)
            np.save(data_folder / f"fake_{which.lower()}_{split.lower()}.npy", arr)

    return dataset_root


# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------


def test_folder_name_helpers():
    """Ensure input/output folder name helpers return correct names."""
    assert _normalize_data_folder_name("HR") == "HR_Data"
    assert _normalize_data_folder_name("LR") == "LR_Data"
    assert _normalize_data_folder_name("Original") == "Original_Data"

    assert _ipf_output_folder_name("HR") == "HR_IPF_Images"
    assert _ipf_output_folder_name("LR") == "LR_IPF_Images"
    assert _ipf_output_folder_name("Original") == "Original_IPF_Images"


def test_build_tasks(temp_dataset):
    """Check that IPF tasks are built correctly from dummy dataset."""
    tasks = _build_ipf_tasks(temp_dataset, ["Train"], ["HR", "Original"], "ALL")
    assert len(tasks) == 2  # one HR and one Original dummy file
    assert all(str(t[0]).endswith(".npy") for t in tasks)
    assert all(str(t[1]).endswith(".png") for t in tasks)
    assert "HR_IPF_Images" in tasks[0][1] or "Original_IPF_Images" in tasks[0][1]


def test_ipf_dir_check_false(temp_dataset):
    """Initially no IPF image dirs should exist, so check should return False."""
    assert not _ipf_dir_exists_and_populated(
        temp_dataset, ["Train", "Val", "Test"], ["HR", "LR", "Original"]
    )


def test_save_dataset_ipfs_creates_ipfs(temp_dataset):
    """Run save_dataset_ipfs and verify IPF image output folders are created."""
    save_dataset_ipfs(
        dataset_root=str(temp_dataset),
        splits=("Train", "Val", "Test"),
        which_list=("HR", "LR", "Original"),
        ref_dir="ALL",
        overwrite=True,
        num_workers=1,  # for deterministic testing
    )

    for split in ["Train", "Val", "Test"]:
        for which in ["HR", "LR", "Original"]:
            ipf_dir = temp_dataset / split / _ipf_output_folder_name(which)
            assert ipf_dir.exists()
            pngs = list(ipf_dir.glob("*.png"))
            assert len(pngs) >= 1
