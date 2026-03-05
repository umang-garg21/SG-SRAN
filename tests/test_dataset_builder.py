# -*-coding:utf-8 -*-
"""
File:        test_dataset_builder.py
Created at:  2025/10/17 13:32:01
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""


import os
import numpy as np
import pytest

from builders.dataset_builder import build_quaternion_sr_dataset


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a temporary dataset directory with fake quaternion .npy files."""
    d = tmp_path / "fake_data"
    d.mkdir()
    for i in range(5):
        arr = np.random.randn(4, 64, 64).astype(np.float32)
        np.save(d / f"quat_{i:03d}.npy", arr)
    return d


def test_dataset_builder_creates_structure(tmp_path, tmp_dataset):
    out_dir = tmp_path / "out"
    dataset_info = build_quaternion_sr_dataset(
        hr_glob=str(tmp_dataset / "*.npy"),
        out_root=str(out_dir),
        dataset_name="FakeDataset",
        scale=2,
        symmetry="Oh",
        reduce_to_fz=True,
    )

    # Check metadata file
    assert os.path.exists(os.path.join(out_dir, "FakeDataset", "dataset_info.json"))

    # Check HR/LR directories exist
    for s in ["Train", "Val", "Test"]:
        for sub in ["HR_Data", "LR_Data", "Original_Data"]:
            assert os.path.isdir(os.path.join(out_dir, "FakeDataset", s, sub))

    # Check patch shape key
    assert "patch_shape" in dataset_info


def test_dataset_builder_patch_files_exist(tmp_path, tmp_dataset):
    out_dir = tmp_path / "out2"
    build_quaternion_sr_dataset(
        hr_glob=str(tmp_dataset / "*.npy"),
        out_root=str(out_dir),
        dataset_name="FakeDataset2",
        scale=2,
    )

    hr_files = list((out_dir / "FakeDataset2" / "Train" / "HR_Data").glob("*.npy"))
    lr_files = list((out_dir / "FakeDataset2" / "Train" / "LR_Data").glob("*.npy"))

    assert len(hr_files) > 0
    assert len(hr_files) == len(lr_files)
