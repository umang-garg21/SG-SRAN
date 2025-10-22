import os
import glob
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from visualization.ipf_render import render_ipf_image
from utils.quat_ops import to_spatial_quat
from utils.symmetry_utils import resolve_symmetry

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _normalize_data_folder_name(which: str) -> str:
    """
    Return correct input data folder name for a given type.
    Ensures 'Original_Data' is used for ORIGINAL.
    """
    which_upper = which.upper()
    if which_upper == "ORIGINAL":
        return "Original_Data"
    return f"{which_upper}_Data"  # HR_Data / LR_Data


def _ipf_output_folder_name(which: str) -> str:
    """
    Return correct IPF output folder name.
    Ensures 'Original_IPF_Images' is used for ORIGINAL (capital O only).
    """
    which_upper = which.upper()
    if which_upper == "ORIGINAL":
        return "Original_IPF_Images"
    return f"{which_upper}_IPF_Images"


def _build_ipf_tasks(dataset_root, splits, which_list, ref_dir):
    """Build list of (input_file, output_file) IPF rendering tasks."""
    tasks = []
    for split in splits:
        split_cap = split.capitalize()
        if split_cap not in ("Train", "Val", "Test"):
            raise ValueError(f"Invalid split: {split_cap}")

        for which in which_list:
            # ✅ Proper input/output mapping
            data_folder_name = _normalize_data_folder_name(which)
            output_folder_name = _ipf_output_folder_name(which)

            data_dir = os.path.join(dataset_root, split_cap, data_folder_name)
            if not os.path.isdir(data_dir):
                print(f"[IPF] Missing folder: {data_dir}")
                continue

            files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            if not files:
                print(f"[IPF] No files found in {data_dir}")
                continue

            out_dir = os.path.join(dataset_root, split_cap, output_folder_name)
            os.makedirs(out_dir, exist_ok=True)

            for fp in files:
                base = os.path.splitext(os.path.basename(fp))[0]
                out_png = os.path.join(out_dir, f"{base}_ref_{ref_dir.lower()}.png")
                tasks.append((fp, out_png))
    return tasks


def _ipf_dir_exists_and_populated(dataset_root, splits, which_list):
    """
    Check whether all IPF image directories already exist and have PNGs.
    This allows skipping rendering if everything is already generated.
    """
    for split in splits:
        split_cap = split.capitalize()
        for which in which_list:
            output_folder_name = _ipf_output_folder_name(which)
            ipf_dir = os.path.join(dataset_root, split_cap, output_folder_name)
            if not os.path.isdir(ipf_dir):
                return False
            if not any(fname.endswith(".png") for fname in os.listdir(ipf_dir)):
                return False
    return True


def _process_single_ipf(fp, out_png, sym_class, ref_dir, include_key, overwrite):
    """Single IPF rendering task."""
    if not overwrite and os.path.exists(out_png):
        return
    arr_hw4 = to_spatial_quat(np.load(fp, mmap_mode="r"))
    render_ipf_image(
        arr_hw4,
        sym_class,
        out_png=out_png,
        ref_dir=ref_dir,
        include_key=include_key,
        overwrite=overwrite,
        format_input=True,
    )


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------


def save_dataset_ipfs(
    dataset_root: str,
    splits=("Train", "Val", "Test"),
    which_list=("HR", "LR", "Original"),
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    num_workers: int = 4,
):
    """
    Save IPF images for all splits/which in a quaternion SR dataset.

    Looks directly in:
        {split}/{HR_Data, LR_Data, Original_Data}
    and saves to:
        {split}/{HR_IPF_Images, LR_IPF_Images, Original_IPF_Images}

    Skips execution if IPF folders already exist and contain PNGs
    unless overwrite=True.
    """
    # --- Validate symmetry ---
    info_path = os.path.join(dataset_root, "dataset_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"Missing dataset_info.json at {info_path}")
    with open(info_path, "r") as f:
        info = json.load(f)

    sym_class = resolve_symmetry(info.get("symmetry", "Oh"))

    ref_dir = ref_dir.upper()
    if ref_dir not in ("X", "Y", "Z", "ALL"):
        raise ValueError("ref_dir must be 'X','Y','Z','ALL'")

    # --- Early exit check ---
    if (
        _ipf_dir_exists_and_populated(dataset_root, splits, which_list)
        and not overwrite
    ):
        print(f"[Rendering IPFs] Skipping: IPF images already exist in {dataset_root}")
        return

    # --- Build tasks ---
    tasks = _build_ipf_tasks(dataset_root, splits, which_list, ref_dir)
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("[Rendering IPFs] No images found to process.")
        return

    print(f"[Rendering IPFs] {total_tasks} total images to process...")

    # --- Run with tqdm ---
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [
                ex.submit(
                    _process_single_ipf,
                    *task,
                    sym_class,
                    ref_dir,
                    include_key,
                    overwrite,
                )
                for task in tasks
            ]
            for _ in tqdm(
                as_completed(futs), total=total_tasks, desc="Rendering IPFs", unit="img"
            ):
                try:
                    _.result()
                except Exception as e:
                    print(f"[Rendering IPFs] Error: {e}")
    else:
        for task in tqdm(tasks, total=total_tasks, desc="Rendering IPFs", unit="img"):
            try:
                _process_single_ipf(*task, sym_class, ref_dir, include_key, overwrite)
            except Exception as e:
                print(f"[Rendering IPFs] Error on {task[0]}: {e}")

    print(f"[Rendering IPFs] Completed saving IPF images -> {dataset_root}")
