import numpy as np
from scipy.io import savemat
from utils.symmetry_utils import resolve_symmetry

from utils.quat_ops import to_scalar_first


data = np.load("Open_718_Train_hr_x_normal_0.npy")


data = to_scalar_first(data)


input_data = data[:, :, :]
output_data = input_data


from visualization.visualize_sr_results import render_input_output_side_by_side

# Render side-by-side comparison
comparison_path = "comparison.png"


sym_class = resolve_symmetry("cubic")


print(f"Rendering comparison to: {comparison_path}")
render_input_output_side_by_side(
    input_q_arr=input_data,
    output_q_arr=output_data,
    sym_class=sym_class,
    out_png=comparison_path,
    ref_dir="ALL",
    include_key=True,
    overwrite=True,
    format_input=True,
    dpi=300,
)

# savemat("sr_save_qrbsa_1d_model_best_minimum_angle_transformation.mat", {"data": data})


from builders.dataset_builder import build_quaternion_sr_dataset
from visualization.save_dataset_ipfs import save_dataset_ipfs
import os
from training.data_loading import QuaternionDataset
from tqdm import tqdm

# Example usage
dataset_out_root = "/data/warren/materials/EBSD"
dataset_name = "IN718_FZ_2D_SR_x4"
dataset_dir = os.path.join(dataset_out_root, dataset_name)

train_ds = QuaternionDataset(
    dataset_root=dataset_dir,
    split="Train",
    preload=True,
    preload_torch=True,  # preload as CPU torch tensors
    pin_memory=True,
)


LR, HR = train_ds[0]

LR[0, :, :]
