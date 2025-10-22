# from builders.dataset_builder import build_quaternion_sr_dataset
# from visualization.save_dataset_ipfs import save_dataset_ipfs
# import os
# from training.data_loading import QuaternionDataset
# from tqdm import tqdm

# # Example usage
# dataset_out_root = "/data/warren/materials/EBSD"
# dataset_name = "IN718_FZ_2D_SR_x4"
# dataset_dir = os.path.join(dataset_out_root, dataset_name)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from utils.quat_ops import (
    normalize_quaternions,
    enforce_hemisphere,
    reduce_to_fz_min_angle,
)
from orix import plot as orix_plot


def plot_fz_ipf_3D(q_flat, sym_class, ref_dir="Z"):
    sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class
    q_flat = normalize_quaternions(q_flat, axis=-1)
    q_flat = enforce_hemisphere(q_flat, scalar_first=True)

    # Reduce to FZ
    q_fz, op_map = reduce_to_fz_min_angle(q_flat, sym=sym, return_op_map=True)
    ori_fz = Orientation(q_fz, symmetry=sym)
    ori_orig = Orientation(q_flat, symmetry=sym)

    outside_mask = op_map != 0
    frac_outside = outside_mask.mean() * 100

    # Reference direction
    ref_map = {
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
        "Z": Vector3d.zvector(),
    }
    v_ref = ref_map[ref_dir.upper()]

    # Compute poles in 3D
    v_plot = (ori_fz * v_ref).data  # (N, 3)
    v_orig = (ori_orig * v_ref).data

    # IPF colors
    ckey = orix_plot.IPFColorKeyTSL(sym.laue)
    ckey.direction = v_ref
    colors = ckey.orientation2color(ori_fz)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Inside FZ originally
    ax.scatter(
        v_plot[~outside_mask, 0],
        v_plot[~outside_mask, 1],
        v_plot[~outside_mask, 2],
        c=colors[~outside_mask],
        s=15,
        alpha=0.7,
        marker="o",
        label="Inside FZ originally",
    )

    # Mapped from outside FZ
    ax.scatter(
        v_plot[outside_mask, 0],
        v_plot[outside_mask, 1],
        v_plot[outside_mask, 2],
        c=colors[outside_mask],
        s=25,
        alpha=0.9,
        marker="^",
        edgecolors="black",
        label="Mapped from outside FZ",
    )

    # Outside original positions
    ax.scatter(
        v_orig[outside_mask, 0],
        v_orig[outside_mask, 1],
        v_orig[outside_mask, 2],
        c=colors[outside_mask],
        s=25,
        alpha=0.9,
        marker="D",
        edgecolors="black",
        label="Outside FZ originally",
    )

    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.4, alpha=0.5)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(f"3D Pole Distribution — Outside FZ: {frac_outside:.2f}%")

    plt.tight_layout()
    plt.show()


from utils.quat_ops import get_dummy_quats
from orix.quaternion import Orientation, symmetry as SYM

dummy_quats = get_dummy_quats(resolution_deg=5.0, pg=SYM.O)  # cubic FZ
plot_fz_ipf_3D(dummy_quats, "O", ref_dir="Z")

# from torch.utils.data import DataLoader
# import torch

# train_ds = QuaternionDataset(
#     dataset_root=dataset_dir,
#     split="Train",
#     preload=True,
#     preload_torch=True,  # preload as CPU torch tensors
#     pin_memory=True,
# )

# from orix.quaternion import symmetry as SYM
# from visualization.ipf_render import render_sr_hr_side_by_side

# q_arr = train_ds.get_numpy_spatial_quat(0)[1]
# from utils.quat_ops import format_quaternions
# from visualization.ipf_render import render_ipf_rgb


# render_sr_hr_side_by_side(q_arr, q_arr, SYM.Oh)


# if torch.cuda.is_available():
#     torch.cuda.init()
#     _ = torch.cuda.current_device()

# train_loader = DataLoader(
#     train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
# )

# # for lr, hr in tqdm(train_loader):
# #     lr = lr.to("cuda", non_blocking=True)
# #     hr = hr.to("cuda", non_blocking=True)


# from post_processing.post_process import run_postprocess_from_config

# # from postprocess.run_postprocess_from_config import run_postprocess_from_config

# run_postprocess_from_config("experiments/IN718/debug_x4_kss_4", max_samples=8)

# dataset_info = build_quaternion_sr_dataset(
#     hr_dirs={
#         "Train": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/*.npy",
#         "Val": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Val/HR_Images/preprocessed_imgs_all_Blocks/*.npy",
#         "Test": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/*.npy",
#     },
#     out_root=dataset_out_root,
#     dataset_name=dataset_name,
#     scale=4,
#     # take_first=2,
#     symmetry="Oh",
#     normalize=True,
#     hemisphere=True,
#     reduce_to_fz=True,
#     creator="Warren Zamudio",
#     contact="wzamudio@ucsb.edu",
# )


# dataset_info = build_quaternion_sr_dataset(
#     hr_dirs={
#         "Train": "/data/warren/materials/EBSD/IN718_2D_SR_x4/Train/Original_Data/*.npy", v
#         "Val": "/data/warren/materials/EBSD/IN718_2D_SR_x4/Val/Original_Data/*.npy",
#         "Test": "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/*.npy",
#     },
#     out_root=dataset_out_root,
#     dataset_name=dataset_name,
#     scale=4,
#     take_first=5,
#     symmetry="Oh",
#     normalize=True,
#     hemisphere=True,
#     reduce_to_fz=True,
#     creator="Warren Zamudio",
#     contact="wzamudio@ucsb.edu",
# )
# save_dataset_ipfs(
#     dataset_root=dataset_dir,
#     splits=("Train", "Val", "Test"),
#     which_list=("HR", "LR", "Original"),
#     ref_dir="ALL",
#     include_key=True,
#     overwrite=False,
#     num_workers=16,
# )
