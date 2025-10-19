# -*-coding:utf-8 -*-
"""
File:        generate_npy.py
Created at:  2025/10/13 12:39:27
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import h5py
import numpy as np

import matplotlib.pyplot as plt
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

from orix.quaternion import Orientation, symmetry as SYM
from orix.quaternion import Quaternion

# file_path = "/mnt/c/data/warren/EBSD/3D_EBSD_Dataset_Voxelized/718_final_yield.dream3d"  # change to your file
file_path = (
    "/mnt/c/data/warren/EBSD/3D_EBSD_Dataset_Voxelized/718_fz_orientations.dream3d"
)

with h5py.File(file_path, "r") as f:

    def print_tree(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"\nGROUP:   {name}\n")
        elif isinstance(obj, h5py.Dataset):
            print(f"      DATASET:   {name}  shape={obj.shape}  dtype={obj.dtype}")

    f.visititems(print_tree)


with h5py.File(file_path, "r") as f:
    quat = f["DataStructure/ImageDataContainer/CellData/Quaternions"]
    print("Shape:", quat.shape)
    print("Dtype:", quat.dtype)

    # Read into memory
    quats = quat[()]


with h5py.File(file_path, "r") as f:
    quat = f["DataStructure/ImageDataContainer/CellData/Quats_FZ"]
    print("Shape:", quat.shape)
    print("Dtype:", quat.dtype)

    # Read into memory
    quats_fz = quat[()]


quats_training = np.load(
    "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/Open_718_Test_hr_x_block_0.npy"
)





# np.allclose(quats, quats_fz, rtol=1e-05, atol=1e-3)


# with h5py.File(file_path, "r") as f:
#     quat = f["/DataContainers/ImageDataContainer/CellData/EulerAngles"]
#     print("Shape:", quat.shape)
#     print("Dtype:", quat.dtype)

#     # Read into memory
#     euler_angles = quat[()]

# np.save("euler_angles.npy", euler_angles)


# euler_angles = np.load("euler_angles.npy")


# def euler_to_ipf(euler_angles, axis="Z"):
#     ori = Orientation.from_euler(
#         np.asarray(euler_angles), symmetry=SYM.Oh, direction="lab2crystal"
#     )

#     key = IPFColorKeyTSL(SYM.Oh, direction=getattr(Vector3d, f"{axis.lower()}vector")())
#     return key.orientation2color(ori.inv())


# # Get slice
# euler_angles_slice = euler_angles[:, :, 358, :]
# print(f"euler_angles slice shape: {euler_angles_slice.shape}")

# plt.figure(figsize=(6, 6))
# plt.imshow(euler_to_ipf(euler_angles_slice, axis="Z"))
# plt.axis("off")
# plt.title("RGB Image from Euler Angles crystal2lab")
# plt.show()


# quat_slice = Quaternion.from_euler(euler_angles_slice, direction="crystal2lab")

# quat_slice = quat_slice.data

# ori = Orientation(np.asarray(quat_slice), symmetry=SYM.Oh)

ori_fz = ori.map_into_symmetry_reduced_zone()

# key = IPFColorKeyTSL(SYM.Oh, direction=getattr(Vector3d, f"{"z".lower()}vector")())

# print("Quats from Euler: No Change")
# # Plot Scalar first
# plt.figure(figsize=(6, 6))
# plt.imshow(
#     IPFColorKeyTSL(
#         SYM.Oh, direction=getattr(Vector3d, f"{"z".lower()}vector")()
#     ).orientation2color(ori.inv())
# )
# plt.axis("off")
# plt.title("RGB Image from Quaternions created from Euler Angles")
# plt.show()


# key.orientation2color(ori.inv())

# def quaternion_to_ipf(quats, axis="Z"):
#     ori = Orientation(np.asarray(quats), symmetry=SYM.Oh)

#     key = IPFColorKeyTSL(SYM.Oh, direction=getattr(Vector3d, f"{axis.lower()}vector")())
#     return key.orientation2color(ori.inv())


# print("Quats from Euler: No Change")
# # Plot Scalar first
# plt.figure(figsize=(6, 6))
# plt.imshow(quaternion_to_ipf(quat_slice, axis="Z"))
# plt.axis("off")
# plt.title("RGB Image from Quaternions created from Euler Angles")
# plt.show()


"""
Quaternions!!
"""

# with h5py.File(file_path, "r") as f:
#     quat = f["DataStructure/ImageDataContainer/CellData/Quaternions"]
#     print("Shape:", quat.shape)
#     print("Dtype:", quat.dtype)

#     # Read into memory
#     their_quats = quat[()]

#     np.save("their_quats.npy", their_quats)


# my_quats_fz = np.load("my_quats_fz.npy")

# my_quats = np.load("my_quats.npy")

# quats = np.load(
#     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/Open_718_Test_hr_x_block_0.npy"
# )

# # quats = their_quats
# print(f"quats shape: {quats.shape}")


# quat_slice = quats
# # Get slice
# # quat_slice = quats[:, :, 358, :]
# print(f"quats slice shape: {quat_slice.shape}")

# # Fix scalar last
# if float(np.mean(np.abs(quat_slice[..., -1]))) > float(
#     np.mean(np.abs(quat_slice[..., 0]))
# ):
#     print("Flipping scalar to S, <X,Y,Z>")
#     q_out = np.concatenate([quat_slice[..., 3:4], quat_slice[..., 0:3]], axis=-1)


# def quaternion_to_ipf(quats, axis="Z"):
#     ori = Orientation(np.asarray(quats), symmetry=SYM.Oh)

#     key = IPFColorKeyTSL(SYM.Oh, direction=getattr(Vector3d, f"{axis.lower()}vector")())
#     return key.orientation2color(ori)


# print("SLICE CHANGED - Scalar First")
# # Plot Scalar first
# plt.figure(figsize=(6, 6))
# plt.imshow(quaternion_to_ipf(q_out, axis="Z"))
# plt.axis("off")
# plt.title("RGB Image from Quaternions")
# plt.show()


# # Plot Scalar last
# print("SLICE NO CHANGE - Scalar Last")

# plt.figure(figsize=(6, 6))
# plt.imshow(quaternion_to_ipf(quat_slice, axis="Z"))
# plt.axis("off")
# plt.title("RGB Image from Quaternions")
# plt.show()


"""
My normal plotting stuff
"""

# WHAT I NORMALLY PLOT WITH

# # quats_formatted = _format_quaternions(quat_slice)
# render_ipf_image(q_out, SYM.Oh)

# print("SLICE NO CHANGE - Scalar Last")
# render_ipf_image(quat_slice, SYM.Oh)


# _DIRS = {"X": Vector3d((1, 0, 0)), "Y": Vector3d((0, 1, 0)), "Z": Vector3d((0, 0, 1))}

# def render_ipf_image(
#     arr_hw4: np.ndarray,
#     sym_class,
#     out_png: Optional[str] = None,
#     ref_dir: str = "ALL",
#     include_key: bool = True,
#     overwrite: bool = False,
# ):
#     """Render quaternion orientation array to an IPF image with consistent formatting."""

#     if arr_hw4.shape[-1] != 4:
#         raise ValueError(f"Expected (H,W,4) input, got {arr_hw4.shape}")

#     if out_png and not overwrite and os.path.exists(out_png):
#         return out_png

#     ori = Orientation(arr_hw4)
#     ori.symmetry = sym_class
#     ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

#     ref_dir_lc = ref_dir.lower()
#     show_all = ref_dir_lc == "all"

#     ncols = 3 if show_all else 1
#     key_cols = 1 if include_key else 0
#     fig_cols = ncols + key_cols
#     wr = [1] * ncols + ([0.9] if include_key else [])

#     fig = plt.figure(
#         constrained_layout=False,
#         figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
#     )
#     gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.25)
#     axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

#     if show_all:
#         for name, ax in zip(("X", "Y", "Z"), axes):
#             ckey.direction = _DIRS[name]
#             ax.imshow(ckey.orientation2color(~ori))
#             ax.set_aspect("equal", adjustable="box")
#             ax.set_title(f"IPF-{name}")
#             ax.axis("off")
#     else:
#         ref = ref_dir.upper()
#         if ref not in _DIRS:
#             raise ValueError("ref_dir must be 'X','Y','Z','ALL'")
#         ckey.direction = _DIRS[ref]
#         axes[0].imshow(ckey.orientation2color(~ori))
#         axes[0].set_aspect("equal", adjustable="box")
#         axes[0].set_title(f"IPF-{ref}")
#         axes[0].axis("off")

#     if include_key:
#         ax_ipf = fig.add_subplot(
#             gs[0, -1], projection="ipf", symmetry=ori.symmetry.laue
#         )
#         ax_ipf.plot_ipf_color_key()
#         ax_ipf.set_title("")

#     if out_png:
#         os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#         fig.savefig(out_png, bbox_inches="tight")
#         plt.close(fig)
#         return out_png
