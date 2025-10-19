# from builders.dataset_builder import build_quaternion_sr_dataset
# from visualization.save_dataset_ipfs import save_dataset_ipfs
# import os
# from training.data_loading import QuaternionDataset
# from tqdm import tqdm

# # Example usage
# dataset_out_root = "/data/warren/materials/EBSD"
# dataset_name = "IN718_FZ_2D_SR_x4"
# dataset_dir = os.path.join(dataset_out_root, dataset_name)


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


from post_processing.post_process import run_postprocess_from_config

# from postprocess.run_postprocess_from_config import run_postprocess_from_config

run_postprocess_from_config("experiments/IN718/debug_x4_kss_4", max_samples=8)

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
