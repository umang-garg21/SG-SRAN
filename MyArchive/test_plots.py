# if __name__ == "__main__":
#     # Example usage
#     dataset_out_root = "/data/warren/materials/EBSD"
#     dataset_name = "IN718_2D_SR_x4"
#     dataset_dir = os.path.join(dataset_out_root, dataset_name)

#     # dataset_info = build_quaternion_sr_dataset(
#     #     hr_dirs={
#     #         "Train": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/*.npy",
#     #         "Val": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Val/HR_Images/preprocessed_imgs_all_Blocks/*.npy",
#     #         "Test": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/*.npy",
#     #     },
#     #     out_root=dataset_out_root,
#     #     dataset_name=dataset_name,
#     #     scale=4,
#     #     symmetry="Oh",
#     #     creator="Warren Zamudio",
#     #     contact="wzamudio@ucsb.edu",
#     # )

#     # save_dataset_ipfs(
#     #     dataset_root=dataset_dir,
#     #     splits=("Train", "Val", "Test"),
#     #     which_list=("HR", "LR"),
#     #     ref_dir="ALL",
#     #     include_key=True,
#     #     overwrite=False,
#     #     num_workers=16,  # adjust for CPU cores
#     # )

#     train_ds = QuaternionDataset(dataset_dir, split="Train")

#     # plot_unfolded_ipf_with_symmetry_and_fz(
#     #     q_spatial,
#     #     ref_dir="X",
#     #     tol_deg=0,
#     #     out_png="fz_patch_debug/indx_179/ipf_X_unfolded.png",
#     # )

#     # indx = 254
#     indx = 179
#     _, q_spatial = train_ds.get_numpy_spatial_quat(indx)

#     render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

#     q_fz_min, op_map = reduce_to_fz_min_angle_fast(
#         q_spatial, symmetry=SYM.Oh, batch_size=100000000
#     )

#     render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

#     result = find_and_render_worst_fz_region(
#         q_spatial,
#         train_ds.sym_class,
#         patch_size=128,
#         out_dir=f"fz_debug/indx_{indx}",
#         base_name="quats",
#         ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
#         tol_deg=0.1,
#     )
#     q_fz_min, op_map = reduce_to_fz_min_angle_fast(
#         q_spatial, symmetry=SYM.Oh, batch_size=100000000
#     )
#     result = find_and_render_worst_fz_region(
#         q_fz_min,
#         train_ds.sym_class,
#         patch_size=128,
#         out_dir=f"fz_debug/indx_{indx}_FZ",
#         base_name="quats_fz",
#         ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
#         tol_deg=0.1,
#     )
#     # quat_scalar_first = _format_quaternions(
#     #     np.load(
#     #         "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/Open_718_Train_hr_x_block_327.npy"
#     #     ),
#     #     normalize=False,
#     #     enforce_hemisphere=False,
#     # )

#     # q_spatial = _to_spatial_quat(quat_scalar_first)

#     q_fz_min, op_map = reduce_to_fz_min_angle_fast(
#         q_spatial, symmetry=SYM.Oh, batch_size=100000000
#     )
#     q_fz_min_format = _to_spatial_quat(_format_quaternions(q_fz_min))

#     print("Unique operators used:", np.unique(op_map))
#     print("Swapped pixel count:", np.sum(op_map != 0))
#     render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
#     render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

#     result = find_and_render_worst_fz_region(
#         q_fz_min_format,
#         train_ds.sym_class,
#         patch_size=300,
#         out_dir=f"fz_debug/indx_327",
#         base_name="quats",
#         ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
#         tol_deg=0.1,
#     )


# #     # q_spatial.shape

# #     # q_spatial[:, :, 0]

# #     render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

# #     q_fz_norm = _to_spatial_quat(_format_quaternions(q_spatial_fz))

# #     render_ipf_image(q_fz_norm, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

# #     q_spatial.shape

# #     ori = Orientation(q_spatial.reshape(-1, 4), symmetry=train_ds.sym_class)

# #     ori_fz, idx_left, idx_right = ori.map_into_symmetry_reduced_zone_with_ops(True)

# #     q_spatial_fz = ori_fz.data.reshape(q_spatial.shape)

# #     test_q = q_spatial[72, 75]
# #     ori = Orientation(test_q, symmetry=SYM.Oh)

# #     ori_fz = ori.map_into_symmetry_reduced_zone()

# #     test_q_fz = ori_fz.data

# #     test_q * SYM.Oh.data[1]


# # q_fz_min, op_map = reduce_to_fz_min_angle_fast(q_spatial, symmetry=SYM.Oh, batch_size=250000)
# # print("Unique operators used:", np.unique(op_map))
# # print("Swapped pixel count:", np.sum(op_map != 0))
# # render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)


# # ori = Orientation(q_spatial.reshape(-1, 4), symmetry=SYM.Oh)
# # ori_fz = ori.map_into_symmetry_reduced_zone()

# # q_spatial_fz = ori_fz.data.reshape(q_spatial.shape)

# # render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

# # render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
# # render_ipf_image(q_spatial_fz, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)


# # plot_unfolded_ipf_with_symmetry_and_fz(
# #     q_spatial,
# #     ref_dir="Z",
# #     tol_deg=0,
# #     out_png="fz_patch_debug/indx_179_FZ/ipf_Z_unfolded.png",
# # )

# # plot_unfolded_ipf_with_symmetry_and_fz(
# #     q_spatial,
# #     ref_dir="X",
# #     tol_deg=0,
# #     out_png="fz_patch_debug/indx_179/ipf_X_unfolded.png",
# # )

# # plot_unfolded_ipf_with_symmetry_and_fz(
# #     q_spatial,
# #     ref_dir="Y",
# #     tol_deg=0,
# #     out_png="fz_patch_debug/indx_179/ipf_Y_unfolded.png",
# # )

# # train_ds.check_integrity(10000, sample_all=True)

# # save_dataset_ipf_summary(
# #     train_ds,
# #     out_png=os.path.join(dataset_dir, "Train", "IPF_Z_HR.png"),
# #     which="HR",
# #     ref_dir="Z",
# #     n_total=5214208,
# #     per_file_max=4096,
# #     include_key=False,
# # )

# # val_ds = QuaternionDataset(dataset_dir, split="Val")
# # test_ds = QuaternionDataset(dataset_dir, split="Test")

# # # a = np.load(
# # #     "/data/warren/materials/EBSD/IN718_2D_SR_x4/Train/HR_Data/IN718_2d_sr_x4_train_hr_x_block_1.npy"
# # # )

# # a = np.load(
# #     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_0.npy"
# # )

# # q = np.concatenate([a[..., 3:4], a[..., :3]], axis=-1)

# # render_ipf_image(a, SYM.O, ref_dir="ALL", include_key=True, overwrite=True)
# # render_ipf_image(q, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
