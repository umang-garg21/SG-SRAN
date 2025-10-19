# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from matplotlib.colors import to_rgb
# from skimage.color import label2rgb
# from orix.quaternion import Orientation, Misorientation, symmetry


# import numpy as np
# from orix.quaternion import Orientation, symmetry


# def check_quaternion_array(
#     arr: np.ndarray,
#     sym_class: str = "Oh",
#     canonicalize_hemisphere: bool = True,
#     check_fz: bool = True,
#     tol_deg: float = 1e-3,
#     return_fz: bool = False,
# ):
#     """
#     Verify, normalize, and analyze quaternion arrays with Fundamental Zone (FZ) diagnostics.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input quaternion array (H,W,4) or (4,H,W).
#     name : str, optional
#         Label for print diagnostics.
#     sym_class : str, default="Oh"
#         ORIX symmetry class name (e.g., "Oh", "D6h", "C1").
#     canonicalize_hemisphere : bool, default=True
#         If True, flips quaternions to ensure w >= 0 (hemisphere consistency).
#     check_fz : bool, default=True
#         If True, quantifies and prints fraction outside FZ.
#     tol_deg : float, default=1e-3
#         Tolerance (degrees) for FZ equivalence.
#     return_fz : bool, default=False
#         If True, returns FZ-mapped quaternions as well.

#     Returns
#     -------
#     arr2 : np.ndarray
#         Normalized, scalar-first quaternions (H,W,4).
#     stats : dict
#         Summary metrics (norms, hemisphere fraction, FZ fraction, etc.)
#     arr_fz : np.ndarray, optional
#         If `return_fz=True`, the quaternions reduced to FZ.
#     """

#     print(f"Shape: {arr.shape}, dtype: {arr.dtype}")

#     if arr.ndim != 3:
#         raise ValueError("Expected 3D quaternion array (H,W,4) or (4,H,W).")

#     if arr.shape[-1] == 4:
#         arr_hw4 = arr
#         fmt = "(H,W,4)"
#     elif arr.shape[0] == 4:
#         arr_hw4 = np.moveaxis(arr, 0, -1)
#         fmt = "(4,H,W) --> converted to (H,W,4)"
#     else:
#         raise ValueError(f"Unrecognized shape {arr.shape}")
#     print(f"Format interpreted as: {fmt}")

#     # Detect and reorder if vector-first (x,y,z,w)
#     if np.mean(np.abs(arr_hw4[..., 0])) < np.mean(np.abs(arr_hw4[..., -1])):
#         print("Detected vector-first format; reordering to scalar-first (w,x,y,z)")
#         arr_hw4 = arr_hw4[..., [3, 0, 1, 2]]

#     # ---- Normalize ----
#     norms = np.linalg.norm(arr_hw4, axis=-1, keepdims=True)
#     arr2 = arr_hw4 / np.clip(norms, 1e-12, None)

#     # Hemisphere canonicalization (w >= 0)
#     if canonicalize_hemisphere:
#         mask_flip = arr2[..., 0] < 0
#         if np.any(mask_flip):
#             arr2[mask_flip] *= -1
#         frac_flipped = np.mean(mask_flip)
#         print(
#             f"Hemisphere canonicalization: flipped {frac_flipped:.2%} of quaternions."
#         )

#     # ---- Norm + scalar diagnostics ----
#     n_stats = np.squeeze(norms)
#     print(
#         f"Norm stats: min={n_stats.min():.6f}, max={n_stats.max():.6f}, mean={n_stats.mean():.6f}"
#     )

#     w_mean = np.mean(arr2[..., 0])
#     w_neg_frac = np.mean(arr2[..., 0] < 0)
#     print(f"Scalar part (w): mean={w_mean:.4f}, fraction negative={w_neg_frac:.3f}")

#     vec_norms = np.linalg.norm(arr2[..., 1:], axis=-1)
#     print(
#         f"Vector norm mean={vec_norms.mean():.4f}, scalar mean={np.abs(arr2[...,0]).mean():.4f}"
#     )

#     median_q = np.median(arr2.reshape(-1, 4), axis=0)
#     print(f"Median quaternion: {median_q}")

#     if np.any(np.isnan(arr2)):
#         print("Contains NaN values!")
#     else:
#         print("No NaNs found.")

#     stats = {
#         "norm_min": float(n_stats.min()),
#         "norm_max": float(n_stats.max()),
#         "norm_mean": float(n_stats.mean()),
#         "frac_negative_w": float(w_neg_frac),
#         "median_quaternion": median_q.tolist(),
#     }

#     # ---- Fundamental Zone diagnostics ----
#     frac_out_fz = np.nan
#     mean_angle_deg = np.nan
#     max_angle_deg = np.nan

#     if check_fz:
#         try:
#             sym = getattr(symmetry, sym_class)
#             ori = Orientation(arr2, symmetry=sym)
#             ori_fz = ori.map_into_symmetry_reduced_zone()

#             # Misorientation between original and reduced
#             mis = ori_fz.inv() * ori
#             ang = mis.angle
#             outside_mask = ang > np.deg2rad(tol_deg)

#             frac_out_fz = float(outside_mask.mean())
#             mean_angle_deg = float(np.rad2deg(ang.mean()))
#             max_angle_deg = float(np.rad2deg(ang.max()))

#             print(f"Fraction outside FZ: {frac_out_fz:.3f}")
#             print(f"Mean Δ: {mean_angle_deg:.3f}°, Max Δ: {max_angle_deg:.3f}°")

#         except Exception as e:
#             print(f"FZ check failed: {e}")

#     stats.update(
#         {
#             "frac_outside_FZ": frac_out_fz,
#             "mean_angle_deg": mean_angle_deg,
#             "max_angle_deg": max_angle_deg,
#         }
#     )

#     if return_fz and check_fz and "ori_fz" in locals():
#         return arr2, stats, np.asarray(ori_fz.data, dtype=np.float32)
#     else:
#         return arr2, stats


# arr = np.load(
#     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_100.npy",
#     mmap_mode="r",
# )


# arr = np.load(
#     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_0.npy",
#     mmap_mode="r",
# )

# arr_fixed, stats, arr_fz = check_quaternion_array(
#     arr,
#     sym_class="Oh",
#     canonicalize_hemisphere=True,
#     check_fz=True,
#     return_fz=True,
# )

# from orix import plot  # Register orix' projections with Matplotlib
# from orix.vector import Vector3d


# v = Vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# kwargs = {"projection": "ipf", "direction": v}
# plt.rcParams["axes.grid"] = True

# v2 = Vector3d([[1, 1, 2], [1, 1, -1]])

# fig_kwargs = {"figsize": (5, 5)}
# labels = ["x", "y", None]
# v2.scatter(
#     axes_labels=labels, show_hemisphere_label=True, figure_kwargs=fig_kwargs
# )  # default hemisphere is "upper"
# v2.scatter(
#     hemisphere="lower",
#     axes_labels=labels,
#     show_hemisphere_label=True,
#     figure_kwargs=fig_kwargs,
# )
# v2.scatter(hemisphere="both", axes_labels=labels, c=["C0", "C1"])


# from orix.crystal_map import Phase
# from orix.vector import Miller, Vector3d

# cubic = Phase(point_group="m-3m")
# print(cubic, "\n", cubic.structure.lattice.abcABG())


# t100 = Miller(uvw=[1, 0, 0], phase=cubic)
# t100.symmetrise(unique=True)

# t100.multiplicity

# t6 = Miller(uvw=[[0, 0, 1]], phase=cubic)
# t6.draw_circle()

# t6
# t6[0].cross(t6[1]).draw_circle()


# t6.scatter()
# t6.multiplicity

# t7, idx = t6.symmetrise(unique=True, return_index=True)
# labels = plot.format_labels(t7.uvw, ("[", "]"))

# # Get an array with one color per family of vectors
# colors = np.array([f"C{i}" for i in range(t6.size)])[idx]

# t7.scatter(c=colors, vector_labels=labels, text_kwargs={"offset": (0, 0.02)})


# symmetry.Oh.fundamental_sector.scatter()


# import numpy as np
# import matplotlib.pyplot as plt
# from orix.quaternion import Orientation, symmetry
# from orix.vector import Miller, Vector3d
# from orix import plot


# def to_scalar_first_unit(arr_hw4_xyzw: np.ndarray) -> np.ndarray:
#     """Convert quaternions [x,y,z,w] → [w,x,y,z], normalize, enforce w≥0."""
#     if arr_hw4_xyzw.shape[-1] != 4:
#         raise ValueError("Expected (...,4) quaternion array.")
#     w = arr_hw4_xyzw[..., 3:4]
#     v = arr_hw4_xyzw[..., :3]
#     q = np.concatenate([w, v], axis=-1)
#     q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
#     flip = q[..., 0] < 0
#     q[flip] *= -1
#     return q.astype(np.float32)


# def plot_quaternion_directions_fz(
#     arr_hw4, sym_class="Oh", max_points=5000, direction="001"
# ):
#     """
#     Plot quaternion dataset orientations in the stereographic projection
#     alongside the fundamental zone for the given symmetry group.

#     Parameters
#     ----------
#     arr_hw4 : np.ndarray
#         Quaternion array, shape (H,W,4) or (4,H,W), scalar-first or last.
#     sym_class : str, default="Oh"
#         ORIX symmetry label (e.g., "Oh", "D6h", "Td").
#     max_points : int, default=5000
#         Max orientations to sample for clarity.
#     direction : str, default="001"
#         Crystal direction to plot (like "001" → [0,0,1] pole figure).

#     Returns
#     -------
#     None (plots figure)
#     """

#     # --- Format quaternion array ---
#     if arr_hw4.shape[0] == 4 and arr_hw4.ndim == 3:
#         arr_hw4 = np.moveaxis(arr_hw4, 0, -1)
#     arr_hw4 = to_scalar_first_unit(arr_hw4)
#     q_flat = arr_hw4.reshape(-1, 4)
#     if q_flat.shape[0] > max_points:
#         idx = np.random.choice(q_flat.shape[0], max_points, replace=False)
#         q_flat = q_flat[idx]

#     # --- Build Orientation objects ---
#     sym = getattr(symmetry, sym_class)
#     ori = Orientation(q_flat, symmetry=sym)
#     ori_fz = ori.map_into_symmetry_reduced_zone()

#     # --- Convert quaternions to directions ---
#     # Interpret quaternions as rotations acting on a crystal direction [uvw]
#     ref_dir = np.array(
#         [[int(direction[0]), int(direction[1]), int(direction[2])]], dtype=float
#     )
#     v_ref = Vector3d(ref_dir)
#     v_rot = ori_fz * v_ref

#     # --- Prepare FZ template using symmetrically equivalent Miller indices ---
#     cubic = Miller(uvw=[[1, 0, 0], [1, 1, 0], [1, 1, 1]], phase=None)
#     cubic.phase = type("Phase", (), {"point_group": sym_class})()  # dummy Phase

#     # --- Plot ---
#     plt.rcParams.update(
#         {
#             "figure.figsize": (7, 7),
#             "font.size": 14,
#             "lines.markersize": 6,
#             "axes.grid": True,
#         }
#     )

#     fig, ax = plt.subplots(subplot_kw={"projection": "stereographic"})

#     # Plot FZ representative directions
#     colors = ["C0", "C1", "C2"]
#     cubic.scatter(
#         c=colors, figure=fig, marker="*", s=100, axes_labels=["RD", "TD", None]
#     )
#     ax.show_hemisphere_label()
#     ax.set_title(f"Cubic Fundamental Zone ({sym_class})")

#     # Plot dataset orientation poles
#     ax.scatter(v_rot, c="red", s=5, alpha=0.6, label=f"Quaternion {direction} poles")
#     ax.legend(loc="upper right", frameon=True, fontsize=10)

#     plt.tight_layout()
#     plt.show()


# import matplotlib.pyplot as plt
# from orix.vector import Vector3d


# arr_hw4, ori_fz = sample_quaternions_in_fz(n=4000, symmetry_class="Oh", shape=(64, 64))
# print(arr_hw4.shape, arr_hw4.dtype)
# # (64, 64, 4) float32

# # visualize
# from orix.quaternion import symmetry

# visualize_fz_quaternions(ori_fz, symmetry.Oh)


# def visualize_fz_quaternions(ori_fz, sym):
#     """Show orientations within the Oh fundamental zone."""
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(projection="stereographic")
#     ax.scatter(ori_fz * Vector3d.zvector(), c="C3", s=10, alpha=0.7)
#     ax.restrict_to_sector(sym.fundamental_sector, edgecolor="black", lw=1.5, pad=3)
#     ax.set_title("Random orientations in Oh fundamental zone")
#     # ax.set_labels("RD", "TD", None)
#     # ax.show_hemisphere_label()
#     plt.show()


# arr_hw4, ori_fz = sample_quaternions_in_fz(n=1000, symmetry_class="Oh", shape=(10, 10))

# arr_hw4 = arr

# # arr_hw4 = arr_hw4[..., [3, 0, 1, 2]]  # → scalar-first

# sym_class = "Oh"
# # Ensure normalized + scalar-first
# if arr_hw4.shape[0] == 4 and arr_hw4.ndim == 3:
#     arr_hw4 = np.moveaxis(arr_hw4, 0, -1)


# arr_hw4 = arr_hw4 / np.linalg.norm(arr_hw4, axis=-1, keepdims=True)
# flip = arr_hw4[..., 0] < 0
# arr_hw4[flip] *= -1
# q = arr_hw4.reshape(-1, 4)


# # %matplotlib inline

# import matplotlib.pyplot as plt
# import numpy as np

# from orix.quaternion import Orientation, symmetry
# from orix.sampling import get_sample_fundamental
# from orix.vector import Vector3d


# plt.rcParams.update(
#     {
#         "axes.grid": True,
#         "figure.figsize": (15, 5),
#         "font.size": 20,
#         "lines.linewidth": 2,
#     }
# )
# directions = Vector3d(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
# pg432 = symmetry.O
# resolution2 = 3
# rot_quat = get_sample_fundamental(resolution2, point_group=pg432, method="quaternion")
# ori_quat = Orientation(rot_quat, symmetry=pg432)

# rot_quat


# ori_quat2 = ori_quat.get_random_sample(1000)

# ori = Orientation(rot_quat, symmetry=pg432)
# ori_fz = ori.map_into_symmetry_reduced_zone()
# # How many original points were *already* in the FZ?
# tol = np.deg2rad(0.1)
# frac_outside_original = (((~ori_fz).outer(ori)).angle > tol).mean()
# print("Fraction outside FZ (original vs. FZ-reduced):", frac_outside_original)


# ori_quat2 = ori_quat.get_random_sample(1000)


# a = ~ori_quat2

# a.scatter("ipf", direction=directions, c="C0", s=5)

# ori_quat2.scatter("ipf", direction=directions, c="C0", s=5)


# ori_quat2 == ori_fz

# ori_quat2
# ori_fz = ori_quat2.map_into_symmetry_reduced_zone()

# sym_class = "O"

# sym = ori_quat2.symmetry


# fig, ax = plt.subplots(subplot_kw={"projection": "stereographic"}, figsize=(6, 6))
# ax.scatter(ori_fz * Vector3d.zvector(), c="red", s=10, alpha=0.6, label="Quat data")

# sector = sym.fundamental_sector
# color = "black"
# lw = 1.5
# alpha = 0.9
# zorder = 5
# # Temporarily adapt the sector pole to the stereographic projection
# original_pole = deepcopy(sector._pole)
# sector._pole = ax.pole
# edges = sector.edges
# sector._pole = original_pole

# # Project FZ edges onto stereographic plane
# x, y, _ = ax._pretransform_input((edges,))
# patch = mpatches.PathPatch(
#     mpath.Path(np.column_stack([x, y]), closed=True),
#     facecolor="none",
#     edgecolor=color,
#     linewidth=lw,
#     alpha=alpha,
#     zorder=zorder,
# )
# ax.add_patch(patch)

# ax.set_labels("RD", "TD", None)
# ax.show_hemisphere_label()
# # ax.legend()
# plt.show()


# sym = getattr(symmetry, sym_class)
# ori = Orientation(q, symmetry=sym)
# ori_fz = ori.map_into_symmetry_reduced_zone()

# fig, ax = plt.subplots(subplot_kw={"projection": "stereographic"}, figsize=(6, 6))
# ax.scatter(ori * Vector3d.zvector(), c="red", s=10, alpha=0.6, label="Quat data")

# sector = sym.fundamental_sector
# color = "black"
# lw = 1.5
# alpha = 0.9
# zorder = 5
# # Temporarily adapt the sector pole to the stereographic projection
# original_pole = deepcopy(sector._pole)
# sector._pole = ax.pole
# edges = sector.edges
# sector._pole = original_pole

# # Project FZ edges onto stereographic plane
# x, y, _ = ax._pretransform_input((edges,))
# patch = mpatches.PathPatch(
#     mpath.Path(np.column_stack([x, y]), closed=True),
#     facecolor="none",
#     edgecolor=color,
#     linewidth=lw,
#     alpha=alpha,
#     zorder=zorder,
# )
# ax.add_patch(patch)

# # # Optionally expand axis limits slightly
# # pad_angle = np.deg2rad(pad)
# # verts = sector.vertices
# # center = sector.center
# # verts_rot = verts.rotate(center.cross(verts), pad_angle)
# # x_pad, y_pad = ax._projection.vector2xy(verts_rot)
# # pad_min = 0.01
# # if x_pad.size:
# #     ax.set(
# #         xlim=(min(np.min(x_pad), np.min(x)) - pad_min,
# #                 max(np.max(x_pad), np.max(x)) + pad_min),
# #         ylim=(min(np.min(y_pad), np.min(y)) - pad_min,
# #                 max(np.max(y_pad), np.max(y)) + pad_min),
# #     )
# # ax.set_title(f"{sym_class} Fundamental Zone (lines only)")
# ax.set_labels("RD", "TD", None)
# ax.show_hemisphere_label()
# # ax.legend()
# plt.show()


# from orix.crystal_map import Phase
# from orix.quaternion import Rotation
# from orix.vector import Miller

# phase = Phase(point_group="m-3m")
# t = Miller.from_highest_indices(phase, uvw=[1, 1, 1])
# t = t.in_fundamental_sector()
# t = t.unit.unique(use_symmetry=True).round()
# print(t)

# fig = t.scatter(
#     vector_labels=[str(vi).replace(".", "") for vi in t.coordinates],
#     text_kwargs={
#         "size": 15,
#         "offset": (0, 0.03),
#         "bbox": {"fc": "w", "pad": 2, "alpha": 0.75},
#     },
#     return_figure=True,
# )


# r"""
# ========================
# Plot symmetry operations
# ========================

# This example shows how to draw proper symmetry operations :math:`s`
# (no reflections or inversions).
# """

# import matplotlib.pyplot as plt

# from orix import plot
# from orix.vector import Vector3d

# marker_size = 200
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection="stereographic")
# ax.set_title("432", pad=20)
# # 4-fold (outer markers will be clipped a bit...)
# v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
# # ax.symmetry_marker(v4fold, fold=4, c="C4", s=marker_size)
# ax.draw_circle(v4fold, color="blue")
# # 3-fold
# v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
# # ax.symmetry_marker(v3fold, fold=3, c="C3", s=marker_size)
# ax.draw_circle(v3fold, color="red")
# # 2-fold
# # fmt: off
# v2fold = Vector3d(
#     [
#         [ 1,  0, 1],
#         [ 0,  1, 1],
#         [-1,  0, 1],
#         [ 0, -1, 1],
#         [ 1,  1, 0],
#         [-1, -1, 0],
#         [-1,  1, 0],
#         [ 1, -1, 0],
#     ]
# )
# # fmt: on
# # ax.symmetry_marker(v2fold, fold=2, c="red", s=marker_size)
# ax.draw_circle(v2fold, color="blue")


# # fig.axes[0].restrict_to_sector(t.phase.point_group.fundamental_sector)


# # # -----------------------------
# # # --- Core Orientation Utils ---
# # # -----------------------------
# # def to_scalar_first_unit(arr_hw4_xyzw: np.ndarray) -> np.ndarray:
# #     """Convert (H,W,4) quaternions [x,y,z,w] -> [w,x,y,z], normalize, enforce w≥0."""
# #     if arr_hw4_xyzw.shape[-1] != 4:
# #         raise ValueError(f"Expected (...,4), got {arr_hw4_xyzw.shape}")
# #     # reorder
# #     w = arr_hw4_xyzw[..., 3:4]
# #     v = arr_hw4_xyzw[..., :3]
# #     q = np.concatenate([w, v], axis=-1).astype(np.float64, copy=False)
# #     # normalize
# #     n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
# #     q /= n
# #     # hemisphere canonicalization
# #     flip = q[..., 0] < 0
# #     q[flip] *= -1.0
# #     return q.astype(np.float32, copy=False)


# # def reduce_to_fz_hw4(q_hw4_wxyz: np.ndarray, sym) -> np.ndarray:
# #     """Reduce orientations to the fundamental zone (FZ) using ORIX Misorientation API."""
# #     ori = Orientation(q_hw4_wxyz, symmetry=sym)
# #     M = Misorientation(ori.data, symmetry=(sym, sym))
# #     Mf = M.map_into_symmetry_reduced_zone()
# #     return np.asarray(Mf.data, dtype=np.float32)


# # # ---------------------------------------
# # # --- Dataset → Orientation operations ---
# # # ---------------------------------------
# # def get_orientation_map(ds, idx: int, which: str = "HR"):
# #     """Return full orientation map from a QuaternionPairDataset as an Orix Orientation."""
# #     lr_np, hr_np = ds[idx]  # both (4,H,W)
# #     hr_np = np.moveaxis(hr_np.numpy(), 0, -1)
# #     lr_np = np.moveaxis(lr_np.numpy(), 0, -1)
# #     arr = hr_np if which.upper() == "HR" else lr_np
# #     arr = to_scalar_first_unit(arr)
# #     return Orientation(arr.reshape(-1, 4), symmetry=ds.sym_class)


# # def sample_orientations(ori: Orientation, max_points=2000):
# #     """Randomly subsample orientations to avoid huge NxN distance matrices."""
# #     if ori.size > max_points:
# #         idx = np.random.choice(ori.size, max_points, replace=False)
# #         return ori[idx]
# #     return ori


# # # --------------------------------
# # # --- Clustering + Visualization ---
# # # --------------------------------
# # def cluster_and_plot(ori, eps=np.deg2rad(15), min_samples=10, max_points=3000):
# #     """Compare orientation clustering with and without symmetry reduction."""
# #     if ori.size > max_points:
# #         idx = np.random.choice(ori.size, max_points, replace=False)
# #         ori = ori[idx]

# #     # --- Without symmetry
# #     ori_no_sym = Orientation(ori.data, symmetry=symmetry.C1)
# #     mori1 = (~ori_no_sym).outer(ori_no_sym)
# #     D1 = mori1.angle
# #     db1 = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D1)
# #     labels1 = db1.labels_

# #     # --- With symmetry
# #     mori2 = (~ori).outer(ori)
# #     mori2.symmetry = ori.symmetry
# #     mori2 = mori2.map_into_symmetry_reduced_zone()
# #     D2 = mori2.angle.astype(np.float32)
# #     db2 = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D2)
# #     labels2 = db2.labels_

# #     print(f"No Symmetry Labels: {np.unique(labels1)}")
# #     print(f"With Symmetry Labels: {np.unique(labels2)}")

# #     ncol = max(labels1.max(), labels2.max()) + 1
# #     colors = [to_rgb(f"C{i}") for i in range(ncol)]
# #     colors_naive = label2rgb(labels1, colors=colors, bg_label=-1)
# #     colors_sym = label2rgb(labels2, colors=colors, bg_label=-1)

# #     # Trick plotting to use cubic FZ
# #     ori_no_sym.symmetry = ori.symmetry

# #     fig = plt.figure(figsize=(12, 6))
# #     ori_no_sym.scatter(figure=fig, position=(1, 2, 1), c=colors_naive, s=5)
# #     plt.gca().set_title("Clustering (no symmetry)")
# #     ori.scatter(figure=fig, position=122, c=colors_sym, s=5)
# #     plt.gca().set_title("Clustering (with FZ reduction)")
# #     plt.show()


# # # --------------------------------------------
# # # --- FZ reduction visualization diagnostics ---
# # # --------------------------------------------
# # def check_fz_reduction(arr_hw4, sym_class, tol=np.deg2rad(1e-3)):
# #     """
# #     Visualize and quantify how many orientations are outside the FZ.

# #     Parameters
# #     ----------
# #     arr_hw4 : np.ndarray
# #         Quaternion array (H,W,4), scalar-first.
# #     sym_class : orix.quaternion.symmetry.Symmetry
# #         Symmetry class (e.g., Oh, D6h, etc.)
# #     tol : float
# #         Angular tolerance (radians) for determining inside/outside FZ.

# #     Returns
# #     -------
# #     frac_outside : float
# #         Fraction of orientations outside the FZ.
# #     mean_angle_deg : float
# #         Mean misorientation angle in degrees.
# #     max_angle_deg : float
# #         Maximum misorientation angle in degrees.
# #     """
# #     ori = Orientation(arr_hw4, symmetry=sym_class)
# #     ori_fz = ori.map_into_symmetry_reduced_zone()

# #     mis = ori_fz.inv() * ori
# #     ang = mis.angle
# #     outside_mask = ang > tol

# #     frac_outside = float(outside_mask.mean())
# #     mean_angle = float(np.rad2deg(ang.mean()))
# #     max_angle = float(np.rad2deg(ang.max()))

# #     print(f"Fraction outside FZ: {frac_outside:.3f}")
# #     print(f"Mean misorientation: {mean_angle:.3f}°")
# #     print(f"Max misorientation: {max_angle:.3f}°")

# #     # --- Visualization ---
# #     fig = plt.figure(figsize=(12, 6))
# #     ori.scatter(figure=fig, position=(1, 2, 1), s=1, c="C0")
# #     plt.gca().set_title("Before FZ reduction")
# #     ori_fz.scatter(figure=fig, position=122, s=1, c="C1")
# #     plt.gca().set_title("After FZ reduction")

# #     plt.suptitle(
# #         f"Fraction outside FZ: {frac_outside:.3f}, Mean Δ={mean_angle:.2f}°, Max Δ={max_angle:.2f}°",
# #         fontsize=12,
# #     )
# #     plt.show()
# #     return frac_outside, mean_angle, max_angle


# # def antipodal_delta_rad(q1, q2):
# #     """Compute antipodal-invariant quaternion misorientation angle (radians)."""
# #     s = np.abs(np.sum(q1 * q2, axis=-1)).clip(-1.0, 1.0)
# #     return 2.0 * np.arccos(s)


# # # Load HR quaternion map
# # arr = np.load(
# #     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_0.npy",
# #     mmap_mode="r",
# # )
# # arr_wxyz = to_scalar_first_unit(arr)

# # # Check FZ coverage
# # frac_out, mean_ang, max_ang = check_fz_reduction(arr_wxyz, symmetry.Oh)

# # # Optionally cluster
# # ori = Orientation(arr_wxyz.reshape(-1, 4), symmetry=symmetry.Oh)
# # cluster_and_plot(sample_orientations(ori, max_points=3000))


# #     def get_orientation_map(ds, idx: int, which: str = "HR"):
# #         """Return full orientation map from dataset as Orix Orientation."""
# #         lr_np, hr_np = ds.get_numpy_hw4(idx)  # both (H,W,4)
# #         arr = hr_np if which.upper() == "HR" else lr_np
# #         return Orientation(arr.reshape(-1, 4), symmetry=ds.sym_class)

# #     def sample_orientations(ori, max_points=2000):
# #         """Randomly subsample orientations to avoid huge NxN distance matrices."""
# #         if ori.size > max_points:
# #             idx = np.random.choice(ori.size, max_points, replace=False)
# #             return ori[idx]
# #         return ori

# #     def cluster_and_plot(ori, eps=np.deg2rad(17), min_samples=5, max_points=3):
# #         if ori.size > max_points:
# #             idx = np.random.choice(ori.size, max_points, replace=False)
# #             ori = ori[idx]

# #         # Remove symmetry by setting it to point group 1 (identity operation)
# #         ori_without_symmetry = Orientation(ori.data, symmetry=symmetry.C1)
# #         mori1 = (~ori_without_symmetry).outer(ori_without_symmetry)

# #         D1 = mori1.angle
# #         db1 = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D1)
# #         labels1 = db1.labels_
# #         print("No Symmetry Labels:", np.unique(labels1))

# #         # --- With symmetry ---
# #         mori2 = (~ori).outer(ori)

# #         mori2.symmetry = ori.symmetry
# #         mori2 = mori2.map_into_symmetry_reduced_zone()
# #         D2 = mori2.angle

# #         db2 = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(
# #             D2.astype(np.float32)
# #         )
# #         labels2 = db2.labels_
# #         print("With Symmetry Labels:", np.unique(labels2))

# #         color_names = [
# #             to_rgb(f"C{i}") for i in range(max(labels1.max(), labels2.max()))
# #         ]  # ['C0', 'C1', ...]

# #         colors_naive = label2rgb(labels1, colors=color_names, bg_label=-1)
# #         colors = label2rgb(labels2, colors=color_names, bg_label=-1)

# #         # Set symmetry to "trick" the scatter plot to use the Oh fundamental zone
# #         ori_without_symmetry.symmetry = ori.symmetry

# #         # Create figure with 2 panels side by side
# #         fig = plt.figure(figsize=(12, 6))

# #         # Left: clustering without symmetry
# #         ori_without_symmetry.scatter(
# #             figure=fig,
# #             position=(1, 2, 1),
# #             c=colors_naive,
# #             s=10,
# #         )
# #         plt.gca().set_title("Clustering (no symmetry)")

# #         ori.scatter(figure=fig, position=122, c=colors, s=10)
# #         # Right: clustering with FZ reduction
# #         # ori.scatter(
# #         #     projection="ipf", figure=fig, position=122, c=colors, s=10, return_figure=False
# #         # )
# #         plt.gca().set_title("Clustering (with FZ reduction)")

# #         plt.show()

# #     def check_fz_reduction(arr_hw4, sym_class, tol=np.deg2rad(1e-3)):
# #         """
# #         Visualize and quantify how many orientations are outside the FZ.

# #         Parameters
# #         ----------
# #         arr_hw4 : np.ndarray
# #             Quaternion array (H,W,4).
# #         sym_class : orix.quaternion.symmetry.Symmetry
# #             Symmetry class (e.g., Oh).
# #         tol : float
# #             Misorientation tolerance in radians.

# #         Returns
# #         -------
# #         frac_outside : float
# #             Fraction of orientations outside FZ.
# #         mean_angle_deg : float
# #             Mean misorientation angle (deg).
# #         max_angle_deg : float
# #             Max misorientation angle (deg).
# #         """
# #         # Build orientations
# #         ori = Orientation(arr_hw4, symmetry=sym_class)
# #         ori_fz = ori.map_into_symmetry_reduced_zone()

# #         # Misorientation between original and reduced
# #         mis = ori_fz.inv() * ori
# #         ang = mis.angle  # radians

# #         outside_mask = ang > tol
# #         frac_outside = outside_mask.mean()
# #         mean_angle = np.rad2deg(ang.mean())
# #         max_angle = np.rad2deg(ang.max())

# #         # --- Plot side-by-side IPF maps ---
# #         fig = plt.figure(figsize=(12, 6))

# #         # Left: before
# #         ori.scatter(figure=fig, position=(1, 2, 1), s=1, c="C0")
# #         plt.gca().set_title("Before FZ reduction")

# #         # Right: after
# #         ori_fz.scatter(figure=fig, position=122, s=1, c="C1")
# #         plt.gca().set_title("After FZ reduction")

# #         plt.suptitle(
# #             f"Fraction outside FZ: {frac_outside:.3f}, "
# #             f"Mean Δ: {mean_angle:.2f}°, Max Δ: {max_angle:.2f}°",
# #             fontsize=12,
# #         )
# #         plt.show()

# #         return frac_outside, mean_angle, max_angle

# #     def to_scalar_first_unit(arr_hw4_xyzw):
# #         """(H,W,4) [x,y,z,w] -> [w,x,y,z], unit, hemisphere w>=0."""
# #         # reorder
# #         w = arr_hw4_xyzw[..., 3:4]
# #         v = arr_hw4_xyzw[..., :3]
# #         q = np.concatenate([w, v], axis=-1).astype(np.float64, copy=False)
# #         # normalize
# #         n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
# #         q /= n
# #         # hemisphere (enforce w>=0)
# #         flip = q[..., 0] < 0
# #         q[flip] *= -1.0
# #         return q.astype(np.float32, copy=False)

# #     def reduce_to_fz_hw4(q_hw4_wxyz, sym):
# #         # Version-safe: reduce via Misorientation API
# #         ori = Orientation(q_hw4_wxyz, symmetry=sym)
# #         M = Misorientation(ori.data, symmetry=(sym, sym))
# #         Mf = M.map_into_symmetry_reduced_zone()
# #         return np.asarray(Mf.data, dtype=np.float32)

# #     def antipodal_delta_rad(q1, q2):
# #         # δ = 2 arccos(|<q1,q2>|), robust to sign
# #         s = np.abs(np.sum(q1 * q2, axis=-1)).clip(-1.0, 1.0)
# #         return 2.0 * np.arccos(s)

# #     # Load dataset (Train split as example)
# #     ds = QuaternionDataset(cfg["dataset_dir"], split="Test")

# #     # Get HR orientation map for one index
# #     ori = get_orientation_map(ds, idx=0, which="HR")

# #     # Run clustering + visualization
# #     cluster_and_plot(ori, eps=np.deg2rad(8), min_samples=5, max_points=4000)

# #     from orix.quaternion import Orientation
# #     from orix import plot as orix_plot

# #     _, hr_np = ds.get_numpy_hw4(0)
# #     # hr_np is your memmap (H,W,4) in [x,y,z,w]
# #     q_std = to_scalar_first_unit(hr_np)  # -> [w,x,y,z]
# #     q_fz = reduce_to_fz_hw4(q_std, ds.sym_class)  # FZ representative

# #     ang = antipodal_delta_rad(q_std, q_fz)  # radians
# #     frac_outside = float((ang > 1e-3).mean())
# #     print("Fraction outside FZ:", frac_outside)
# #     print("Mean Δ (deg):", float(np.rad2deg(ang.mean())))
# #     print("Max  Δ (deg):", float(np.rad2deg(ang.max())))

# #     plt.rcParams["axes.grid"] = False

# #     # We'll want our plots to look a bit larger than the default size
# #     new_params = {
# #         "figure.facecolor": "w",
# #         "figure.figsize": (20, 7),
# #         "lines.markersize": 10,
# #         "font.size": 15,
# #         "axes.grid": True,
# #     }
# #     plt.rcParams.update(new_params)

# #     ds = QuaternionDataset(cfg["dataset_dir"], split="Test")
# #     O2 = get_orientation_map(
# #         ds=QuaternionDataset(cfg["dataset_dir"], split="Test"), idx=2, which="HR"
# #     )
# #     S = SYM.Oh
# #     ipfkey = orix_plot.IPFColorKeyTSL(S)
# #     O2.symmetry = ipfkey.symmetry
# #     rgb_z = ipfkey.orientation2color(O2)
# #     O2.scatter("ipf", c=rgb_z, direction=ipfkey.direction)


# # import numpy as np


# # # Example usage
# # arr = np.load(
# #     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_0.npy",
# #     mmap_mode="r",
# # )

# # if np.mean(np.abs(arr[..., 0])) < np.mean(np.abs(arr[..., -1])):
# #     print(" Detected vector-first format; reordering to scalar-first (s,x,y,z)")
# #     arr = arr[..., [3, 0, 1, 2]]


# # # --- Normalize (important for ORIX orientation consistency) ---
# # arr2 = arr / np.linalg.norm(arr, axis=-1, keepdims=True)


# # # --- Define symmetry (e.g., cubic Oh for FCC) ---
# # sym = symmetry.Oh

# # # --- Create Orientation object ---
# # ori = Orientation(arr, symmetry=sym)

# # # --- Map into the Fundamental Zone (FZ) ---
# # ori_fz = ori.map_into_symmetry_reduced_zone()

# # # --- Check fraction already in FZ ---
# # frac_in_fz = np.mean(ori == ori_fz)

# # print(f"Fraction already in FZ: {frac_in_fz:.3f}")


# # check_quaternion_array(arr)

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from orix.quaternion import Orientation, symmetry

# # # --- Load your quaternion map ---
# # # arr = np.load("example_quaternion.npy", mmap_mode="r")  # (H,W,4) or (4,H,W)
# # if arr.shape[0] == 4:
# #     arr = np.moveaxis(arr, 0, -1)
# # arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)  # normalize


# # # --- Create Orientation object ---
# # sym = symmetry.Oh  # cubic (m-3m), typical for FCC
# # ori = Orientation(arr, symmetry=sym)

# # # --- Reduce to Fundamental Zone (FZ) ---
# # ori_fz = ori.in_FZ()
# # frac_in_fz = (ori == ori_fz).mean()
# # print(f"Fraction already in FZ: {frac_in_fz:.3f}")

# # # --- Plot quaternions in Rodrigues space (FZ visualization) ---
# # fig = plt.figure(figsize=(6, 6))
# # ax = fig.add_subplot(111, projection="rodrigues", symmetry=sym)
# # ax.scatter(ori, s=1, alpha=0.4, color="gray", label="Raw")
# # ax.scatter(ori_fz, s=1, alpha=0.8, color="red", label="In FZ")
# # ax.plot_FZ_boundary()
# # ax.set_title("FCC (Oh) Fundamental Zone — Rodrigues Space")
# # ax.legend()
# # plt.show()


# # # import numpy as np
# # # from orix.quaternion import Orientation, symmetry

# # # # Example cubic symmetry
# # # sym = symmetry.Oh

# # # # Create orientation
# # # q = train_set.get_numpy_hw4(0)[1]
# # # ori = Orientation(q, symmetry=sym)

# # # # Map into fundamental zone (old API)
# # # ori_fz = ori.map_into_symmetry_reduced_zone()

# # # print("Original:", ori)
# # # print("Fundamental zone representative:", ori_fz)
