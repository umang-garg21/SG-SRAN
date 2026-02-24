import os
import numpy as np
import torch

from visualization.ipf_render import render_ipf_image
from utils.quat_ops import quat_left_multiply_numpy, to_spatial_quat
from utils.symmetry_utils import resolve_symmetry


# def visualize_fcc_sym_ops_on_hr():
from training.data_loading import QuaternionDataset


# def visualize_fcc_sym_ops_on_hr():
# Dataset location and split
dataset_out_root = "/data/warren/materials/EBSD"
dataset_name = "IN718_FZ_2D_SR_x4"
dataset_dir = os.path.join(dataset_out_root, dataset_name)
test_ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

# Load first HR quaternion field (4,H,W), scalar-first
q_hr = test_ds[0][1]  # (4,H,W) (assuming [0][1] is HR)

q_hr.shape

if hasattr(q_hr, "cpu"):
    q_hr_np = q_hr.cpu().numpy()
else:
    q_hr_np = np.array(q_hr)
# Convert to (H,W,4) for IPF rendering

q_hr_np = to_spatial_quat(q_hr_np)

# Resolve FCC symmetry object and get operators
fcc_sym = resolve_symmetry("O")
fcc_ops = fcc_sym.data  # (N_ops, 4)

# Output directory
out_dir = "outputs/ipf_fcc_sym_ops"
os.makedirs(out_dir, exist_ok=True)

# Apply all symmetry operations at once
q_img_syms = quat_left_multiply_numpy(
    q_hr_np,
    fcc_ops,
    layout="quat_last",
)  # (N_ops, H, W, 4)

# Render and save each symmetry-applied image
for i in range(q_img_syms.shape[0]):
    q_img_sym = q_img_syms[i]
    out_png = os.path.join(out_dir, f"ipf_hr_fcc_sym_op_{i:02d}.png")
    render_ipf_image(
        q_img_sym,
        fcc_sym,
        out_png=out_png,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
    )
    print(f"Saved: {out_png}")


print(f"Saved FCC symmetry operation visualizations to {out_dir}")
# return q_img_syms


# # =============================================================================
# # Entry point
# # =============================================================================

# if __name__ == "__main__":
#     visualize_fcc_sym_ops_on_hr()


# ---------------------------------------------------------------------------
# Debugging
# ---------------------------------------------------------------------------

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Orientation
from orix.vector import Vector3d
from orix import plot as orix_plot

from utils.quat_ops import format_quaternions, assert_quaternion_shape

sym_class = fcc_sym

fcc_sym[2]

q_arr = q_img_syms[2]
# Shape: q_arr: (H, W, 4) Scalar FIRST
q_arr.shape
# ---------------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------------
from utils.quat_ops import (
    to_quat_spatial,
    to_scalar_first,
    normalize_quaternions,
    enforce_hemisphere,
    reduce_to_fz_min_angle,
    to_scalar_last,
    to_spatial_quat,
)

assert_quaternion_shape(q_arr)


# 1. Force quaternion-first and scalar-first internally
q_out = to_quat_spatial(q_arr)
# Shape: (4, H, W) Scalar Last
q_out.shape

q_out_2 = to_scalar_first(q_out)
# Shape: (4,H, W) Scalar First
q_out[:, 0, 0]

# q_out2 = q_out

# q = normalize_quaternions(q_out, axis=0, eps=1e-12)


# (q == q_out).all()

# (q == q_out2).all()

# (q_out2 == q_out).all()


# # 2. Normalization & hemisphere (skip if FZ reduction)
# if not reduce_fz:
#     if normalize:
#         q_out = normalize_quaternions(q_out, axis=0, eps=eps)
#     if hemisphere:
#         enforce_hemisphere(q_out, scalar_first=True)

# # 3. FZ reduction (always scalar-first)
# if reduce_fz:
#     if sym is None:
#         raise ValueError("`sym` must be provided when reduce_fz=True")
#     if isinstance(sym, str):
#         sym = resolve_symmetry(sym)

#     q_fz = reduce_to_fz_min_angle(
#         q_out,
#         sym=sym,
#         normalize=normalize,
#         hemisphere=hemisphere,
#         return_op_map=False,
#         eps=eps,
#     )

# # 4. Scalar position
# if not scalar_first:
#     q_out = to_scalar_last(q_out)

# 5. Final layout
# if quat_first:
#     q_out = to_quat_spatial(q_out)
# else:
q_out = to_spatial_quat(q_out)

q_out.shape
# Shape: (H, W, 4) Scalar First

# return q_out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

q_out

q_arr = q_out
q_arr.shape


# q_arr = q_img_syms[2]
# q_arr = q_hr_np

# Predefined reference directions
_DIRS = {
    "X": Vector3d((1, 0, 0)),
    "Y": Vector3d((0, 1, 0)),
    "Z": Vector3d((0, 0, 1)),
}

ori = Orientation(q_arr)
ori.symmetry = fcc_sym
ckey = orix_plot.IPFColorKeyTSL(fcc_sym.laue)

ref_dir = "ALL"
ref_dir_lc = ref_dir.lower()
show_all = ref_dir_lc == "all"
directions = ("X", "Y", "Z") if show_all else (ref_dir.upper(),)

rgb_list = []
for d in directions:
    if d not in _DIRS:
        raise ValueError(f"Invalid ref_dir '{d}'. Must be one of X,Y,Z,ALL.")
    ckey.direction = _DIRS[d]
    rgb_list.append(ckey.orientation2color(ori))


include_key = True

# Render RGB map(s)
rgb_out = rgb_list
show_all = isinstance(rgb_out, list)
ncols = 3 if show_all else 1
key_cols = 1 if include_key else 0
fig_cols = ncols + key_cols
wr = [1] * ncols + ([0.9] if include_key else [])

fig = plt.figure(
    constrained_layout=False,
    figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
)
gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.25)
axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

if show_all:
    for name, rgb, ax in zip(("X", "Y", "Z"), rgb_out, axes):
        ax.imshow(rgb)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"IPF-{name}")
        ax.axis("off")
else:
    ref = ref_dir.upper()
    axes[0].imshow(rgb_out)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(f"IPF-{ref}")
    axes[0].axis("off")

if include_key:
    ax_ipf = fig.add_subplot(gs[0, -1], projection="ipf", symmetry=sym_class.laue)
    ax_ipf.plot_ipf_color_key()
    ax_ipf.set_title("")