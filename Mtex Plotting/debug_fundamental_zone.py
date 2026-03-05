#!/usr/bin/env python3
"""
============================================================
Debug Quaternion Convention from .mat (converted from .npy)
Python equivalent of debug_fundamental_zone.m

Determines:
  - component order (xyzw vs wxyz)
  - whether stored orientation matches MTEX (crystal->specimen) or Bunge (specimen->crystal)

MTEX convention: crystal -> specimen (inverse of Bunge)
Bunge convention: specimen -> crystal
They are inverses. Euler angles reported are the usual Bunge ZXZ angles.
============================================================
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from orix.quaternion import Orientation
from orix.quaternion.symmetry import Oh, C1
from orix import plot as orix_plot
from orix.vector import Vector3d

# ============================================================
# Load data
# ============================================================
mat = sio.loadmat("Open_718_Test_hr_x_block_0.mat")
data = mat["data"].astype(np.float64)   # expects (H, W, 4)

H, W, C = data.shape
assert C == 4, f"Expected 4 quaternion components, got {C}"
print(f"Shape: {H} x {W} x {C}\n")

# ============================================================
# Basic stats per component
# ============================================================
flat = data.reshape(-1, 4)
for i in range(4):
    col = flat[:, i]
    print(
        f"Component {i+1}: mean = {col.mean():+.4f}  "
        f"std = {col.std():.4f}  "
        f"min = {col.min():+.4f}  "
        f"max = {col.max():+.4f}"
    )
print()

# ============================================================
# STEP 1 — Guess component order (xyzw vs wxyz)
# ============================================================
# The scalar (w) component tends to have the largest absolute mean
# because unit quaternions with small rotations have w ~ 1.
means = np.abs(flat.mean(axis=0))
scalar_idx = int(np.argmax(means))   # 0-indexed
print(f"Likely scalar component index: {scalar_idx + 1}")

if scalar_idx == 3:
    print("-> Most likely stored as [x y z w] (scalar last)\n")
    # Rearrange to [w, x, y, z]
    q_wxyz = np.concatenate(
        [data[..., 3:4], data[..., 0:1], data[..., 1:2], data[..., 2:3]],
        axis=-1,
    )
elif scalar_idx == 0:
    print("-> Most likely stored as [w x y z] (scalar first)\n")
    q_wxyz = data.copy()
else:
    print("WARNING: Scalar not clearly at index 1 or 4 — manual inspection required")
    q_wxyz = data.copy()

# Normalize
norms = np.linalg.norm(q_wxyz, axis=-1, keepdims=True)
q_wxyz = q_wxyz / norms

q_flat = q_wxyz.reshape(-1, 4)   # (N, 4)  [w, x, y, z]  — orix/Bunge convention

# ============================================================
# STEP 2 — FZ reduction via Oh symmetry (follows notebook approach)
# ============================================================
# Oh has 48 symmetry operators. For each quaternion q we compute
#   s_inv ⊗ q  for all s in Oh,
# enforce w >= 0, and pick the representative with maximum w
# (= minimum rotation angle = deepest in the fundamental zone).

oh_ops     = np.asarray(Oh.data, dtype=np.float64)   # (48, 4) [w,x,y,z]
oh_ops_inv = oh_ops.copy()
oh_ops_inv[:, 1:] *= -1   # conjugate = inverse for unit quaternions


def _quat_mul(a, b):
    """Hamilton product a ⊗ b.  a,b: (..., 4) [w,x,y,z]."""
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        axis=-1,
    )


def reduce_to_fz(q, sym_inv):
    """
    Reduce quaternions q to the Oh fundamental zone.

    Parameters
    ----------
    q       : (N, 4) unit quaternions [w,x,y,z]
    sym_inv : (G, 4) inverse symmetry operators [w,x,y,z]

    Returns
    -------
    (N, 4) FZ representatives with w >= 0 and maximum w.
    """
    N = q.shape[0]
    # Broadcast: (G, 1, 4) ⊗ (1, N, 4) → (G, N, 4)
    cands = _quat_mul(sym_inv[:, np.newaxis], q[np.newaxis])
    # Enforce w >= 0
    cands *= np.where(cands[..., 0:1] < 0, -1.0, 1.0)
    # Pick the candidate with the largest w for each quaternion
    best = np.argmax(cands[..., 0], axis=0)           # (N,)
    return cands[best, np.arange(N)]                   # (N, 4)


print("Reducing all quaternions to the fundamental zone...")
q_fz = reduce_to_fz(q_flat, oh_ops_inv)               # (N, 4)
print("Done.\n")

# ============================================================
# STEP 3 — IPF colors  (IPFColorKeyTSL, same as notebook)
# ============================================================
ori_fz = Orientation(q_fz, symmetry=(Oh, C1))

ckey = orix_plot.IPFColorKeyTSL(Oh.laue)
ref_dirs = {
    "X": Vector3d.xvector(),
    "Y": Vector3d.yvector(),
    "Z": Vector3d.zvector(),
}

print("Computing IPF colors for X, Y, Z reference directions...")
ipf_colors = {}
for name, vec in ref_dirs.items():
    ckey.direction = vec
    ipf_colors[name] = ckey.orientation2color(ori_fz)   # (N, 3) float RGB in [0,1]
print("Done.\n")

# ============================================================
# FIGURE 1 — Spatial IPF maps  (H × W image, every pixel = its IPF color)
# ============================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("Spatial IPF Color Map — all quaternions", fontsize=13)

for ax, name in zip(axes1, ["X", "Y", "Z"]):
    rgb = np.clip(ipf_colors[name].reshape(H, W, 3), 0.0, 1.0)
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    ax.set_title(f"IPF-{name}", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig("ipf_spatial_map.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → ipf_spatial_map.png\n")

# ============================================================
# FIGURE 2 — IPF stereographic scatter  (all points, small markers)
# ============================================================
fig2, axes2 = plt.subplots(
    1, 4,
    figsize=(17, 5),
    subplot_kw={"projection": "ipf", "symmetry": Oh.laue},
)
fig2.suptitle(
    f"IPF Scatter — all {H*W} quaternions, FZ-reduced (Oh / m-3m)",
    fontsize=12,
)

for ax, name in zip(axes2[:3], ["X", "Y", "Z"]):
    ckey.direction = ref_dirs[name]
    ax.scatter(
        ori_fz,
        c=ipf_colors[name],
        s=2,          # small markers so you can distinguish grains
        alpha=0.6,
        rasterized=True,   # fast rendering for large N
        linewidths=0,
    )
    ax.set_title(f"IPF-{name}", fontsize=11)

# Color key panel
ckey.direction = Vector3d.zvector()
axes2[3].plot_ipf_color_key(annotated=True)
axes2[3].set_title("Color key (Z)", fontsize=11)

plt.tight_layout()
plt.savefig("ipf_scatter.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → ipf_scatter.png")
