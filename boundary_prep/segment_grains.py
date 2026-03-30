# -*- coding:utf-8 -*-
"""
File:        test_grain_segmentation_on_ipf.py
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu

Description:
    Full grain segmentation test script for quaternion EBSD maps.
    Includes:
      ✔ symmetry-aware misorientation
      ✔ CUDA-accelerated misorientation + GB mask
      ✔ graph-based grain segmentation
      ✔ small-grain cleanup
      ✔ KAM map computation
      ✔ IPF rendering + overlays

    Input:  (1,4,H,W) quaternion field, scalar-first
    Output: grain_labels.png, gb_mask.png, KAM.png, IPF images
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from orix.quaternion import symmetry as SYM

from training.data_loading import QuaternionDataset
from visualization.ipf_render import render_ipf_image, render_ipf_rgb


# ======================================================================
# QUATERNION HELPERS
# ======================================================================


def qnorm(q):
    """Normalize quaternion tensor (B,4,H,W)."""
    return q / (q.norm(dim=1, keepdim=True).clamp_min(1e-12))


def quat_mul(q1, q2):
    """Hamilton product with broadcasting. (...,4) x (...,4) → (...,4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@torch.jit.script
def misorientation_angle_sym(q1, q2, sym_ops):
    """
    Symmetry-aware misorientation, fully vectorized.
    q1, q2: (...,4)
    sym_ops: (G,4,4)
    returns: angle in radians (...,)
    """
    # q2 variants: (...,G,4)
    q2var = torch.einsum("gij,...j->...gi", sym_ops, q2)
    dots = torch.sum(q1.unsqueeze(-2) * q2var, dim=-1).abs().clamp(-1, 1)
    best = dots.max(dim=-1).values
    return 2.0 * torch.acos(best)


# ======================================================================
# MISORIENTATION NEIGHBOR PAIRS
# ======================================================================


def neighbor_misorientation(q, sym_ops):
    """
    q: (1,4,H,W)
    returns:
      mis_x: (H,W-1)
      mis_y: (H-1,W)
    """
    _, _, H, W = q.shape
    qHW = q.squeeze(0).permute(1, 2, 0)  # (H,W,4)

    # horizontal
    qL = qHW[:, :-1]
    qR = qHW[:, 1:]
    mis_x = misorientation_angle_sym(qL, qR, sym_ops)

    # vertical
    qU = qHW[:-1, :]
    qD = qHW[1:, :]
    mis_y = misorientation_angle_sym(qU, qD, sym_ops)

    return mis_x, mis_y


# ======================================================================
# GRAPH-BASED SEGMENTATION
# ======================================================================


def segment_grains_graph(q, sym_ops, thr_deg=5.0):
    """
    q: (1,4,H,W)
    Returns: grain_labels(H,W), num_grains
    """
    mis_x, mis_y = neighbor_misorientation(q, sym_ops)
    thr = np.deg2rad(thr_deg)

    mis_x_np = mis_x.cpu().numpy()
    mis_y_np = mis_y.cpu().numpy()

    _, _, H, W = q.shape
    N = H * W
    rows, cols = [], []

    def idx(i, j):
        return i * W + j

    # horizontal
    mask_x = mis_x_np <= thr
    ii, jj = np.where(mask_x)
    p1 = idx(ii, jj)
    p2 = idx(ii, jj + 1)
    rows += list(p1) + list(p2)
    cols += list(p2) + list(p1)

    # vertical
    mask_y = mis_y_np <= thr
    ii, jj = np.where(mask_y)
    p1 = idx(ii, jj)
    p2 = idx(ii + 1, jj)
    rows += list(p1) + list(p2)
    cols += list(p2) + list(p1)

    data = np.ones(len(rows), np.uint8)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    nG, labels = connected_components(A, directed=False)
    return labels.reshape(H, W).astype(np.int32), nG


# ======================================================================
# SMALL-GRAIN CLEANUP
# ======================================================================


# ----------------------------------------------------------
# Generate neighbor offsets
# ----------------------------------------------------------
def generate_offsets(radius=1, connectivity=8):
    offs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if connectivity == 4 and abs(dy) + abs(dx) != 1:
                continue
            offs.append((dy, dx))
    return offs


# ----------------------------------------------------------
# Full CUDA small-grain cleanup (vectorized)
# ----------------------------------------------------------
def cleanup_small_grains_cuda(labels, q, sym_ops, min_pixels=150, max_iter=3):
    """
    Fully CUDA-based, vectorized small-grain cleanup.
    labels: (H,W) int64  torch tensor on CUDA
    q:      (1,4,H,W)    torch tensor on CUDA
    sym_ops:(G,4,4)      torch tensor on CUDA
    """

    device = q.device
    labels = labels.clone()  # keep on GPU
    _, _, H, W = q.shape

    # Quaternion field (H,W,4)
    qHW = q.squeeze(0).permute(1, 2, 0)  # (H,W,4)

    offsets = generate_offsets(radius=1, connectivity=4)  # 4-neighbor cleanup

    for _ in range(max_iter):

        # Count grain sizes (GPU)
        unique, counts = torch.unique(labels, return_counts=True)
        small = unique[counts < min_pixels]

        if small.numel() == 0:
            break

        # boolean mask of small-grain pixels
        small_mask = torch.isin(labels, small).bool()  # (H,W)

        # Get (H,W,4) center quats
        q_center = qHW[small_mask]  # (Ns,4)
        Ns = q_center.shape[0]

        # Map flattened indices back to (i,j)
        idxs = torch.nonzero(small_mask, as_tuple=False)  # (Ns,2)

        # ------------------------------
        # Gather neighbors for *all* Ns pixels
        # ------------------------------
        neigh_quats = []
        neigh_labels = []
        valid_masks = []

        for dy, dx in offsets:
            ny = idxs[:, 0] + dy
            nx = idxs[:, 1] + dx

            valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)

            # sanitize invalid neighbor coords
            ny = torch.clamp(ny, 0, H - 1)
            nx = torch.clamp(nx, 0, W - 1)

            qN = qHW[ny, nx]  # (Ns,4)
            labN = labels[ny, nx]  # (Ns,)

            neigh_quats.append(qN)
            neigh_labels.append(labN)
            valid_masks.append(valid)

        # Stack neighbors: (Ns,K,4), (Ns,K), (Ns,K)
        neigh_quats = torch.stack(neigh_quats, dim=1)
        neigh_labels = torch.stack(neigh_labels, dim=1)
        valid_masks = torch.stack(valid_masks, dim=1)

        # Expand q_centers to match neighbors
        qC = q_center.unsqueeze(1)  # (Ns,1,4)
        qC = qC.expand(-1, neigh_quats.shape[1], -1)  # (Ns,K,4)

        # ------------------------------
        # Compute misorientation (Ns,K)
        # ------------------------------
        ang = misorientation_angle_sym(qC, neigh_quats, sym_ops)  # (Ns,K)

        # Mask out invalid neighbors
        ang_masked = torch.where(valid_masks, ang, torch.full_like(ang, 999.0))

        # ------------------------------
        # Compute best neighbor grain ID for each small-grain pixel
        # ------------------------------
        best_idx = torch.argmin(ang_masked, dim=1)  # (Ns,)
        best_labels = neigh_labels[torch.arange(Ns), best_idx]  # (Ns,)

        # Assign each small-grain pixel to its best neighbor
        labels[small_mask] = best_labels

    # Compact grain numbering
    uniq = torch.unique(labels)
    remap = {int(o): i for i, o in enumerate(uniq.tolist())}
    remapped = labels.clone()
    for old, new in remap.items():
        remapped[labels == old] = new

    return remapped


def compute_kam(q, labels, sym_ops, radius=1):
    """
    Kernel Average Misorientation (KAM).
    q: (1,4,H,W)  torch tensor on CPU or GPU
    labels: (H,W) numpy array
    sym_ops: (G,4,4) torch tensor
    """
    device = q.device
    _, _, H, W = q.shape

    # (H,W,4)
    qHW = q.squeeze(0).permute(1, 2, 0)  # torch
    kam = torch.zeros((H, W), device=device, dtype=torch.float32)

    for i in range(H):
        for j in range(W):

            g0 = labels[i, j]
            q0 = qHW[i, j].unsqueeze(0).unsqueeze(0)  # (1,1,4) torch

            mis_list = []

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        continue
                    if labels[ni, nj] != g0:
                        continue

                    qN = qHW[ni, nj].unsqueeze(0).unsqueeze(0)  # (1,1,4)

                    # Torch → Torch → scalar
                    ang = misorientation_angle_sym(q0, qN, sym_ops)[0, 0]
                    mis_list.append(ang)

            if len(mis_list) > 0:
                kam[i, j] = torch.stack(mis_list).mean()
            else:
                kam[i, j] = 0.0

    return kam.cpu().numpy()


def compute_kam_cuda(q, labels, sym_ops, radius=2, connectivity=8, eps=1e-6):

    device = q.device
    dtype = q.dtype
    _, _, H, W = q.shape

    labels = labels.to(device).long()
    qHW = q.squeeze(0).permute(1, 2, 0)  # (H,W,4)

    offsets = generate_offsets(radius, connectivity)
    K = len(offsets)

    q_center = qHW.unsqueeze(2)  # (H,W,1,4)

    neigh_quats = []
    neigh_labels = []
    valid_masks = []

    for dy, dx in offsets:
        qN = torch.zeros_like(qHW)
        lN = torch.zeros_like(labels)
        valid = torch.zeros((H, W), dtype=torch.bool, device=device)

        sy0 = max(0, -dy)
        sy1 = min(H, H - dy)
        dy0 = max(0, dy)
        dy1 = min(H, H + dy)
        sx0 = max(0, -dx)
        sx1 = min(W, W - dx)
        dx0 = max(0, dx)
        dx1 = min(W, W + dx)

        if (sy0 < sy1) and (sx0 < sx1):
            qN[dy0:dy1, dx0:dx1] = qHW[sy0:sy1, sx0:sx1]
            lN[dy0:dy1, dx0:dx1] = labels[sy0:sy1, sx0:sx1]
            valid[dy0:dy1, dx0:dx1] = True

        neigh_quats.append(qN)
        neigh_labels.append(lN)
        valid_masks.append(valid)

    neigh_quats = torch.stack(neigh_quats, dim=2)  # (H,W,K,4)
    neigh_labels = torch.stack(neigh_labels, dim=2)  # (H,W,K)
    valid_masks = torch.stack(valid_masks, dim=2)  # (H,W,K)

    qC = q_center.expand(-1, -1, K, -1)  # (H,W,K,4)

    ang = misorientation_angle_sym(qC, neigh_quats, sym_ops)  # (H,W,K)

    same = labels.unsqueeze(2) == neigh_labels
    mask = same & valid_masks

    m = mask.float()
    ang_m = ang * m
    sum_ang = ang_m.sum(dim=2)
    count = m.sum(dim=2)

    kam = sum_ang / (count + eps)
    kam = torch.where(count > 0, kam, torch.zeros_like(kam))

    return kam


# ======================================================================
# VISUALIZATION HELPERS
# ======================================================================


def plot_labels(labels, out_png):
    H, W = labels.shape
    lab = labels.astype(float)
    lab = lab / (lab.max() + 1e-6)
    rgb = plt.cm.jet(lab)[..., :3]

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Grain labels")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_mask(mask, out_png, title="Mask"):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_scalar_map(arr, out_png, title=""):
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="inferno")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# If not already imported:
# from visualization.ipf_render import render_ipf_rgb  # returns RGB array


import numpy as np
from scipy.ndimage import binary_dilation


def compute_gb_mask(labels_np, connectivity=4, dilate=1):
    """
    Compute grain boundary (GB) mask from integer label map.

    labels_np: (H,W) numpy array
    connectivity: 4 or 8
    dilate: number of iterations to thicken the boundaries
    """
    H, W = labels_np.shape
    gb = np.zeros((H, W), dtype=bool)

    # Horizontal boundaries
    diff_h = labels_np[:, 1:] != labels_np[:, :-1]
    gb[:, 1:] |= diff_h
    gb[:, :-1] |= diff_h

    # Vertical boundaries
    diff_v = labels_np[1:, :] != labels_np[:-1, :]
    gb[1:, :] |= diff_v
    gb[:-1, :] |= diff_v

    if connectivity == 8:
        # NE-SW
        diff_d1 = labels_np[1:, 1:] != labels_np[:-1, :-1]
        gb[1:, 1:] |= diff_d1
        gb[:-1, :-1] |= diff_d1
        # NW-SE
        diff_d2 = labels_np[1:, :-1] != labels_np[:-1, 1:]
        gb[1:, :-1] |= diff_d2
        gb[:-1, 1:] |= diff_d2

    if dilate > 0:
        gb = binary_dilation(gb, iterations=dilate)

    return gb


import matplotlib.pyplot as plt
import numpy as np


def compute_thin_gb_mask(labels_np, connectivity=4):
    """
    Compute a 1-pixel-thick grain boundary mask from label map.
    """

    H, W = labels_np.shape
    gb = np.zeros((H, W), dtype=bool)

    # Horizontal boundaries (one side ONLY)
    gb[:, 1:] |= labels_np[:, 1:] != labels_np[:, :-1]

    # Vertical boundaries
    gb[1:, :] |= labels_np[1:, :] != labels_np[:-1, :]

    if connectivity == 8:
        # Diagonals (one side only)
        gb[1:, 1:] |= labels_np[1:, 1:] != labels_np[:-1, :-1]
        gb[1:, :-1] |= labels_np[1:, :-1] != labels_np[:-1, 1:]

    return gb


def overlay_gb_on_ipf(ipf_rgb, gb_mask, out_png):
    """
    Overlay black grain boundaries on IPF RGB image.

    ipf_rgb: (H,W,3) array in [0,255] or [0,1]
    gb_mask: (H,W) bool
    out_png: output file path
    """
    # convert to float
    img = ipf_rgb.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    # Copy image
    overlay = img.copy()

    # Paint GB pixels black
    overlay[gb_mask] = [0.0, 0.0, 0.0]

    # Plot + save
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


import torch


def smooth_labels_cuda(labels, radius=1, iterations=2):
    """
    Majority-based boundary smoothing.
    Works fully on CUDA.
    Does NOT merge grains.
    """

    device = labels.device
    H, W = labels.shape
    lab = labels.clone()

    # Define neighbor window (3x3 for radius=1, 5x5 for radius=2)
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((dy, dx))
    K = len(offsets)

    # Precompute thin boundary mask (on CUDA)
    boundary_mask = torch.from_numpy(compute_thin_gb_mask(labels.cpu().numpy())).to(
        device
    )

    for _ in range(iterations):

        neigh_stack = []

        for dy, dx in offsets:
            shifted = torch.zeros_like(lab)

            sy0 = max(0, -dy)
            sy1 = min(H, H - dy)
            dy0 = max(0, dy)
            dy1 = min(H, H + dy)

            sx0 = max(0, -dx)
            sx1 = min(W, W - dx)
            dx0 = max(0, dx)
            dx1 = min(W, W + dx)

            if sy1 > sy0 and sx1 > sx0:
                shifted[dy0:dy1, dx0:dx1] = lab[sy0:sy1, sx0:sx1]

            neigh_stack.append(shifted)

        # Stack: (H, W, K)
        neigh = torch.stack(neigh_stack, dim=2)

        # Flatten for mode along last dim
        neigh_flat = neigh.view(-1, K)  # (H*W, K)
        maj = torch.mode(neigh_flat, dim=1).values  # (H*W,)
        maj = maj.view(H, W)

        # Only smooth boundary pixels!
        lab[boundary_mask] = maj[boundary_mask]

    return lab


import torch
import torch.nn.functional as F


def curvature_smooth_labels_cuda(labels, iterations=30, lam=0.15):
    """
    Curvature flow smoothing of grain boundaries.
    This produces MTEX/Dream3D-style smooth boundaries.

    labels: (H,W) int64, CUDA
    iterations: how many smoothing steps
    lam: smoothing strength per step (0.1–0.2)
    """

    device = labels.device
    H, W = labels.shape

    # We maintain the region map as float for diffusion
    region = labels.float()

    # 2D Laplacian kernel for curvature motion
    lap_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,3,3)

    for _ in range(iterations):
        # Compute Laplacian
        lap = F.conv2d(region.unsqueeze(0).unsqueeze(0), lap_kernel, padding=1)
        lap = lap.squeeze(0).squeeze(0)

        # Gradient descent step: region += lam * curvature
        region = region + lam * lap

        # Snap region map back to nearest grain label AFTER smoothing step
        # (this preserves topology, avoids all merging)
        # Compute distances to all grain IDs
        unique = torch.unique(labels)
        # (G, H, W)
        dist = torch.abs(region.unsqueeze(0) - unique.view(-1, 1, 1))
        nearest = unique[torch.argmin(dist, dim=0)]

        region = nearest.float()

    return region.long()


# ======================================================================
# MAIN TEST FUNCTION
# ======================================================================


def test_grain_segmentation_on_ipf():

    # ============================
    # Load quaternion dataset
    # ============================
    dataset_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4"
    dataset_dir = os.path.join(dataset_root, dataset_name)

    ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

    q_lr = ds[0][0].unsqueeze(0)  # (1,4,H,W)
    q_lr = qnorm(q_lr)
    device = q_lr.device
    dtype = q_lr.dtype

    # ============================
    # Load symmetry
    # ============================
    sym_np = np.load("/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy")
    sym_ops = torch.tensor(sym_np, dtype=dtype, device=device)  # (G,4,4)

    # For IPF
    sym_class = SYM.O

    # ===========================================================
    # 1. CPU segmentation (SciPy needs CPU)
    # ===========================================================
    labels_np, nG = segment_grains_graph(q_lr.cpu(), sym_ops.cpu(), thr_deg=3.0)
    print(f"Initial grains: {nG}")

    # ===========================================================
    # 2. Convert to GPU for cleanup + KAM
    # ===========================================================
    labels = torch.from_numpy(labels_np).long().to(q_lr.device)

    # ===========================================================
    # 3. CUDA Cleanup (fast!)
    # ===========================================================

    labels = cleanup_small_grains_cuda(
        labels,  # torch (H,W)
        q_lr,  # torch (1,4,H,W), CUDA
        sym_ops,  # torch (G,4,4), CUDA
        min_pixels=3,
        max_iter=1,
    )
    # labels = curvature_smooth_labels_cuda(labels, iterations=20, lam=0.2)
    # labels_np = labels_smooth.cpu().numpy()

    # # Smooth grain labels (Dream3D style)
    # labels = smooth_labels_cuda(labels, radius=1, iterations=1)

    # Convert for plotting
    # labels_np = labels_smoothed.cpu().numpy()

    # back to CPU numpy ONLY for plotting
    labels_np = labels.cpu().numpy()

    nG2 = np.unique(labels_np).size
    print(f"Grains after cleanup: {nG2}")

    # ============================
    # KAM computation
    # ============================
    kam = compute_kam_cuda(q_lr, labels, sym_ops, radius=2, connectivity=8)
    kam_np = kam.cpu().numpy()
    print("Computed KAM map.")
    # ============================
    # Output directory
    # ============================
    out_dir = "outputs/grain_segmentation_full"
    os.makedirs(out_dir, exist_ok=True)

    # ============================
    # Render IPF
    # ============================
    q_lr_np = q_lr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    render_ipf_image(
        q_lr_np,
        sym_class,
        out_png=f"{out_dir}/ipf_lr.png",
        ref_dir="Z",
        include_key=True,
        overwrite=True,
    )

    ipf_rgb = render_ipf_rgb(
        q_lr_np,
        sym_class,
        ref_dir="Z",
    )

    # ---------- Compute grain boundary mask ----------
    # gb_mask = compute_gb_mask(labels_np, connectivity=1, dilate=0)
    gb_mask = compute_thin_gb_mask(labels_np, connectivity=4)
    overlay_gb_on_ipf(ipf_rgb, gb_mask, f"{out_dir}/ipf_with_black_gb.png")

    # ============================
    # Plots
    # ============================
    plot_labels(labels_np, f"{out_dir}/grain_labels.png")
    plot_mask(gb_mask, f"{out_dir}/gb_mask.png", "GB Mask")

    plot_scalar_map(kam_np, f"{out_dir}/kam_map.png", "KAM")

    print(f"Saved grain segmentation results → {out_dir}")

    return labels, gb_mask, q_lr_np, kam


# ---------------------------------------------------------------------------
# JUST added for testing
# ---------------------------------------------------------------------------


def compute_gb_mask(labels_lr: torch.Tensor) -> torch.Tensor:
    lab = labels_lr
    H, W = lab.shape

    gb = torch.zeros((H, W), dtype=torch.bool, device=lab.device)
    gb[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    gb[1:, :] |= lab[1:, :] != lab[:-1, :]

    return gb.float()


@torch.no_grad()
def smooth_gb_and_sdf(
    gb_lr: torch.Tensor,
    scale: int,
    sigma: float = 2.0,
    iters: int = 12,
) -> torch.Tensor:
    device = gb_lr.device
    H, W = gb_lr.shape
    Hh, Wh = H * scale, W * scale

    gb_hr = F.interpolate(
        gb_lr[None, None],
        size=(Hh, Wh),
        mode="nearest",
    )[0, 0]

    kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    dist = gb_hr.clone()
    for _ in range(iters):
        dist = F.conv2d(dist[None, None], kernel, padding=1)[0, 0]

    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    half = size // 2

    coords = torch.arange(size, device=device) - half
    g1 = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)[None, None]

    sdf_hr = F.conv2d(dist[None, None], g2, padding=half)[0, 0]
    sdf_hr = sdf_hr / (sdf_hr.max() + 1e-8)

    return sdf_hr


@torch.no_grad()
def build_sdf(labels_lr: torch.Tensor, scale: int) -> torch.Tensor:
    """
    LR labels -> LR GB mask -> smooth HR SDF.
    """
    gb_lr = compute_gb_mask(labels_lr)  # (H,W) float 0/1
    sdf_hr = smooth_gb_and_sdf(gb_lr, scale)  # (H*scale, W*scale) float in [0,1]
    return sdf_hr


@torch.no_grad()
def build_smooth_hr_labels(
    labels_lr: torch.Tensor,
    scale: int,
    sdf_shift: float = 0.8,
) -> torch.Tensor:
    """
    Build HR labels with *curved* boundaries by:
      1) building a global smooth SDF from LR grain boundaries,
      2) computing its normals,
      3) sampling LR labels at coordinates shifted slightly along
         those normals (sub-pixel in LR space).

    labels_lr : (H,W) long
    scale     : upsampling factor
    sdf_shift : how far to nudge along the SDF normal, in *LR pixels*
                (0.5–1.5 is a good range to try)

    Returns
    -------
    labels_hr : (H*scale, W*scale) long
    """
    device = labels_lr.device
    H, W = labels_lr.shape
    Hh, Wh = H * scale, W * scale

    # 1) Smooth SDF on HR grid from existing helpers
    sdf_hr = build_sdf(labels_lr, scale=scale).to(device)  # (Hh,Wh), float

    # 2) Central-difference gradient on HR (same shape as sdf_hr)
    dy = torch.zeros_like(sdf_hr)
    dx = torch.zeros_like(sdf_hr)

    # central diff interior, forward/backward at borders
    dy[1:-1, :] = 0.5 * (sdf_hr[2:, :] - sdf_hr[:-2, :])
    dy[0, :] = sdf_hr[1, :] - sdf_hr[0, :]
    dy[-1, :] = sdf_hr[-1, :] - sdf_hr[-2, :]

    dx[:, 1:-1] = 0.5 * (sdf_hr[:, 2:] - sdf_hr[:, :-2])
    dx[:, 0] = sdf_hr[:, 1] - sdf_hr[:, 0]
    dx[:, -1] = sdf_hr[:, -1] - sdf_hr[:, -2]

    # Normalize to get unit normals
    norm = torch.sqrt(dx * dx + dy * dy + 1e-12)
    nx = dx / norm
    ny = dy / norm

    # 3) HR sampling grid in *LR* coordinates
    #    (standard pixel-center mapping)
    idx_y = torch.arange(Hh, device=device, dtype=sdf_hr.dtype)
    idx_x = torch.arange(Wh, device=device, dtype=sdf_hr.dtype)
    y_base = (idx_y + 0.5) / scale - 0.5  # (Hh,)
    x_base = (idx_x + 0.5) / scale - 0.5  # (Wh,)

    y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing="ij")  # (Hh,Wh)

    # 4) Shift coords slightly *against* the gradient (into grain interiors)
    y_shift = y_grid - sdf_shift * ny
    x_shift = x_grid - sdf_shift * nx

    # 5) Nearest-neighbor lookup in LR label map
    y_nn = torch.round(y_shift).clamp(0, H - 1).long()
    x_nn = torch.round(x_shift).clamp(0, W - 1).long()

    labels_hr = labels_lr[y_nn, x_nn]  # (Hh,Wh), still integer grain IDs

    return labels_hr.long()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    scale = 4
    labels_lr, gb_mask, q_lr_np, kam = test_grain_segmentation_on_ipf()

    labels_hr = build_smooth_hr_labels(labels_lr, scale=scale, sdf_shift=0.7)
    sdf_hr = build_sdf(labels_lr, scale=scale)

    plt.imshow(labels_lr)
    plt.show()
    plt.imshow(labels_lr[20:35, :])
    plt.show()
    plt.imshow(labels_hr)
    plt.show()
    plt.imshow(labels_hr[65:125, :])
    plt.show()
    plt.imshow(labels_hr[110:125, 70:85])
    plt.show()
    labels_hr[110:125, 70:85]

    # ============================
    # Load quaternion dataset
    # ============================
    dataset_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4"
    dataset_dir = os.path.join(dataset_root, dataset_name)

    ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

    q_lr = ds[0][1].unsqueeze(0)  # (1,4,H,W)
    q_lr = qnorm(q_lr)
    device = q_lr.device
    dtype = q_lr.dtype

    # ============================
    # Load symmetry
    # ============================
    sym_np = np.load("/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy")
    sym_ops = torch.tensor(sym_np, dtype=dtype, device=device)  # (G,4,4)

    # For IPF
    sym_class = SYM.O

    # ===========================================================
    # 1. CPU segmentation (SciPy needs CPU)
    # ===========================================================
    labels_np, nG = segment_grains_graph(q_lr.cpu(), sym_ops.cpu(), thr_deg=3.0)
    print(f"Initial grains: {nG}")

    # ===========================================================
    # 2. Convert to GPU for cleanup + KAM
    # ===========================================================
    labels = torch.from_numpy(labels_np).long().to(q_lr.device)

    # ===========================================================
    # 3. CUDA Cleanup (fast!)
    # ===========================================================

    labels = cleanup_small_grains_cuda(
        labels,  # torch (H,W)
        q_lr,  # torch (1,4,H,W), CUDA
        sym_ops,  # torch (G,4,4), CUDA
        min_pixels=3,
        max_iter=1,
    )

    plt.imshow(labels_hr)
    plt.show()

    plt.imshow(labels)
    plt.show()

    # import matplotlib.pyplot as plt
    # import numpy as np

    # ipf_hr = render_ipf_rgb(
    #     q_lr.squeeze(0).permute(1, 2, 0).cpu().numpy(), sym_class, ref_dir="Z"
    # )  # (H,W,3) uint8
    # overlay = ipf_hr.copy().astype(float)

    # # paint mismatches red
    # overlay[gb_diff.cpu().numpy()] = [255, 0, 0]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(overlay.astype(np.uint8))
    # plt.title("Boundary Mismatch Overlay: smooth vs segmented HR")
    # plt.axis("off")
    # plt.show()


# plt.imshow(abs(labels_hr-labels))
# # labels = curvature_smooth_labels_cuda(labels, iterations=20, lam=0.2)
# # labels_np = labels_smooth.cpu().numpy()

# # # Smooth grain labels (Dream3D style)
# # labels = smooth_labels_cuda(labels, radius=1, iterations=1)

# # Convert for plotting
# # labels_np = labels_smoothed.cpu().numpy()

# # back to CPU numpy ONLY for plotting
# labels_np = labels.cpu().numpy()

# nG2 = np.unique(labels_np).size
# print(f"Grains after cleanup: {nG2}")

# # ============================
# # KAM computation
# # ============================
# kam = compute_kam_cuda(q_lr, labels, sym_ops, radius=2, connectivity=8)
# kam_np = kam.cpu().numpy()
# print("Computed KAM map.")
# # ============================
# # Output directory
# # ============================
# out_dir = "outputs/grain_segmentation_full"
# os.makedirs(out_dir, exist_ok=True)

# # ============================
# # Render IPF
# # ============================
# q_lr_np = q_lr.squeeze(0).permute(1, 2, 0).cpu().numpy()
# render_ipf_image(
#     q_lr_np,
#     sym_class,
#     out_png=f"{out_dir}/ipf_lr.png",
#     ref_dir="Z",
#     include_key=True,
#     overwrite=True,
# )

# ipf_rgb = render_ipf_rgb(
#     q_lr_np,
#     sym_class,
#     ref_dir="Z",
# )

# # ---------- Compute grain boundary mask ----------
# # gb_mask = compute_gb_mask(labels_np, connectivity=1, dilate=0)
# gb_mask = compute_thin_gb_mask(labels_np, connectivity=4)
# overlay_gb_on_ipf(ipf_rgb, gb_mask, f"{out_dir}/ipf_with_black_gb.png")

# # ============================
# # Plots
# # ============================
# plot_labels(labels_np, f"{out_dir}/grain_labels.png")
# plot_mask(gb_mask, f"{out_dir}/gb_mask.png", "GB Mask")

# plot_scalar_map(kam_np, f"{out_dir}/kam_map.png", "KAM")

# print(f"Saved grain segmentation results → {out_dir}")


## BASE ONE

# # -*- coding:utf-8 -*-
# """
# File:        test_grain_segmentation_on_ipf.py
# Author:      Warren Zamudio
# Contact:     wzamudio@ucsb.edu
# Description:
#     Robust MTEX/Dream3D-style grain segmentation test
#     from a quaternion field (4,H,W) using symmetry-aware
#     misorientation + graph connectivity.
# """

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from orix.quaternion import symmetry as SYM

# from training.data_loading import QuaternionDataset
# from visualization.ipf_render import render_ipf_image


# # ============================================================
# # Quaternion utilities
# # ============================================================


# def qnorm(q):
#     return q / (q.norm(dim=1, keepdim=True).clamp_min(1e-12))


# def quat_mul(q1, q2):
#     w1, x1, y1, z1 = q1.unbind(-1)
#     w2, x2, y2, z2 = q2.unbind(-1)
#     return torch.stack(
#         [
#             w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
#         ],
#         dim=-1,
#     )


# def misorientation_angle_sym(q1, q2, sym_ops):
#     """
#     q1, q2: (...,4)
#     sym_ops: (G,4,4) torch
#     Returns angle (...,)
#     """
#     # (G,...,4) = (G,4,4)@(4,...)
#     q2var = torch.einsum("gij,...j->...gi", sym_ops, q2)  # (...,G,4)
#     dots = torch.sum(q1.unsqueeze(-2) * q2var, dim=-1)  # (...,G)
#     dots = dots.abs().clamp(-1, 1)
#     best = dots.max(dim=-1).values
#     return 2.0 * torch.acos(best)


# # ============================================================
# # Grain segmentation
# # ============================================================


# def neighbor_misorientation(q, sym_ops):
#     """
#     q: (1,4,H,W)
#     Returns:
#       mis_x: (H,W-1)
#       mis_y: (H-1,W)
#     """
#     _, _, H, W = q.shape
#     qHW = q.squeeze(0).permute(1, 2, 0)  # (H,W,4)

#     q0 = qHW[:, :-1]  # left
#     q1 = qHW[:, 1:]  # right
#     mis_x = misorientation_angle_sym(q0, q1, sym_ops)  # (H,W-1)

#     q0 = qHW[:-1, :]
#     q1 = qHW[1:, :]
#     mis_y = misorientation_angle_sym(q0, q1, sym_ops)  # (H-1,W)

#     return mis_x, mis_y


# def segment_grains_graph(q, sym_ops, thr_deg=5.0):
#     """
#     q: (1,4,H,W)
#     Returns:
#         grain_labels: (H,W) np int32
#     """
#     _, _, H, W = q.shape
#     mis_x, mis_y = neighbor_misorientation(q, sym_ops)
#     thr = np.deg2rad(thr_deg)

#     mis_x_np = mis_x.cpu().numpy()
#     mis_y_np = mis_y.cpu().numpy()

#     # Build graph edges
#     rows, cols = [], []

#     def idx(i, j):
#         return i * W + j

#     # horizontal
#     mask_x = mis_x_np <= thr
#     ii, jj = np.where(mask_x)
#     p1 = idx(ii, jj)
#     p2 = idx(ii, jj + 1)
#     rows += list(p1) + list(p2)
#     cols += list(p2) + list(p1)

#     # vertical
#     mask_y = mis_y_np <= thr
#     ii, jj = np.where(mask_y)
#     p1 = idx(ii, jj)
#     p2 = idx(ii + 1, jj)
#     rows += list(p1) + list(p2)
#     cols += list(p2) + list(p1)

#     nPix = H * W
#     data = np.ones(len(rows), np.uint8)

#     from scipy.sparse import csr_matrix
#     from scipy.sparse.csgraph import connected_components

#     A = csr_matrix((data, (rows, cols)), shape=(nPix, nPix))
#     nGrains, labels = connected_components(A, directed=False)
#     return labels.reshape(H, W).astype(np.int32), nGrains


# # ============================================================
# # Visualization helpers
# # ============================================================


# def plot_labels_rgb(labels, out_png):
#     """
#     Convert integer grain labels → RGB visualization.
#     """
#     H, W = labels.shape
#     lab = labels.astype(np.float32)
#     lab = lab / lab.max() if lab.max() > 0 else lab
#     rgb = plt.cm.jet(lab)[..., :3]  # Jet as quick coloring

#     plt.figure(figsize=(6, 6))
#     plt.imshow(rgb)
#     plt.title("Grain Labels")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     plt.close()


# def plot_mask(mask, out_png, title="GB mask"):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask, cmap="gray")
#     plt.title(title)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     plt.close()


# # ============================================================
# # TEST ENTRY
# # ============================================================


# def test_grain_segmentation_on_ipf():
#     from training.data_loading import QuaternionDataset

#     # ===========================================
#     # Load dataset
#     # ===========================================
#     dataset_root = "/data/warren/materials/EBSD"
#     dataset_name = "IN718_FZ_2D_SR_x4"
#     dataset_dir = os.path.join(dataset_root, dataset_name)

#     ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

#     # LR quaternion field
#     q_lr = ds[0][1].unsqueeze(0)  # (1,4,H,W)
#     q_lr = qnorm(q_lr)

#     device = q_lr.device
#     dtype = q_lr.dtype

#     # ===========================================
#     # Load symmetry operators (4×4)
#     # ===========================================
#     sym_np = np.load("/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy")
#     sym_ops = torch.tensor(sym_np, dtype=dtype, device=device)  # (24,4,4)

#     # For IPF
#     sym_class = SYM.O

#     # ===========================================
#     # Segment grains
#     # ===========================================
#     labels, nG = segment_grains_graph(q_lr, sym_ops, thr_deg=5.0)
#     print(f"Found {nG} grains")

#     H, W = labels.shape

#     # GB mask for visualization
#     mis_x, mis_y = neighbor_misorientation(q_lr, sym_ops)
#     mis_x_np = mis_x.cpu().numpy()
#     mis_y_np = mis_y.cpu().numpy()
#     thr = np.deg2rad(5.0)

#     gb_mask = np.zeros((H, W), bool)
#     gb_mask[:, 1:] |= mis_x_np > thr
#     gb_mask[:, :-1] |= mis_x_np > thr
#     gb_mask[1:, :] |= mis_y_np > thr
#     gb_mask[:-1, :] |= mis_y_np > thr

#     # ===========================================
#     # Output directory
#     # ===========================================
#     out_dir = "outputs/grain_seg_test"
#     os.makedirs(out_dir, exist_ok=True)

#     # ===========================================
#     # Render LR IPF
#     # ===========================================
#     q_lr_np = q_lr.squeeze(0).permute(1, 2, 0).cpu().numpy()

#     render_ipf_image(
#         q_lr_np,
#         sym_class,
#         out_png=f"{out_dir}/ipf_lr.png",
#         ref_dir="Z",
#         include_key=True,
#         overwrite=True,
#     )

#     # ===========================================
#     # Grain label visualization
#     # ===========================================
#     plot_labels_rgb(labels, f"{out_dir}/grain_labels.png")
#     plot_mask(gb_mask, f"{out_dir}/gb_mask.png", "GB mask")

#     print(f"Saved results to: {out_dir}")
#     return labels, q_lr_np


# # ============================================================
# # Entry point
# # ============================================================

# if __name__ == "__main__":
#     test_grain_segmentation_on_ipf()