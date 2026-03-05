# -*-coding:utf-8 -*-
"""
File:        loss_functions.py
Created at:  2025/10/18 13:57:00
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""
import torch
import torch.nn.functional as F
import numpy as np
from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion import Orientation, symmetry as SYM
from utils.symmetry_utils import resolve_symmetry


# ============================================================
# Quaternion left multiplication (Torch)
# ============================================================
def quat_left_multiply_torch(
    q_right: torch.Tensor,
    q_left: torch.Tensor,
    eps: float = 1e-12,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Left multiply a set of symmetry operators with a quaternion field (Torch).

    Parameters
    ----------
    q_right : torch.Tensor
        Quaternion tensor of shape (4,*spatial) or (*spatial,4).
    q_left : torch.Tensor
        Operator quaternions of shape (M, 4).
    eps : float
        Numerical floor for normalization.
    normalize : bool
        If True, normalize output quaternions.

    Returns
    -------
    out : torch.Tensor
        Quaternion tensor of shape (M, 4, *spatial).
    """
    # Convert to (*spatial,4)
    if q_right.shape[0] == 4:
        q_right = torch.moveaxis(q_right, 0, -1)

    orig_spatial = q_right.shape[:-1]
    N = int(torch.prod(torch.tensor(orig_spatial), dtype=torch.long))
    M = q_left.shape[0]

    flat = q_right.reshape(N, 4)

    # left operator components
    w0, x0, y0, z0 = [q_left[:, i].unsqueeze(1) for i in range(4)]
    # right quaternion components
    w1, x1, y1, z1 = [flat[:, i].unsqueeze(0) for i in range(4)]

    out = torch.empty((M, N, 4), dtype=torch.float32, device=q_right.device)
    out[:, :, 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    out[:, :, 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    out[:, :, 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    out[:, :, 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    if normalize:
        norms = torch.linalg.norm(out, dim=2, keepdim=True)
        out = out / torch.clamp(norms, min=eps)

    out = out.view(M, *orig_spatial, 4)
    out = torch.moveaxis(out, -1, 1)  # (M,4,*spatial)
    return out


# # ============================================================
# # Reduce to Fundamental Zone (FZ)
# # ============================================================
# def reduce_to_fz_min_angle_torch_fast(
#     q: torch.Tensor,
#     sym: object,
#     normalize: bool = True,
#     hemisphere: bool = True,
#     return_op_map: bool = False,
#     eps: float = 1e-12,
# ):
#     """
#     Fast reduction of quaternions to the Fundamental Zone using max scalar
#     criterion (min misorientation), fully on GPU.

#     Parameters
#     ----------
#     q : torch.Tensor
#         Quaternion tensor of shape (B,4,H,W) or (B,H,W,4).
#     sym : orix symmetry object or str
#         Symmetry group.
#     normalize : bool
#         Whether to normalize quaternions.
#     hemisphere : bool
#         Enforce hemisphere convention (w>=0).
#     return_op_map : bool
#         Return symmetry operator index.
#     eps : float
#         Numerical stability epsilon.

#     Returns
#     -------
#     q_fz : torch.Tensor
#     op_map : torch.Tensor (if requested)
#     """
#     # Handle layout
#     orig_first = q.shape[1] == 4
#     if orig_first:
#         q_spatial = q.permute(0, 2, 3, 1)  # (B,H,W,4)
#     else:
#         q_spatial = q

#     B, H, W, _ = q_spatial.shape

#     # Normalize + hemisphere
#     if normalize:
#         q_spatial = F.normalize(q_spatial, p=2, dim=-1, eps=eps)
#     if hemisphere:
#         mask = q_spatial[:, 0, :, :] < 0
#         q_spatial[mask] = -q_spatial[mask]

#     # Resolve symmetry operators
#     if isinstance(sym, str):
#         sym = resolve_symmetry(sym)
#     sym_ops = torch.as_tensor(
#         sym.data if isinstance(sym.data, np.ndarray) else sym.data.cpu().numpy(),
#         dtype=torch.float32,
#         device=q.device,
#     )  # (M,4)

#     M = sym_ops.shape[0]

#     # Broadcast left multiplication
#     # (M, B*H*W, 4)
#     q_flat = q_spatial.reshape(-1, 4)
#     N = q_flat.shape[0]

#     w0, x0, y0, z0 = [sym_ops[:, i].unsqueeze(1) for i in range(4)]
#     w1, x1, y1, z1 = [q_flat[:, i].unsqueeze(0) for i in range(4)]

#     out = torch.empty((M, N, 4), dtype=torch.float32, device=q.device)
#     out[:, :, 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
#     out[:, :, 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
#     out[:, :, 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
#     out[:, :, 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

#     out = F.normalize(out, p=2, dim=-1, eps=eps)

#     # Pick max scalar part
#     w_vals = out[..., 0]  # (M,N)
#     best_idx = torch.argmax(w_vals, dim=0)  # (N,)

#     # Gather best quaternion
#     best_idx_exp = best_idx.unsqueeze(-1).expand(-1, 4)
#     out_perm = out.permute(1, 0, 2).contiguous()  # (N,M,4)
#     q_fz = out_perm.gather(1, best_idx_exp.unsqueeze(1)).squeeze(1)  # (N,4)
#     q_fz = q_fz.view(B, H, W, 4)

#     if orig_first:
#         q_fz = q_fz.permute(0, 3, 1, 2)


#     if return_op_map:
#         best_idx = best_idx.view(B, H, W)
#         return q_fz, best_idx
#     return q_fz
def reduce_to_fz_min_angle_torch_fast(
    q: torch.Tensor,
    sym: object,
    normalize: bool = True,
    hemisphere: bool = True,
    return_op_map: bool = False,
    eps: float = 1e-12,
):
    """
    Fast symmetry reduction for quaternions of shape (B, 4, H, W),
    using max scalar part criterion (min misorientation angle).
    Fully GPU and batched.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion tensor of shape (B,4,H,W).
    sym : object or str
        Symmetry group or name (e.g. 'Oh').
    normalize : bool
        Normalize quaternions.
    hemisphere : bool
        Flip hemisphere so scalar part >= 0.
    return_op_map : bool
        If True, return index of chosen symmetry operator.
    eps : float
        Numerical epsilon.

    Returns
    -------
    q_fz : torch.Tensor
        Reduced quaternions, same shape as q.
    op_map : torch.Tensor (optional)
        Index of symmetry operator chosen per pixel, shape (B,H,W).
    """
    B, C, H, W = q.shape
    N = B * H * W  # total number of pixels

    # Normalize & hemisphere
    if normalize:
        q = F.normalize(q, p=2, dim=1, eps=eps)

    if hemisphere:
        mask = (q[:, 0, :, :] < 0).unsqueeze(1)  # (B,1,H,W)
        q = torch.where(mask, -q, q)

    # Resolve symmetry operators
    if isinstance(sym, str):
        sym = resolve_symmetry(sym)
    sym_ops = torch.as_tensor(
        sym.data if isinstance(sym.data, np.ndarray) else sym.data.cpu().numpy(),
        dtype=torch.float32,
        device=q.device,
    )  # (M,4)
    # Bunge convention: s⁻¹ ⊗ q  (unit quat inverse = conjugate: negate vector part)
    sym_ops_inv = torch.cat([sym_ops[:, :1], -sym_ops[:, 1:]], dim=-1)

    # Flatten quaternion field to (4, N)
    q_flat = q.view(B, C, -1).reshape(C, N)

    # Left multiply with s⁻¹: output (M,4,N)
    cand = quat_left_multiply_torch(q_flat, sym_ops_inv, eps=eps, normalize=True)

    # Pick symmetry op with max scalar part
    w_vals = cand[:, 0, :]  # (M, N)
    best_idx = torch.argmax(w_vals, dim=0)  # (N,)

    # Gather best quaternions
    M = sym_ops.shape[0]
    cand_perm = cand.permute(2, 0, 1).contiguous()  # (N, M, 4)
    best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)
    q_best = cand_perm.gather(1, best_idx_exp).squeeze(1)  # (N,4)

    # Reshape back to (B,4,H,W)
    q_best = q_best.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    if return_op_map:
        op_map = best_idx.view(B, H, W)
        return q_best, op_map
    return q_best


# def reduce_to_fz_safe(q: torch.Tensor, sym) -> torch.Tensor:
#     """
#     Shape-safe wrapper around reduce_to_fz_min_angle_torch.
#     Supports (N,4) and (B,4,H,W) shapes.
#     """
#     if q.dim() == 2 and q.shape[1] == 4:
#         # Reshape to fake image shape for compatibility
#         N = q.shape[0]
#         q_expanded = q.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1,4,N,1)
#         q_fz_exp = reduce_to_fz_min_angle_torch(q_expanded, sym, return_op_map=False)
#         q_fz = q_fz_exp.squeeze(0).permute(1, 0).squeeze(-1)  # (N,4)
#         return q_fz
#     elif q.dim() == 3 and q.shape[0] == 4:
#         # handle (4, H, W)
#         q_expanded = q.unsqueeze(0)  # (1,4,H,W)
#         q_fz_exp = reduce_to_fz_min_angle_torch(q_expanded, sym, return_op_map=False)
#         q_fz = q_fz_exp.squeeze(0)  # (4,H,W)
#         return q_fz
#     else:
#         # already in compatible shape
#         return reduce_to_fz_min_angle_torch(q, sym, return_op_map=False)


def safe_normalize(q):
    norm = torch.norm(q, p=2, dim=1, keepdim=True)
    # Avoid division by zero by clamping the norm
    norm = torch.max(norm, torch.ones_like(norm) * 1e-8)
    return q / norm


def rotational_distance_loss(q_pred, q_target, eps: float = 1e-12):
    # supports (N,4) or (B,4,H,W)
    if q_pred.dim() > 2:
        # B, C, H, W = q_pred.shape
        qp = q_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        qt = q_target.permute(0, 2, 3, 1).reshape(-1, 4)
    else:
        qp, qt = q_pred, q_target

    qp = F.normalize(qp, p=2, dim=1, eps=eps)
    qt = F.normalize(qt, p=2, dim=1, eps=eps)

    w1, x1, y1, z1 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
    w2, x2, y2, z2 = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]

    # r = qt ⊗ conj(qp)
    rw = w2 * w1 + x2 * (-x1) + y2 * (-y1) + z2 * (-z1)
    rx = w2 * (-x1) + x2 * w1 + y2 * (-z1) + z2 * (y1)
    ry = w2 * (-y1) + x2 * (z1) + y2 * w1 + z2 * (-x1)
    rz = w2 * (-z1) + x2 * (-y1) + y2 * (x1) + z2 * w1

    rw = rw.abs()  # hemisphere
    v_norm = torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
    angle = 2.0 * torch.atan2(v_norm, torch.clamp(rw, min=eps))
    return angle.mean()


def orientation_gradient_loss(
    q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Orientation gradient loss (fixed): L1 loss between gradient magnitudes of quaternion fields.
    Pads gradient outputs to match original size.
    """
    # This function is kept for compatibility but not used directly in the
    # build_loss dispatch when using the optimized module below.
    # Finite difference filters (constructed per-call previously) are
    # intentionally cheap here, but for heavy training loops prefer the
    # RotationalDistanceOrientationLoss module which caches kernels as buffers.
    kernel_x = torch.tensor([[[[-1, 1]]]], dtype=q_pred.dtype, device=q_pred.device)
    kernel_y = torch.tensor([[[[-1], [1]]]], dtype=q_pred.dtype, device=q_pred.device)

    def grad_mag(q):
        gx = F.conv2d(q, kernel_x.expand(q.size(1), 1, 1, 2), groups=q.size(1))
        gy = F.conv2d(q, kernel_y.expand(q.size(1), 1, 2, 1), groups=q.size(1))

        # pad back to original shape (right and bottom edges)
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")  # (left, right, top, bottom)
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")

        gmag = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
        return gmag

    grad_pred = grad_mag(q_pred)
    grad_target = grad_mag(q_target)

    return F.l1_loss(grad_pred, grad_target)


class RotationalDistanceOrientationLoss(torch.nn.Module):
    """
    Combined rotational distance + orientation-gradient loss as a Module.
    This caches the finite-difference kernels as buffers to avoid allocating
    small tensors on the device each batch (which is slow and can force
    synchronizations).
    """

    def __init__(self, weight: float = 0.05):
        super().__init__()
        # store small kernels as buffers (float32); they will be moved to the
        # correct device when this Module is .to(device)
        self.register_buffer("kernel_x", torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float32))
        self.register_buffer("kernel_y", torch.tensor([[[[-1.0], [1.0]]]], dtype=torch.float32))
        self.weight = weight

    def grad_mag(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # kernels are registered buffers and reside on the same device/dtype as
        # this module when it's moved via .to(device)
        kx = self.kernel_x
        ky = self.kernel_y
        gx = F.conv2d(q, kx.expand(q.size(1), 1, 1, 2).to(q.dtype), groups=q.size(1))
        gy = F.conv2d(q, ky.expand(q.size(1), 1, 2, 1).to(q.dtype), groups=q.size(1))

        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")

        gmag = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
        return gmag

    def rotational_distance(self, q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # reuse the existing function for rotational distance
        return rotational_distance_loss(q_pred, q_target, eps=eps)

    def forward(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        rd = self.rotational_distance(q_pred, q_target)
        grad_pred = self.grad_mag(q_pred)
        grad_target = self.grad_mag(q_target)
        og = F.l1_loss(grad_pred, grad_target)
        return rd + self.weight * og


class EdgeWeightedRotationalDistanceOrientationLoss(torch.nn.Module):
    """
    Combined rotational distance + orientation-gradient loss as a Module.
    This caches the finite-difference kernels as buffers to avoid allocating
    small tensors on the device each batch (which is slow and can force
    synchronizations).
    """

    def __init__(self, group_quats, edge_factor=20.0, grad_loss_weight=0.05, entropy_factor=0.2):
        super().__init__()
        # Register small kernels for gradient calculation
        self.register_buffer("kernel_x", torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float32))
        self.register_buffer("kernel_y", torch.tensor([[[[-1.0], [1.0]]]], dtype=torch.float32))
        
        self.entropy_factor = entropy_factor
        # Register Symmetry Group (24, 4)
        # Ensure input is a tensor
        group_quats = np.load(group_quats) if isinstance(group_quats, str) else group_quats
        if not torch.is_tensor(group_quats):
            group_quats = torch.tensor(group_quats, dtype=torch.float32)
        self.register_buffer("group_quats", group_quats)
        
        self.edge_factor = edge_factor
        self.grad_loss_weight = grad_loss_weight

    def grad_mag(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # kernels are registered buffers and reside on the same device/dtype as
        # this module when it's moved via .to(device)
        kx = self.kernel_x
        ky = self.kernel_y
        gx = F.conv2d(q, kx.expand(q.size(1), 1, 1, 2).to(q.dtype), groups=q.size(1))
        gy = F.conv2d(q, ky.expand(q.size(1), 1, 2, 1).to(q.dtype), groups=q.size(1))

        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")

        gmag = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
        return gmag

    def rotational_distance(self, q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # reuse the existing function for rotational distance
        return rotational_distance_loss(q_pred, q_target, eps=eps)

    def forward(self, q_pred: torch.Tensor, q_target: torch.Tensor, selection_weights=None) -> torch.Tensor:
        rd = self.rotational_distance(q_pred, q_target)
        entropy_factor = self.entropy_factor if hasattr(self, 'entropy_factor') else 0.0
        grad_pred = self.grad_mag(q_pred)
        grad_target = self.grad_mag(q_target)

        # Edge weighting based on symmetry group
        # Sum channels to get a single intensity map
        target_edges = grad_target.sum(dim=1) # (B, H, W)
            
        # Normalize to [0, 1] using tanh to squash high gradients
        edge_prob = torch.tanh(target_edges * 5.0)
        # Create Weight Map:
        # Interior (prob~0) -> 1.0
        # Boundary (prob~1) -> edge_factor (e.g., 20.0)
        weights = 1.0 + (self.edge_factor - 1.0) * edge_prob

        weighted_pixel_loss= (rd * weights.unsqueeze(1)).mean()  # (B,1,H,W)
        rd = weighted_pixel_loss.mean()
        og = F.l1_loss(grad_pred, grad_target)

        total_loss= weighted_pixel_loss + self.grad_loss_weight * og
        if selection_weights is not None:
            entropy= -torch.sum(selection_weights * torch.log(selection_weights + 1e-12), dim=1)  # (B,H,W)
            entropy_loss = entropy.mean()

            total_loss += entropy_factor*entropy_loss

        return total_loss


class SymmetryAwareRotationalLoss(torch.nn.Module):
    def __init__(self, sym_group_path, weight_gradient=0.05):
        super().__init__()
        # Load symmetries: (24, 4)
        syms = torch.tensor(np.load(sym_group_path), dtype=torch.float32)
        self.register_buffer("syms", syms)
        self.weight_gradient = weight_gradient
        
        # Gradient kernels
        self.register_buffer("kernel_x", torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float32))
        self.register_buffer("kernel_y", torch.tensor([[[[-1.0], [1.0]]]], dtype=torch.float32))

    def grad_mag(self, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Increased eps to 1e-6 for safety
        gx = F.conv2d(q, self.kernel_x.expand(q.size(1), 1, 1, 2).to(q.dtype), groups=q.size(1))
        gy = F.conv2d(q, self.kernel_y.expand(q.size(1), 1, 2, 1).to(q.dtype), groups=q.size(1))
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
        return torch.sqrt(gx.pow(2) + gy.pow(2) + eps)

    def forward(self, q_pred, q_target, eps=1e-6): # <--- CHANGED DEFAULT TO 1e-6
        """
        q_pred: (B, 4, H, W)
        q_target: (B, 4, H, W)
        """
        # 1. Strict Flattening with Contiguous Memory
        qp = q_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        qt = q_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        
        # Safe normalization to prevent 0-division gradient explosion
        qp = F.normalize(qp, dim=1, eps=eps)
        qt = F.normalize(qt, dim=1, eps=eps)

        # 2. Compute Relative Rotation: Q_rel = q_target * conj(q_pred)
        w1, x1, y1, z1 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
        w2, x2, y2, z2 = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]

        rw = w2 * w1 - x2 * (-x1) - y2 * (-y1) - z2 * (-z1)
        rx = w2 * (-x1) + x2 * w1 + y2 * (-z1) - z2 * (-y1)
        ry = w2 * (-y1) - x2 * (-z1) + y2 * w1 + z2 * (-x1)
        rz = w2 * (-z1) + x2 * (-y1) - y2 * (-x1) + z2 * w1
        
        # 3. Apply Symmetries
        s_w, s_x, s_y, s_z = self.syms[:, 0], self.syms[:, 1], self.syms[:, 2], self.syms[:, 3]
        
        # Reshape for broadcasting
        rw = rw.reshape(-1, 1)
        rx = rx.reshape(-1, 1)
        ry = ry.reshape(-1, 1)
        rz = rz.reshape(-1, 1)

        s_w = s_w.reshape(1, -1)
        s_x = s_x.reshape(1, -1)
        s_y = s_y.reshape(1, -1)
        s_z = s_z.reshape(1, -1)
        
        # Real part of (S * Q_rel)
        w_dist = (rw * s_w - rx * s_x - ry * s_y - rz * s_z)
        
        # 4. Find Minimum Rotation
        # We want to maximize the real part (closest to Identity)
        max_w_val, _ = torch.max(torch.abs(w_dist), dim=1) 

        # --- CRITICAL FIX: Safe Clamping for acos ---
        # Ensure values stay strictly within (-1, 1)
        # 1e-6 leaves enough margin for float32 gradients to remain finite.
        max_w_val = torch.clamp(max_w_val, min=-(1.0 - eps), max=(1.0 - eps))
        
        angle = 2.0 * torch.acos(max_w_val)
        loss_rot = angle.mean()

        # 5. Gradient Loss
        grad_pred = self.grad_mag(q_pred, eps=eps)
        grad_target = self.grad_mag(q_target, eps=eps)
        loss_grad = F.l1_loss(grad_pred, grad_target)

        return loss_rot + self.weight_gradient * loss_grad


class SimpleSymmetryRotationalLoss(torch.nn.Module):
    """
    Simple symmetry-aware rotational loss.

    - No gradient kernels
    - No edge weighting
    - Just min-over-symmetry geodesic angle
    """

    def __init__(self, sym_group_path: str):
        super().__init__()
        syms = torch.tensor(np.load(sym_group_path), dtype=torch.float32)
        self.register_buffer("syms", syms)

    def forward(self, q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Supports (N,4) and (B,4,H,W)
        if q_pred.dim() > 2:
            qp = q_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
            qt = q_target.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        else:
            qp = q_pred.contiguous().view(-1, 4)
            qt = q_target.contiguous().view(-1, 4)

        qp = F.normalize(qp, dim=1, eps=eps)
        qt = F.normalize(qt, dim=1, eps=eps)

        # Relative quaternion q_rel = qt * conj(qp)
        w1, x1, y1, z1 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
        w2, x2, y2, z2 = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]

        rw = w2 * w1 - x2 * (-x1) - y2 * (-y1) - z2 * (-z1)
        rx = w2 * (-x1) + x2 * w1 + y2 * (-z1) - z2 * (-y1)
        ry = w2 * (-y1) - x2 * (-z1) + y2 * w1 + z2 * (-x1)
        rz = w2 * (-z1) + x2 * (-y1) - y2 * (-x1) + z2 * w1

        s_w, s_x, s_y, s_z = self.syms[:, 0], self.syms[:, 1], self.syms[:, 2], self.syms[:, 3]

        rw = rw.reshape(-1, 1)
        rx = rx.reshape(-1, 1)
        ry = ry.reshape(-1, 1)
        rz = rz.reshape(-1, 1)

        s_w = s_w.reshape(1, -1)
        s_x = s_x.reshape(1, -1)
        s_y = s_y.reshape(1, -1)
        s_z = s_z.reshape(1, -1)

        w_dist = (rw * s_w - rx * s_x - ry * s_y - rz * s_z)
        max_w_val, _ = torch.max(torch.abs(w_dist), dim=1)
        max_w_val = torch.clamp(max_w_val, min=-(1.0 - eps), max=(1.0 - eps))

        angle = 2.0 * torch.acos(max_w_val)
        return angle.mean()
    
def rotational_distance_orientation_loss(
    q_pred: torch.Tensor, q_target: torch.Tensor
) -> torch.Tensor:
    """
    Orientation gradient loss (fixed): L1 loss between gradient magnitudes of quaternion fields.
    Pads gradient outputs to match original size.
    """
    # Finite difference filters
    return rotational_distance_loss(
        q_pred=q_pred, q_target=q_target
    ) + 0.05 * orientation_gradient_loss(q_pred=q_pred, q_target=q_target)


# def rotational_distance_loss(q_pred, q_target):
#     """
#     Compute the rotational distance between two quaternions.

#     Parameters:
#     -----------
#     q_pred : torch.Tensor
#         Predicted quaternion tensor of shape (N, 4), where N is the batch size.
#     q_target : torch.Tensor
#         Target quaternion tensor of shape (N, 4), where N is the batch size.

#     Returns:
#     --------
#     torch.Tensor
#         The mean rotational distance loss.
#     """
#     eps = 1e-4  # Small epsilon to avoid numerical issues with acos

#     # Normalize the quaternions to ensure they are unit quaternions
#     q_pred = safe_normalize(q_pred)
#     q_target = safe_normalize(q_target)

#     # Compute the dot product between the predicted and target quaternions
#     dot_product = torch.sum(q_pred * q_target, dim=1)

#     # Clamp dot product to the valid range for acos [-1, 1], with an added epsilon for stability
#     dot_product = torch.clamp(dot_product, min=-1.0 + eps, max=1.0 - eps)

#     # Compute the angle (rotational distance)
#     rotational_distance = 2 * torch.acos(torch.abs(dot_product))

#     # Mean of the rotational distance
#     return rotational_distance.mean()


def fz_reduced_rotational_distance_loss(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    sym: str | object,
    eps: float = 1e-9,
) -> torch.Tensor:

    q_pred = safe_normalize(q_pred)
    q_target = safe_normalize(q_target)

    # Use fast symmetry reduction
    q_pred_fz = reduce_to_fz_min_angle_torch_fast(q_pred, sym)
    q_target_fz = reduce_to_fz_min_angle_torch_fast(q_target, sym)

    return rotational_distance_loss(q_pred_fz, q_target_fz)


# Example usage within a model
def build_loss(cfg):
    # Get the loss type and symmetry from the configuration
    loss_type = cfg.get("loss", "rotational_distance").lower()
    # "symmetry_group" is the canonical config key; fall back to "symmetry" then "O"
    symmetry = cfg.get("symmetry_group", cfg.get("symmetry", "O"))

    # Resolve symmetry if it's a string (e.g., 'Oh') or pass it as an object
    resolved_symmetry = resolve_symmetry(symmetry)

    if loss_type == "fz_reduced_rotational_distance":
        return lambda q_pred, q_target: fz_reduced_rotational_distance_loss(
            q_pred, q_target, sym=resolved_symmetry
        )
    elif loss_type == "rotational_distance":
        return rotational_distance_loss
    elif loss_type == "rotational_distance_orientation":
        # Return a Module instance that caches kernels as buffers. This
        # reduces CPU/GPU sync overhead caused by allocating small tensors
        # on-device each training step.
        return RotationalDistanceOrientationLoss()
    elif loss_type == "edge_weighted_rotational_distance_orientation":
        return EdgeWeightedRotationalDistanceOrientationLoss(
            group_quats=cfg.get("symmetry_group_path", "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"),
            edge_factor=cfg.get("edge_factor", 20.0),
            grad_loss_weight=cfg.get("weight_gradient", 0.05),
            entropy_factor=cfg.get("entropy_factor", 0.2)
        )
    elif loss_type == "symmetry_aware_rotational":
        return SymmetryAwareRotationalLoss(
            sym_group_path=cfg.get("symmetry_group_path", "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"),
            weight_gradient=cfg.get("weight_gradient", 0.05)
        )
    elif loss_type == "simple_symmetry_rotational":
        return SimpleSymmetryRotationalLoss(
            sym_group_path=cfg.get("symmetry_group_path", "symmetry_groups/O_group.npy")
        )
    elif loss_type == "l1":
        return torch.nn.L1Loss()
    elif loss_type == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":

    from orix.quaternion import symmetry as SYM

    q_pred = torch.randn(5, 4, 128, 128)
    q_target = torch.randn(5, 4, 128, 128)

    # Loss with cubic symmetry (Oh)
    print(fz_reduced_rotational_distance_loss(q_pred, q_pred, sym=SYM.Oh))
