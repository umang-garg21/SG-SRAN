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
        mask = q[:, 0] < 0
        q[mask] = -q[mask]

    # Resolve symmetry operators
    if isinstance(sym, str):
        sym = resolve_symmetry(sym)
    sym_ops = torch.as_tensor(
        sym.data if isinstance(sym.data, np.ndarray) else sym.data.cpu().numpy(),
        dtype=torch.float32,
        device=q.device,
    )  # (M,4)

    # Flatten quaternion field to (4, N)
    q_flat = q.view(B, C, -1).reshape(C, N)

    # Left multiply: output (M,4,N)
    cand = quat_left_multiply_torch(q_flat, sym_ops, eps=eps, normalize=True)

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


def rotational_distance_loss(q_pred, q_target):
    """
    Compute the rotational distance between two quaternions.

    Parameters:
    -----------
    q_pred : torch.Tensor
        Predicted quaternion tensor of shape (N, 4), where N is the batch size.
    q_target : torch.Tensor
        Target quaternion tensor of shape (N, 4), where N is the batch size.

    Returns:
    --------
    torch.Tensor
        The mean rotational distance loss.
    """
    eps = 1e-9  # Small epsilon to avoid numerical issues with acos

    # Normalize the quaternions to ensure they are unit quaternions
    q_pred = safe_normalize(q_pred)
    q_target = safe_normalize(q_target)

    # Compute the dot product between the predicted and target quaternions
    dot_product = torch.sum(q_pred * q_target, dim=1)

    # Clamp dot product to the valid range for acos [-1, 1], with an added epsilon for stability
    dot_product = torch.clamp(dot_product, min=-1.0 + eps, max=1.0 - eps)

    # Compute the angle (rotational distance)
    rotational_distance = 2 * torch.acos(torch.abs(dot_product))

    # Mean of the rotational distance
    return rotational_distance.mean()


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
    symmetry = cfg.get("symmetry", "Oh")  # Default to 'Oh' symmetry if not provided

    # Resolve symmetry if it's a string (e.g., 'Oh') or pass it as an object
    resolved_symmetry = resolve_symmetry(symmetry)

    if loss_type == "fz_reduced_rotational_distance":
        return lambda q_pred, q_target: fz_reduced_rotational_distance_loss(
            q_pred, q_target, sym=resolved_symmetry
        )
    elif loss_type == "rotational_distance":
        return rotational_distance_loss
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

    q_pred = torch.randn(5, 4, 1, 1)
    q_target = torch.randn(5, 4, 1, 1)

    # Loss with cubic symmetry (Oh)
    print(rotational_distance_loss(q_pred, q_pred))
