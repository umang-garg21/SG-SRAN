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


# ============================================================
# Reduce to Fundamental Zone (FZ)
# ============================================================
def reduce_to_fz_min_angle_torch(
    q: torch.Tensor,
    sym: object,
    normalize: bool = True,
    hemisphere: bool = True,
    return_op_map: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce quaternions to the Fundamental Zone (FZ) of a given symmetry group
    using minimum misorientation angle criterion (PyTorch version).

    Parameters
    ----------
    q : torch.Tensor
        Quaternion tensor of shape (B,4,H,W) or (B,H,W,4).
    sym : object or str
        Symmetry group (or name).
    normalize : bool
        Whether to normalize quaternions.
    hemisphere : bool
        Whether to enforce hemisphere convention (w>=0).
    return_op_map : bool
        Return operator index map.
    eps : float
        Numerical floor.

    Returns
    -------
    q_fz : torch.Tensor
    op_map : torch.Tensor (if requested)
    """
    # Handle layout
    orig_first = q.shape[1] == 4
    if orig_first:
        q_spatial = q.permute(0, 2, 3, 1)  # (B,H,W,4)
    else:
        q_spatial = q

    B, H, W, _ = q_spatial.shape

    # Normalize + hemisphere
    if normalize:
        q_spatial = F.normalize(q_spatial, p=2, dim=-1, eps=eps)

    if hemisphere:
        # If the scalar component (w) is negative, flip the entire quaternion
        q_spatial[..., 0] = torch.sign(q_spatial[..., 0]) * torch.abs(q_spatial[..., 0])

    # Resolve symmetry
    if isinstance(sym, str):
        sym = resolve_symmetry(sym)

    if isinstance(sym.data, np.ndarray):
        sym_data = torch.tensor(sym.data, dtype=torch.float32, device=q.device)
    else:
        sym_data = sym.data.to(torch.float32).to(q.device)

    region = OrientationRegion.from_symmetry(sym)

    flat = q_spatial.reshape(-1, 4)
    N = flat.shape[0]
    M = sym_data.shape[0]
    # Convert flat tensor to numpy for ORIX
    flat_np = flat.detach().cpu().numpy()
    region = OrientationRegion.from_symmetry(sym)

    # Early exit if already in FZ
    if (Orientation(flat_np, sym) < region).all():
        q_out = q_spatial if not orig_first else q
        if return_op_map:
            return q_out, torch.zeros(
                q_spatial.shape[:-1], dtype=torch.int32, device=q.device
            )
        return q_out

    # 2. Apply symmetry ops
    ops = sym_data
    cand = quat_left_multiply_torch(q_spatial, ops, eps=eps)

    # Convert candidate to numpy for FZ check
    cand_flat = cand.reshape(M * N, 4)
    cand_flat_np = cand_flat.detach().cpu().numpy()

    inside_mask = (Orientation(cand_flat_np, sym) < region).reshape(M, N)

    # # Flatten
    # flat = q_spatial.reshape(-1, 4)
    # N = flat.shape[0]
    # M = sym_data.shape[0]

    # # Check if already in FZ (fast path)
    # if (Orientation(flat.cpu(), sym) < region).all():
    #     q_fz = q_spatial if not orig_first else q
    #     if return_op_map:
    #         op_map = torch.zeros((B, H, W), dtype=torch.int32, device=q.device)
    #         return q_fz, op_map
    #     return q_fz

    # # Apply symmetry ops
    # cand = quat_left_multiply_torch(flat, sym_data, eps=eps)  # (M,4,N)
    # cand = cand.permute(0, 2, 1).contiguous().reshape(M * N, 4)  # (M*N,4)

    # inside_mask = (Orientation(cand, sym) < region).reshape(M, N)

    cand = cand.reshape(M, N, 4)
    w_vals = cand[..., 0]
    w_vals[~inside_mask] = -float("inf")

    best_idx = torch.argmax(w_vals, dim=0)  # (N,)
    best_idx_exp = best_idx.unsqueeze(-1).expand(-1, 4)
    q_fz = cand.permute(1, 0, 2).gather(1, best_idx_exp.unsqueeze(1)).squeeze(1)

    q_fz = q_fz.reshape(B, H, W, 4)
    if orig_first:
        q_fz = q_fz.permute(0, 3, 1, 2)  # back to (B,4,H,W)

    if return_op_map:
        op_map = best_idx.reshape(B, H, W)
        return q_fz, op_map
    return q_fz


def reduce_to_fz_safe(q: torch.Tensor, sym) -> torch.Tensor:
    """
    Shape-safe wrapper around reduce_to_fz_min_angle_torch.
    Supports (N,4) and (B,4,H,W) shapes.
    """
    if q.dim() == 2 and q.shape[1] == 4:
        # Reshape to fake image shape for compatibility
        N = q.shape[0]
        q_expanded = q.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1,4,N,1)
        q_fz_exp = reduce_to_fz_min_angle_torch(q_expanded, sym, return_op_map=False)
        q_fz = q_fz_exp.squeeze(0).permute(1, 0).squeeze(-1)  # (N,4)
        return q_fz
    elif q.dim() == 3 and q.shape[0] == 4:
        # handle (4, H, W)
        q_expanded = q.unsqueeze(0)  # (1,4,H,W)
        q_fz_exp = reduce_to_fz_min_angle_torch(q_expanded, sym, return_op_map=False)
        q_fz = q_fz_exp.squeeze(0)  # (4,H,W)
        return q_fz
    else:
        # already in compatible shape
        return reduce_to_fz_min_angle_torch(q, sym, return_op_map=False)


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
    eps = 1e-4  # Small epsilon to avoid numerical issues with acos

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

    # Normalize along the quaternion components (dim=1, which is the 4 components of the quaternion)
    q_pred = F.normalize(q_pred, p=2, dim=1, eps=eps)
    q_target = F.normalize(q_target, p=2, dim=1, eps=eps)

    # Reduce to Fundamental Zone
    q_pred_fz = reduce_to_fz_safe(q_pred, sym)
    q_target_fz = reduce_to_fz_safe(q_target, sym)

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

    q_pred = torch.randn(5, 4, 128, 128)
    q_target = torch.randn(5, 4, 128, 128)

    # Loss with cubic symmetry (Oh)
    print(fz_reduced_rotational_distance_loss(q_pred, q_pred, sym=SYM.Oh))
