import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.quat_utils.Qops_with_QSN import conv2d, Residual_SA
from einops import rearrange 
# ─── requirements ───────────────────────────────────────────────────────────────
# pip install torch e3nn==0.7.4              # e3nn just for rotation utilities
# ────────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from e3nn import o3
from model.qrbsa_1d import QRBSA_1D

# --------‑‑ helper: rotate the XYZ/vector part of the input tensor ‑‑-----------
def apply_rotation(x, R, vec_idx=(0, 1, 2)):
    """
    x:  (B, C, H, W) input; vec_idx are the channels that form a 3‑vector.
    R:  (3, 3) rotation matrix.
    """
    v = x[:, vec_idx, ...]                       # (B, 3, H, W)
    v_rot = torch.einsum('ij,bjhw->bihw', R, v)
    x_r = x.clone()
    x_r[:, vec_idx, ...] = v_rot
    return x_r

# --------‑‑ helper: enforce unique quaternion sign (optional but recommended) ‑
def canonical_quat(q):
    # make scalar part non‑negative so q ≡ −q collapses to one representative
    mask = (q[..., 0:1] < 0).float()
    return q * (1.0 - 2.0 * mask)

# --------‑‑ wrapper itself ‑‑---------------------------------------------------
import scipy.spatial.transform

def get_full_octahedral_group():
    # Generate all 24 rotation matrices of the octahedral group using scipy
    group = scipy.spatial.transform.Rotation.create_group('O')
    return torch.tensor(group.as_matrix(), dtype=torch.float32)  # (24, 3, 3)

ROT_MATS = get_full_octahedral_group()

def make_model(args):
    return so3reynolds_qrbsa_1d(args, rot_mats=ROT_MATS, vec_idx=(0, 1, 2), canonicalise=True)

class so3reynolds_qrbsa_1d(nn.Module):
    def __init__(self, args,
                 rot_mats=ROT_MATS,
                 vec_idx=(0, 1, 2),
                 canonicalise=True):
        super().__init__()
        self.backbone = QRBSA_1D(args)
        self.R = rot_mats
        self.vec_idx = vec_idx
        self.canon = canonicalise

    def forward(self, x):
        outs = []
        for R in self.R.to(x.device):
            x_r = apply_rotation(x, R, self.vec_idx)
            o   = self.backbone(x_r)
            if self.canon:
                o = canonical_quat(o)            # ensures one‑to‑one SO3 rep
            outs.append(o)
        return torch.stack(outs, 0).mean(0)       # same shape as backbone output
