import sys
import torch
import math
import importlib.util
from pathlib import Path

# Load the target module by file path since the folder name starts with a digit
repo_root = Path(__file__).resolve().parents[1]
mod_path = repo_root / 'e3nn_experimentation' / '4x4_superresolution_transformers' / 'equiformer_ebsdSr.py'
spec = importlib.util.spec_from_file_location('equiformer_ebsdSr', str(mod_path))
eq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eq)

def quat_angle_diff(q1, q2):
    # q: (N,4) wxyz; compute minimal rotation angle between quaternions
    q1 = q1 / q1.norm(dim=1, keepdim=True)
    q2 = q2 / q2.norm(dim=1, keepdim=True)
    # relative
    w1, x1, y1, z1 = q1.unbind(dim=1)
    w2, x2, y2, z2 = q2.unbind(dim=1)
    # quaternion multiply q_rel = q2 * conj(q1)
    w = w2*w1 + x2*x1 + y2*y1 + z2*z1
    w = torch.clamp(torch.abs(w), -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    return angle


def run_ab_test(device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    physics = eq.FCCPhysics(device)
    encoder = eq.FCCEncoder(physics)
    model = eq.CrystalTransformerSR(physics, scale_factor=4, depth=1).to(device)

    # Dummy low-res image
    H, W = 8, 8
    N = H * W
    q_low = torch.randn(N, 4, device=device)
    q_low = q_low / q_low.norm(dim=1, keepdim=True)

    # Encode
    f0, f1, f4, f6 = encoder(q_low, img_shape=(H, W))

    # Prepare image-shaped coeffs
    f4_img = f4
    f6_img = f6
    f0_img = f0

    # Baseline: decoder on original low-res coeffs (flattened)
    f4_flat = f4_img.permute(0,2,3,1).reshape(-1, f4_img.size(1))
    f6_flat = f6_img.permute(0,2,3,1).reshape(-1, f6_img.size(1))
    q_baseline = model.decoder(f4_flat, f6_flat)

    # 1) Default embedding
    model.embedding.debug_center_only = False
    emb_default = model.embedding(torch.cat([f4_img, f6_img], dim=1), boundary_map=f0_img)
    emb_temp = emb_default.permute(0,2,3,1)
    coeffs_def = model.proj(emb_temp)
    coeffs_flat_def = coeffs_def.permute(0,3,1,2).reshape(-1, model.coeff_dim).clone()
    dis_all = (~model.mask_all).to(coeffs_flat_def.device)
    if dis_all.any():
        coeffs_flat_def[:, dis_all] = 0.0
    f4_def = coeffs_flat_def[:, :model.f4_len]
    f6_def = coeffs_flat_def[:, model.f4_len:]
    q_def = model.decoder(f4_def, f6_def)

    # 2) Center-only embedding
    model.embedding.debug_center_only = True
    emb_center = model.embedding(torch.cat([f4_img, f6_img], dim=1), boundary_map=f0_img)
    emb_temp = emb_center.permute(0,2,3,1)
    coeffs_ctr = model.proj(emb_temp)
    coeffs_flat_ctr = coeffs_ctr.permute(0,3,1,2).reshape(-1, model.coeff_dim).clone()
    dis_all = (~model.mask_all).to(coeffs_flat_ctr.device)
    if dis_all.any():
        coeffs_flat_ctr[:, dis_all] = 0.0
    f4_ctr = coeffs_flat_ctr[:, :model.f4_len]
    f6_ctr = coeffs_flat_ctr[:, model.f4_len:]
    q_ctr = model.decoder(f4_ctr, f6_ctr)

    # Compare angle diffs against baseline
    ang_def = quat_angle_diff(q_baseline, q_def)
    ang_ctr = quat_angle_diff(q_baseline, q_ctr)

    print(f"Baseline->Default: mean={ang_def.mean().item():.4f} rad, deg={ang_def.mean().item()*180/math.pi:.3f}, max_deg={ang_def.max().item()*180/math.pi:.3f}")
    print(f"Baseline->Center:  mean={ang_ctr.mean().item():.4f} rad, deg={ang_ctr.mean().item()*180/math.pi:.3f}, max_deg={ang_ctr.max().item()*180/math.pi:.3f}")

    # also print a few sample distances
    for i in range(5):
        print(f"sample {i}: def_deg={(ang_def[i].item()*180/math.pi):.3f}, ctr_deg={(ang_ctr[i].item()*180/math.pi):.3f}")

if __name__ == '__main__':
    run_ab_test()
