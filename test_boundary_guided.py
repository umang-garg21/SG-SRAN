"""Quick smoke test for BoundaryGuidedUpsample and FCCAutoEncoderSRBoundaryGuided."""
import torch
import torch.nn.functional as F

from models.SR_boundary_guided import BoundaryGuidedUpsample, FCCAutoEncoderSRBoundaryGuided

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ── 1. BoundaryGuidedUpsample standalone ─────────────────────────────────────
up = BoundaryGuidedUpsample(upsample_factor=4, window_size=3,
                              init_sigma=0.5, init_lambda=2.0).to(device)
n_params = sum(p.numel() for p in up.parameters() if p.requires_grad)
print(f"BoundaryGuidedUpsample trainable params: {n_params}")

H, W = 8, 8
f4 = torch.randn(H * W, 9, device=device)
f6 = torch.randn(H * W, 13, device=device)
with torch.no_grad():
    f4_hr, f6_hr, hr_shape = up(f4, f6, (H, W))
print(f"LR {H}x{W} -> HR {hr_shape}  f4:{f4_hr.shape}  f6:{f6_hr.shape}")

# ── 2. Boundary map in [0, 1] ─────────────────────────────────────────────────
bdry = BoundaryGuidedUpsample._compute_boundary_map(
    f4.unsqueeze(0), f6.unsqueeze(0), H, W
)
bmin, bmax = float(bdry.min()), float(bdry.max())
print(f"boundary map range: [{bmin:.4f}, {bmax:.4f}]  (expected [0, 1])")
assert 0.0 <= bmin and bmax <= 1.0, "boundary map out of [0,1]!"

# ── 3. epoch-0 ≈ NN upsample (zero-init FCTP) ────────────────────────────────
feat = torch.cat([f4, f6], dim=-1)
feat_img = feat.reshape(H, W, 22).permute(2, 0, 1).unsqueeze(0)
nn_up = (F.interpolate(feat_img, scale_factor=4, mode="nearest")
           .squeeze(0).permute(1, 2, 0).reshape(-1, 22))
diff = (torch.cat([f4_hr, f6_hr], dim=-1) - nn_up).abs().max().item()
print(f"epoch-0 diff from NN upsample: {diff:.2e}  (should be ~0)")
assert diff < 1e-5, f"epoch-0 mismatch: {diff}"

# ── 4. Full model forward_sr ──────────────────────────────────────────────────
model = FCCAutoEncoderSRBoundaryGuided(
    device=device,
    upsample_factor=4,
    lr_conv_kernel_size_1=5,
    lr_conv_kernel_size_2=9,
).to(device)
n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FCCAutoEncoderSRBoundaryGuided trainable params: {n_model}")

lr_q = torch.randn(H * W, 4, device=device)
lr_q = lr_q / lr_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
with torch.no_grad():
    q_sr = model.forward_sr(lr_q, lr_shape=(H, W))
print(f"forward_sr output: {q_sr.shape}  (expected ({H*W*16}, 4) = ({H*16*W*16}, 4))")
assert q_sr.shape == (H * W * 16, 4), f"unexpected shape {q_sr.shape}"

# ── 5. feature_loss_sr gradient flows ────────────────────────────────────────
hr_q = torch.randn(H * 4 * W * 4, 4, device=device)
hr_q = hr_q / hr_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
loss = model.feature_loss_sr(lr_q, hr_q, lr_shape=(H, W))
loss.backward()
print(f"feature_loss_sr: {loss.item():.6f}  (gradient OK)")

print("\nALL TESTS PASSED")
