import torch
import torch.nn.functional as F
import importlib.util
import os
import numpy as np

root = os.path.dirname(os.path.dirname(__file__))
path_equi = os.path.join(root, 'e3nn_experimentation', '4x4_superresolution_transformers', 'equiformer_ebsdSr.py')

spec = importlib.util.spec_from_file_location('equiformer', path_equi)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
physics = mod.FCCPhysics(device)

# instantiate encoder and model
enc = mod.FCCEncoder(physics)
model = mod.CrystalTransformerSR(physics, scale_factor=4, depth=1)
model.to(device)
enc.to(device)

# small test image
B = 1
H = 3
W = 3
N = H * W
q = torch.randn(N, 4, device=device)
q = q / q.norm(dim=1, keepdim=True)

# get encoder image-shaped outputs
with torch.no_grad():
    f0_img, f1_img, f4_img, f6_img = enc(q, img_shape=(H, W))

# concat as model expects
x = torch.cat([f4_img, f6_img], dim=1)

emb = model.embedding
print('\n=== BoundarySuppressedConv inspector ===')
print('Device:', device)
print('Input shapes: x', x.shape, 'f0', f0_img.shape)
print('Embedding kernel_size, padding, stride, K2:', emb.kernel_size, emb.padding, emb.stride, emb.K2)

# Print irreps channel grouping for irreps_in
print('\nIrreps_in groups (start, end, mul, ir.l):')
idx = 0
for mul, ir in emb.irreps_in:
    dim = mul * (2 * ir.l + 1)
    print(' ', idx, idx + dim, ' mul=', mul, ' l=', ir.l)
    idx += dim
print(' total in_dim:', emb.in_dim)

# Perform unfold exactly as forward
x_in = x.clone().to(device)
B0, C, H0, W0 = x_in.shape
x_unfolded = F.unfold(x_in, kernel_size=emb.kernel_size, padding=emb.padding, stride=emb.stride)
L = x_unfolded.shape[-1]
x_unfolded = x_unfolded.view(B0, C, emb.K2, L)
print('\nx_unfolded shape:', x_unfolded.shape)
print('L (num patches):', L)

# show first patch (centered) values for first channel group
print('\nSample x_unfolded [batch=0, channel=0, kernel_positions, first 3 patches]:')
print(x_unfolded[0, 0, :, :min(3, L)])

# kernel weights and softmax
kernel_raw = emb.kernel_weights.detach().cpu().numpy()
print('\nkernel_weights raw:', kernel_raw)
weights = torch.softmax(emb.kernel_weights, dim=0).view(1, 1, emb.K2, 1)
print('weights softmax:', weights.view(-1).detach().cpu().numpy())

# boundary map processing
bmap = f0_img.clone().to(device)
b_unfold = F.unfold(bmap, kernel_size=emb.kernel_size, padding=emb.padding, stride=emb.stride)
b_unfold = b_unfold.view(B0, 1, emb.K2, L)
print('\nboundary_unfold shape:', b_unfold.shape)
print('boundary patch sample (first 3):', b_unfold[0,0,:, :min(3,L)])

# suppression
suppression = torch.sigmoid(-10.0 * (b_unfold - emb.boundary_threshold))
print('\nsuppression stats: min, mean, max:', suppression.min().item(), suppression.mean().item(), suppression.max().item())
print('suppression sample (first patch):', suppression[0,0,:,0].cpu().numpy())

# weights * suppression
weights_supp = weights.to(device) * suppression
sums = weights_supp.sum(dim=2)
print('\nweights after suppression sums per patch (first 3):', sums[0,0,:min(3,L)].detach().cpu().numpy())

# normalized weights (as in forward)
weights_norm = weights_supp / (weights_supp.sum(dim=2, keepdim=True) + 1e-8)
print('\nweights_norm sample (kernel positions for first patch):', weights_norm[0,0,:,0].detach().cpu().numpy())
print('sum(weights_norm) for first patch:', weights_norm[0,0,:,0].sum().item())

# aggregated output vector calculation
x_agg = (x_unfolded * weights_norm).sum(dim=2)
print('\nAggregated x_agg shape (B,L,C):', x_agg.shape)
print('Aggregated first pixel vector (channels 0..7):', x_agg[0, :min(3,L), :8].detach().cpu().numpy())

# Compare aggregated center-pixel to original center pixel (if stride==1 and padding preserves center)
# Find mapping from patch index to (h,w)
H_out = (H0 + 2 * emb.padding - emb.kernel_size) // emb.stride + 1
W_out = (W0 + 2 * emb.padding - emb.kernel_size) // emb.stride + 1
print('\nComputed H_out,W_out:', H_out, W_out, ' H_out*W_out=', H_out*W_out)

# show correspondence for pixel (1,1) (center)
center_idx = (1) * W_out + 1 if (H_out>1 and W_out>1) else 0
print('center_idx:', center_idx)
orig_center = x_in[0, :, 1:2, 1:2].reshape(-1).cpu().numpy()
print(' original center channels first 8:', orig_center[:8])
print(' aggregated center channels first 8:', x_agg[0, center_idx, :8].detach().cpu().numpy())

# Print a per-irrep slice from aggregated vs original for visual inspection
print('\nPer-irrep comparison (orig vs agg) for center pixel:')
idx = 0
for mul, ir in emb.irreps_in:
    dim = mul * (2 * ir.l + 1)
    orig_slice = orig_center[idx:idx+min(8,dim)]
    agg_slice = x_agg[0, center_idx, idx:idx+min(8,dim)].detach().cpu().numpy()
    print(f' ir.l={ir.l:2d} mul={mul:2d} idx={idx:3d}:{idx+dim:3d} orig[:6]={orig_slice[:6]} agg[:6]={agg_slice[:6]}')
    idx += dim

print('\nDone.')
