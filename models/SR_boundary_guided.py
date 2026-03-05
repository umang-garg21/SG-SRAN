# -*- coding: utf-8 -*-
"""
models/SR_boundary_guided.py
-----------------------------
Equivariant SR upsampler guided by a grain-boundary probability map derived
from O(3)-invariant l=0 pairwise similarities of LR crystal features.

Boundary map (computed at upsample time from LR features)
  For adjacent LR pixels i and j, the l=0 cross-invariant is:
    sim_ij = (f4_n_i · f4_n_j + f6_n_i · f6_n_j) / 2  ∈ [-1, 1]
  This equals the l=0 projection of (fi ⊗ fj) — a scalar O(3)-invariant.
  Same grain → sim ≈ +1 (b ≈ 0).  Boundary crossing → sim ≈ -1 (b ≈ 1).

  Per-pixel boundary probability:
    b[p] = 1 − mean_over_incident_edges( sim_ij )   ∈ [0, 1]

Upsampling (per HR pixel p_hr at LR sub-pixel position (sy, sx))
  1. Gather window_size×window_size LR neighbourhood N_k around parent pixel.
  2. Score each candidate k ∈ N:
       log_score_k  = −d_k² / (2σ²)   ← distance rolloff   (σ = exp(log_σ) learnable)
                    − λ · b_k          ← boundary gate       (λ = exp(log_λ) learnable)
                    + γ · sim_k        ← intra-grain boost   (γ = exp(log_γ) learnable)
       where d_k is the LR-unit distance from p_hr to pixel k, and
       sim_k = (f4_nn_n·f4_k_n + f6_nn_n·f6_k_n)/2 is the orientation similarity
       between the HR pixel's NN feature (its parent LR pixel) and candidate k.
       Cross-boundary candidates are doubly penalised (high b_k, low sim_k).
  3. attn_k = softmax_k(log_score_k).
  4. Equivariant value: val_k = FCTP_val(feat_lr_k, SH(dir_k))
       dir_k = (sx−ox_k, sy−oy_k, 0)  ← direction from k to p_hr in LR-pixel units.
  5. context = Σ_k attn_k · val_k          ← equivariant weighted sum.
  6. out = FCTP_out(feat_nn, context) + feat_nn   ← equivariant mix + NN residual.

Init: FCTP_val and FCTP_out weights = 0 → pure NN upsample at epoch 0.

Equivariance
  Boundary map is O(3)-invariant (inner product of same-l irreps after normalise).
  Distance rolloff is a scalar function of spatial coordinates → invariant.
  Boundary gate is invariant (boundary map is invariant).
  FCTP is equivariant.  Weighted sum of equivariant vectors is equivariant.
  NN upsample applies identically to every irrep channel → equivariant.

Memory (4× SR, 128×128 HR, window=3×3, B=2):
  feat_win:  2 × 16384 × 9 × 22 × 4B ≈  25 MB
  vals:       same
  attn:       2 × 16384 × 9 × 4B     ≈   1 MB
  sh_cache:  16 × 9 × 6 × 4B         ≈  3 KB (precomputed once)
  Total extra beyond NN upsample:      ≈ 55 MB  (fits easily on A100)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

from models.SR_double_conv_SRattn import FCCAutoEncoderSR, EquivariantSpatialConv
from models.SR_grain_attn import LRBlockAttentionBlock


# ──────────────────────────────────────────────────────────────────────────────
class BoundaryGuidedUpsample(nn.Module):
	"""
	LR → HR equivariant upsampler guided by boundary map derived from
	O(3)-invariant l=0 pairwise similarities (cross-inner products) of
	adjacent LR crystal features.

	Args:
	    upsample_factor: Spatial scale factor (e.g. 4 for ×4 SR).
	    window_size:     Side length of the LR pixel neighbourhood gathered
	                     per HR pixel (default 3 → 3×3 = 9 candidates).
	    init_sigma:      Initial rolloff scale in LR-pixel units (default 0.5).
	                     σ = exp(log_σ), learnable.  Smaller → tighter locality.
	    init_lambda:     Initial boundary-gate strength (default 2.0).
	                     λ = exp(log_λ), learnable.  Larger → harder boundary.
	    init_gamma:      Initial intra-grain similarity boost (default 1.0).
	                     γ = exp(log_γ), learnable.  Larger → stronger within-grain preference.
	"""

	_SH_IRREPS = Irreps("1x0e + 1x2e")   # 1+5 = 6 SH components (even-parity only)

	def __init__(
		self,
		upsample_factor: int  = 4,
		window_size:     int  = 3,
		init_sigma:      float = 0.5,
		init_lambda:     float = 2.0,
		init_gamma:      float = 1.0,
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.window_size     = int(window_size)
		self.half_w          = window_size // 2
		self.K               = window_size * window_size

		self.irreps_feat = Irreps("1x4e + 1x6e")

		# ── Learnable rolloff sigma, boundary-gate lambda, intra-grain gamma ─
		self.log_sigma  = nn.Parameter(torch.tensor(math.log(float(init_sigma))))
		self.log_lambda = nn.Parameter(torch.tensor(math.log(float(init_lambda))))
		self.log_gamma  = nn.Parameter(torch.tensor(math.log(float(init_gamma))))

		# ── Equivariant value transform: feat_lr ⊗ SH(dir) → feat ─────────
		# zero-init → output = NN upsample at epoch 0
		self.tp_val = FullyConnectedTensorProduct(
			self.irreps_feat, self._SH_IRREPS, self.irreps_feat,
			shared_weights=True,
		)
		with torch.no_grad():
			self.tp_val.weight.data.zero_()

		# ── Equivariant output mixing: feat_nn ⊗ context → feat ────────────
		# zero-init → output = NN upsample at epoch 0
		self.tp_out = FullyConnectedTensorProduct(
			self.irreps_feat, self.irreps_feat, self.irreps_feat,
			shared_weights=True,
		)
		with torch.no_grad():
			self.tp_out.weight.data.zero_()

		# ── Precomputed geometry buffers (fixed, shape depends on r and W) ──
		# Registered as non-persistent buffers so they move with .to() but
		# are NOT saved in the state_dict (they're cheap to rebuild).
		r  = self.upsample_factor
		K  = self.K
		hw = self.half_w
		W  = self.window_size

		ofs = torch.arange(-hw, hw + 1, dtype=torch.float32)
		oy, ox = torch.meshgrid(ofs, ofs, indexing="ij")
		oy = oy.reshape(K)   # (K,)
		ox = ox.reshape(K)   # (K,)

		# ── Direction-sign masks for edge-crossing boundary (K,) float ──────
		# Used in forward() to select the correct h/v edge for each window slot k.
		self.register_buffer("_oy",  oy,  persistent=False)
		self.register_buffer("_ox",  ox,  persistent=False)
		self.register_buffer("_oxp", (ox > 0).float(), persistent=False)  # 1 if neighbor is right
		self.register_buffer("_oxn", (ox < 0).float(), persistent=False)  # 1 if neighbor is left
		self.register_buffer("_oyp", (oy > 0).float(), persistent=False)  # 1 if neighbor is below
		self.register_buffer("_oyn", (oy < 0).float(), persistent=False)  # 1 if neighbor is above

		# Sub-pixel positions within one LR-pixel tile for each HR pixel index
		sub_y = torch.arange(r, dtype=torch.float32) / r   # (r,)  ∈ [0, 1)
		sub_x = torch.arange(r, dtype=torch.float32) / r   # (r,)

		# dy[sy_idx, k] = sub_y[sy_idx] − oy[k]   (direction y component from k to p_hr)
		# dx[sx_idx, k] = sub_x[sx_idx] − ox[k]   (direction x component from k to p_hr)
		dy = sub_y.view(r, 1) - oy.view(1, K)   # (r, K)
		dx = sub_x.view(r, 1) - ox.view(1, K)   # (r, K)

		# dsq_cache[sy_idx, sx_idx, k] = d² from HR pixel to window slot k in LR units
		dsq = dy.unsqueeze(1).pow(2) + dx.unsqueeze(0).pow(2)   # (r, r, K)

		# Direction vectors from LR pixel k to HR sub-pixel (sy, sx)
		dir_y = dy.unsqueeze(1).expand(r, r, K)    # (r, r, K)
		dir_x = dx.unsqueeze(0).expand(r, r, K)    # (r, r, K)  NOTE: use dx row→sx_idx
		# (0-indexed: dir_x[sy_idx, sx_idx, k] = sub_x[sx_idx] - ox[k])
		# Recompute dir_x correctly using the dx slice along sx dimension:
		dy_full = dy.unsqueeze(1).expand(r, r, K)   # (r, r, K): dy[sy_idx, k]
		dx_full = dx.unsqueeze(0).expand(r, r, K)   # (r, r, K): dx[sx_idx, k]

		dirs = torch.stack([
			dx_full,                           # x = col direction
			dy_full,                           # y = row direction
			torch.zeros(r, r, K),              # z = 0 (in-plane)
		], dim=-1)                             # (r, r, K, 3)

		norms   = dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		center  = (dx_full.pow(2) + dy_full.pow(2)) < 1e-8   # (r, r, K) bool
		dirs_n  = dirs / norms
		dirs_n[center] = torch.tensor([0.0, 0.0, 1.0])        # z-axis for zero offset

		with torch.no_grad():
			sh = o3.spherical_harmonics(
				self._SH_IRREPS,
				dirs_n.reshape(-1, 3),
				normalize=False,
			).reshape(r, r, K, 6)   # (r, r, K, 6)

		# persistent=False: not saved in state_dict, rebuilt on load
		self.register_buffer("_sh_cache",  sh,  persistent=False)    # (r, r, K, 6)
		self.register_buffer("_dsq_cache", dsq, persistent=False)    # (r, r, K)

	# ── static: boundary map ──────────────────────────────────────────────────
	@staticmethod
	def _compute_boundary_map(
		f4: torch.Tensor,   # (B, H*W, 9)
		f6: torch.Tensor,   # (B, H*W, 13)
		H:  int,
		W:  int,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Boundary probability map from l=0 invariants of adjacent LR pixels.

		  sim_ij = (f4_n_i·f4_n_j + f6_n_i·f6_n_j) / 2  ∈ [-1, 1]
		  b_ij   = (1 − sim_ij) / 2                       ∈ [0, 1]
		  b[p]   = mean of incident-edge boundary values   ∈ [0, 1]

		Returns:
		    bdry_node : (B, H, W)   per-pixel average boundary probability
		    bdry_h    : (B, H, W-1) horizontal edge boundary (between col j and j+1)
		    bdry_v    : (B, H-1, W) vertical edge boundary   (between row i and i+1)
		"""
		B      = f4.shape[0]
		device = f4.device
		dtype  = f4.dtype

		f4n = F.normalize(f4.reshape(B, H, W, 9),  dim=-1)   # (B, H, W, 9)
		f6n = F.normalize(f6.reshape(B, H, W, 13), dim=-1)   # (B, H, W, 13)

		# Horizontal edges: similarity between (i,j) and (i,j+1)
		sim_h = ((f4n[:, :, :-1, :] * f4n[:, :, 1:, :]).sum(-1)
		         + (f6n[:, :, :-1, :] * f6n[:, :, 1:, :]).sum(-1)) / 2   # (B,H,W-1)
		bdry_h = (1.0 - sim_h) / 2                                        # ∈ [0, 1]

		# Vertical edges: similarity between (i,j) and (i+1,j)
		sim_v = ((f4n[:, :-1, :, :] * f4n[:, 1:, :, :]).sum(-1)
		         + (f6n[:, :-1, :, :] * f6n[:, 1:, :, :]).sum(-1)) / 2   # (B,H-1,W)
		bdry_v = (1.0 - sim_v) / 2                                        # ∈ [0, 1]

		# Per-pixel probability = mean of incident edge values
		bdry = torch.zeros(B, H, W, device=device, dtype=dtype)
		count = torch.zeros(B, H, W, device=device, dtype=dtype)
		bdry[:, :, :-1] += bdry_h;  count[:, :, :-1] += 1
		bdry[:, :,  1:] += bdry_h;  count[:, :,  1:] += 1
		bdry[:, :-1, :] += bdry_v;  count[:, :-1, :] += 1
		bdry[:,  1:, :] += bdry_v;  count[:,  1:, :] += 1

		bdry_node = bdry / count.clamp_min(1.0)   # (B, H, W) ∈ [0, 1]
		return bdry_node, bdry_h, bdry_v

	def forward(
		self,
		f4:        torch.Tensor,
		f6:        torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""
		Args:
		    f4:        (H*W, 9)  or (B, H*W, 9)    l=4 LR features
		    f6:        (H*W, 13) or (B, H*W, 13)   l=6 LR features
		    img_shape: (H, W)                       LR spatial dimensions
		Returns:
		    f4_out:   (rH*rW, 9)  or (B, rH*rW, 9)
		    f6_out:   (rH*rW, 13) or (B, rH*rW, 13)
		    hr_shape: (rH, rW)
		"""
		H, W   = img_shape
		r      = self.upsample_factor
		Hr, Wr = H * r, W * r
		C      = 22
		K      = self.K
		hw     = self.half_w
		device = f4.device
		dtype  = f4.dtype

		batched = f4.dim() == 3
		if not batched:
			f4 = f4.unsqueeze(0)
			f6 = f6.unsqueeze(0)
		B = f4.shape[0]

		# ── 1.  Boundary map from l=0 cross-invariants ──────────────────────
		# b[p] ∈ [0,1]: 0 = same grain interior, 1 = grain boundary
		bdry_lr, bdry_h, bdry_v = self._compute_boundary_map(f4, f6, H, W)
		# bdry_lr : (B, H, W)   per-pixel average (used for bilinear gate)
		# bdry_h  : (B, H, W-1) horizontal edge crossings
		# bdry_v  : (B, H-1, W) vertical edge crossings

		# ── 2.  Pack features into image form ───────────────────────────────
		feat_lr  = torch.cat([f4, f6], dim=-1)                         # (B, H*W, 22)
		feat_img = feat_lr.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)

		# ── 3.  Nearest-neighbour upsample — equivariant base + residual ────
		feat_nn      = F.interpolate(feat_img, scale_factor=r, mode="nearest")  # (B, C, Hr, Wr)
		feat_nn_flat = feat_nn.permute(0, 2, 3, 1).reshape(B * Hr * Wr, C)     # (B·N, C)

		# ── 4.  Gather W×W LR feature patches at each LR pixel ──────────────
		#   unfold gives (B, C, H, W, ws, ws).
		#   Must permute C to last dim BEFORE merging K dims, otherwise the
		#   reshape scrambles C with the spatial window indices.
		feat_padded = F.pad(feat_img, [hw] * 4, mode="replicate")        # (B, C, H+2hw, W+2hw)
		feat_unf    = (feat_padded
		               .unfold(2, self.window_size, 1)
		               .unfold(3, self.window_size, 1))                   # (B, C, H, W, ws, ws)
		feat_win_lr = (feat_unf
		               .permute(0, 2, 3, 4, 5, 1)   # (B, H, W, ws, ws, C)
		               .reshape(B, H, W, K * C)      # (B, H, W, K*C)
		               .permute(0, 3, 1, 2))          # (B, K*C, H, W)
		feat_win_hr = F.interpolate(feat_win_lr, scale_factor=r, mode="nearest")  # (B, K*C, Hr, Wr)
		feat_win    = (feat_win_hr
		               .permute(0, 2, 3, 1)           # (B, Hr, Wr, K*C)
		               .reshape(B, Hr * Wr, K, C))    # (B, N, K, C)

		# ── 5.  Per-edge-crossing boundary for each center→k direction ───────
		# Uses the boundary of the EDGE that must be CROSSED to reach k, not
		# the per-node average at k.  A neighbor in a different grain's interior
		# has b_node≈0 but the crossing edge to it has b_edge≈1 — this correctly
		# blocks cross-grain attention in all cases.
		#
		# For center (cy,cx) → neighbor k at (cy+oy[k], cx+ox[k]):
		#   ox[k]>0: right h-edge  b_right[cy,cx] = bdry_h[cy,cx]
		#   ox[k]<0: left  h-edge  b_left[cy,cx]  = bdry_h[cy,cx-1]  (pad left)
		#   oy[k]>0: down  v-edge  b_down[cy,cx]  = bdry_v[cy,cx]
		#   oy[k]<0: up    v-edge  b_up[cy,cx]    = bdry_v[cy-1,cx]  (pad top)
		#   diagonal: max of the two cardinal edge values crossed
		b_right = F.pad(bdry_h, [0, 1], value=0.0)           # (B, H, W)
		b_left  = F.pad(bdry_h, [1, 0], value=0.0)           # (B, H, W)
		b_down  = F.pad(bdry_v, [0, 0, 0, 1], value=0.0)     # (B, H, W)
		b_up    = F.pad(bdry_v, [0, 0, 1, 0], value=0.0)     # (B, H, W)
		oxp = self._oxp;  oxn = self._oxn                     # (K,)
		oyp = self._oyp;  oyn = self._oyn                     # (K,)
		# (B, H, W, K): h and v crossing component for each (center, k) pair
		h_comp = (b_right.unsqueeze(-1) * oxp
		        + b_left.unsqueeze(-1)  * oxn)                # (B, H, W, K)
		v_comp = (b_down.unsqueeze(-1)  * oyp
		        + b_up.unsqueeze(-1)    * oyn)                # (B, H, W, K)
		bdry_edge_lr = torch.maximum(h_comp, v_comp)          # (B, H, W, K)  ∈ [0, 1]
		# NN-upsample: each HR pixel inherits its LR parent's edge-crossing values
		bdry_edge_hr = F.interpolate(
		    bdry_edge_lr.permute(0, 3, 1, 2),                 # (B, K, H, W)
		    scale_factor=r, mode="nearest",
		)                                                      # (B, K, Hr, Wr)
		bdry_win = (bdry_edge_hr
		            .permute(0, 2, 3, 1)                       # (B, Hr, Wr, K)
		            .reshape(B, Hr * Wr, K))                   # (B, N, K)

		# ── 6.  Retrieve precomputed geometry for current device / dtype ─────
		sh_cache  = self._sh_cache.to(device=device, dtype=dtype)   # (r, r, K, 6)
		dsq_cache = self._dsq_cache.to(device=device, dtype=dtype)  # (r, r, K)

		# Sub-pixel indices: each HR pixel maps to one of r² patterns
		sy_idx = torch.arange(Hr, device=device) % r   # (Hr,)
		sx_idx = torch.arange(Wr, device=device) % r   # (Wr,)

		# Index into cache: (Hr, Wr, K)
		k_idx  = torch.arange(K, device=device)
		i_y    = sy_idx.view(Hr, 1, 1).expand(Hr, Wr, K)
		i_x    = sx_idx.view(1, Wr, 1).expand(Hr, Wr, K)
		i_k    = k_idx.view(1, 1, K).expand(Hr, Wr, K)

		dsq_hr = dsq_cache[i_y, i_x, i_k]              # (Hr, Wr, K)
		sh_hr  = sh_cache[i_y, i_x, i_k]               # (Hr, Wr, K, 6)

		# ── 7.  Attention scores: distance rolloff + boundary gate + intra-grain boost ──
		# Intra-grain similarity: cosine similarity between each HR pixel's NN-upsampled
		# feature (= its parent LR pixel orientation) and each LR window candidate.
		# Same-grain candidates score high (+γ); cross-boundary candidates doubly
		# penalised (high b_k from boundary gate AND low sim_k from here).
		Nr = Hr * Wr
		f4_ref   = F.normalize(feat_nn_flat[..., :9],  dim=-1).reshape(B, Nr, 1,  9)
		f6_ref   = F.normalize(feat_nn_flat[..., 9:],  dim=-1).reshape(B, Nr, 1, 13)
		f4_win_n = F.normalize(feat_win[..., :9],  dim=-1)    # (B, Nr, K, 9)
		f6_win_n = F.normalize(feat_win[..., 9:],  dim=-1)    # (B, Nr, K, 13)
		sim      = ((f4_ref * f4_win_n).sum(-1)
		            + (f6_ref * f6_win_n).sum(-1)) / 2.0       # (B, Nr, K) ∈ [-1, 1]
		sim_hr   = sim.reshape(B, Hr, Wr, K)                   # (B, Hr, Wr, K)

		sigma       = torch.exp(self.log_sigma).clamp(max=0.75)   # prevent attention from spreading too wide
		lam         = torch.exp(self.log_lambda)
		gamma       = torch.exp(self.log_gamma)
		log_rolloff = -dsq_hr.unsqueeze(0) / (2.0 * sigma * sigma)   # (1, Hr, Wr, K)
		log_gate    = -lam * bdry_win.reshape(B, Hr, Wr, K)          # (B, Hr, Wr, K)
		log_boost   = gamma * sim_hr                                  # (B, Hr, Wr, K)

		log_scores = log_rolloff + log_gate + log_boost               # (B, Hr, Wr, K)

		# Float32 softmax for stability under AMP
		attn = torch.softmax(
		    log_scores.reshape(B * Hr * Wr, K).float(), dim=-1
		).to(dtype)                                                   # (B·N, K)

		# ── 8.  Equivariant value transform ──────────────────────────────────
		# sh_hr is the same for every batch item — expand once
		sh_flat   = (sh_hr.unsqueeze(0)
		             .expand(B, Hr, Wr, K, 6)
		             .reshape(B * Hr * Wr * K, 6))           # (B·N·K, 6)

		feat_flat = feat_win.reshape(B * Hr * Wr * K, C)    # (B·N·K, C)

		vals = self.tp_val(feat_flat, sh_flat)               # (B·N·K, C)  equivariant
		vals = vals.reshape(B * Hr * Wr, K, C)              # (B·N, K, C)

		# ── 9.  Attention-weighted aggregate → equivariant context ───────────
		context = (attn.unsqueeze(-1) * vals).sum(dim=1)    # (B·N, C)

		# ── 10. Boundary-gated bilinear base: captures intra-grain gradients ────
		# Standard bilinear upsampling captures orientation gradients within grains.
		# Gate by the center-pixel's boundary probability (NN-upsampled from LR):
		#   t = (1 − b)²  →  1.0 deep inside grain,  0.0 at grain boundary
		# Interior pixels use bilinear (captures gradient); boundary pixels fall
		# back to NN (no cross-grain blending).  At epoch 0, tp_out weight = 0
		# so out = feat_base_flat, which is a clean gradient-preserving base.
		feat_bilinear_hr   = F.interpolate(
		    feat_img, scale_factor=r, mode="bilinear", align_corners=False
		)                                                                    # (B, C, Hr, Wr)
		feat_bilinear_flat = feat_bilinear_hr.permute(0, 2, 3, 1).reshape(B * Nr, C)  # (B·N, C)
		bdry_nn            = F.interpolate(
		    bdry_lr.unsqueeze(1).float(), scale_factor=r, mode="nearest"
		).squeeze(1).to(dtype)                                              # (B, Hr, Wr)
		bdry_nn_flat       = bdry_nn.reshape(B * Nr, 1)                    # (B·N, 1)
		t                  = (1.0 - bdry_nn_flat).pow(2)                   # (B·N, 1)
		feat_base_flat     = t * feat_bilinear_flat + (1.0 - t) * feat_nn_flat  # (B·N, C)

		# ── 11. Output: equivariant mix + boundary-gated bilinear residual ───
		out = self.tp_out(feat_base_flat, context) + feat_base_flat          # (B·N, C)
		out = out.reshape(B, Hr * Wr, C)

		if not batched:
			out = out.squeeze(0)   # (N, C)

		return out[..., :9], out[..., 9:], (Hr, Wr)


# ──────────────────────────────────────────────────────────────────────────────
class FCCAutoEncoderSRBoundaryGuided(FCCAutoEncoderSR):
	"""FCCAutoEncoderSR (double-conv LR) with boundary-guided upsampling.

	Replaces EquivariantTransposeConv with BoundaryGuidedUpsample.
	Full pipeline:

	  LR → encode
	    → conv_lr1 (k1)
	    → conv_lr2 (k2)
	    → BoundaryGuidedUpsample          ← NEW: boundary-gated distance-rolloff
	    → conv_hr (3×3)
	    → decode → FZ-reduce

	The upsampler computes an l=0 boundary map from the LR features at the
	upsample boundary and uses it to weight a learnable gathering of LR
	features, with a learned distance rolloff.  Same-grain pixels are upsampled
	with high fidelity; boundary pixels naturally blend from the correct grain.

	At epoch 0: BoundaryGuidedUpsample ≡ NN upsample (FCTP weights zero-init).

	At epoch 0: BoundaryGuidedUpsample ≡ NN upsample (FCTP weights zero-init).
	Followed by LRBlockAttentionBlock HR refinement (zero-init delta → identity at epoch 0).

	New config keys:
	    bg_window_size:     W×W LR neighbourhood per HR pixel (default 3).
	    bg_init_sigma:      Initial rolloff scale in LR pixels (default 0.5).
	    bg_init_lambda:     Initial boundary gate strength     (default 2.0).
	    bg_init_gamma:      Initial intra-grain similarity boost (default 1.0).
	    hr_attn_num_blocks: Number of stacked HR attention blocks (default 2).
	    hr_attn_num_channels: Hidden channel multiplier C for HR attention FCTP (default 8).
	    hr_attn_block_size:   Spatial block size in HR pixels (default 16).
	"""

	_SH_IRREPS = Irreps("1x0e + 1x2e")

	def __init__(
		self,
		device                   = None,
		upsample_factor:  int    = 4,
		upsample_residual: bool  = False,
		lr_shape                 = None,
		lr_conv_kernel_size      = None,
		lr_conv_kernel_size_1    = None,
		lr_conv_kernel_size_2    = None,
		decoder_backend: str     = "lookup",
		decoder_config           = None,
		bg_window_size:  int     = 3,
		bg_init_sigma:   float   = 0.5,
		bg_init_lambda:  float   = 2.0,
		bg_init_gamma:   float   = 1.0,
		hr_attn_num_blocks: int  = 2,
		hr_attn_num_channels: int = 8,
		hr_attn_block_size: int  = 16,
		**decoder_kwargs,
	):
		# Build all inherited layers (conv_lr1, conv_lr2, upsample_conv as
		# EquivariantTransposeConv placeholder, conv_hr, encoder, decoder).
		super().__init__(
			device=device,
			upsample_factor=upsample_factor,
			upsample_residual=upsample_residual,
			upsampler="conv",          # placeholder; replaced immediately below
			lr_shape=lr_shape,
			lr_conv_kernel_size=lr_conv_kernel_size,
			lr_conv_kernel_size_1=lr_conv_kernel_size_1,
			lr_conv_kernel_size_2=lr_conv_kernel_size_2,
			decoder_backend=decoder_backend,
			decoder_config=decoder_config,
			**decoder_kwargs,
		)
		# Replace the inherited EquivariantTransposeConv with BoundaryGuidedUpsample
		self.upsample_conv = BoundaryGuidedUpsample(
			upsample_factor=self.upsample_factor,
			window_size=int(bg_window_size),
			init_sigma=float(bg_init_sigma),
			init_lambda=float(bg_init_lambda),
			init_gamma=float(bg_init_gamma),
		)

		# ── HR block-local attention refinement (same pattern as double_conv_attn) ─
		self.hr_attn_block_size = int(hr_attn_block_size)
		self.hr_attn_blocks = nn.ModuleList([
			LRBlockAttentionBlock(num_channels=int(hr_attn_num_channels))
			for _ in range(int(hr_attn_num_blocks))
		])
		self._cached_hr_block_shape: tuple[int, int] | None = None
		self._cached_hr_sh_block:    torch.Tensor | None    = None

		# ── Second HR conv: applied after grain-gated attention ──────────────
		self.conv_hr2 = EquivariantSpatialConv(kernel_size=3)

	@torch.no_grad()
	def _get_hr_sh_block(
		self,
		block_h: int,
		block_w: int,
		device:  torch.device,
		dtype:   torch.dtype,
	) -> torch.Tensor:
		"""(block_h*block_w, 6) SH of normalised relative positions within one HR block."""
		if (
			self._cached_hr_block_shape == (block_h, block_w)
			and self._cached_hr_sh_block is not None
			and self._cached_hr_sh_block.device == device
		):
			return self._cached_hr_sh_block.to(dtype)
		ys   = torch.linspace(-1.0, 1.0, block_h, device=device)
		xs   = torch.linspace(-1.0, 1.0, block_w, device=device)
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
		zs   = torch.zeros(block_h * block_w, device=device)
		dirs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), zs], dim=-1)
		dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		sh   = o3.spherical_harmonics(self._SH_IRREPS, dirs, normalize=False)
		self._cached_hr_block_shape = (block_h, block_w)
		self._cached_hr_sh_block    = sh
		return sh.to(dtype)

	def _forward_sr_features(
		self,
		f4_lr: torch.Tensor,
		f6_lr: torch.Tensor,
		lr_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""BoundaryGuidedUpsample → conv_hr → HR block-local attention blocks."""
		f4_hr, f6_hr, hr_shape = super()._forward_sr_features(f4_lr, f6_lr, lr_shape)

		Hr, Wr = hr_shape
		device = f4_hr.device
		dtype  = f4_hr.dtype

		batched = f4_hr.dim() == 3
		if not batched:
			f4_hr = f4_hr.unsqueeze(0)
			f6_hr = f6_hr.unsqueeze(0)
		B = f4_hr.shape[0]

		# ── Grain-interior mask: suppress HR attention at grain boundaries ────
		# Boundary probability from upsampled features; quadratic so suppression
		# is sharp near boundaries (b≈1 → mask≈0) but full inside grains (b≈0 → mask≈1).
		# This prevents HR attention from blending cross-boundary information while
		# still allowing full coherence enforcement deep inside each grain.
		b_hr, _, _ = BoundaryGuidedUpsample._compute_boundary_map(f4_hr, f6_hr, Hr, Wr)
		grain_mask = (1.0 - b_hr).pow(2)   # (B, Hr, Wr)

		block_h = min(self.hr_attn_block_size, Hr)
		block_w = min(self.hr_attn_block_size, Wr)
		pad_h   = (-Hr) % block_h
		pad_w   = (-Wr) % block_w
		Hr_pad  = Hr + pad_h
		Wr_pad  = Wr + pad_w

		feat = torch.cat([f4_hr, f6_hr], dim=-1)   # (B, Hr*Wr, 22)

		if pad_h > 0 or pad_w > 0:
			feat_2d = feat.reshape(B, Hr, Wr, 22).permute(0, 3, 1, 2)
			feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
			feat    = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, 22)
			gm_2d   = grain_mask.unsqueeze(1)
			gm_2d   = F.pad(gm_2d, (0, pad_w, 0, pad_h), mode="reflect")
			grain_mask_flat = gm_2d.squeeze(1).reshape(B, Hr_pad * Wr_pad, 1)
		else:
			grain_mask_flat = grain_mask.reshape(B, Hr * Wr, 1)

		sh_block = self._get_hr_sh_block(block_h, block_w, device, dtype)   # (Nb, 6)

		for attn_block in self.hr_attn_blocks:
			delta = attn_block(feat, sh_block, Hr_pad, Wr_pad, block_h, block_w)
			feat  = feat + grain_mask_flat * delta   # apply only inside grains

		if pad_h > 0 or pad_w > 0:
			feat = feat.reshape(B, Hr_pad, Wr_pad, 22)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, 22)

		if not batched:
			feat = feat.squeeze(0)

		f4_out, f6_out = self.conv_hr2(feat[..., :9], feat[..., 9:], hr_shape)
		return f4_out, f6_out, hr_shape
