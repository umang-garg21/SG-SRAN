# -*- coding: utf-8 -*-
"""
File:        SR_global_attn.py
Description: Global equivariant LR attention upsampler for FCC EBSD super-resolution.

Pipeline
--------
  f4_lr, f6_lr
    → [LRSelfAttentionBlock × num_attn_blocks]  (global self-attention in LR space)
    → EquivariantTransposeConv                  (upsample LR → HR)
    → [conv_hr in FCCAutoEncoderSR]             (equivariant spatial conv at HR)
    → f4_hr, f6_hr

All attention is done in the LR feature space where the attention matrix is
O(N_lr²) — manageable and globally mixes every LR pixel with every other.
The transpose conv upsample is physically initialised (bilinear) and is the
only step that changes resolution.

LRSelfAttentionBlock (stacked num_attn_blocks times, independent weights):
  1. Compute O(3)-invariant scores from the CURRENT 1× irrep features:
       scores = s4·(f4ₙ@f4ₙᵀ) + s6·(f6ₙ@f6ₙᵀ) + pb + pbᵀ
     (each block re-scores from its updated features → attention pattern evolves)
  2. Expand to Cx4e+Cx6e hidden channels via o3.Linear.
  3. Equivariant value transform in hidden space: FCTP(h, sh_lr) → h.
  4. Global aggregation: ctx = attn @ vals.
  5. Equivariant output mix: FCTP(h, ctx) → h.
  6. Contract back + residual: lin_out(h_out) + feat.

Equivariance:
  • F.normalize of same-l irreps is O(3)-equivariant; inner product of two
    normalised same-l irreps is an O(3) invariant → scores are invariant.
  • Position bias adds scalars to scores — equivariance unaffected.
  • FCTP(h, sh_spatial): sh encodes image geometry, not crystal orientation.
    Output transforms like h → equivariant.
  • Weighted sum of equivariant vectors is equivariant.
  • o3.Linear maps Nx4e→Mx4e, Nx6e→Mx6e separately → equivariant.

Parameter count per block (C = num_channels):
  tp_val: 6·C²   (C paths × C_sh=1, mode uvw)
  tp_out: 8·C³   (8 CG path types, fully connected channels)
  C=8  → 4,521 / block    C=16 → 34,377 / block    C=32 → 270,985 / block

Memory (x4 SR, 64×64 LR):
  Attention matrix per block: 4096² × 4 B ≈ 64 MB (float32)  — A100 ✓
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear as IrrepsLinear

from models.SR_conv import EquivariantTransposeConv, FCCAutoEncoderSR


# ──────────────────────────────────────────────────────────────────────────────
class LRSelfAttentionBlock(nn.Module):
	"""Single global equivariant self-attention block operating in LR feature space.

	Scores are computed from the current 1× irrep features (invariant inner
	products).  Mixing uses multi-channel hidden irreps for expressivity.
	Zero-init on the output projection ensures each block starts as identity.
	"""

	def __init__(self, num_channels: int = 8):
		super().__init__()
		C = int(num_channels)

		self.irreps_feat = Irreps("1x4e + 1x6e")
		self.irreps_h    = Irreps(f"{C}x4e + {C}x6e")
		self.sh_irreps   = Irreps("1x0e + 1x2e")

		# Learnable log-scale per irrep-l (init: 1/√d_l → scores ∈ [-1, 1])
		self.log_s4 = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(9.0))))
		self.log_s6 = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(13.0))))

		# Symmetric position bias: SH(abs_pos) → scalar
		self.pos_bias = nn.Linear(6, 1, bias=True)
		nn.init.zeros_(self.pos_bias.weight)
		nn.init.zeros_(self.pos_bias.bias)

		# Channel expansion: 1x4e+1x6e → Cx4e+Cx6e
		self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)

		# Value transform in hidden space: h ⊗ sh_lr → h  (6·C² weights)
		self.tp_val = FullyConnectedTensorProduct(
			self.irreps_h, self.sh_irreps, self.irreps_h, shared_weights=True,
		)

		# Output mix in hidden space: h ⊗ ctx → h  (8·C³ weights)
		self.tp_out = FullyConnectedTensorProduct(
			self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True,
		)

		# Channel contraction + zero-init → block returns zero delta at epoch 0
		self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
		with torch.no_grad():
			self.lin_out.weight.data.zero_()

	def forward(self, feat: torch.Tensor, sh_lr: torch.Tensor) -> torch.Tensor:
		"""
		Args:
		    feat:  (B, N_lr, 22)  current LR features (1x4e + 1x6e, equivariant)
		    sh_lr: (N_lr,  6)     fixed SH of LR absolute grid positions
		Returns:
		    (B, N_lr, 22)  attention delta — caller adds the residual: feat + delta
		"""
		B, N, _ = feat.shape
		dtype = feat.dtype

		# ── O(3)-invariant attention scores ─────────────────────────────────
		s4 = torch.exp(self.log_s4)
		s6 = torch.exp(self.log_s6)

		# Scores from CURRENT features; (B, N, d) @ (B, d, N) → (B, N, N)
		f4_n = F.normalize(feat[..., :9],  dim=-1)   # (B, N_lr, 9)
		f6_n = F.normalize(feat[..., 9:],  dim=-1)   # (B, N_lr, 13)
		scores = (s4 * torch.bmm(f4_n, f4_n.transpose(-2, -1))
		          + s6 * torch.bmm(f6_n, f6_n.transpose(-2, -1)))  # (B, N_lr, N_lr)

		# Symmetric position bias: (N_lr, 1) broadcast to (B, N_lr, N_lr)
		pb     = self.pos_bias(sh_lr)                       # (N_lr, 1)
		scores = scores + (pb + pb.T).unsqueeze(0)          # (B, N_lr, N_lr)

		# Float32 softmax for numerical stability under AMP
		attn = torch.softmax(scores.float(), dim=-1).to(dtype)   # (B, N_lr, N_lr)

		# ── Equivariant value / output mix — flatten B*N for e3nn ops ────────
		feat_flat = feat.reshape(B * N, -1)                                        # (B*N, 22)
		sh_flat   = sh_lr.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)        # (B*N, 6)

		h      = self.lin_in(feat_flat).reshape(B, N, -1)                          # (B, N, C·22)
		vals   = self.tp_val(h.reshape(B * N, -1), sh_flat).reshape(B, N, -1)     # (B, N, C·22)
		ctx    = torch.bmm(attn, vals)                                             # (B, N, C·22)  GLOBAL MIXING
		h_out  = self.tp_out(h.reshape(B * N, -1), ctx.reshape(B * N, -1)).reshape(B, N, -1)  # (B, N, C·22)
		# Return delta only (zero-init lin_out → delta = 0 at epoch 0)
		return self.lin_out(h_out.reshape(B * N, -1)).reshape(B, N, -1)           # (B, N, 22)


# ──────────────────────────────────────────────────────────────────────────────
class GlobalEquivariantAttentionUpsample(nn.Module):
	"""Global LR self-attention + transpose-conv upsample.

	Stacks `num_attn_blocks` independent LRSelfAttentionBlocks in LR feature
	space (global O(N_lr²) mixing), then applies EquivariantTransposeConv to
	upsample to HR resolution.  The post-upsample spatial conv is handled by
	`conv_hr` in FCCAutoEncoderSR (inherited, not duplicated here).

	Args:
	    upsample_factor:  Spatial scale factor (e.g. 4 for x4 SR).
	    num_channels:     Hidden channel multiplicity C for FCTP.
	                      Dominant cost: 8·C³ per block.
	    num_attn_blocks:  Number of stacked attention blocks (each independent).
	"""

	_SH_IRREPS = Irreps("1x0e + 1x2e")   # 6 components, even parity

	def __init__(
		self,
		upsample_factor:  int = 4,
		num_channels:     int = 8,
		num_attn_blocks:  int = 2,
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.num_channels    = int(num_channels)
		self.num_attn_blocks = int(num_attn_blocks)

		# Stack of independent global self-attention blocks (LR space)
		self.attn_blocks = nn.ModuleList([
			LRSelfAttentionBlock(num_channels=self.num_channels)
			for _ in range(self.num_attn_blocks)
		])

		# Equivariant transpose-conv upsample LR → HR
		self.upsample = EquivariantTransposeConv(
			upsample_factor=self.upsample_factor,
			use_residual=False,
		)

		# Lazy LR position-SH cache (instance attrs, not parameters/buffers)
		self._cached_lr_shape: tuple[int, int] | None = None
		self._cached_sh_lr:    torch.Tensor | None    = None

	@torch.no_grad()
	def _get_sh_lr(self, H: int, W: int, device: torch.device) -> torch.Tensor:
		"""(H*W, 6) SH of absolute normalised LR grid positions; lazy-cached."""
		if (
			self._cached_lr_shape == (H, W)
			and self._cached_sh_lr is not None
			and self._cached_sh_lr.device == device
		):
			return self._cached_sh_lr
		ys = torch.linspace(-1.0, 1.0, H, device=device)
		xs = torch.linspace(-1.0, 1.0, W, device=device)
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
		zs   = torch.zeros(H * W, device=device)
		dirs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), zs], dim=-1)
		dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		sh   = o3.spherical_harmonics(self._SH_IRREPS, dirs, normalize=False)
		self._cached_lr_shape = (H, W)
		self._cached_sh_lr    = sh
		return sh

	def forward(
		self,
		f4: torch.Tensor,
		f6: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""
		Args:
		    f4:        (H*W, 9) or (B, H*W, 9)    l=4 irrep features at LR resolution
		    f6:        (H*W, 13) or (B, H*W, 13)  l=6 irrep features at LR resolution
		    img_shape: (H, W)                      LR spatial dimensions
		Returns:
		    f4_out:   (rH*rW, 9) or (B, rH*rW, 9)
		    f6_out:   (rH*rW, 13) or (B, rH*rW, 13)
		    hr_shape: (rH, rW)
		"""
		H, W   = img_shape
		device = f4.device
		dtype  = f4.dtype

		batched = f4.dim() == 3
		if not batched:
			f4 = f4.unsqueeze(0)
			f6 = f6.unsqueeze(0)

		# Position SH for ALL blocks (computed once, reused)
		sh_lr = self._get_sh_lr(H, W, device).to(dtype)   # (N_lr, 6)

		# N global self-attention blocks — explicit residual after every block
		feat = torch.cat([f4, f6], dim=-1)                 # (B, N_lr, 22)
		for block in self.attn_blocks:
			feat = feat + block(feat, sh_lr)               # (B, N_lr, 22)

		# Equivariant transpose-conv upsample to HR (batch-aware)
		f4_hr, f6_hr, hr_shape = self.upsample(feat[..., :9], feat[..., 9:], img_shape)
		# f4_hr: (B, rH*rW, 9), f6_hr: (B, rH*rW, 13)

		if not batched:
			f4_hr = f4_hr.squeeze(0)
			f6_hr = f6_hr.squeeze(0)

		return f4_hr, f6_hr, hr_shape


# ──────────────────────────────────────────────────────────────────────────────
class FCCAutoEncoderSRGlobalAttn(FCCAutoEncoderSR):
	"""FCCAutoEncoderSR with global equivariant LR attention.

	Full pipeline:
	  LR quats → encode → conv_lr
	    → [N LRSelfAttentionBlocks]          (global attention, LR space)
	    → EquivariantTransposeConv           (upsample)
	    → conv_hr                            (equivariant spatial conv, HR space)
	    → decode → FZ-reduce

	All training methods (feature_loss_sr, forward_sr) inherited from
	FCCAutoEncoderSR unchanged.

	Args:
	    num_channels:    Hidden channel multiplicity C.  8·C³ weights per block.
	    num_attn_blocks: Number of stacked LR attention blocks.
	"""

	def __init__(
		self,
		device:          str | torch.device | None = None,
		upsample_factor: int  = 4,
		num_channels:    int  = 8,
		num_attn_blocks: int  = 2,
		decoder_backend: str  = "lookup",
		decoder_config:  dict | None = None,
		**decoder_kwargs,
	):
		super().__init__(
			device=device,
			upsample_factor=upsample_factor,
			upsampler="conv",   # placeholder — replaced immediately below
			decoder_backend=decoder_backend,
			decoder_config=decoder_config,
			**decoder_kwargs,
		)
		self.upsample_conv = GlobalEquivariantAttentionUpsample(
			upsample_factor=self.upsample_factor,
			num_channels=int(num_channels),
			num_attn_blocks=int(num_attn_blocks),
		)
