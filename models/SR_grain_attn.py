# -*- coding: utf-8 -*-
"""
File:        SR_grain_attn.py
Description: Block-local equivariant LR attention upsampler for FCC EBSD super-resolution.

Pipeline
--------
  f4_lr, f6_lr
    → [LRBlockAttentionBlock × num_attn_blocks]  (block-local self-attention in LR space)
    → EquivariantTransposeConv                   (upsample LR → HR)
    → [conv_hr in FCCAutoEncoderSR]              (equivariant spatial conv at HR)
    → f4_hr, f6_hr

Attention is confined to non-overlapping spatial blocks of ~block_size×block_size LR
pixels, so the attention matrix per block is O(Nb²) where Nb = block_h × block_w.

LRBlockAttentionBlock (stacked num_attn_blocks times, independent weights):
  1. Partition (B, H*W, 22) → (B·num_blocks, Nb, 22); pad H/W if needed.
  2. Compute O(3)-invariant scores within each block from the CURRENT features:
       scores = s4·(f4ₙ@f4ₙᵀ) + s6·(f6ₙ@f6ₙᵀ) + pb + pbᵀ
     Position bias uses normalised relative positions inside each block.
  3. Expand to Cx4e+Cx6e hidden channels via o3.Linear.
  4. Equivariant value transform: FCTP(h, sh_block) → h.
  5. Block-local aggregation: ctx = attn @ vals.
  6. Equivariant output mix: FCTP(h, ctx) → h.
  7. Contract back + residual: lin_out(h_out) + feat.
  8. Reshape back to (B, H*W, 22), unpad.

Equivariance: identical reasoning to SR_global_attn.py — inner products of
normalised same-l irreps are O(3)-invariant; FCTP and weighted sums preserve
equivariance; position bias adds only scalars.

Memory (x4 SR, 64×64 LR, 16×16 blocks → 16 blocks):
  Attention matrix per block: 256² × 4 B ≈ 256 KB   — negligible compared to global.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear as IrrepsLinear

from models.SR_conv import EquivariantTransposeConv, FCCAutoEncoderSR


# ──────────────────────────────────────────────────────────────────────────────
class LRBlockAttentionBlock(nn.Module):
	"""Single block-local equivariant self-attention block in LR feature space.

	Identical architecture to LRSelfAttentionBlock (SR_global_attn.py) but
	attention is restricted to non-overlapping spatial blocks of the LR grid
	instead of the full grid.  All blocks share weights; only the block-relative
	position SH changes.
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

		# Symmetric position bias: SH(block-relative pos) → scalar
		self.pos_bias = nn.Linear(6, 1, bias=True)
		nn.init.zeros_(self.pos_bias.weight)
		nn.init.zeros_(self.pos_bias.bias)

		# Channel expansion: 1x4e+1x6e → Cx4e+Cx6e
		self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)

		# Value transform in hidden space: h ⊗ sh_block → h  (6·C² weights)
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

	def forward(
		self,
		feat:    torch.Tensor,
		sh_block: torch.Tensor,
		H:       int,
		W:       int,
		block_h: int,
		block_w: int,
	) -> torch.Tensor:
		"""
		Args:
		    feat:     (B, H*W, 22)             LR features (H, W already padded).
		    sh_block: (block_h*block_w, 6)     SH of block-relative positions.
		    H, W:     (padded) LR spatial dims. Must be multiples of block_h, block_w.
		    block_h, block_w: block size in pixels.
		Returns:
		    (B, H*W, 22)  attention delta.
		"""
		B, N, C22 = feat.shape
		num_bh = H // block_h
		num_bw = W // block_w
		Nb     = block_h * block_w
		Bb     = B * num_bh * num_bw
		dtype  = feat.dtype

		# Partition: (B, H*W, 22) → (Bb, Nb, 22)
		feat_blocks = (
			feat.reshape(B, num_bh, block_h, num_bw, block_w, C22)
			    .permute(0, 1, 3, 2, 4, 5)          # (B, num_bh, num_bw, block_h, block_w, C22)
			    .reshape(Bb, Nb, C22)
		)

		# ── O(3)-invariant attention scores (block-local) ─────────────────────
		s4 = torch.exp(self.log_s4)
		s6 = torch.exp(self.log_s6)

		f4_n = F.normalize(feat_blocks[..., :9],  dim=-1)   # (Bb, Nb, 9)
		f6_n = F.normalize(feat_blocks[..., 9:],  dim=-1)   # (Bb, Nb, 13)
		scores = (s4 * torch.bmm(f4_n, f4_n.transpose(-2, -1))
		          + s6 * torch.bmm(f6_n, f6_n.transpose(-2, -1)))  # (Bb, Nb, Nb)

		# Symmetric position bias (same for all blocks — weight-shared)
		pb     = self.pos_bias(sh_block)                       # (Nb, 1)
		scores = scores + (pb + pb.T).unsqueeze(0)             # (Bb, Nb, Nb)

		# Float32 softmax for numerical stability under AMP
		attn = torch.softmax(scores.float(), dim=-1).to(dtype)   # (Bb, Nb, Nb)

		# ── Equivariant value / output mix ────────────────────────────────────
		sh_flat   = sh_block.unsqueeze(0).expand(Bb, Nb, -1).reshape(Bb * Nb, -1)   # (Bb*Nb, 6)
		feat_flat = feat_blocks.reshape(Bb * Nb, C22)

		h      = self.lin_in(feat_flat).reshape(Bb, Nb, -1)
		vals   = self.tp_val(h.reshape(Bb * Nb, -1), sh_flat).reshape(Bb, Nb, -1)
		ctx    = torch.bmm(attn, vals)                                               # (Bb, Nb, Ch22)
		h_out  = self.tp_out(h.reshape(Bb * Nb, -1), ctx.reshape(Bb * Nb, -1)).reshape(Bb, Nb, -1)

		delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C22)

		# Reassemble: (Bb, Nb, 22) → (B, H*W, 22)
		delta = (
			delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C22)
			             .permute(0, 1, 3, 2, 4, 5)   # (B, num_bh, block_h, num_bw, block_w, C22)
			             .reshape(B, H * W, C22)
		)
		return delta


# ──────────────────────────────────────────────────────────────────────────────
class BlockEquivariantAttentionUpsample(nn.Module):
	"""Block-local LR self-attention + transpose-conv upsample.

	Stacks `num_attn_blocks` independent LRBlockAttentionBlocks in LR feature
	space.  Attention is restricted to non-overlapping spatial blocks of
	~block_size×block_size LR pixels.  The LR image is zero-padded to the
	nearest multiple of block_size before attention and cropped after.

	Args:
	    upsample_factor:  Spatial scale factor (e.g. 4 for x4 SR).
	    num_channels:     Hidden channel multiplicity C for FCTP.
	    num_attn_blocks:  Number of stacked attention blocks (each independent).
	    block_size:       Target block width/height in LR pixels (default 16).
	                      Actual block dims are min(block_size, H) × min(block_size, W).
	"""

	_SH_IRREPS = Irreps("1x0e + 1x2e")   # 6 components, even parity

	def __init__(
		self,
		upsample_factor: int = 4,
		num_channels:    int = 8,
		num_attn_blocks: int = 2,
		block_size:      int = 16,
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.num_channels    = int(num_channels)
		self.num_attn_blocks = int(num_attn_blocks)
		self.block_size      = int(block_size)

		# Stack of independent block-local self-attention blocks
		self.attn_blocks = nn.ModuleList([
			LRBlockAttentionBlock(num_channels=self.num_channels)
			for _ in range(self.num_attn_blocks)
		])

		# Equivariant transpose-conv upsample LR → HR
		self.upsample = EquivariantTransposeConv(
			upsample_factor=self.upsample_factor,
			use_residual=False,
		)

		# Lazy block-SH cache (keyed on (block_h, block_w, device))
		self._cached_block_shape:  tuple[int, int] | None = None
		self._cached_sh_block:     torch.Tensor | None    = None

	@torch.no_grad()
	def _get_sh_block(self, block_h: int, block_w: int, device: torch.device) -> torch.Tensor:
		"""(block_h*block_w, 6) SH of normalised relative positions within one block."""
		if (
			self._cached_block_shape == (block_h, block_w)
			and self._cached_sh_block is not None
			and self._cached_sh_block.device == device
		):
			return self._cached_sh_block
		ys = torch.linspace(-1.0, 1.0, block_h, device=device)
		xs = torch.linspace(-1.0, 1.0, block_w, device=device)
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
		zs   = torch.zeros(block_h * block_w, device=device)
		dirs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), zs], dim=-1)
		dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		sh   = o3.spherical_harmonics(self._SH_IRREPS, dirs, normalize=False)
		self._cached_block_shape = (block_h, block_w)
		self._cached_sh_block    = sh
		return sh

	def forward(
		self,
		f4: torch.Tensor,
		f6: torch.Tensor,
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
		device = f4.device
		dtype  = f4.dtype

		batched = f4.dim() == 3
		if not batched:
			f4 = f4.unsqueeze(0)
			f6 = f6.unsqueeze(0)
		B = f4.shape[0]

		# Block dims: clamp to image size
		block_h = min(self.block_size, H)
		block_w = min(self.block_size, W)

		# Pad H, W to multiples of block_h, block_w
		pad_h = (-H) % block_h
		pad_w = (-W) % block_w
		H_pad = H + pad_h
		W_pad = W + pad_w

		feat = torch.cat([f4, f6], dim=-1)   # (B, H*W, 22)

		if pad_h > 0 or pad_w > 0:
			# reflect-pad in spatial dims
			feat_2d = feat.reshape(B, H, W, 22).permute(0, 3, 1, 2)   # (B, 22, H, W)
			feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
			feat    = feat_2d.permute(0, 2, 3, 1).reshape(B, H_pad * W_pad, 22)

		# Block-relative position SH (same for all blocks and images)
		sh_block = self._get_sh_block(block_h, block_w, device).to(dtype)   # (Nb, 6)

		# N block-local self-attention blocks — explicit residual after every block
		for block in self.attn_blocks:
			feat = feat + block(feat, sh_block, H_pad, W_pad, block_h, block_w)

		# Crop padding back to original LR shape
		if pad_h > 0 or pad_w > 0:
			feat = feat.reshape(B, H_pad, W_pad, 22)[:, :H, :W, :].reshape(B, H * W, 22)

		# Equivariant transpose-conv upsample to HR
		f4_hr, f6_hr, hr_shape = self.upsample(feat[..., :9], feat[..., 9:], img_shape)

		if not batched:
			f4_hr = f4_hr.squeeze(0)
			f6_hr = f6_hr.squeeze(0)

		return f4_hr, f6_hr, hr_shape


# ──────────────────────────────────────────────────────────────────────────────
class FCCAutoEncoderSRBlockAttn(FCCAutoEncoderSR):
	"""FCCAutoEncoderSR with block-local equivariant LR attention.

	Full pipeline:
	  LR quats → encode → conv_lr
	    → [N LRBlockAttentionBlocks]         (block-local attention, LR space)
	    → EquivariantTransposeConv           (upsample)
	    → conv_hr                            (equivariant spatial conv, HR space)
	    → decode → FZ-reduce

	Attention is confined to non-overlapping ~block_size×block_size LR-pixel
	blocks, making memory cost O(Nb²) per block instead of O(N_lr²) globally.

	Args:
	    num_channels:    Hidden channel multiplicity C.  8·C³ weights per block.
	    num_attn_blocks: Number of stacked LR attention blocks.
	    block_size:      Target block side length in LR pixels (default 16).
	"""

	def __init__(
		self,
		device:          str | torch.device | None = None,
		upsample_factor: int  = 4,
		num_channels:    int  = 8,
		num_attn_blocks: int  = 2,
		block_size:      int  = 16,
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
		self.upsample_conv = BlockEquivariantAttentionUpsample(
			upsample_factor=self.upsample_factor,
			num_channels=int(num_channels),
			num_attn_blocks=int(num_attn_blocks),
			block_size=int(block_size),
		)
