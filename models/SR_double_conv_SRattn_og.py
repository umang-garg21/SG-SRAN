
class AttentionBlock(nn.Module):

	def __init__(self, irreps_in: Irreps, num_channels: int = 8, tp_out_chunk_size: int | None = None,):
		super().__init__()
		C = int(num_channels)
		self.irreps_feat = Irreps(irreps_in)
		assert self.irreps_feat == self.irreps_feat.regroup(), "irreps_in must be regrouped (same-l terms contiguous); call Irreps(irreps_in).regroup() before passing"
		# Hidden irreps: scale each term's multiplicity by C
		self.irreps_h = Irreps([(mul * C, ir) for mul, ir in self.irreps_feat])
		self.tp_out_chunk_size = tp_out_chunk_size

		# One learnable log-scale per unique l, init: 1/√(2l+1)
		ls = sorted({ir.l for _, ir in self.irreps_feat})
		self.attn_ls = ls
		self.log_scales = nn.ParameterList([
			nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(float(2 * l + 1)))))
			for l in ls
		])
		# Pre-compute slices into feat dim grouped by l
		l_to_segs: dict[int, list[tuple[int, int]]] = {l: [] for l in ls}
		offset = 0
		for mul, ir in self.irreps_feat:
			dim = mul * ir.dim
			l_to_segs[ir.l].append((offset, offset + dim))
			offset += dim
		# Store as list aligned with attn_ls: (start, end) spanning all segs for that l
		self.attn_slices: list[tuple[int, int]] = [
			(segs[0][0], segs[-1][1]) for segs in (l_to_segs[l] for l in ls)
		]

		# Pairwise position bias via inner product of block-position SH features
		self.pos_bias = nn.Linear(1, 1, bias=True)
		nn.init.zeros_(self.pos_bias.weight)
		nn.init.zeros_(self.pos_bias.bias)

		# Channel expansion: irreps_feat → irreps_h
		self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)

		# Output mix in hidden space: h ⊗ ctx → h
		self.tp_out = FullyConnectedTensorProduct(
			self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True,
		)

		# Channel contraction + zero-init → block returns zero delta at epoch 0
		self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
		with torch.no_grad():
			self.lin_out.weight.data.zero_()

	def forward(
		self,
		feat:     torch.Tensor,
		sh_block: torch.Tensor,
		H:        int,
		W:        int,
		block_h:  int,
		block_w:  int,
	) -> torch.Tensor:
		"""
		Args:
		    feat:     (B, H*W, C_feat)  HR features (H, W already padded).
		    sh_block: (Nb, SH_dim)      SH position features for block pixels.
		    H, W:     (padded) HR spatial dims. Must be multiples of block_h, block_w.
		    block_h, block_w: block size in pixels.
		Returns:
		    (B, H*W, C_feat)  attention delta.
		"""
		B, N, C_feat = feat.shape
		num_bh = H // block_h
		num_bw = W // block_w
		Nb     = block_h * block_w
		Bb     = B * num_bh * num_bw
		dtype  = feat.dtype

		# Partition: (B, H*W, C_feat) → (Bb, Nb, C_feat)
		feat_blocks = (
			feat.reshape(B, num_bh, block_h, num_bw, block_w, C_feat)
			    .permute(0, 1, 3, 2, 4, 5)
			    .reshape(Bb, Nb, C_feat)
		)

		# ── O(3)-invariant attention scores (block-local) ─────────────────────
		# Sum per-l: s_l * normalize(feat_l) @ normalize(feat_l).T
		scores = torch.zeros(Bb, Nb, Nb, dtype=dtype, device=feat.device)
		for (start, end), log_s in zip(self.attn_slices, self.log_scales):
			s   = torch.exp(log_s)
			f_n = F.normalize(feat_blocks[..., start:end], dim=-1)
			scores = scores + s * torch.bmm(f_n, f_n.transpose(-2, -1))

		# Pairwise position bias via inner product of block-pixel SH features
		pb     = self.pos_bias((sh_block @ sh_block.T).unsqueeze(-1)).squeeze(-1)  # (Nb, Nb)
		scores = scores + pb.unsqueeze(0)                                          # (Bb, Nb, Nb)

		# Float32 softmax for numerical stability under AMP
		attn = torch.softmax(scores.float(), dim=-1).to(dtype)   # (Bb, Nb, Nb)

		# ── Equivariant value / output mix ────────────────────────────────────
		feat_flat = feat_blocks.reshape(Bb * Nb, C_feat)
		h   = self.lin_in(feat_flat).reshape(Bb, Nb, -1)
		ctx = torch.bmm(attn, h)                                  # (Bb, Nb, Ch)

		h_flat   = h.reshape(Bb * Nb, -1)
		ctx_flat = ctx.reshape(Bb * Nb, -1)
		if self.tp_out_chunk_size is not None:
			chunks = h_flat.split(self.tp_out_chunk_size, dim=0)
			ctx_chunks = ctx_flat.split(self.tp_out_chunk_size, dim=0)
			h_out_flat = torch.cat([self.tp_out(hc, cc) for hc, cc in zip(chunks, ctx_chunks)], dim=0)
		else:
			h_out_flat = self.tp_out(h_flat, ctx_flat)
		h_out = h_out_flat.reshape(Bb, Nb, -1)

		delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C_feat)

		# Reassemble: (Bb, Nb, C_feat) → (B, H*W, C_feat)
		delta = (
			delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C_feat)
			             .permute(0, 1, 3, 2, 4, 5)
			             .reshape(B, H * W, C_feat)
		)
		return delta

