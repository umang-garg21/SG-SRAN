import math
import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

try:
	from models.local_iso_embedding import build_local_iso_fcc_embedding
except Exception:
	try:
		from local_iso_embedding import build_local_iso_fcc_embedding
	except Exception:
		_local_iso_path = Path(__file__).resolve().parent / "local_iso_embedding.py"
		_spec = importlib.util.spec_from_file_location(
			"_repo_local_iso_embedding",
			_local_iso_path,
		)
		if _spec is None or _spec.loader is None:
			raise ImportError(f"Could not load local iso embedding module: {_local_iso_path}")
		_mod = importlib.util.module_from_spec(_spec)
		_spec.loader.exec_module(_mod)
		build_local_iso_fcc_embedding = _mod.build_local_iso_fcc_embedding


class FCCPhysics(nn.Module):
	def __init__(self, device: str = "cpu"):
		super().__init__()
		self.device = device

		inv_sqrt_2 = 1.0 / math.sqrt(2.0)
		half = 0.5
		self.fcc_syms_inv = torch.tensor(
			[
				[1, 0, 0, 0],
				[0, -1, 0, 0],
				[0, 0, -1, 0],
				[0, 0, 0, -1],
				[inv_sqrt_2, -inv_sqrt_2, 0, 0],
				[inv_sqrt_2, 0, -inv_sqrt_2, 0],
				[inv_sqrt_2, 0, 0, -inv_sqrt_2],
				[inv_sqrt_2, inv_sqrt_2, 0, 0],
				[inv_sqrt_2, 0, inv_sqrt_2, 0],
				[inv_sqrt_2, 0, 0, inv_sqrt_2],
				[0, -inv_sqrt_2, -inv_sqrt_2, 0],
				[0, -inv_sqrt_2, 0, -inv_sqrt_2],
				[0, 0, -inv_sqrt_2, -inv_sqrt_2],
				[0, -inv_sqrt_2, inv_sqrt_2, 0],
				[0, 0, -inv_sqrt_2, inv_sqrt_2],
				[0, -inv_sqrt_2, 0, inv_sqrt_2],
				[half, -half, -half, -half],
				[half, half, half, -half],
				[half, half, -half, half],
				[half, -half, half, half],
				[half, -half, -half, half],
				[half, -half, half, -half],
				[half, half, -half, -half],
				[half, half, half, half],
			],
			dtype=torch.float32,
			device=device,
		)


class LocalIsoFCCEncoder(nn.Module):
	"""FCC encoder backed by LocalIsoCrystalEmbedding irreps features."""

	def __init__(
		self,
		device: str | torch.device = "cpu",
	):
		super().__init__()
		self.embedding = build_local_iso_fcc_embedding(
			dtype=torch.float32,
			device=device,
		).eval()
		self.irreps_a1 = self.embedding.irreps_a1
		self.irreps_full = self.embedding.irreps_full
		self.out_dim_a1 = int(self.irreps_a1.dim)
		self.out_dim_full = int(self.irreps_full.dim)

	def _to_embedding_device(self, quats_passive: torch.Tensor) -> torch.Tensor:
		quats_passive = quats_passive.to(
			device=self.embedding.group_mats.device,
			dtype=self.embedding.group_mats.dtype,
		)
		return quats_passive

	def forward_a1(self, quats_passive: torch.Tensor) -> torch.Tensor:
		quats_passive = self._to_embedding_device(quats_passive)
		return self.embedding.forward_irreps_passive(
			quats_passive,
			active_only=True,
		)

	def forward_full(self, quats_passive: torch.Tensor) -> torch.Tensor:
		quats_passive = self._to_embedding_device(quats_passive)
		return self.embedding.forward_irreps_passive(
			quats_passive,
			active_only=False,
		)

	def forward(self, quats_passive: torch.Tensor) -> torch.Tensor:
		return self.forward_a1(quats_passive)


class FastLookupLocalIsoFCCDecoder(nn.Module):
	"""Nearest-neighbor lookup decoder for local-iso FCC irreps features."""

	def __init__(
		self,
		device: torch.device,
		lookup_resolution: int = 1,
		table_chunk_size: int = 8192,
		lookup_npy_path: str | None = None,
	):
		super().__init__()
		self.device = torch.device(device)
		self.lookup_resolution = int(lookup_resolution)
		self.table_chunk_size = max(256, int(table_chunk_size))

		if self.lookup_resolution < 1:
			raise ValueError(
				f"lookup_resolution must be >= 1, got {lookup_resolution}"
			)

		self.lookup_npy_path = self._resolve_lookup_path(lookup_npy_path)
		if not self.lookup_npy_path.exists():
			raise FileNotFoundError(
				f"Local-iso lookup table not found: {self.lookup_npy_path}"
			)

		table_quats, table_feat, table_feat_norm = self._load_lookup_file()
		self.register_buffer("table_quats", table_quats)
		self.register_buffer("table_feat", table_feat)
		self.register_buffer("table_feat_norm", table_feat_norm)

	def _resolve_lookup_path(self, lookup_npy_path: str | None) -> Path:
		if lookup_npy_path is not None and str(lookup_npy_path).strip() != "":
			return Path(lookup_npy_path).expanduser().resolve()
		fname = f"local_iso_lookup_O_res{self.lookup_resolution}_irreps.npy"
		return (Path.cwd() / "symmetry_groups" / fname).resolve()

	def _load_lookup_file(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		import numpy as np

		arr = np.load(str(self.lookup_npy_path), allow_pickle=False)
		if arr.ndim != 2 or arr.shape[1] < (4 + 1 + 1):
			raise ValueError(
				f"Invalid local-iso lookup shape {arr.shape} in {self.lookup_npy_path}."
			)

		t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
		table_quats = t[:, :4]
		table_feat = t[:, 4:-1]
		table_feat_norm = t[:, -1]
		return table_quats, table_feat, table_feat_norm

	def forward(self, feat: torch.Tensor) -> torch.Tensor:
		query_feat = feat.to(torch.float32)
		query_norm = (query_feat * query_feat).sum(dim=-1, keepdim=True)

		batch_size = query_feat.shape[0]
		table_n = self.table_feat.shape[0]
		best_dist = torch.full(
			(batch_size,),
			float("inf"),
			dtype=query_feat.dtype,
			device=query_feat.device,
		)
		best_idx = torch.zeros((batch_size,), dtype=torch.long, device=query_feat.device)

		for start in range(0, table_n, self.table_chunk_size):
			end = min(start + self.table_chunk_size, table_n)
			feat_chunk = self.table_feat[start:end]
			norm_chunk = self.table_feat_norm[start:end].unsqueeze(0)
			dots = query_feat @ feat_chunk.transpose(0, 1)
			dist = query_norm + norm_chunk - 2.0 * dots

			chunk_best_dist, chunk_best_idx = torch.min(dist, dim=1)
			improved = chunk_best_dist < best_dist
			best_dist = torch.where(improved, chunk_best_dist, best_dist)
			best_idx = torch.where(improved, chunk_best_idx + start, best_idx)

		return self.table_quats[best_idx]


class EquivariantSpatialConv(nn.Module):
	"""
	Equivariant spatial convolution layer that mixes features from nearby pixels
	while preserving O(3) symmetry.

	Treats the local-iso embedding as a single irreps feature vector and mixes
	neighbourhood information via Clebsch-Gordan tensor products.
	"""

	def __init__(
		self,
		kernel_size: int = 3,
		irreps_io: Irreps | str = "1x4e + 1x6e",
	):
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2

		self.irreps_in = Irreps(irreps_io)
		self.total_dim = int(self.irreps_in.dim)

		self.tp = FullyConnectedTensorProduct(
			self.irreps_in,
			self.irreps_in,
			self.irreps_in,
			shared_weights=True,
		)
		# Learnable 3×3 spatial kernel for neighbour aggregation
		self.spatial_weights = nn.Parameter(
			torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)
		)

	def forward(
		self,
		features: torch.Tensor,
		img_shape: tuple[int, int],
	) -> torch.Tensor:
		H, W = img_shape
		batched = features.dim() == 3
		if not batched:
			features = features.unsqueeze(0)
		B = features.shape[0]
		N = features.shape[1]
		C = features.shape[-1]
		if C != self.total_dim:
			raise ValueError(f"Expected feature dim {self.total_dim}, got {C}.")
		if N != H * W:
			raise ValueError(f"Expected N={H*W} from img_shape, got N={N}.")

		# Reshape to image grid and gather neighbours via learned spatial kernel.
		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
		feat_padded = F.pad(
			feat_img,
			(self.padding, self.padding, self.padding, self.padding),
			mode="replicate",
		)
		patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
		w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
		neighbour_feats = (patches * w).sum(dim=(-1, -2))
		neighbour_flat = neighbour_feats.permute(0, 2, 3, 1).reshape(B * H * W, C)
		features_flat = features.reshape(B * H * W, C)

		out = self.tp(features_flat, neighbour_flat) + features_flat
		out = out.reshape(B, H * W, C)
		if not batched:
			out = out.squeeze(0)
		return out


class EquivariantUpsampleConv(nn.Module):
	"""
	Equivariant upsample convolution for EBSD super-resolution.

	Pipeline:
	  1. Nearest-neighbour upsample — copy each LR pixel's irreps r×r times.
	     F.interpolate(mode='nearest') applies identically to every channel,
	     so it is equivariant by construction.
	  2. SH-informed equivariant 2×2 neighbourhood aggregation at HR:
	     for each HR pixel gather its 2×2 patch, couple each neighbour's irreps
	     with the fixed even-l spherical harmonics evaluated at that kernel
	     direction, and sum → context vector.
	  3. tp(feat_self, context) → output.
	  4. Residual from the step-1 (NN upsample) output.

	Parity note:
	  This assumes even-parity feature irreps and uses even-l SH terms
	  (`1x0e + 1x2e`) for neighbourhood encoding.

	Init: both TP weights = 0  →  output equals the clean NN upsample residual.
	"""

	def __init__(
		self,
		upsample_factor: int = 4,
		irreps_feat: Irreps | str = "1x4e + 1x6e",
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)

		self.irreps_feat = Irreps(irreps_feat)
		self.total_dim = int(self.irreps_feat.dim)
		self.sh_irreps   = Irreps("1x0e + 1x2e")  # even-only SH: 6 components

		# Fixed 2×2 kernel directions in the z=0 plane (x=col, y=row).
		# Unfold order (kH, kW): (0,0), (0,1), (1,0), (1,1)
		s = 1.0 / math.sqrt(2)
		kernel_dirs = torch.tensor(
			[[-s, -s, 0.0], [+s, -s, 0.0], [-s, +s, 0.0], [+s, +s, 0.0]],
			dtype=torch.float32,
		)
		sh_kernel = o3.spherical_harmonics(self.sh_irreps, kernel_dirs, normalize=False)  # (4, 6)
		self.register_buffer("sh_kernel", sh_kernel)

		# TP aggregation: feat_j ⊗ SH_j → feat  (active CG paths: 4e⊗0e→4e, 4e⊗2e→4e/6e, 6e⊗0e→6e, 6e⊗2e→4e/6e)
		self.tp_aggregate = FullyConnectedTensorProduct(
			self.irreps_feat, self.sh_irreps, self.irreps_feat, shared_weights=True,
		)
		with torch.no_grad():
			self.tp_aggregate.weight.data.zero_()

		# TP mixing: feat_self ⊗ context → output
		self.tp = FullyConnectedTensorProduct(
			self.irreps_feat, self.irreps_feat, self.irreps_feat, shared_weights=True,
		)
		with torch.no_grad():
			self.tp.weight.data.zero_()

	def forward(
		self,
		features: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, tuple[int, int]]:
		H, W = img_shape
		r = self.upsample_factor
		C = self.total_dim
		Hr, Wr = H * r, W * r
		N = Hr * Wr

		batched = features.dim() == 3
		if not batched:
			features = features.unsqueeze(0)
		B = features.shape[0]
		if features.shape[-1] != C:
			raise ValueError(f"Expected feature dim {C}, got {features.shape[-1]}.")
		if features.shape[1] != H * W:
			raise ValueError(f"Expected N={H*W} from img_shape, got N={features.shape[1]}.")

		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
		feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode="nearest")

		feat_padded = F.pad(feat_hr, [0, 1, 0, 1], mode="replicate")
		patches = feat_padded.unfold(2, 2, 1).unfold(3, 2, 1)
		patches = patches.reshape(B, C, Hr, Wr, 4)
		patches_flat = patches.permute(0, 2, 3, 4, 1).reshape(B * N, 4, C)

		sh_exp = self.sh_kernel.unsqueeze(0).expand(B * N, -1, -1).reshape(B * N * 4, -1)
		agg_flat = self.tp_aggregate(
			patches_flat.reshape(B * N * 4, C),
			sh_exp,
		)
		context = agg_flat.reshape(B * N, 4, C).sum(dim=1)

		feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
		out_features = self.tp(feat_flat, context)
		out_features = out_features + feat_flat
		out_features = out_features.reshape(B, N, C)

		if not batched:
			out_features = out_features.squeeze(0)
		return out_features, (Hr, Wr)


class EquivariantTransposeConv(nn.Module):
	"""
	Equivariant upsampler using a learned depthwise transpose convolution.

	Pipeline:
	  1. ConvTranspose2d (stride=r, groups=C) — learned scalar upsample per
	     channel, initialized close to bilinear interpolation.  Depthwise
	     (groups=C) means no cross-channel mixing, so equivariance is preserved.
	  2. Learnable 3×3 scalar spatial aggregation at HR to gather context.
	  3. FullyConnectedTensorProduct(feat_self, context) → equivariant mixing.
	"""

	def __init__(
		self,
		kernel_size: int = 3,
		upsample_factor: int = 4,
		use_residual: bool = False,
		irreps_io: Irreps | str = "1x4e + 1x6e",
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.use_residual = bool(use_residual)

		self.irreps_io = Irreps(irreps_io)
		self.total_dim = int(self.irreps_io.dim)
		C = self.total_dim

		# Depthwise transpose conv — each feature channel gets its own
		# scalar kernel, so no cross-channel mixing (equivariance preserved)
		tp_kernel = upsample_factor + 2
		tp_pad = (tp_kernel - upsample_factor) // 2
		self.transpose_conv = nn.ConvTranspose2d(
			in_channels=C,
			out_channels=C,
			kernel_size=tp_kernel,
			stride=upsample_factor,
			padding=tp_pad,
			output_padding=0,
			groups=C,
			bias=False,
		)
		with torch.no_grad():
			self._init_bilinear()

		# Scalar 3×3 spatial aggregation at HR
		self.spatial_weights = nn.Parameter(
			torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)
		)

		# Equivariant tensor product for feature mixing at HR
		self.tp = FullyConnectedTensorProduct(
			self.irreps_io,
			self.irreps_io,
			self.irreps_io,
			shared_weights=True,
		)

	def _init_bilinear(self):
		r = self.upsample_factor
		k = self.transpose_conv.kernel_size[0]
		bilinear_1d = torch.zeros(k)
		center = (k - 1) / 2.0
		for i in range(k):
			bilinear_1d[i] = max(0, 1 - abs(i - center) / r)
		bilinear_2d = bilinear_1d.unsqueeze(1) * bilinear_1d.unsqueeze(0)
		bilinear_2d = bilinear_2d / bilinear_2d.sum()
		self.transpose_conv.weight.data[:] = bilinear_2d.unsqueeze(0).unsqueeze(0)

	def forward(
		self,
		features: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, tuple[int, int]]:
		H, W = img_shape
		r = self.upsample_factor
		Hr, Wr = H * r, W * r

		batched = features.dim() == 3
		if not batched:
			features = features.unsqueeze(0)
		B = features.shape[0]

		C = features.shape[-1]
		if C != self.total_dim:
			raise ValueError(f"Expected feature dim {self.total_dim}, got {C}.")
		if features.shape[1] != H * W:
			raise ValueError(f"Expected N={H*W} from img_shape, got N={features.shape[1]}.")

		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)

		feat_hr = self.transpose_conv(feat_img)
		feat_hr = feat_hr[:, :, :Hr, :Wr]

		feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
		patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
		w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
		context_img = (patches * w).sum(dim=(-1, -2))

		N = Hr * Wr
		feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
		context_flat = context_img.permute(0, 2, 3, 1).reshape(B * N, C)

		out = self.tp(feat_flat, context_flat)

		if self.use_residual:
			out = out + feat_flat

		out = out.reshape(B, N, C)
		if not batched:
			out = out.squeeze(0)
		return out, (Hr, Wr)


class EquivariantAttentionUpsample(nn.Module):
	"""
	Equivariant attention-based upsampler for EBSD super-resolution.

	Pipeline:
	  1. Nearest-neighbour upsample — equivariant by construction (F.interpolate
	     with mode='nearest' applies identically to every irrep channel).
	  2. Extract a k×k local neighbourhood at each HR pixel.
	  3. Compute O(3)-invariant attention scores per neighbour:
	       score_j = TP_score(feat_i, feat_j)  →  1x0e  (invariant scalar).
	  4. Pre-compute fixed SH encodings of the k×k spatial kernel directions
	     (even-l only: 1x0e + 1x2e = 6 components for even-parity irreps).
	  5. Transform each neighbour with the SH direction:
	       val_j = TP_val(feat_j, sh_j)  →  irreps_feat  (equivariant).
	  6. Context = Σ_j softmax(score_j) · val_j  (weighted sum — equivariant).
	  7. Output = TP_out(feat_i, context) + feat_i  (equivariant + residual).

	Equivariance proof sketch:
	  • TP_score maps to 1x0e → rotation-invariant attention weights.
	  • softmax of invariants is invariant.
	  • TP_val(feat_j, sh_j): sh_j is a fixed spatial encoding (does NOT rotate
	    with the crystal orientation), so the output transforms like feat_j.
	  • A linearly weighted sum of equivariant vectors is equivariant.
	  • TP_out maps two equivariant inputs → equivariant output.

	Init: TP_val and TP_out weights = 0  →  output equals the clean NN upsample
	residual at epoch 0 (safe training start).
	"""

	def __init__(
		self,
		upsample_factor: int = 4,
		k_size: int = 3,
		irreps_feat: Irreps | str = "1x4e + 1x6e",
	):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.k_size = int(k_size)
		self.padding = k_size // 2
		self.K = k_size * k_size  # neighbourhood size

		self.irreps_feat = Irreps(irreps_feat)
		self.total_dim = int(self.irreps_feat.dim)
		self.sh_irreps   = Irreps("1x0e + 1x2e")   # 6 components, even parity only

		# Build fixed SH encodings of the k×k kernel directions in the z=0 plane.
		# Convention: x = column offset, y = row offset, z = 0.
		# unfold ordering for (kH, kW): row-major → k = kH*k_size + kW.
		kH_range = torch.arange(k_size, dtype=torch.float32) - k_size // 2
		kW_range = torch.arange(k_size, dtype=torch.float32) - k_size // 2
		grid_H, grid_W = torch.meshgrid(kH_range, kW_range, indexing="ij")
		ky = grid_H.reshape(-1)   # row offset → y
		kx = grid_W.reshape(-1)   # col offset → x
		kz = torch.zeros_like(kx)
		dirs = torch.stack([kx, ky, kz], dim=-1)  # (K, 3)

		# Center direction is (0,0,0) — undefined; use z-axis so SH is well-defined.
		center = (ky == 0) & (kx == 0)
		dirs[center] = torch.tensor([0.0, 0.0, 1.0])

		# Normalise non-center directions.
		norms = dirs.norm(dim=-1, keepdim=True)
		not_center = norms.squeeze(-1) > 1e-8
		dirs[not_center] = dirs[not_center] / norms[not_center]

		sh_kernel = o3.spherical_harmonics(self.sh_irreps, dirs, normalize=False)  # (K, 6)
		self.register_buffer("sh_kernel", sh_kernel)

		# TP 1: invariant attention score  feat ⊗ feat → 1x0e.
		self.tp_score = FullyConnectedTensorProduct(
			self.irreps_feat, self.irreps_feat, Irreps("1x0e"),
			shared_weights=True,
		)

		# TP 2: equivariant value encoding  feat ⊗ sh → feat.
		self.tp_val = FullyConnectedTensorProduct(
			self.irreps_feat, self.sh_irreps, self.irreps_feat,
			shared_weights=True,
		)
		with torch.no_grad():
			self.tp_val.weight.data.zero_()

		# TP 3: equivariant output mixing  feat_self ⊗ context → output
		self.tp_out = FullyConnectedTensorProduct(
			self.irreps_feat, self.irreps_feat, self.irreps_feat,
			shared_weights=True,
		)
		with torch.no_grad():
			self.tp_out.weight.data.zero_()

	def forward(
		self,
		features: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, tuple[int, int]]:
		H, W = img_shape
		r = self.upsample_factor
		Hr, Wr = H * r, W * r
		N = Hr * Wr
		K = self.K
		C = self.total_dim

		batched = features.dim() == 3
		if not batched:
			features = features.unsqueeze(0)
		B = features.shape[0]
		if features.shape[-1] != C:
			raise ValueError(f"Expected feature dim {C}, got {features.shape[-1]}.")
		if features.shape[1] != H * W:
			raise ValueError(f"Expected N={H*W} from img_shape, got N={features.shape[1]}.")

		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)

		feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode="nearest")

		feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)

		# Extract kxk patches around each HR pixel.
		feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
		patches = feat_padded.unfold(2, self.k_size, 1).unfold(3, self.k_size, 1)
		patches = patches.reshape(B, C, Hr, Wr, K)
		patches_flat = patches.permute(0, 2, 3, 4, 1).reshape(B * N, K, C)

		queries = feat_flat.unsqueeze(1).expand(B * N, K, C).reshape(B * N * K, C)
		keys = patches_flat.reshape(B * N * K, C)
		scores = self.tp_score(queries.contiguous(), keys)
		scores = scores.reshape(B * N, K)
		attn = torch.softmax(scores, dim=1)

		sh_exp = self.sh_kernel.unsqueeze(0).expand(B * N, K, -1).reshape(B * N * K, 6)
		vals = self.tp_val(keys, sh_exp.contiguous())
		vals = vals.reshape(B * N, K, C)

		context = (attn.unsqueeze(-1) * vals).sum(dim=1)

		out = self.tp_out(feat_flat, context) + feat_flat
		out = out.reshape(B, N, C)
		if not batched:
			out = out.squeeze(0)

		return out, (Hr, Wr)


class FCCAutoEncoder(nn.Module):
	"""
	Local-iso FCC autoencoder wrapper.

	Pipeline:
	  1) encode passive quaternion -> local-iso irreps feature vector
	  2) optional spatial equivariant feature mixing
	  3) lookup decode feature vector -> passive quaternion
	  4) reduce to FCC fundamental zone
	"""

	def __init__(
		self,
		device: str | torch.device | None = None,
		grid_res: int = 100_000,
		decoder_backend: str = "lookup",
		decoder_config: dict[str, Any] | None = None,
		**decoder_kwargs: Any,
	):
		super().__init__()
		if device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.physics = FCCPhysics(str(self.device))
		dcfg = dict(decoder_config or {})
		dcfg.update(decoder_kwargs)

		def dget(key: str, default: Any) -> Any:
			return dcfg.get(key, default)

		self.encoder = LocalIsoFCCEncoder(
			device=self.device,
		)
		self.irreps_a1 = self.encoder.irreps_a1
		self.irreps_full = self.encoder.irreps_full
		self.feature_dim_a1 = int(self.encoder.out_dim_a1)
		self.feature_dim = int(self.encoder.out_dim_full)

		# First e3nn layer requested: irreps_a1 -> irreps_full.
		self.lift_layer = o3.Linear(self.irreps_a1, self.irreps_full)
		self.feature_irreps = self.irreps_full

		self.conv_layer = EquivariantSpatialConv(
			kernel_size=3,
			irreps_io=self.feature_irreps,
		)

		backend = str(decoder_backend).lower()
		self.decoder_backend = backend

		if backend in {"lookup", "fast_lookup", "local_iso_lookup", "cubochoric_lookup"}:
			self.decoder = FastLookupLocalIsoFCCDecoder(
				device=self.device,
				lookup_resolution=int(dget("decoder_lookup_resolution", 1)),
				table_chunk_size=int(dget("decoder_lookup_chunk_size", 8192)),
				lookup_npy_path=dget("decoder_lookup_npy_path", None),
			)
			table_dim = int(self.decoder.table_feat.shape[-1])
			if table_dim != self.feature_dim:
				raise ValueError(
					f"Lookup feature dim ({table_dim}) does not match irreps_full dim "
					f"({self.feature_dim})."
				)
		else:
			raise ValueError(
				"local-iso autoencoder supports lookup decoder backends only "
				"('lookup', 'fast_lookup', 'local_iso_lookup')."
			)

	@staticmethod
	def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
		return quats / norm

	def _load_learnable_decoder_checkpoint(self, ckpt_path: str, strict: bool = True) -> None:
		path = Path(ckpt_path).expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(f"Learnable decoder checkpoint not found: {path}")

		try:
			blob = torch.load(str(path), map_location=self.device, weights_only=True)
		except TypeError:
			blob = torch.load(str(path), map_location=self.device)

		if isinstance(blob, dict) and "decoder_state_dict" in blob:
			state_dict = blob["decoder_state_dict"]
		elif isinstance(blob, dict):
			state_dict = blob
		else:
			raise ValueError(f"Unsupported learnable decoder checkpoint format: {path}")

		load_result = self.decoder.load_state_dict(state_dict, strict=bool(strict))
		if hasattr(load_result, "missing_keys") and len(load_result.missing_keys) > 0:
			print(
				"[FCCAutoEncoder] learnable decoder missing keys: "
				f"{load_result.missing_keys[:8]}"
			)
		if hasattr(load_result, "unexpected_keys") and len(load_result.unexpected_keys) > 0:
			print(
				"[FCCAutoEncoder] learnable decoder unexpected keys: "
				f"{load_result.unexpected_keys[:8]}"
			)

	@staticmethod
	def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
		w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
		w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
		return torch.stack(
			[
				w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
				w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
				w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
				w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
			],
			dim=1,
		)

	def encode_a1(self, quats: torch.Tensor) -> torch.Tensor:
		return self.encoder.forward_a1(quats)

	def encode_full_target(self, quats: torch.Tensor) -> torch.Tensor:
		return self.encoder.forward_full(quats)

	def lift_to_full(self, features_a1: torch.Tensor) -> torch.Tensor:
		return self.lift_layer(features_a1)

	def encode(self, quats: torch.Tensor) -> torch.Tensor:
		features_a1 = self.encode_a1(quats)
		return self.lift_to_full(features_a1)

	def feature_loss(
		self,
		quats: torch.Tensor,
		img_shape: tuple[int, int],
		normalize_input: bool = True,
	) -> torch.Tensor:
		quats = quats.to(self.device)
		if normalize_input:
			quats = self._normalize_quaternions(quats)
		with torch.no_grad():
			feat_a1 = self.encode_a1(quats).detach()
			feat_tgt = self.encode_full_target(quats).detach()
		feat_seed = self.lift_to_full(feat_a1)
		feat_out = self.conv_layer(feat_seed, img_shape)
		return F.mse_loss(feat_out, feat_tgt)
	
	def decode(self, features: torch.Tensor) -> torch.Tensor:
		q_bunge = self.decoder(features)
		return self.reduce_to_fz(q_bunge)

	def reduce_to_fz(
		self,
		quats: torch.Tensor,
		return_op_map: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
		quats = self._normalize_quaternions(quats)
		batch_size = quats.shape[0]

		q_expanded = quats.unsqueeze(1).expand(-1, 24, -1)
		syms = self.physics.fcc_syms_inv.unsqueeze(0).expand(batch_size, -1, -1)

		q_flat = q_expanded.reshape(-1, 4)
		s_flat = syms.reshape(-1, 4)
		# Bunge convention: s⁻¹ ⊗ q  (left orbit under crystal symmetry)
		# fcc_syms_inv already stores the inverse symmetries, so use directly
		fam = self.quat_mul(s_flat, q_flat).view(batch_size, 24, 4)
		fam = self._normalize_quaternions(fam.reshape(-1, 4)).view(batch_size, 24, 4)

		w_abs = fam[..., 0].abs()
		best_idx = torch.argmax(w_abs, dim=1)
		batch_idx = torch.arange(batch_size, device=quats.device)
		q_fz = fam[batch_idx, best_idx]
		q_fz = torch.where(q_fz[:, :1] < 0, -q_fz, q_fz)
		q_fz = self._normalize_quaternions(q_fz)
		if return_op_map:
			return q_fz, best_idx
		return q_fz
	
	def forward(
		self,
		quats: torch.Tensor,
		img_shape: tuple[int, int] | None = None,
		normalize_input: bool = True,
	) -> torch.Tensor:
		quats = quats.to(self.device)
		if quats.dim() != 2 or quats.shape[-1] != 4:
			raise ValueError(
				f"FCCAutoEncoder expects (N,4), got {tuple(quats.shape)}"
			)
		if normalize_input:
			quats = self._normalize_quaternions(quats)
		features = self.encode(quats)
		if img_shape is not None:
			features = self.conv_layer(features, img_shape)
		return self.decode(features)
	
	@staticmethod
	def _sample_fz_quaternions(
		resolution: int = 1,
		method: str = "cubochoric",
		device: torch.device | None = None,
	) -> torch.Tensor:
		try:
			import numpy as np
			from orix.quaternion import symmetry
			from orix.sampling import get_sample_fundamental
		except Exception as exc:
			raise ImportError(
				"FZ sampling requires `orix` and `numpy` to be installed."
			) from exc

		rot = get_sample_fundamental(
			int(resolution),
			point_group=symmetry.Oh,
			method=str(method),
		)

		raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
		if raw.ndim != 2:
			raw = raw.reshape(-1, 4)
		if raw.shape[-1] != 4 and raw.shape[0] == 4:
			raw = raw.T
		if raw.shape[-1] != 4:
			raise ValueError(f"Unexpected sampled quaternion shape: {tuple(raw.shape)}")

		q = torch.as_tensor(raw, dtype=torch.float32)
		q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
		q = torch.where(q[..., :1] < 0, -q, q)
		if device is not None:
			q = q.to(device)
		return q

	@torch.no_grad()
	def export_fz_encoding_table(
		self,
		csv_path: str,
		resolution: int = 3,
		sampling_method: str = "cubochoric",
		include_decode: bool = True,
		binary_path: str | None = None,
		decode_chunk_size: int = 4096,
	) -> dict[str, Any]:
		"""
		Export FZ quaternion samples and their encoded/decoded representations.

		Saved columns include:
		- q_w, q_x, q_y, q_z
		- features_a1_*
		- features_full_*
		- features_lifted_*
		- (optional) q_dec_* and decode quality metrics
		"""
		quats = self._sample_fz_quaternions(
			resolution=resolution,
			method=sampling_method,
			device=self.device,
		)
		quats = self._normalize_quaternions(quats)
		features_a1 = self.encode_a1(quats)
		features_full = self.encode_full_target(quats)
		features_lifted = self.lift_to_full(features_a1)

		payload: dict[str, Any] = {
			"quats": quats.detach().cpu(),
			"features_a1": features_a1.detach().cpu(),
			"features_full": features_full.detach().cpu(),
			"features_lifted": features_lifted.detach().cpu(),
			"resolution": int(resolution),
			"sampling_method": str(sampling_method),
			"num_rows": int(quats.shape[0]),
		}

		if binary_path is not None:
			bin_file = Path(binary_path)
			bin_file.parent.mkdir(parents=True, exist_ok=True)
			torch.save(payload, str(bin_file))

		return payload


class FCCAutoEncoderSR(FCCAutoEncoder):
	"""
	FCC super-resolution autoencoder.

	Extends FCCAutoEncoder with an upsample stage and a second conv layer so
	that the full pipeline operates at HR resolution:

	  LR quats → encode → conv_lr → upsample → conv_hr → decode

	Training objective (feature-space, fully differentiable):
	  MSE(feat_hr_model, feat_hr_enc)
	where `feat_hr_enc` is the local-iso irreps feature vector of the
	ground-truth HR quaternions (encoder frozen via no_grad).

	The encoder and decoder are both non-differentiable / frozen during
	training.  Only conv_lr, upsample_conv, and conv_hr are trained.
	"""

	def __init__(
		self,
		device: str | torch.device | None = None,
		upsample_factor: int = 4,
		upsample_residual: bool = False,
		upsampler: str = "conv",
		decoder_backend: str = "lookup",
		decoder_config: dict | None = None,
		**decoder_kwargs,
	):
		# Builds physics, encoder, conv_layer (→ used as conv_lr), decoder
		super().__init__(
			device=device,
			decoder_backend=decoder_backend,
			decoder_config=decoder_config,
			**decoder_kwargs,
		)
		self.upsample_factor = int(upsample_factor)
		# self.conv_layer inherited from FCCAutoEncoder serves as the LR conv
		upsampler_type = str(upsampler).lower()
		if upsampler_type == "attention":
			self.upsample_conv = EquivariantAttentionUpsample(
				upsample_factor=self.upsample_factor,
				irreps_feat=self.feature_irreps,
			)
		else:
			self.upsample_conv = EquivariantTransposeConv(
				upsample_factor=self.upsample_factor,
				use_residual=bool(upsample_residual),
				irreps_io=self.feature_irreps,
			)
		self.conv_hr = EquivariantSpatialConv(
			kernel_size=3,
			irreps_io=self.feature_irreps,
		)

	def feature_loss_sr(
		self,
		lr_quats: torch.Tensor,
		hr_quats: torch.Tensor,
		lr_shape: tuple[int, int],
		normalize_input: bool = True,
	) -> torch.Tensor:
		"""SR training loss in local-iso feature space."""
		lr_quats = lr_quats.to(self.device)
		hr_quats = hr_quats.to(self.device)

		batched = lr_quats.dim() == 3
		if batched:
			B = lr_quats.shape[0]
			lr_flat = lr_quats.reshape(-1, 4)
			hr_flat = hr_quats.reshape(-1, 4)
		else:
			B = 1
			lr_flat = lr_quats
			hr_flat = hr_quats

		if normalize_input:
			lr_flat = self._normalize_quaternions(lr_flat)
			hr_flat = self._normalize_quaternions(hr_flat)

		with torch.no_grad():
			feat_lr_a1_flat = self.encode_a1(lr_flat).detach()
			feat_hr_tgt_flat = self.encode_full_target(hr_flat).detach()
		feat_lr_seed_flat = self.lift_to_full(feat_lr_a1_flat)
		feat_dim = int(feat_lr_seed_flat.shape[-1])

		if batched:
			feat_lr = feat_lr_seed_flat.reshape(B, -1, feat_dim)
			feat_hr_tgt = feat_hr_tgt_flat.reshape(B, -1, feat_dim)
		else:
			feat_lr = feat_lr_seed_flat
			feat_hr_tgt = feat_hr_tgt_flat

		feat_conv = self.conv_layer(feat_lr, lr_shape)
		feat_up, hr_shape = self.upsample_conv(feat_conv, lr_shape)
		feat_hr = self.conv_hr(feat_up, hr_shape)

		return F.mse_loss(feat_hr, feat_hr_tgt)

	def forward_sr(
		self,
		lr_quats: torch.Tensor,
		lr_shape: tuple[int, int],
		normalize_input: bool = True,
	) -> torch.Tensor:
		"""
		Inference: LR quaternions → HR-resolution quaternions.

		Args:
		    lr_quats: (H_lr*W_lr, 4)
		    lr_shape: (H_lr, W_lr)
		Returns:
		    (H_hr*W_hr, 4)  FZ-reduced quaternions at HR resolution
		"""
		lr_quats = lr_quats.to(self.device)
		if normalize_input:
			lr_quats = self._normalize_quaternions(lr_quats)
		feat_lr = self.encode(lr_quats)
		feat_conv = self.conv_layer(feat_lr, lr_shape)
		feat_up, hr_shape = self.upsample_conv(feat_conv, lr_shape)
		feat_hr = self.conv_hr(feat_up, hr_shape)
		return self.decode(feat_hr)

	def forward(
		self,
		quats: torch.Tensor,
		img_shape: tuple[int, int] | None = None,
		normalize_input: bool = True,
	) -> torch.Tensor:
		"""Forward pass. When img_shape is given treats quats as LR and runs
		the full SR pipeline; otherwise falls back to the base autoencoder."""
		if img_shape is not None:
			return self.forward_sr(quats, lr_shape=img_shape, normalize_input=normalize_input)
		return super().forward(quats, img_shape=None, normalize_input=normalize_input)
