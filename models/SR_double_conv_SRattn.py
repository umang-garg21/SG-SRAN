import math
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

from models.SR_grain_attn import LRBlockAttentionBlock


def wigner_D_cuda(
	l: int,
	alpha: torch.Tensor,
	beta: torch.Tensor,
	gamma: torch.Tensor,
) -> torch.Tensor:
	"""CUDA-compatible wrapper for e3nn's wigner_D function."""
	alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
	device = alpha.device

	alpha = alpha[..., None, None] % (2 * math.pi)
	beta = beta[..., None, None] % (2 * math.pi)
	gamma = gamma[..., None, None] % (2 * math.pi)

	X = o3._wigner.so3_generators(l).to(device)
	return (
		torch.matrix_exp(alpha * X[1])
		@ torch.matrix_exp(beta * X[0])
		@ torch.matrix_exp(gamma * X[1])
	)


class FCCPhysics(nn.Module):
	def __init__(self, device: str = "cpu"):
		super().__init__()
		self.device = device

		self.s4 = torch.zeros(9, device=device)
		self.s4[4] = 0.7638
		self.s4[8] = 0.6455

		self.s6 = torch.zeros(13, device=device)
		self.s6[6] = 0.3536
		self.s6[10] = -0.9354

		inv_sqrt_2 = 1 / math.sqrt(2)
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

		# self.fcc_syms = torch.tensor(
		# 	[
		# 		[1, 0, 0, 0],
		# 		[0, 1, 0, 0],
		# 		[0, 0, 1, 0],
		# 		[0, 0, 0, 1],
		# 		[inv_sqrt_2, inv_sqrt_2, 0, 0],
		# 		[inv_sqrt_2, 0, inv_sqrt_2, 0],
		# 		[inv_sqrt_2, 0, 0, inv_sqrt_2],
		# 		[inv_sqrt_2, -inv_sqrt_2, 0, 0],
		# 		[inv_sqrt_2, 0, -inv_sqrt_2, 0],
		# 		[inv_sqrt_2, 0, 0, -inv_sqrt_2],
		# 		[0, inv_sqrt_2, inv_sqrt_2, 0],
		# 		[0, inv_sqrt_2, 0, inv_sqrt_2],
		# 		[0, 0, inv_sqrt_2, inv_sqrt_2],
		# 		[0, inv_sqrt_2, -inv_sqrt_2, 0],
		# 		[0, 0, inv_sqrt_2, -inv_sqrt_2],
		# 		[0, inv_sqrt_2, 0, -inv_sqrt_2],
		# 		[half, half, half, half],
		# 		[half, -half, -half, half],
		# 		[half, -half, half, -half],
		# 		[half, half, -half, -half],
		# 		[half, half, half, -half],
		# 		[half, half, -half, half],
		# 		[half, -half, half, half],
		# 		[half, -half, -half, -half],
		# 	],
		# 	dtype=torch.float32,
		# 	device=device,
		# )


class FCCEncoder(nn.Module):
	def __init__(self, physics: FCCPhysics):
		super().__init__()
		self.physics = physics

	def forward(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		#TODO: check passive vs active e3nn convention and invert quaternions if needed
		R = o3.quaternion_to_matrix(quats)
		alpha, beta, gamma = o3.matrix_to_angles(R)

		D4 = wigner_D_cuda(4, alpha, beta, gamma)
		D6 = wigner_D_cuda(6, alpha, beta, gamma)
		f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
		f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)

		return f4, f6


class OptimizingFCCDecoder(nn.Module):
	"""
	Decode (f4, f6) -> quaternion by directly minimizing:
	  ||D4(q)s4 - f4||^2 + w6 * ||D6(q)s6 - f6||^2

	Uses multi-start Adam over quaternions and picks the best candidate.
	"""

	def __init__(
		self,
		physics: FCCPhysics,
		num_starts: int = 6,
		steps: int = 25,
		lr: float = 0.08,
		w6: float = 0.5,
		eps: float = 1e-12,
		early_stop_tol: float = 1e-6,
		early_stop_patience: int = 3,
		min_steps: int = 6,
		log_optimization: bool = False,
		log_every: int = 1,
	):
		super().__init__()
		self.physics = physics
		self.num_starts = int(num_starts)
		self.steps = int(steps)
		self.lr = float(lr)
		self.w6 = float(w6)
		self.eps = float(eps)
		self.early_stop_tol = float(early_stop_tol)
		self.early_stop_patience = int(early_stop_patience)
		self.min_steps = int(min_steps)
		self.log_optimization = bool(log_optimization)
		self.log_every = max(1, int(log_every))
		self.last_optimization_trace: dict[str, Any] | None = None

		self.register_buffer("s4", physics.s4.clone())
		self.register_buffer("s6", physics.s6.clone())

	@staticmethod
	def _normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
		norm = q.norm(dim=-1, keepdim=True)
		qn = q / norm.clamp_min(eps)

		bad = norm.squeeze(-1) < eps
		if bad.any():
			qn = qn.clone()
			qn[bad] = qn.new_tensor([1.0, 0.0, 0.0, 0.0])
		return qn

	@staticmethod
	def _fix_sign(q: torch.Tensor) -> torch.Tensor:
		return torch.where(q[..., :1] < 0, -q, q)

	def _loss(
		self,
		q: torch.Tensor,
		f4_tgt: torch.Tensor,
		f6_tgt: torch.Tensor,
	) -> torch.Tensor:
		R = o3.quaternion_to_matrix(q)
		alpha, beta, gamma = o3.matrix_to_angles(R)

		D4 = wigner_D_cuda(4, alpha, beta, gamma)
		D6 = wigner_D_cuda(6, alpha, beta, gamma)

		f4_pred = torch.einsum("bij,j->bi", D4, self.s4)
		f6_pred = torch.einsum("bij,j->bi", D6, self.s6)

		l4 = (f4_pred - f4_tgt).pow(2).mean(dim=-1)
		l6 = (f6_pred - f6_tgt).pow(2).mean(dim=-1)
		return l4 + self.w6 * l6

	@torch.no_grad()
	def _init_quats(self, bsz: int, device: torch.device) -> torch.Tensor:
		k = self.num_starts
		q0 = torch.zeros((bsz, k, 4), device=device, dtype=torch.float32)
		q0[..., 0] = 1.0
		if k > 1:
			q_rand = torch.randn((bsz, k - 1, 4), device=device, dtype=torch.float32)
			q_rand = self._normalize_quat(q_rand, self.eps)
			q0[:, 1:, :] = q_rand
		q0 = self._normalize_quat(q0, self.eps)
		q0 = self._fix_sign(q0)
		return q0

	def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		device = f4.device
		bsz = f4.shape[0]
		k = self.num_starts

		f4_tgt = f4.detach().to(torch.float32)
		f6_tgt = f6.detach().to(torch.float32)

		with torch.no_grad():
			q_init = self._init_quats(bsz, device)

		f4_rep = f4_tgt[:, None, :].expand(bsz, k, -1).reshape(bsz * k, -1)
		f6_rep = f6_tgt[:, None, :].expand(bsz, k, -1).reshape(bsz * k, -1)

		with torch.enable_grad():
			u = nn.Parameter(q_init.clone())
			opt = torch.optim.Adam([u], lr=self.lr)
			best_loss = float("inf")
			stale_steps = 0
			loss_history: list[float] = []

			for step_idx in range(self.steps):
				opt.zero_grad(set_to_none=True)
				q = self._normalize_quat(u, self.eps)
				q = self._fix_sign(q)
				q_flat = q.reshape(bsz * k, 4)
				loss = self._loss(q_flat, f4_rep, f6_rep).mean()
				loss.backward()
				opt.step()

				loss_val = float(loss.detach().item())
				loss_history.append(loss_val)
				if self.log_optimization and (
					step_idx == 0
					or (step_idx + 1) % self.log_every == 0
					or step_idx == self.steps - 1
				):
					print(
						f"[OptimizingFCCDecoder] step {step_idx + 1}/{self.steps} "
						f"loss={loss_val:.6e} best={min(best_loss, loss_val):.6e}"
					)
				if best_loss - loss_val > self.early_stop_tol:
					best_loss = loss_val
					stale_steps = 0
				else:
					stale_steps += 1

				if (
					self.early_stop_patience > 0
					and stale_steps >= self.early_stop_patience
					and step_idx + 1 >= self.min_steps
				):
					if self.log_optimization:
						print(
							f"[OptimizingFCCDecoder] early stop at step {step_idx + 1} "
							f"(patience={self.early_stop_patience}, tol={self.early_stop_tol})"
						)
					break

		with torch.no_grad():
			q = self._normalize_quat(u, self.eps)
			q = self._fix_sign(q)
			q_flat = q.reshape(bsz * k, 4)
			loss_vec = self._loss(q_flat, f4_rep, f6_rep).view(bsz, k)

			best_k = torch.argmin(loss_vec, dim=1)
			batch_idx = torch.arange(bsz, device=device)
			q_best = q[batch_idx, best_k, :]
			q_best = self._normalize_quat(q_best, self.eps)
			q_best = self._fix_sign(q_best)
			final_best_loss = float(loss_vec[batch_idx, best_k].mean().item())
			self.last_optimization_trace = {
				"steps_run": len(loss_history),
				"loss_history": loss_history,
				"final_mean_best_loss": final_best_loss,
			}
			if self.log_optimization:
				print(
					f"[OptimizingFCCDecoder] finished steps={len(loss_history)} "
					f"final_mean_best_loss={final_best_loss:.6e}"
				)
			return q_best


class CubochoricOptimizingFCCDecoder(OptimizingFCCDecoder):
	"""
	Optimizing decoder with cubochoric starts (from ORIX fundamental-zone sampling).

	This keeps the same optimization objective as ``OptimizingFCCDecoder`` but
	replaces random initialization with cubochoric quaternion samples.
	"""

	def __init__(
		self,
		physics: FCCPhysics,
		cubochoric_resolution: int = 3,
		**kwargs: Any,
	):
		super().__init__(physics, **kwargs)
		self.cubochoric_resolution = int(cubochoric_resolution)
		if self.cubochoric_resolution < 1:
			raise ValueError(
				f"cubochoric_resolution must be >= 1, got {cubochoric_resolution}"
			)
		cubochoric_quats = self._build_cubochoric_quat_table(
			resolution=self.cubochoric_resolution,
			device=torch.device(physics.device),
		)
		self.register_buffer("cubochoric_quats", cubochoric_quats)

	@staticmethod
	def _build_cubochoric_quat_table(
		resolution: int,
		device: torch.device,
	) -> torch.Tensor:
		try:
			import numpy as np
			from orix.quaternion import symmetry
			from orix.sampling import get_sample_fundamental
		except Exception as exc:
			raise ImportError(
				"Cubochoric decoder requires `orix` and `numpy` to be available."
			) from exc

		rot = get_sample_fundamental(
			int(resolution),
			point_group=symmetry.Oh,
			method="cubochoric",
		)

		raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
		if raw.ndim != 2:
			raw = raw.reshape(-1, 4)
		if raw.shape[-1] != 4 and raw.shape[0] == 4:
			raw = raw.T
		if raw.shape[-1] != 4:
			raise ValueError(
				f"Unexpected cubochoric quaternion shape: {tuple(raw.shape)}"
			)

		q = torch.as_tensor(raw, dtype=torch.float32, device=device)
		q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
		q = torch.where(q[..., :1] < 0, -q, q)
		return q

	@torch.no_grad()
	def _init_quats(self, bsz: int, device: torch.device) -> torch.Tensor:
		k = self.num_starts
		q0 = torch.zeros((bsz, k, 4), device=device, dtype=torch.float32)
		q0[..., 0] = 1.0

		if k > 1:
			table = self.cubochoric_quats.to(device=device, dtype=torch.float32)
			table_n = table.shape[0]
			if table_n > 0:
				idx = torch.randint(0, table_n, (bsz, k - 1), device=device)
				q0[:, 1:, :] = table[idx]
			else:
				q_rand = torch.randn((bsz, k - 1, 4), device=device, dtype=torch.float32)
				q0[:, 1:, :] = self._normalize_quat(q_rand, self.eps)

		q0 = self._normalize_quat(q0, self.eps)
		q0 = self._fix_sign(q0)
		return q0


class FastLookupFCCDecoder(nn.Module):
	"""
	Fast non-iterative decoder based on nearest-neighbor lookup in (f4, f6) space.

	Builds a cubochoric FZ codebook once, encodes each codebook quaternion to
	(f4, f6), and decodes by nearest lookup without per-sample optimization.
	"""

	def __init__(
		self,
		physics: FCCPhysics,
		lookup_resolution: int = 1,
		w6: float = 0.5,
		table_chunk_size: int = 8192,
		lookup_npy_path: str | None = None,
		rebuild_lookup_file: bool = False,
		refine_steps: int = 0,
		refine_lr: float = 0.05,
	):
		super().__init__()
		self.physics = physics
		self.lookup_resolution = int(lookup_resolution)
		self.w6 = float(w6)
		self.table_chunk_size = max(256, int(table_chunk_size))
		self.rebuild_lookup_file = bool(rebuild_lookup_file)
		self.refine_steps = max(0, int(refine_steps))
		self.refine_lr = float(refine_lr)

		if self.lookup_resolution < 1:
			raise ValueError(f"lookup_resolution must be >= 1, got {lookup_resolution}")
		if self.w6 < 0:
			raise ValueError(f"w6 must be >= 0, got {w6}")
		if self.refine_lr <= 0:
			raise ValueError(f"refine_lr must be > 0, got {refine_lr}")

		self.lookup_npy_path = self._resolve_lookup_path(lookup_npy_path)
		self._ensure_lookup_file()
		table_quats, table_feat, table_feat_norm = self._load_lookup_file()

		self.register_buffer("table_quats", table_quats.to(torch.float32))
		self.register_buffer("table_feat", table_feat)
		self.register_buffer("table_feat_norm", table_feat_norm)
		self.register_buffer("s4", physics.s4.clone())
		self.register_buffer("s6", physics.s6.clone())

	@staticmethod
	def _normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
		norm = q.norm(dim=-1, keepdim=True)
		qn = q / norm.clamp_min(eps)

		bad = norm.squeeze(-1) < eps
		if bad.any():
			qn = qn.clone()
			qn[bad] = qn.new_tensor([1.0, 0.0, 0.0, 0.0])
		return torch.where(qn[..., :1] < 0, -qn, qn)


	def _loss(self, q: torch.Tensor, f4_tgt: torch.Tensor, f6_tgt: torch.Tensor) -> torch.Tensor:
		"""Per-sample feature reconstruction objective for quaternion candidates.

		For each quaternion q, predict (f4, f6) through the fixed physics encoder and
		measure weighted MSE to targets:
		    L(q) = MSE(f4_pred, f4_tgt) + w6 * MSE(f6_pred, f6_tgt).
		"""

		#TODO: check passive vs active e3nn convention and invert quaternions if needed
		R = o3.quaternion_to_matrix(q)
		alpha, beta, gamma = o3.matrix_to_angles(R)

		D4 = wigner_D_cuda(4, alpha, beta, gamma)
		D6 = wigner_D_cuda(6, alpha, beta, gamma)

		f4_pred = torch.einsum("bij,j->bi", D4, self.s4)
		f6_pred = torch.einsum("bij,j->bi", D6, self.s6)

		l4 = (f4_pred - f4_tgt).pow(2).mean(dim=-1)
		l6 = (f6_pred - f6_tgt).pow(2).mean(dim=-1)
		return l4 + self.w6 * l6

	def _refine(self, q_seed: torch.Tensor, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		"""Refine lookup quaternions with short gradient-based local optimization.

		Args:
			q_seed: Initial quaternions from nearest-neighbor lookup, shape (B, 4).
			f4, f6: Target features for the same batch, shapes (B, 9) and (B, 13).

		Returns:
			Refined unit quaternions (B, 4), canonicalized to positive scalar part.

		Notes:
			- This optimizes only temporary tensor `u` (the candidate quaternions).
			- Decoder parameters/buffers are fixed; this is test-time local search.
			- Re-normalizing each step keeps candidates on S^3 (valid rotations).
		"""
		# Fast path: no refinement requested.
		if self.refine_steps <= 0:
			return q_seed

		# Targets are treated as constants during refinement.
		f4_tgt = f4.to(torch.float32)
		f6_tgt = f6.to(torch.float32)

		with torch.enable_grad():
			# Optimize quaternion values directly, initialized from lookup seeds.
			u = nn.Parameter(q_seed.detach().clone())
			opt = torch.optim.Adam([u], lr=self.refine_lr)
			for _ in range(self.refine_steps):
				opt.zero_grad(set_to_none=True)
				# Project to valid/canonical unit quaternions before evaluating loss.
				q = self._normalize_quat(u)
				# Mean over batch for a single scalar objective for Adam.
				loss = self._loss(q, f4_tgt, f6_tgt).mean()
				loss.backward()
				opt.step()
		# Return detached normalized/canonicalized refined quaternions.
		return self._normalize_quat(u.detach())

	def _resolve_lookup_path(self, lookup_npy_path: str | None) -> Path:
		if lookup_npy_path is not None and str(lookup_npy_path).strip() != "":
			return Path(lookup_npy_path).expanduser().resolve()
		fname = f"fast_lookup_fz_res{self.lookup_resolution}_w6_{self.w6:.6f}.npy"
		return (Path.cwd() / "symmetry_groups" / fname).resolve()

	def _build_lookup_arrays(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		table_quats = CubochoricOptimizingFCCDecoder._build_cubochoric_quat_table(
			resolution=self.lookup_resolution,
			device=torch.device(self.physics.device),
		)
		encoder = FCCEncoder(self.physics)
		with torch.no_grad():
			table_f4, table_f6 = encoder(table_quats)

		f6_scale = math.sqrt(self.w6) if self.w6 > 0 else 0.0
		table_feat = torch.cat([table_f4, table_f6 * f6_scale], dim=-1).to(torch.float32)
		table_feat_norm = (table_feat * table_feat).sum(dim=-1)
		return table_quats.to(torch.float32), table_feat, table_feat_norm

	def _ensure_lookup_file(self) -> None:
		if self.lookup_npy_path.exists() and not self.rebuild_lookup_file:
			return

		self.lookup_npy_path.parent.mkdir(parents=True, exist_ok=True)
		table_quats, table_feat, table_feat_norm = self._build_lookup_arrays()

		import numpy as np

		packed = torch.cat(
			[
				table_quats,
				table_feat,
				table_feat_norm.unsqueeze(-1),
			],
			dim=-1,
		).detach().cpu().numpy().astype(np.float32, copy=False)
		np.save(str(self.lookup_npy_path), packed)

	def _load_lookup_file(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		import numpy as np

		arr = np.load(str(self.lookup_npy_path), allow_pickle=False)
		if arr.ndim != 2 or arr.shape[1] != (4 + 22 + 1):
			raise ValueError(
				f"Invalid lookup table shape {arr.shape} in {self.lookup_npy_path}. "
				"Expected (N, 27)."
			)

		t = torch.as_tensor(
			arr,
			dtype=torch.float32,
			device=torch.device(self.physics.device),
		)
		table_quats = t[:, :4]
		table_feat = t[:, 4:26]
		table_feat_norm = t[:, 26]
		return table_quats, table_feat, table_feat_norm

	def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		"""Decode by nearest-neighbor lookup in precomputed (f4, f6) feature space.

		Distance metric:
		    ||f4_q - f4_tbl||^2 + w6 * ||f6_q - f6_tbl||^2
		implemented as Euclidean distance on concatenated features
		    [f4, sqrt(w6) * f6].

		The lookup table is scanned in chunks to keep memory bounded.
		"""
		# Use a stable dtype for distance computations and optional refinement.
		f4 = f4.to(torch.float32)
		f6 = f6.to(torch.float32)

		# Build weighted query features so plain L2 matches (l4 + w6*l6) objective.
		f6_scale = math.sqrt(self.w6) if self.w6 > 0 else 0.0
		query_feat = torch.cat([f4, f6 * f6_scale], dim=-1) # This is the (f4, f6) predicted feature vector, scaled for distance metric.
		query_norm = (query_feat * query_feat).sum(dim=-1, keepdim=True)

		batch_size = query_feat.shape[0]
		table_n = self.table_feat.shape[0]

		# Track the best table entry found so far for each query sample.
		best_dist = torch.full(
			(batch_size,),
			float("inf"),
			dtype=query_feat.dtype,
			device=query_feat.device,
		)
		best_idx = torch.zeros((batch_size,), dtype=torch.long, device=query_feat.device)

		# Chunked scan over table rows; avoids allocating a full (B x table_n) matrix.
		for start in range(0, table_n, self.table_chunk_size):
			end = min(start + self.table_chunk_size, table_n)
			feat_chunk = self.table_feat[start:end]
			norm_chunk = self.table_feat_norm[start:end].unsqueeze(0)

			# Squared L2 via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b.
			dots = query_feat @ feat_chunk.transpose(0, 1)
			dist = query_norm + norm_chunk - 2.0 * dots

			# Keep per-query argmin across processed chunks.
			chunk_best_dist, chunk_best_idx = torch.min(dist, dim=1)
			improved = chunk_best_dist < best_dist
			best_dist = torch.where(improved, chunk_best_dist, best_dist)
			best_idx = torch.where(improved, chunk_best_idx + start, best_idx)

		# Gather nearest quaternion seeds from lookup table.
		q_lookup = self.table_quats[best_idx]
		# Optional local optimization to improve over the discrete lookup seed.
		if self.refine_steps > 0:
			q_lookup = self._refine(q_lookup, f4, f6)
		return q_lookup


class EquivariantSpatialConv(nn.Module):
	"""
	Equivariant spatial convolution layer that mixes features from nearby pixels
	while preserving O(3) symmetry.

	Treats f4 and f6 as true l=4 and l=6 irreps and couples them via
	Clebsch-Gordan tensor products, projecting back onto l=4 and l=6 subspaces.
	A learned 3×3 spatial kernel aggregates neighbour information before the TP.
	"""

	def __init__(self, kernel_size: int = 3):
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2

		self.irreps_in = Irreps("1x4e + 1x6e")
		self.tp = FullyConnectedTensorProduct(
			self.irreps_in,
			self.irreps_in,
			Irreps("1x4e + 1x6e"),
			shared_weights=True,
		)
		self.spatial_weights = nn.Parameter(
			torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)
		)

	def forward(
		self,
		f4: torch.Tensor,
		f6: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
		    f4: (H*W, 9) or (B, H*W, 9)
		    f6: (H*W, 13) or (B, H*W, 13)
		    img_shape: (H, W)
		Returns:
		    f4_out, f6_out with same leading shape as input
		"""
		H, W = img_shape
		batched = f4.dim() == 3
		if not batched:
			f4 = f4.unsqueeze(0)
			f6 = f6.unsqueeze(0)
		B = f4.shape[0]

		features = torch.cat([f4, f6], dim=-1)   # (B, H*W, 22)
		C = features.shape[-1]

		# Reshape to image grid and gather neighbours via learned spatial kernel
		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)   # (B, 22, H, W)
		feat_padded = F.pad(feat_img, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
		patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
		w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
		neighbour_feats = (patches * w).sum(dim=(-1, -2))   # (B, 22, H, W)
		neighbour_flat  = neighbour_feats.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*N, 22)
		features_flat   = features.reshape(B * H * W, C)                              # (B*N, 22)

		# Equivariant tensor product: self ⊗ neighbour → (l=4, l=6) + residual
		out = self.tp(features_flat, neighbour_flat) + features_flat   # (B*N, 22)
		out = out.reshape(B, H * W, C)   # (B, N, 22)

		if not batched:
			out = out.squeeze(0)   # (N, 22)
		return out[..., :9], out[..., 9:]


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
	  Features are 1x4e + 1x6e (even parity).  Only even-l SH couple into
	  even-parity outputs, so sh_irreps = "1x0e + 1x2e" (1 + 5 = 6 components).

	Init: both TP weights = 0  →  output equals the clean NN upsample residual.
	"""

	def __init__(self, upsample_factor: int = 4):
		super().__init__()
		self.upsample_factor = int(upsample_factor)

		self.irreps_feat = Irreps("1x4e + 1x6e")
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
		f4: torch.Tensor,
		f6: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""
		Args:
		    f4: (H*W, 9)   l=4 features at LR resolution
		    f6: (H*W, 13)  l=6 features at LR resolution
		    img_shape: (H, W)  LR spatial dimensions
		Returns:
		    f4_out: (rH*rW, 9)   l=4 features at HR resolution
		    f6_out: (rH*rW, 13)  l=6 features at HR resolution
		    hr_shape: (rH, rW)
		"""
		H, W = img_shape
		r = self.upsample_factor
		C = 22
		Hr, Wr = H * r, W * r
		N = Hr * Wr

		# Pack to image: (1, 22, H, W)
		features = torch.cat([f4, f6], dim=-1)
		feat_img = features.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

		# Step 1: nearest-neighbour upsample → (1, 22, Hr, Wr)
		feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode="nearest")

		# Step 2: SH-informed 2×2 neighbourhood aggregation
		# Pad right+bottom by 1 so every HR pixel has a full 2×2 patch
		feat_padded = F.pad(feat_hr, [0, 1, 0, 1], mode="replicate")     # (1, 22, Hr+1, Wr+1)
		patches = feat_padded.unfold(2, 2, 1).unfold(3, 2, 1)            # (1, 22, Hr, Wr, 2, 2)
		patches = patches.reshape(1, C, Hr, Wr, 4)                       # (1, 22, Hr, Wr, 4)
		patches_flat = patches.squeeze(0).permute(1, 2, 3, 0).reshape(N, 4, C)  # (N, 4, 22)

		sh_exp = self.sh_kernel.unsqueeze(0).expand(N, -1, -1).reshape(N * 4, -1)  # (N*4, 6)
		agg_flat = self.tp_aggregate(
			patches_flat.reshape(N * 4, C),  # (N*4, 22)
			sh_exp,                          # (N*4, 6)
		)                                    # (N*4, 22)
		context = agg_flat.reshape(N, 4, C).sum(dim=1)  # (N, 22)

		# Step 3: self features at HR + equivariant TP
		feat_flat = feat_hr.squeeze(0).permute(1, 2, 0).reshape(N, C)  # (N, 22)
		out_features = self.tp(feat_flat, context)                      # (N, 22)

		# Step 4: residual from NN upsample
		f4_out = out_features[:, :9]  + feat_flat[:, :9]
		f6_out = out_features[:, 9:]  + feat_flat[:, 9:]

		# no residual, just the TP output
		# f4_out = out_features[:, :9]  
		# f6_out = out_features[:, 9:]  

		return f4_out, f6_out, (Hr, Wr)


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

	def __init__(self, kernel_size: int = 3, upsample_factor: int = 4, use_residual: bool = False):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.use_residual = bool(use_residual)

		self.irreps_io = Irreps("1x4e + 1x6e")
		C = 22  # 9 (l=4) + 13 (l=6)

		# Depthwise transpose conv — each of the 22 channels gets its own
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
		f4: torch.Tensor,
		f6: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""
		Args:
		    f4: (H*W, 9) or (B, H*W, 9)
		    f6: (H*W, 13) or (B, H*W, 13)
		    img_shape: (H, W)  LR spatial dimensions
		Returns:
		    f4_out: (rH*rW, 9) or (B, rH*rW, 9)
		    f6_out: (rH*rW, 13) or (B, rH*rW, 13)
		    hr_shape: (rH, rW)
		"""
		H, W = img_shape
		r = self.upsample_factor
		Hr, Wr = H * r, W * r

		batched = f4.dim() == 3
		if not batched:
			f4 = f4.unsqueeze(0)
			f6 = f6.unsqueeze(0)
		B = f4.shape[0]

		features = torch.cat([f4, f6], dim=-1)  # (B, H*W, 22)
		C = features.shape[-1]
		feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, 22, H, W)

		# Step 1: learned depthwise transpose conv upsample → (B, 22, rH, rW)
		feat_hr = self.transpose_conv(feat_img)
		feat_hr = feat_hr[:, :, :Hr, :Wr]  # trim to exact target size

		# Step 2: scalar 3×3 neighbour aggregation at HR
		feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
		patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
		w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
		context_img = (patches * w).sum(dim=(-1, -2))   # (B, 22, Hr, Wr)

		N = Hr * Wr
		feat_flat    = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)      # (B*N, 22)
		context_flat = context_img.permute(0, 2, 3, 1).reshape(B * N, C)  # (B*N, 22)

		# Step 3: equivariant tensor product
		out = self.tp(feat_flat, context_flat)  # (B*N, 22)

		if self.use_residual:
			out = out + feat_flat

		out = out.reshape(B, N, C)   # (B, N, 22)
		if not batched:
			out = out.squeeze(0)   # (N, 22)
		return out[..., :9], out[..., 9:], (Hr, Wr)


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
	     (even-l only: 1x0e + 1x2e = 6 components, consistent with even-parity
	     irreps features).
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

	def __init__(self, upsample_factor: int = 4, k_size: int = 3):
		super().__init__()
		self.upsample_factor = int(upsample_factor)
		self.k_size = int(k_size)
		self.padding = k_size // 2
		self.K = k_size * k_size  # neighbourhood size

		self.irreps_feat = Irreps("1x4e + 1x6e")
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

		# TP 1: invariant attention score  feat ⊗ feat → 1x0e
		#   Active CG paths: 4e⊗4e→0e (1 path), 6e⊗6e→0e (1 path) — 2 weights total.
		self.tp_score = FullyConnectedTensorProduct(
			self.irreps_feat, self.irreps_feat, Irreps("1x0e"),
			shared_weights=True,
		)

		# TP 2: equivariant value encoding  feat ⊗ sh → feat
		#   Active paths include: 4e⊗0e→4e, 4e⊗2e→{4e,6e}, 6e⊗0e→6e, 6e⊗2e→{4e,6e}.
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
		f4: torch.Tensor,
		f6: torch.Tensor,
		img_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""
		Args:
		    f4: (H*W, 9)   l=4 features at LR resolution
		    f6: (H*W, 13)  l=6 features at LR resolution
		    img_shape: (H, W)  LR spatial dimensions
		Returns:
		    f4_out: (rH*rW, 9)   l=4 features at HR resolution
		    f6_out: (rH*rW, 13)  l=6 features at HR resolution
		    hr_shape: (rH, rW)
		"""
		H, W = img_shape
		r = self.upsample_factor
		Hr, Wr = H * r, W * r
		N = Hr * Wr
		K = self.K
		C = 22

		# Pack to image: (1, C, H, W)
		features = torch.cat([f4, f6], dim=-1)
		feat_img = features.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

		# Step 1: nearest-neighbour upsample → (1, C, Hr, Wr)
		feat_hr = F.interpolate(feat_img, scale_factor=float(r), mode="nearest")

		# Flatten to (N, C)
		feat_flat = feat_hr.squeeze(0).permute(1, 2, 0).reshape(N, C)

		# Step 2: extract k×k patches around each HR pixel
		# pad: replicate at borders so every pixel has a full k×k neighbourhood
		feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")  # (1,C,Hr+2p,Wr+2p)
		patches = feat_padded.unfold(2, self.k_size, 1).unfold(3, self.k_size, 1)
		# patches: (1, C, Hr, Wr, k_size, k_size)
		patches = patches.reshape(1, C, Hr, Wr, K)
		# (C, Hr, Wr, K) → (Hr, Wr, K, C) → (N, K, C)
		patches_flat = patches.squeeze(0).permute(1, 2, 3, 0).reshape(N, K, C)

		# Step 3: O(3)-invariant attention scores
		queries = feat_flat.unsqueeze(1).expand(N, K, C).reshape(N * K, C)  # lazy expand
		keys    = patches_flat.reshape(N * K, C)
		scores  = self.tp_score(queries.contiguous(), keys)  # (N*K, 1)
		scores  = scores.reshape(N, K)                       # (N, K)
		attn    = torch.softmax(scores, dim=1)               # (N, K) — invariant weights

		# Steps 4 & 5: transform neighbour values with spatial SH direction
		sh_exp = self.sh_kernel.unsqueeze(0).expand(N, K, -1).reshape(N * K, 6)  # (N*K, 6)
		vals   = self.tp_val(keys, sh_exp.contiguous())  # (N*K, C) equivariant
		vals   = vals.reshape(N, K, C)

		# Step 6: context = attention-weighted sum
		context = (attn.unsqueeze(-1) * vals).sum(dim=1)  # (N, C)

		# Step 7: equivariant output mixing + residual from NN upsample
		out = self.tp_out(feat_flat, context) + feat_flat  # (N, C)

		return out[:, :9], out[:, 9:], (Hr, Wr)


class FCCAutoEncoder(nn.Module):
	"""
	Physics-based FCC autoencoder wrapper.

	This class reproduces the core behavior from the
	`run_physics_decoder_test` pipeline in simple_encoder_decoder:
	  1) encode quaternion -> (f4, f6)
	  2) decode -> canonical quaternion
	  3) match decoded quaternion to the closest FCC symmetry variant
	  4) optionally compute reconstruction distance + misorientation stats
	"""

	def __init__(
		self,
		device: str | torch.device | None = None,
		grid_res: int = 100_000,
		decoder_backend: str = "optimizing",
		decoder_config: dict[str, Any] | None = None,
		**decoder_kwargs: Any,
	):
		super().__init__()
		if device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.physics = FCCPhysics(str(self.device))
		self.encoder = FCCEncoder(self.physics)
		self.conv_layer = EquivariantSpatialConv(kernel_size=3)
		dcfg = dict(decoder_config or {})
		dcfg.update(decoder_kwargs)

		def dget(key: str, default: Any) -> Any:
			return dcfg.get(key, default)

		backend = str(decoder_backend).lower()
		self.decoder_backend = backend
	
		if backend in {"lookup", "fast_lookup", "cubochoric_lookup"}:
			import pdb; pdb.set_trace()
			self.decoder = FastLookupFCCDecoder(
				self.physics,
				lookup_resolution=int(dget("decoder_lookup_resolution", 3)),
				w6=float(dget("decoder_w6", 0.5)),
				table_chunk_size=int(dget("decoder_lookup_chunk_size", 8192)),
				lookup_npy_path=dget("decoder_lookup_npy_path", None),
				rebuild_lookup_file=bool(dget("decoder_lookup_rebuild", False)),
				refine_steps=int(dget("decoder_lookup_refine_steps", 10)),
				refine_lr=float(dget("decoder_lookup_refine_lr", 0.05)),
			)
		else:
			raise ValueError(f"Unknown decoder_backend: {decoder_backend}")

	@staticmethod
	def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
		return quats / norm

	@staticmethod
	def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
		return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)

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

	# e3nn requires active convention (rotations act on vectors from the left), but Bunge convention is passive.
	# So we convert between conventions by conjugating the quaternion (negating vector part) before encoding and after decoding.
	@staticmethod
	def _to_active_convention(quats: torch.Tensor) -> torch.Tensor:
		"""Convert Bunge (passive) quaternion to active convention by conjugation."""
		return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)

	@staticmethod
	def _from_active_convention(quats: torch.Tensor) -> torch.Tensor:
		"""Convert active quaternion back to Bunge (passive) convention by conjugation."""
		return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)

	def encode(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		# Bunge inputs are passive; FCCEncoder expects active convention
		return self.encoder(self._to_active_convention(quats))

	def feature_loss(
		self,
		quats: torch.Tensor,
		img_shape: tuple[int, int],
		normalize_input: bool = True,
	) -> torch.Tensor:
		"""
		MSE loss in (f4, f6) feature space — fully differentiable w.r.t. conv_layer.

		The encoder is frozen (no gradients) so only conv_layer parameters receive
		gradients.  This is the correct training signal for the conv layer because
		FastLookupFCCDecoder is non-differentiable (lookup table).

		Loss = MSE(f4_conv, f4_enc) + w6 * MSE(f6_conv, f6_enc)
		"""
		quats = quats.to(self.device)
		if normalize_input:
			quats = self._normalize_quaternions(quats)
		with torch.no_grad():
			f4, f6 = self.encode(quats)
		f4_tgt = f4.detach()
		f6_tgt = f6.detach()
		f4_out, f6_out = self.conv_layer(f4_tgt, f6_tgt, img_shape)
		w6 = float(getattr(self.decoder, "w6", 0.5))
		return F.mse_loss(f4_out, f4_tgt) + w6 * F.mse_loss(f6_out, f6_tgt)
	
	# Make sure to do s^-1 ⊗ q before using FZ reduction, because we do it in Bunge convention

	def decode(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		# Decoder operates in active convention; convert result back to Bunge
		q_bunge = self._from_active_convention(self.decoder(f4, f6))
		# Apply FZ reduction in Bunge convention: s⁻¹ ⊗ q (left multiplication)
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
		# Stage 1: physics encoder quaternion -> (f4, f6)
		f4, f6 = self.encode(quats)
		# Stage 2: symmetry-aware spatial conv (requires img_shape)
		if img_shape is not None:
			f4, f6 = self.conv_layer(f4, f6, img_shape)
		# Stage 3: decoder (f4, f6) -> quaternion, includes FZ reduction
		return self.decode(f4, f6)
	
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
		- f4_0..f4_8
		- f6_0..f6_12
		- (optional) q_dec_* and decode quality metrics
		"""
		quats = self._sample_fz_quaternions(
			resolution=resolution,
			method=sampling_method,
			device=self.device,
		)
		quats = self._normalize_quaternions(quats)
		f4, f6 = self.encode(quats)

		payload: dict[str, Any] = {
			"quats": quats.detach().cpu(),
			"f4": f4.detach().cpu(),
			"f6": f6.detach().cpu(),
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
	  MSE(f4_hr_model, f4_hr_enc) + w6 * MSE(f6_hr_model, f6_hr_enc)
	where (f4_hr_enc, f6_hr_enc) are the irreps obtained by encoding the
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
		lr_shape: tuple[int, int] | None = None,
		lr_conv_kernel_size: int | None = None,
		lr_conv_kernel_size_1: int | None = None,
		lr_conv_kernel_size_2: int | None = None,
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

		def _resolve_k(explicit_k, fallback_k, lr_shape, default=3):
			"""Resolve kernel size: explicit → fallback → lr_shape-derived → default, enforced odd."""
			if explicit_k is not None:
				k = int(explicit_k)
			elif fallback_k is not None:
				k = int(fallback_k)
			elif lr_shape is not None:
				k = max(3, min(lr_shape) // 16)
			else:
				k = default
			return k if k % 2 == 1 else k + 1

		k1 = _resolve_k(lr_conv_kernel_size_1, lr_conv_kernel_size, lr_shape)
		k2 = _resolve_k(lr_conv_kernel_size_2, lr_conv_kernel_size, lr_shape)

		# Override inherited conv_layer (lr1) and add conv_lr2 with their respective kernels
		self.conv_layer = EquivariantSpatialConv(kernel_size=k1)
		self.conv_lr2   = EquivariantSpatialConv(kernel_size=k2)
		upsampler_type = str(upsampler).lower()
		if upsampler_type == "attention":
			self.upsample_conv = EquivariantAttentionUpsample(
				upsample_factor=self.upsample_factor,
			)
		else:
			self.upsample_conv = EquivariantTransposeConv(
				upsample_factor=self.upsample_factor,
				use_residual=bool(upsample_residual),
			)
		self.conv_hr = EquivariantSpatialConv(kernel_size=3)

	def _forward_sr_features(
		self,
		f4_lr: torch.Tensor,
		f6_lr: torch.Tensor,
		lr_shape: tuple[int, int],
	) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
		"""Trainable SR pipeline: conv_lr1 → conv_lr2 → upsample → conv_hr.

		Override in subclasses to add post-conv_hr processing.
		Returns (f4_hr, f6_hr, hr_shape).
		"""
		f4, f6           = self.conv_layer(f4_lr, f6_lr, lr_shape)
		f4, f6           = self.conv_lr2(f4, f6, lr_shape)
		f4, f6, hr_shape = self.upsample_conv(f4, f6, lr_shape)
		f4, f6           = self.conv_hr(f4, f6, hr_shape)
		return f4, f6, hr_shape

	def feature_loss_sr(
		self,
		lr_quats: torch.Tensor,
		hr_quats: torch.Tensor,
		lr_shape: tuple[int, int],
		normalize_input: bool = True,
	) -> torch.Tensor:
		"""
		SR training loss in (f4, f6) feature space.

		Encoder is frozen; only conv_layer (LR), upsample_conv, and conv_hr
		receive gradient updates.

		Args:
		    lr_quats:  (H_lr*W_lr, 4) or (B, H_lr*W_lr, 4)  LR quaternions
		    hr_quats:  (H_hr*W_hr, 4) or (B, H_hr*W_hr, 4)  HR ground-truth quaternions
		    lr_shape:  (H_lr, W_lr)
		Returns:
		    scalar MSE loss
		"""
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

		# Encode both under no_grad; targets are fully detached
		with torch.no_grad():
			f4_lr_flat,     f6_lr_flat     = self.encode(lr_flat)
			f4_hr_tgt_flat, f6_hr_tgt_flat = self.encode(hr_flat)

		if batched:
			f4_lr     = f4_lr_flat.detach().reshape(B, -1, 9)
			f6_lr     = f6_lr_flat.detach().reshape(B, -1, 13)
			f4_hr_tgt = f4_hr_tgt_flat.detach().reshape(B, -1, 9)
			f6_hr_tgt = f6_hr_tgt_flat.detach().reshape(B, -1, 13)
		else:
			f4_lr     = f4_lr_flat.detach()
			f6_lr     = f6_lr_flat.detach()
			f4_hr_tgt = f4_hr_tgt_flat.detach()
			f6_hr_tgt = f6_hr_tgt_flat.detach()

		f4_hr, f6_hr, _ = self._forward_sr_features(f4_lr, f6_lr, lr_shape)

		w6 = float(getattr(self.decoder, "w6", 0.5))
		return F.mse_loss(f4_hr, f4_hr_tgt) + w6 * F.mse_loss(f6_hr, f6_hr_tgt)

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
		f4_lr, f6_lr        = self.encode(lr_quats)
		f4_hr, f6_hr, _     = self._forward_sr_features(f4_lr, f6_lr, lr_shape)
		return self.decode(f4_hr, f6_hr)

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


# ──────────────────────────────────────────────────────────────────────────────
class FCCAutoEncoderSRDoubleConvAttn(FCCAutoEncoderSR):
	"""FCCAutoEncoderSR (double-conv) + HR equivariant self-attention refinement.

	Pipeline:
	  LR → encode → conv_lr (k1) → conv_lr2 (k2)
	    → EquivariantTransposeConv   (upsample 4×)
	    → conv_hr (3×3 scalar)
	    → [LRBlockAttentionBlock × num_hr_attn_blocks]  ← NEW, in HR space
	    → decode → FZ-reduce

	The HR attention blocks compute O(3)-invariant orientation-similarity scores:
	  score_ij = s4·(f4_i·f4_j) + s6·(f6_i·f6_j)
	and use them to weight neighbourhood aggregation.  Same-grain pixels
	(high similarity) reinforce their orientation; cross-grain pixels (low
	similarity) do not bleed across boundaries.  This sharpens staircase
	boundaries left by the scalar conv_hr without sacrificing equivariance.

	At epoch 0 each HR attention block contributes zero delta (lin_out zero-init)
	so output = conv_hr, identical to the base double-conv model.

	Args:
	    num_hr_attn_blocks:   Number of stacked HR attention blocks (default 2).
	    hr_attn_num_channels: Hidden channel multiplier C for FCTP (default 8).
	    hr_attn_block_size:   Spatial block size in HR pixels (default 16 = 4 LR
	                          pixels at 4× scale, spans one full LR grain boundary).
	"""

	_SH_IRREPS = Irreps("1x0e + 1x2e")

	def __init__(
		self,
		device=None,
		upsample_factor: int = 4,
		upsample_residual: bool = False,
		upsampler: str = "conv",
		lr_shape=None,
		lr_conv_kernel_size=None,
		lr_conv_kernel_size_1=None,
		lr_conv_kernel_size_2=None,
		decoder_backend: str = "lookup",
		decoder_config=None,
		num_hr_attn_blocks: int = 2,
		hr_attn_num_channels: int = 8,
		hr_attn_block_size: int = 16,
		**decoder_kwargs,
	):
		super().__init__(
			device=device,
			upsample_factor=upsample_factor,
			upsample_residual=upsample_residual,
			upsampler=upsampler,
			lr_shape=lr_shape,
			lr_conv_kernel_size=lr_conv_kernel_size,
			lr_conv_kernel_size_1=lr_conv_kernel_size_1,
			lr_conv_kernel_size_2=lr_conv_kernel_size_2,
			decoder_backend=decoder_backend,
			decoder_config=decoder_config,
			**decoder_kwargs,
		)
		self.hr_attn_block_size = int(hr_attn_block_size)
		self.hr_attn_blocks = nn.ModuleList([
			LRBlockAttentionBlock(num_channels=int(hr_attn_num_channels))
			for _ in range(int(num_hr_attn_blocks))
		])
		self._cached_hr_block_shape: tuple[int, int] | None = None
		self._cached_hr_sh_block:    torch.Tensor | None    = None

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
		"""conv_lr1 → conv_lr2 → upsample → conv_hr → HR attention blocks."""
		f4_hr, f6_hr, hr_shape = super()._forward_sr_features(f4_lr, f6_lr, lr_shape)

		Hr, Wr = hr_shape
		device = f4_hr.device
		dtype  = f4_hr.dtype

		batched = f4_hr.dim() == 3
		if not batched:
			f4_hr = f4_hr.unsqueeze(0)
			f6_hr = f6_hr.unsqueeze(0)
		B = f4_hr.shape[0]

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

		sh_block = self._get_hr_sh_block(block_h, block_w, device, dtype)   # (Nb, 6)

		for attn_block in self.hr_attn_blocks:
			feat = feat + attn_block(feat, sh_block, Hr_pad, Wr_pad, block_h, block_w)

		if pad_h > 0 or pad_w > 0:
			feat = feat.reshape(B, Hr_pad, Wr_pad, 22)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, 22)

		if not batched:
			feat = feat.squeeze(0)

		return feat[..., :9], feat[..., 9:], hr_shape
