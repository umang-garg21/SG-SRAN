import math
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from e3nn import o3


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
		if self.refine_steps <= 0:
			return q_seed

		f4_tgt = f4.to(torch.float32)
		f6_tgt = f6.to(torch.float32)

		with torch.enable_grad():
			u = nn.Parameter(q_seed.detach().clone())
			opt = torch.optim.Adam([u], lr=self.refine_lr)
			for _ in range(self.refine_steps):
				opt.zero_grad(set_to_none=True)
				q = self._normalize_quat(u)
				loss = self._loss(q, f4_tgt, f6_tgt).mean()
				loss.backward()
				opt.step()
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
		f4 = f4.to(torch.float32)
		f6 = f6.to(torch.float32)
		f6_scale = math.sqrt(self.w6) if self.w6 > 0 else 0.0
		query_feat = torch.cat([f4, f6 * f6_scale], dim=-1)
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

		q_lookup = self.table_quats[best_idx]
		if self.refine_steps > 0:
			q_lookup = self._refine(q_lookup, f4, f6)
		return q_lookup


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
		dcfg = dict(decoder_config or {})
		dcfg.update(decoder_kwargs)

		def dget(key: str, default: Any) -> Any:
			return dcfg.get(key, default)

		backend = str(decoder_backend).lower()
		self.decoder_backend = backend
	
		if backend in {"lookup", "fast_lookup", "cubochoric_lookup"}:
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
	
	def forward(self, quats: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
		quats = quats.to(self.device)
		if quats.dim() != 2 or quats.shape[-1] != 4:
			raise ValueError(
				f"FCCAutoEncoder follows simple encoder/decoder behavior and expects (N,4), got {tuple(quats.shape)}"
			)
		if normalize_input:
			quats = self._normalize_quaternions(quats)
		f4, f6 = self.encode(quats)
		return self.decode(f4, f6)  # decode now includes FZ reduction
	
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




