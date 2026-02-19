import math
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
		self.fcc_syms = torch.tensor(
			[
				[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1],
				[inv_sqrt_2, inv_sqrt_2, 0, 0],
				[inv_sqrt_2, 0, inv_sqrt_2, 0],
				[inv_sqrt_2, 0, 0, inv_sqrt_2],
				[inv_sqrt_2, -inv_sqrt_2, 0, 0],
				[inv_sqrt_2, 0, -inv_sqrt_2, 0],
				[inv_sqrt_2, 0, 0, -inv_sqrt_2],
				[0, inv_sqrt_2, inv_sqrt_2, 0],
				[0, inv_sqrt_2, 0, inv_sqrt_2],
				[0, 0, inv_sqrt_2, inv_sqrt_2],
				[0, inv_sqrt_2, -inv_sqrt_2, 0],
				[0, 0, inv_sqrt_2, -inv_sqrt_2],
				[0, inv_sqrt_2, 0, -inv_sqrt_2],
				[half, half, half, half],
				[half, -half, -half, half],
				[half, -half, half, -half],
				[half, half, -half, -half],
				[half, half, half, -half],
				[half, half, -half, half],
				[half, -half, half, half],
				[half, -half, -half, -half],
			],
			dtype=torch.float32,
			device=device,
		)


class FCCEncoder(nn.Module):
	def __init__(self, physics: FCCPhysics):
		super().__init__()
		self.physics = physics

	def forward(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		R = o3.quaternion_to_matrix(quats)
		alpha, beta, gamma = o3.matrix_to_angles(R)

		D4 = wigner_D_cuda(4, alpha, beta, gamma)
		D6 = wigner_D_cuda(6, alpha, beta, gamma)
		f4 = torch.einsum("bij,j->bi", D4, self.physics.s4)
		f6 = torch.einsum("bij,j->bi", D6, self.physics.s6)

		return f4, f6


class SphericalSamplingDecoder(nn.Module):
	def __init__(self, physics: FCCPhysics, grid_res: int = 1_000_000):
		super().__init__()
		self.n_fib_samples = grid_res
		self.physics = physics

		self.grid_vecs = self._fibonacci_sphere(
			samples=self.n_fib_samples,
			device=physics.device,
		)
		self.Y4_grid = o3.spherical_harmonics(4, self.grid_vecs, normalize=True)

	def forward(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		del f6
		batch_size = f4.shape[0]

		signal = torch.einsum("bi,gi->bg", f4, self.Y4_grid)
		_, z_indices = torch.max(signal, dim=1)
		z_axis = self.grid_vecs[z_indices]

		dots = torch.einsum(
			"bij,bij->bi",
			self.grid_vecs.unsqueeze(0).expand(batch_size, -1, -1),
			z_axis.unsqueeze(1).expand(-1, self.n_fib_samples, -1),
		)
		mask = dots.abs() < 0.2

		masked_signal = signal.clone()
		masked_signal[~mask] = -float("inf")

		_, x_indices = torch.max(masked_signal, dim=1)
		x_axis = self.grid_vecs[x_indices]

		z_axis = torch.nn.functional.normalize(z_axis, dim=-1)
		proj = torch.sum(x_axis * z_axis, dim=-1, keepdim=True) * z_axis
		x_axis = torch.nn.functional.normalize(x_axis - proj, dim=-1)
		y_axis = torch.cross(z_axis, x_axis, dim=-1)

		R_rec = torch.stack([x_axis, y_axis, z_axis], dim=-1)
		return o3.matrix_to_quaternion(R_rec)

	def _fibonacci_sphere(self, samples: int, device: str) -> torch.Tensor:
		points = []
		phi = math.pi * (3.0 - math.sqrt(5.0))

		for i in range(samples):
			y = 1 - (i / float(samples - 1)) * 2
			radius = math.sqrt(1 - y * y)
			theta = phi * i
			x = math.cos(theta) * radius
			z = math.sin(theta) * radius
			points.append([x, y, z])

		return torch.tensor(points, dtype=torch.float32, device=device)


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
	):
		super().__init__()
		if device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.physics = FCCPhysics(str(self.device))
		self.encoder = FCCEncoder(self.physics)
		self.decoder = SphericalSamplingDecoder(self.physics, grid_res=grid_res)

	@staticmethod
	def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
		return quats / norm

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

	def encode(self, quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		return self.encoder(quats)

	def decode(self, f4: torch.Tensor, f6: torch.Tensor) -> torch.Tensor:
		return self.decoder(f4, f6)

	def forward(self, quats: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
		quats = quats.to(self.device)
		if quats.dim() != 2 or quats.shape[-1] != 4:
			raise ValueError(
				f"FCCAutoEncoder follows simple encoder/decoder behavior and expects (N,4), got {tuple(quats.shape)}"
			)
		if normalize_input:
			quats = self._normalize_quaternions(quats)
		f4, f6 = self.encode(quats)
		return self.decode(f4, f6)

	def match_closest_symmetry(
		self,
		q_decoded: torch.Tensor,
		q_truth: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Match each decoded quaternion to the closest element in its FCC symmetry family.

		Returns
		-------
		closest_quats : (B, 4)
		errors : (B,)
		best_indices : (B,)
		"""
		q_decoded = q_decoded.to(self.device)
		q_truth = q_truth.to(self.device)

		q_rec_expanded = q_decoded.unsqueeze(1).expand(-1, 24, -1)
		fcc_syms_expanded = self.physics.fcc_syms.unsqueeze(0).expand(q_truth.shape[0], -1, -1)

		w1, x1, y1, z1 = (
			q_rec_expanded[..., 0],
			q_rec_expanded[..., 1],
			q_rec_expanded[..., 2],
			q_rec_expanded[..., 3],
		)
		w2, x2, y2, z2 = (
			fcc_syms_expanded[..., 0],
			fcc_syms_expanded[..., 1],
			fcc_syms_expanded[..., 2],
			fcc_syms_expanded[..., 3],
		)

		family = torch.stack(
			[
				w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
				w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
				w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
				w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
			],
			dim=-1,
		)

		q_truth_expanded = q_truth.unsqueeze(1)
		dist_pos = torch.norm(family - q_truth_expanded, dim=-1)
		dist_neg = torch.norm(family + q_truth_expanded, dim=-1)
		min_dist = torch.minimum(dist_pos, dist_neg)

		errors = torch.min(min_dist, dim=1)[0]
		best_indices = torch.argmin(min_dist, dim=1)

		batch_indices = torch.arange(q_truth.shape[0], device=self.device)
		closest_quats = family[batch_indices, best_indices]
		use_neg = dist_neg[batch_indices, best_indices] < dist_pos[batch_indices, best_indices]
		closest_quats[use_neg] = -closest_quats[use_neg]

		return closest_quats, errors, best_indices

	def reconstruct_batch(
		self,
		q_batch: torch.Tensor,
		normalize_input: bool = True,
		return_metrics: bool = True,
	) -> dict[str, torch.Tensor]:
		"""
		Run encode/decode and FCC symmetry matching for one batch.
		"""
		q_batch = q_batch.to(self.device)
		if normalize_input:
			q_batch = self._normalize_quaternions(q_batch)

		f4, f6 = self.encode(q_batch)
		q_canonical = self.decode(f4, f6)
		q_reconstructed, errors, best_indices = self.match_closest_symmetry(
			q_decoded=q_canonical,
			q_truth=q_batch,
		)

		out = {
			"q_input": q_batch,
			"q_canonical": q_canonical,
			"q_reconstructed": q_reconstructed,
			"symmetry_index": best_indices,
		}

		if return_metrics:
			q_conj = torch.stack(
				[
					q_batch[:, 0],
					-q_batch[:, 1],
					-q_batch[:, 2],
					-q_batch[:, 3],
				],
				dim=1,
			)
			error_quats = self.quat_mul(q_reconstructed, q_conj)
			w_errors = error_quats[:, 0].abs().clamp(max=1.0)
			misorientation_deg = 2.0 * torch.acos(w_errors) * 180.0 / math.pi

			out["errors"] = errors
			out["misorientation_deg"] = misorientation_deg

		return out

	@torch.no_grad()
	def reconstruct_all(
		self,
		quats: torch.Tensor,
		batch_size: int | None = None,
		normalize_input: bool = True,
		return_metrics: bool = True,
	) -> dict[str, Any]:
		"""
		Process all quaternions in chunks and return concatenated outputs + summary stats.
		"""
		quats = quats.to(self.device)
		if batch_size is None:
			batch_size = 1000 if self.device.type == "cuda" else 500

		keys = ["q_input", "q_canonical", "q_reconstructed", "symmetry_index"]
		if return_metrics:
			keys.extend(["errors", "misorientation_deg"])

		buckets: dict[str, list[torch.Tensor]] = {k: [] for k in keys}

		for batch_start in range(0, quats.shape[0], batch_size):
			batch_end = min(batch_start + batch_size, quats.shape[0])
			out = self.reconstruct_batch(
				q_batch=quats[batch_start:batch_end],
				normalize_input=normalize_input,
				return_metrics=return_metrics,
			)
			for k in keys:
				buckets[k].append(out[k])

		result = {k: torch.cat(v, dim=0) for k, v in buckets.items()}

		if return_metrics:
			errors = result["errors"]
			mis = result["misorientation_deg"]
			result["stats"] = {
				"num_quats": int(quats.shape[0]),
				"max_error": float(errors.max().item()),
				"mean_error": float(errors.mean().item()),
				"median_error": float(errors.median().item()),
				"max_misorientation_deg": float(mis.max().item()),
				"mean_misorientation_deg": float(mis.mean().item()),
				"median_misorientation_deg": float(mis.median().item()),
			}

		return result


