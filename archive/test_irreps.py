import torch
import math
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

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create random Euler angles for a batch of quaternions
batch_size = 1
alpha = torch.rand(batch_size, device=device) * 2 * math.pi
beta = torch.rand(batch_size, device=device) * 2 * math.pi
gamma = torch.rand(batch_size, device=device) * 2 * math.pi

# Initialize s4 and s6 vectors (matching your physics constants)
s4 = torch.zeros(9, device=device)
s4[4] = 0.7638
s4[8] = 0.6455

s6 = torch.zeros(13, device=device)
s6[6] = 0.3536
s6[10] = -0.9354

# Compute Wigner D matrices
D4 = wigner_D_cuda(4, alpha, beta, gamma)  # Shape: (batch_size, 9, 9)
D6 = wigner_D_cuda(6, alpha, beta, gamma)  # Shape: (batch_size, 13, 13)

# Apply the matrix-vector products

# D4 and D6 have shape (batch_size, 2l+1, 2l+1)

f4 = torch.einsum("bij,j->bi", D4, s4)  # Shape: (batch_size, 9)
f6 = torch.einsum("bij,j->bi", D6, s6)  # Shape: (batch_size, 13)


# 9 x 9  @ 9 x 1 -> 9 x 1




print("f4 shape:", f4.shape)
print("f6 shape:", f6.shape)
print("\nFirst f4 sample:")
print(f4[0])
print("\nFirst f6 sample:")
print(f6[0])

# Verify s4 vector
print("\ns4 vector:")
print(s4)