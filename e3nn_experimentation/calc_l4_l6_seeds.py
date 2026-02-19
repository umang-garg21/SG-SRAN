# import torch
# import numpy as np
# from e3nn import o3

# def compute_invariant_seed(l_order, group_matrices):
#     """
#     Computes the invariant seed vector (A1g) for a given L order and Symmetry Group.

#     Parameters
#     ----------
#     l_order : int
#         Spherical harmonic order (e.g., 4 or 6)
#     group_matrices : torch.Tensor
#         Rotation matrices of the group, shape (N_ops, 3, 3)

#     Returns
#     -------
#     torch.Tensor
#         The normalized invariant seed vector of shape (2l+1,)
#     """
#     device = group_matrices.device

#     # 1. Convert group rotation matrices to Euler angles (alpha, beta, gamma)
#     alpha, beta, gamma = o3.matrix_to_angles(group_matrices)

#     # 2. Compute Wigner D matrices for every operation in the group
#     # Shape: (N_ops, 2l+1, 2l+1)
#     D_matrices = o3.wigner_D(l_order, alpha, beta, gamma)

#     # 3. Compute the Projector P = (1/N) * sum(D(g))
#     # This matrix averages out all non-invariant components.
#     # The invariant vector is the eigenvector of P with eigenvalue = 1.
#     P = D_matrices.mean(dim=0)  # Shape (2l+1, 2l+1)

#     # 4. Eigendecomposition to find the invariant subspace
#     # We use eigh because P is Hermitian (symmetric for real irreps)
#     eigenvalues, eigenvectors = torch.linalg.eigh(P)

#     # 5. Extract the vector with eigenvalue ≈ 1.0
#     # Note: eigh sorts eigenvalues in ascending order, so the largest is last.
#     max_eigenvalue = eigenvalues[-1]
#     seed = eigenvectors[:, -1]

#     # Validation: Check if a valid invariant exists
#     if max_eigenvalue < 0.99:
#         print(f"Warning: No invariant (A1g) component found for L={l_order} (Max Eval: {max_eigenvalue:.4f})")
#         return None

#     # 6. Post-Processing: Enforce sparsity and sign convention
#     # Eigenvectors have arbitrary global phase/sign.
#     # We fix the sign so the center component (index l) is positive, matching physics conventions.
#     center_idx = l_order # The m=0 index
#     if seed[center_idx] < 0:
#         seed = -seed

#     # Zero out tiny numerical noise (e.g., 1e-8) for cleanliness
#     seed[torch.abs(seed) < 1e-6] = 0.0

#     # Renormalize just to be safe
#     seed = seed / torch.norm(seed)

#     return seed

# # ==============================================================================
# # EXECUTION & VERIFICATION
# # ==============================================================================

# # 1. Load your group (mocking the loading part here)
# # You would use: sym_matrices = load_oh_symmetry_group("path/to/Oh.npy")
# # Here I will generate the 24 proper rotations of the Cube (O group) for demonstration
# # (Identity + 90/180 axes + 120 diagonals + 180 diagonals)
# # For the sake of this snippet, let's assume you have the matrices loaded:
# # sym_matrices = torch.tensor(...)

# # -------------------------------------------------------------------------
# # NOTE: Since I don't have your .npy file, I will compute the projector
# # using the analytical property that "averaging over group" works.
# # Below is how you run it in your pipeline.
# # -------------------------------------------------------------------------

# def verify_seeds():
#     # Load your actual symmetry group here
#     # sym_matrices = load_oh_symmetry_group(...)

#     # For this demo, let's look at the Target Values you hard-coded to verify math
#     print(f"{'Index':<5} | {'Computed':<12} | {'Hard-Coded':<12} | {'Match?'}")
#     print("-" * 45)

#     # --- Verify L=4 ---
#     # We know the answer should look like this (from your hard-coded values):
#     s4_expected = torch.zeros(9)
#     s4_expected[0] = (7/12)**0.5
#     s4_expected[4] = (5/24)**0.5
#     s4_expected[8] = (7/12)**0.5 # Wait, your code had s4[0] and s4[8] asymmetric?
#     # Correction: Physics says m=+4 and m=-4 should be symmetric for Cubic.
#     # Your snippet: s4[0]=sqrt(7/12), s4[4]=sqrt(5/24), s4[8]=sqrt(5/24)
#     # That asymmetry in your snippet (0 vs 8) is actually suspicious!
#     # Usually cubic L=4 is: Y4,0 + sqrt(5/14)(Y4,4 + Y4,-4).

#     # Let's run the automatic solver to find the REAL Truth.
#     # (Requires the symmetry group matrix list to run for real)
#     pass


# def get_cubic_seeds(device, sym_group_path="path/to/Oh_group.npy"):
#     # 1. Load Proper Rotations (reuse your filtered loading function)
#     sym_matrices = load_oh_symmetry_group(sym_group_path).to(device)

#     # 2. Compute seeds automatically
#     s4 = compute_invariant_seed(4, sym_matrices)
#     s6 = compute_invariant_seed(6, sym_matrices)

#     return s4, s6


# if __name__ == "__main__":
#     print("Copy the 'compute_invariant_seed' function into your code.")
#     print("Pass your 'sym_matrices' (proper rotations only) to it.")

import torch
import math
from e3nn import o3


def compute_seeds_from_quaternions(sym_quaternions, device="cpu"):
    """
    Computes invariant seeds for L=4 and L=6 using the Reynolds Operator
    averaged over the provided symmetry quaternions.
    """
    sym_quaternions = sym_quaternions.to(device)

    # 1. Convert Quaternions to Euler Angles (Alpha, Beta, Gamma)
    # e3nn requires matrix -> angles
    # Move to CPU first for matrix operations with e3nn
    rot_matrices = o3.quaternion_to_matrix(sym_quaternions.to("cpu"))
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrices)

    seeds = {}

    # 2. Loop for L=4 and L=6
    for l in range(1, 10):
        dim = 2 * l + 1

        # A. Compute Wigner D matrices for all 24 ops
        # Shape: (24, dim, dim)
        # Note: wigner_D internally uses CPU tensors, so keep angles on CPU
        D = o3.wigner_D(l, alpha, beta, gamma)
        D = D.to(device)

        # B. Reynolds Operator (Average over group)
        # P = (1/|G|) * sum(D(g))
        P = D.mean(dim=0)

        # C. Eigendecomposition
        # The invariant is the eigenvector with eigenvalue 1.0
        evals, evecs = torch.linalg.eigh(P)

        # Extract the vector corresponding to the largest eigenvalue (approx 1.0)
        seed = evecs[:, -1]

        # Validation
        if evals[-1] < 0.99:
            print(f"⚠️  WARNING: No invariant found for L={l}")
            continue

        # D. Post-Processing (Standardize Sign/Phase)
        # Convention: Make the center component (m=0) positive
        center_idx = l
        if seed[center_idx] < 0:
            seed = -seed

        # Clean numerical noise
        seed[torch.abs(seed) < 1e-6] = 0.0

        # Normalize
        seed = seed / torch.norm(seed)

        seeds[l] = seed

    return seeds


# =============================================================================
# DEFINE YOUR SYMMETRY GROUP
# =============================================================================
inv_sqrt_2 = 1 / math.sqrt(2)
half = 0.5

fcc_syms = torch.tensor(
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
)

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    seeds = compute_seeds_from_quaternions(
        fcc_syms, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    s4, s6 = seeds[4], seeds[6]
    print("\n Computed Invariant Seed (L=4):")
    print(s4)
    print("\n Computed Invariant Seed (L=6):")
    print(s6)
    print("\n Computed Invariant Seed (L=8):")
    print(seeds[8])
    print("\n Computed Invariant Seed (L=9):")
    print(seeds[9])

    # Optional: Compare with your old values to see the fix
    print("\n--- Verification of Coefficients ---")
    print(
        f"L=4 Center (m=0): {s4[4]:.4f}  (Old code used: {math.sqrt(5/24):.4f} vs {math.sqrt(7/12):.4f})"
    )
    print(f"L=4 Edge (m=4):   {s4[8]:.4f}")
