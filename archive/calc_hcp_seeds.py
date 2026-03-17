import math

import torch
from e3nn import o3


def _angles_from_quaternions(sym_quaternions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot_matrices = o3.quaternion_to_matrix(sym_quaternions.to("cpu"))
    return o3.matrix_to_angles(rot_matrices)


def _reynolds_operator(sym_quaternions: torch.Tensor, l: int, device: str = "cpu") -> torch.Tensor:
    alpha, beta, gamma = _angles_from_quaternions(sym_quaternions)
    D = o3.wigner_D(l, alpha, beta, gamma).to(device)
    return D.mean(dim=0)


def compute_seeds_from_quaternions(sym_quaternions: torch.Tensor, device: str = "cpu") -> dict[int, torch.Tensor]:
    """Compute invariant Reynolds seeds for l=4 and l=6."""
    sym_quaternions = sym_quaternions.to(device)
    seeds: dict[int, torch.Tensor] = {}

    for l in (4, 6):
        P = _reynolds_operator(sym_quaternions, l=l, device=device)
        evals, evecs = torch.linalg.eigh(P)
        seed = evecs[:, -1]

        # Report multiplicity near the top eigenvalue; L=6 can be nearly degenerate.
        top = evals[-1]
        mult = int((evals >= (top - 1e-6)).sum().item())
        print(f"L={l}: max eigenvalue={top:.6f} (top-multiplicity~{mult})")

        if top < 0.99:
            print(f"  WARNING: no robust invariant found for L={l}")
            continue

        center_idx = l
        if seed[center_idx] < 0:
            seed = -seed

        seed[torch.abs(seed) < 1e-6] = 0.0
        seed = seed / torch.norm(seed).clamp_min(1e-12)
        seeds[l] = seed

    return seeds


def build_hcp_syms_mtex() -> torch.Tensor:
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sqrt3_2, -half, 0.0],
            [sqrt3_2, 0.0, 0.0, half],
            [0.0, half, -sqrt3_2, 0.0],
            [half, 0.0, 0.0, sqrt3_2],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -half, -sqrt3_2, 0.0],
            [-half, 0.0, 0.0, sqrt3_2],
            [0.0, -sqrt3_2, -half, 0.0],
            [-sqrt3_2, 0.0, 0.0, half],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def build_hcp_syms_inv_mtex() -> torch.Tensor:
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -sqrt3_2, half, 0.0],
            [sqrt3_2, 0.0, 0.0, -half],
            [0.0, -half, sqrt3_2, 0.0],
            [half, 0.0, 0.0, -sqrt3_2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, half, sqrt3_2, 0.0],
            [-half, 0.0, 0.0, -sqrt3_2],
            [0.0, sqrt3_2, half, 0.0],
            [-sqrt3_2, 0.0, 0.0, -half],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

def build_fcc_syms() -> torch.Tensor:
	inv_sqrt_2 = 1.0 / math.sqrt(2.0)
	half = 0.5
	return torch.tensor(
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


def build_fcc_syms_inv() -> torch.Tensor:
	inv_sqrt_2 = 1.0 / math.sqrt(2.0)
	half = 0.5
	return torch.tensor(
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
	)



def compare_seed_sets(
    seeds_a: dict[int, torch.Tensor],
    seeds_b: dict[int, torch.Tensor],
    label_a: str,
    label_b: str,
    tol: float = 1e-5,
) -> bool:
    """Compare seeds with sign-invariant cosine similarity."""
    shared = sorted(set(seeds_a) & set(seeds_b))
    if not shared:
        print("No overlapping l-orders found to compare.")
        return False

    all_pass = True
    print(f"\nComparing {label_a} vs {label_b} seeds")
    for l in shared:
        a = seeds_a[l].to(torch.float64)
        b = seeds_b[l].to(torch.float64)
        a = a / a.norm().clamp_min(1e-12)
        b = b / b.norm().clamp_min(1e-12)

        abs_cos = float(torch.abs(torch.dot(a, b)).item())
        err = 1.0 - abs_cos
        passed = err <= tol
        all_pass &= passed

        status = "PASS" if passed else "FAIL"
        print(f"  L={l}: {status}  |dot|={abs_cos:.12f}  err={err:.3e}  (tol={tol:.1e})")

    return all_pass


def compare_reynolds_projectors(
    syms: torch.Tensor,
    syms_inv: torch.Tensor,
    l_values: tuple[int, ...] = (4, 6),
    tol: float = 1e-5,
    device: str = "cpu",
) -> bool:
    """Compare Reynolds operators P=(1/|G|)sum_g D^l(g) for syms vs syms_inv."""
    all_pass = True
    print("\nComparing Reynolds projectors (syms vs syms_inv)")

    for l in l_values:
        p_syms = _reynolds_operator(syms.to(device), l=l, device=device).to(torch.float64)
        p_inv = _reynolds_operator(syms_inv.to(device), l=l, device=device).to(torch.float64)
        max_abs = float((p_syms - p_inv).abs().max().item())
        passed = max_abs <= tol
        all_pass &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  L={l}: {status}  max|P_syms - P_inv|={max_abs:.3e}  (tol={tol:.1e})")

    return all_pass


if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hcp_syms = build_hcp_syms_mtex()
    hcp_syms_inv = build_hcp_syms_inv_mtex()

    print("\n--- Compute seeds from syms ---")
    seeds_syms = compute_seeds_from_quaternions(hcp_syms, device=device)
    for l in sorted(seeds_syms):
        print(f"s{l} (syms):\n{seeds_syms[l]}")

    print("\n--- Compute seeds from syms_inv ---")
    seeds_syms_inv = compute_seeds_from_quaternions(hcp_syms_inv, device=device)
    for l in sorted(seeds_syms_inv):
        print(f"s{l} (syms_inv):\n{seeds_syms_inv[l]}")

    ok_seed = compare_seed_sets(
        seeds_syms,
        seeds_syms_inv,
        label_a="syms",
        label_b="syms_inv",
        tol=1e-5,
    )
    ok_proj = compare_reynolds_projectors(
        hcp_syms,
        hcp_syms_inv,
        l_values=(4, 6),
        tol=1e-5,
        device=device,
    )

    print(f"\nSeed check (syms vs syms_inv): {'PASS' if ok_seed else 'FAIL'}")
    print(f"Projector check (syms vs syms_inv): {'PASS' if ok_proj else 'FAIL'}")
    print(f"Overall syms vs syms_inv check: {'PASS' if (ok_seed and ok_proj) else 'FAIL'}")

    print("\n TEST FCC \n")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hcp_syms = build_fcc_syms()
    hcp_syms_inv = build_fcc_syms_inv()

    print("\n--- Compute seeds from syms ---")
    seeds_syms = compute_seeds_from_quaternions(hcp_syms, device=device)
    for l in sorted(seeds_syms):
        print(f"s{l} (syms):\n{seeds_syms[l]}")

    print("\n--- Compute seeds from syms_inv ---")
    seeds_syms_inv = compute_seeds_from_quaternions(hcp_syms_inv, device=device)
    for l in sorted(seeds_syms_inv):
        print(f"s{l} (syms_inv):\n{seeds_syms_inv[l]}")

    ok_seed = compare_seed_sets(
        seeds_syms,
        seeds_syms_inv,
        label_a="syms",
        label_b="syms_inv",
        tol=1e-5,
    )
    ok_proj = compare_reynolds_projectors(
        hcp_syms,
        hcp_syms_inv,
        l_values=(4, 6),
        tol=1e-5,
        device=device,
    )

    print(f"\nSeed check (syms vs syms_inv): {'PASS' if ok_seed else 'FAIL'}")
    print(f"Projector check (syms vs syms_inv): {'PASS' if ok_proj else 'FAIL'}")
    print(f"Overall syms vs syms_inv check: {'PASS' if (ok_seed and ok_proj) else 'FAIL'}")


    