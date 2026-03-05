import torch
import math
from e3nn import o3

# =============================================================================
# 1. YOUR COMPUTED SEEDS (From previous step)
# =============================================================================
def get_cubic_seeds(device="cpu"):
    # L=4 Seed (A1g)
    s4 = torch.zeros(9, device=device)
    s4[4] = 0.7638  # m=0
    s4[8] = 0.6455  # m=+4
    
    # L=6 Seed (A1g)
    s6 = torch.zeros(13, device=device)
    s6[6]  =  0.3536 # m=0
    s6[10] = -0.9354 # m=+4

    return s4, s6

# =============================================================================
# 2. DEFINE PERFECT LATTICES
# =============================================================================
def get_fcc_neighbors():
    """Returns the 12 nearest neighbor vectors for FCC (normalized)."""
    # FCC neighbors are permutations of (±1, ±1, 0)
    # Total 12 vectors.
    vectors = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ]
    return torch.tensor(vectors, dtype=torch.float32)

def get_bcc_neighbors():
    """Returns the 8 nearest neighbor vectors for BCC (normalized)."""
    # BCC neighbors are permutations of (±1, ±1, ±1)
    # Total 8 vectors.
    vectors = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]
    return torch.tensor(vectors, dtype=torch.float32)

def get_sc_neighbors():
    """Returns the 6 nearest neighbor vectors for SC (normalized)."""
    # SC neighbors are permutations of (±1, 0, 0)
    vectors = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    return torch.tensor(vectors, dtype=torch.float32)


# =============================================================================
# 3. CALCULATE FINGERPRINT
# =============================================================================
def compute_lattice_weights(lattice_vectors, s4, s6):
    """
    Projects the lattice vectors onto the L=4 and L=6 seeds.
    Returns the average activation magnitude (The Order Parameter).
    """
    # Normalize input vectors
    lattice_vectors = torch.nn.functional.normalize(lattice_vectors, p=2, dim=1)
    
    # Convert vectors to spherical harmonics Y(r)
    # e3nn uses (x, y, z) -> Y
    sh4 = o3.spherical_harmonics(4, lattice_vectors, normalize=True)
    sh6 = o3.spherical_harmonics(6, lattice_vectors, normalize=True)
    
    # Project onto our Invariant Seeds (Dot Product)
    # "How much does this atom match the cubic shape?"
    proj4 = torch.mv(sh4, s4)
    proj6 = torch.mv(sh6, s6)
    
    # The "Fingerprint" is the average projection over all neighbors
    # (This is equivalent to the Steinhardt Order Parameter Q_l)
    w4 = proj4.mean().item()
    w6 = proj6.mean().item()
    
    return w4, w6

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    s4, s6 = get_cubic_seeds()
    
    # Calculate for FCC
    fcc_vecs = get_fcc_neighbors()
    w4_fcc, w6_fcc = compute_lattice_weights(fcc_vecs, s4, s6)
    
    # Calculate for BCC and SC (for contrast)
    bcc_vecs = get_bcc_neighbors()
    w4_bcc, w6_bcc = compute_lattice_weights(bcc_vecs, s4, s6)

    sc_vecs = get_sc_neighbors()
    w4_sc, w6_sc = compute_lattice_weights(sc_vecs, s4, s6)

    print("="*60)
    print(f"{'Lattice':<10} | {'L=4 Weight (w1)':<20} | {'L=6 Weight (w2)':<20}")
    print("-" * 60)
    print(f"{'FCC':<10} | {w4_fcc:>12.6f}         | {w6_fcc:>12.6f}")
    print(f"{'BCC':<10} | {w4_bcc:>12.6f}         | {w6_bcc:>12.6f}")
    print(f"{'SC':<10}  | {w4_sc:>12.6f}         | {w6_sc:>12.6f}")
    print("="*60)
    
    print("\nINTERPRETATION:")
    print(f"To detect FCC specifically, use these weights:")
    print(f"Score_FCC = ({w4_fcc:.4f}) * |f4| + ({w6_fcc:.4f}) * |f6|")