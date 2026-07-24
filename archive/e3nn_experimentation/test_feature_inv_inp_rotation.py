import torch
import math
from e3nn import o3

def verify_fcc_invariance():
    print("="*60)
    print("VERIFICATION: FCC AWARENESS (Fundamental Zone Folding)")
    print("="*60)
    
    device = "cpu"
    
    # 1. SETUP: The Encoder (Seeds)
    s4 = torch.zeros(9); s4[4] = 0.7638; s4[8] = 0.6455
    s6 = torch.zeros(13); s6[6] = 0.3536; s6[10] = -0.9354
    
    # 2. SETUP: The 24 Symmetry Quaternions
    inv_sqrt_2 = 1 / math.sqrt(2); half = 0.5
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

    # 3. Create ONE Random Quaternion (The "True" Orientation)
   
    q_random = torch.randn(1, 4)
    q_random = q_random / torch.norm(q_random)
    
    # q_random= torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    # q_random = q_random / torch.norm(q_random)

    print(f"Random Input Quaternion: {q_random.numpy()[0]}")
    
    # 4. Generate the 24 Equivalent Quaternions (Crystal Symmetries)
    # IMPORTANT: We multiply on the RIGHT (q * sym) to represent 
    # re-indexing the crystal axes.
    # q_batch[i] = q_random * sym[i]
    
    # Quaternion multiply logic (q1 * q2)
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
        w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)

    # Broadcast q_random to (24, 4)
    q_batch = quat_mul(q_random.repeat(24, 1), fcc_syms)
    
    print(f"Generated {len(q_batch)} equivalent orientations.")
    print(f"Example 1: {q_batch[0].numpy()}")
    print(f"Example 2: {q_batch[5].numpy()} (Visibly different!)")

    # 5. Run Encoder
    R = o3.quaternion_to_matrix(q_batch)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    D4 = o3.wigner_D(4, alpha, beta, gamma)
    D6 = o3.wigner_D(6, alpha, beta, gamma)
    
    f4 = torch.einsum("bij,j->bi", D4, s4)
    f6 = torch.einsum("bij,j->bi", D6, s6)
    
    print("f4 and f6 computed for all 24 orientations.")
    print("f4[0]: ", f4[0])
    print("f4[5]: ", f4[5])
    print("f6[0]: ", f6[0])
    print("f6[5]: ", f6[5])

    # 6. Check Variance
    # We expect f4[0] == f4[1] == ... == f4[23]
    # Even though q[0] != q[1]
    
    f4_dev = torch.norm(f4 - f4[0], dim=1).max()
    f6_dev = torch.norm(f6 - f6[0], dim=1).max()
    
    print("\nRESULTS:")
    print(f"L=4 Max Difference between equivalents: {f4_dev.item():.6e}")
    print(f"L=6 Max Difference between equivalents: {f6_dev.item():.6e}")
    
    if f4_dev < 1e-4 and f6_dev < 1e-4:
        print("\n✓ SUCCESS: The encoder produced a unique latent for all 24 symmetries.")
        print("  This latent represents the 'Canonical Orientation' (Fundamental Zone).")
    else:
        print("\n❌ FAILURE: The encoder is sensitive to symmetry choice.")

if __name__ == "__main__":
    verify_fcc_invariance()