import torch
from e3nn import o3
import math
import numpy as np
import time
import sys
sys.path.append('/data/home/umang/Materials/e3nn_Reynolds')

# ==============================================================================
# 1. PHYSICS LAB SETUP (Same as before)
# ==============================================================================
class FCCPhysicsLab:
    def __init__(self, device='cpu'):
        self.device = device
        self.s4 = torch.zeros(9, device=device); self.s4[4] = 0.7638; self.s4[8] = 0.6455
        self.s6 = torch.zeros(13, device=device); self.s6[6] = 0.3536; self.s6[10] = -0.9354
        self.tp4 = o3.FullTensorProduct("1x4e", "1x4e")
        self.tp6 = o3.FullTensorProduct("1x6e", "1x6e")
        
        # Symmetry group for final verification
        inv_sqrt_2 = 1 / math.sqrt(2); half = 0.5
        self.fcc_syms = torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [inv_sqrt_2, inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0], [inv_sqrt_2, 0, -inv_sqrt_2, 0], [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [0, inv_sqrt_2, inv_sqrt_2, 0], [0, inv_sqrt_2, 0, inv_sqrt_2], [0, 0, inv_sqrt_2, inv_sqrt_2],
            [0, inv_sqrt_2, -inv_sqrt_2, 0], [0, 0, inv_sqrt_2, -inv_sqrt_2], [0, inv_sqrt_2, 0, -inv_sqrt_2],
            [half, half, half, half], [half, -half, -half, half], [half, -half, half, -half], [half, half, -half, -half],
            [half, half, half, -half], [half, half, -half, half], [half, -half, half, half], [half, -half, -half, -half],
        ], dtype=torch.float32, device=device)

# ==============================================================================
# 2. FEATURE GENERATOR
# ==============================================================================
def make_unified_fcc(q, lab):
    R = o3.quaternion_to_matrix(q)
    alpha, beta, gamma = o3.matrix_to_angles(R)
    D4 = o3.wigner_D(4, alpha, beta, gamma)
    D6 = o3.wigner_D(6, alpha, beta, gamma)
    f4 = torch.einsum("bij,j->bi", D4, lab.s4)
    f6 = torch.einsum("bij,j->bi", D6, lab.s6)
    return torch.cat([f4, f6], dim=-1)

# ==============================================================================
# 3. ANISOTROPIC TABLE (The "Complex 3D Table")
# ==============================================================================
class AnisotropicLookupTable:
    def __init__(self, lab, angular_res=100, axis_samples=200):
        """
        Builds a dense database of (Score_L4, Score_L6, TorqueDir) -> w
        KEY FIX: Use separate L4 and L6 scores instead of summing them!
        """
        self.lab = lab
        print(f"Building 3D Anisotropic Table ({angular_res} angles x {axis_samples} axes)...")
        
        # 1. Generate Axes (Fibonacci Sphere for uniform sampling)
        golden_ratio = (1 + 5**0.5)/2
        i = torch.arange(0, axis_samples)
        theta = 2 * torch.pi * i / golden_ratio
        phi = torch.acos(1 - 2*(i+0.5)/axis_samples)
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        axes = torch.stack([x, y, z], dim=1) # (N_axes, 3)
        
        # 2. Generate Angles (0 to 180)
        angles = torch.linspace(0.01, torch.pi, angular_res) # Avoid 0 to avoid div/0 in torque
        
        # 3. Create Grid (Cross Product of Axes and Angles)
        # We simulate every combination
        # Total points = angular_res * axis_samples
        grid_axes = axes.repeat_interleave(angular_res, dim=0) # (Total, 3)
        grid_angles = angles.repeat(axis_samples)              # (Total,)
        
        # 4. Convert to Quaternions
        half_ang = grid_angles / 2.0
        w = torch.cos(half_ang)
        s = torch.sin(half_ang)
        q = torch.zeros(grid_axes.shape[0], 4)
        q[:, 0] = w
        q[:, 1:] = grid_axes * s.unsqueeze(1)
        
        # 5. Run Physics Simulation
        feats = make_unified_fcc(q, lab)
        
        # Interact
        f4 = feats[:, :9]; f6 = feats[:, 9:]
        ref4 = lab.s4.view(1, 9).expand(q.shape[0], -1)
        ref6 = lab.s6.view(1, 13).expand(q.shape[0], -1)
        out4 = lab.tp4(f4, ref4)
        out6 = lab.tp6(f6, ref6)
        
        # 6. Extract Observables - KEEP L4 AND L6 SEPARATE!
        self.scores_L4 = out4[:, 0]                   # L4 Scalar (separate)
        self.scores_L6 = out6[:, 0]                   # L6 Scalar (separate)
        self.scores = out4[:, 0] + out6[:, 0]         # Combined (for backward compat)
        torque_vec = out4[:, 1:4] + out6[:, 1:4]      # The Vector
        self.torque_dirs = torch.nn.functional.normalize(torque_vec, dim=1)
        
        # 7. Store the Truth
        self.true_w = w
        
        # 8. Build keys using SEPARATE L4, L6 scores + torque direction
        # Key is now 5D: [score_L4, score_L6, dir_x, dir_y, dir_z]
        self.keys = torch.cat([
            self.scores_L4.unsqueeze(1), 
            self.scores_L6.unsqueeze(1), 
            self.torque_dirs
        ], dim=1)  # (N, 5)
        
    def _fold_to_fundamental_zone(self, dirs):
        """
        Fold directions into the fundamental zone (first octant, sorted).
        Returns: folded_dirs, signs, permutation indices (for unfolding)
        """
        # Store signs for unfolding
        signs = torch.sign(dirs)
        signs[signs == 0] = 1  # Handle zeros
        
        abs_dirs = torch.abs(dirs)
        sorted_dirs, perm = torch.sort(abs_dirs, dim=1)
        
        return sorted_dirs, signs, perm
    
    def _unfold_from_fundamental_zone(self, folded_dirs, signs, perm):
        """
        Unfold directions back from fundamental zone using stored signs and permutation.
        """
        batch_size = folded_dirs.shape[0]
        
        # Inverse permutation
        inv_perm = torch.argsort(perm, dim=1)
        unfolded = torch.gather(folded_dirs, 1, inv_perm)
        
        # Restore signs
        unfolded = unfolded * signs
        
        return unfolded

    def lookup(self, obs_score_L4, obs_score_L6, obs_torque_vec):
        """
        Finds the w value corresponding to the observed L4/L6 scores and torque.
        Uses SEPARATE L4 and L6 scores for disambiguation!
        """
        obs_dir = torch.nn.functional.normalize(obs_torque_vec, dim=1)
        
        # Make Query Key using separate L4, L6 scores + raw direction
        query_key = torch.cat([
            obs_score_L4.unsqueeze(1), 
            obs_score_L6.unsqueeze(1), 
            obs_dir
        ], dim=1)  # (Batch, 5)
        
        # Nearest Neighbor Search (Brute force is fine for <50k points on CPU)
        # Dist = Euclidean distance in the 5D observable space
        # Weight scores higher because they dominate angle determination
        weights = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0], device=query_key.device)
        
        # Compute distances: (Batch, N_Table)
        dists = torch.cdist(query_key * weights, self.keys * weights)
        
        min_indices = torch.argmin(dists, dim=1)
        
        return self.true_w[min_indices].unsqueeze(1)
    
    def lookup_with_folding(self, obs_score_L4, obs_score_L6, obs_torque_vec):
        """
        Alternative: Finds w using fundamental zone folding (Option 2).
        Folds query to FZ, looks up, then uses original direction for reconstruction.
        Now uses SEPARATE L4 and L6 scores!
        """
        obs_dir = torch.nn.functional.normalize(obs_torque_vec, dim=1)
        
        # Fold to fundamental zone for lookup
        folded_dir, signs, perm = self._fold_to_fundamental_zone(obs_dir)
        
        # Build folded keys for table (on first call, cache this)
        if not hasattr(self, 'folded_keys'):
            table_folded, _, _ = self._fold_to_fundamental_zone(self.torque_dirs)
            self.folded_keys = torch.cat([
                self.scores_L4.unsqueeze(1),
                self.scores_L6.unsqueeze(1),
                table_folded
            ], dim=1)
        
        # Make Query Key using folded direction with separate scores
        query_key = torch.cat([
            obs_score_L4.unsqueeze(1), 
            obs_score_L6.unsqueeze(1), 
            folded_dir
        ], dim=1)
        
        weights = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0], device=query_key.device)
        dists = torch.cdist(query_key * weights, self.folded_keys * weights)
        min_indices = torch.argmin(dists, dim=1)
        
        # Return w and the ORIGINAL direction (not folded) for reconstruction
        return self.true_w[min_indices].unsqueeze(1), obs_dir

# ==============================================================================
# 4. DETERMINISTIC DECODER (With 3D Lookup)
# ==============================================================================
def run_experiment_3d(feature_in, lab, table, use_folding=False):
    batch_size = feature_in.shape[0]
    f4_in = feature_in[:, :9]; f6_in = feature_in[:, 9:]
    ref4 = lab.s4.view(1, 9).expand(batch_size, -1)
    ref6 = lab.s6.view(1, 13).expand(batch_size, -1)
    
    out4 = lab.tp4(f4_in, ref4)
    out6 = lab.tp6(f6_in, ref6)
    
    # Observables - KEEP L4 AND L6 SEPARATE!
    obs_score_L4 = out4[:, 0]
    obs_score_L6 = out6[:, 0]
    obs_torque = out4[:, 1:4] + out6[:, 1:4]
    
    if use_folding:
        # Option 2: Use folding for lookup, but original direction for reconstruction
        w_pred, axis_pred = table.lookup_with_folding(obs_score_L4, obs_score_L6, obs_torque)
    else:
        # Option 1: Use raw direction (no folding)
        w_pred = table.lookup(obs_score_L4, obs_score_L6, obs_torque)
        axis_pred = torch.nn.functional.normalize(obs_torque, dim=1)
    
    # Reconstruct
    sin_half = torch.sqrt(torch.clamp(1.0 - w_pred**2, min=0))
    xyz_pred = axis_pred * sin_half
    
    return torch.cat([w_pred, xyz_pred], dim=-1)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    from orix.crystal_map import Phase
    from visualization.ipf_render import render_input_output_comparison
    
    torch.set_printoptions(precision=4, sci_mode=False)
    device = torch.device("cpu")  # CPU for stability
    lab = FCCPhysicsLab(device=device)
    
    # Build the lookup table
    table = AnisotropicLookupTable(lab, angular_res=200, axis_samples=1000)
    
    print("\n" + "="*70)
    print("DETERMINISTIC FCC AUTOENCODER - FULL IMAGE TEST")
    print("="*70)
    
    # 1. Load quaternions from file
    file_path = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/preprocessed_imgs_1D/Open_718_Train_hr_x_normal_0.npy"
    q_numpy = np.load(file_path)
    print(f"Loaded quaternion data with shape: {q_numpy.shape}")
    
    # Convert to torch tensor
    q_all = torch.tensor(q_numpy, dtype=torch.float32, device=device)
    
    # Handle different input shapes
    is_image = False
    img_shape = None
    if q_all.dim() == 1:
        q_all = q_all.unsqueeze(0)
    elif q_all.dim() == 3:
        is_image = True
        img_shape = q_all.shape[:2]  # (H, W)
        original_shape = q_all.shape
        q_all = q_all.reshape(-1, 4)  # (H*W, 4)
        print(f"Reshaped from {original_shape} to {q_all.shape}")
    
    # Normalize all quaternions
    q_all = q_all / torch.norm(q_all, dim=1, keepdim=True)
    num_quats = q_all.shape[0]
    print(f"Processing {num_quats} quaternions...")
    
    # 2. Process in batches
    batch_size = 5000
    q_reconstructed_all = []
    all_overlaps = []
    
    start_time = time.time()
    
    for batch_start in range(0, num_quats, batch_size):
        batch_end = min(batch_start + batch_size, num_quats)
        q_batch = q_all[batch_start:batch_end]
        
        # Encode
        feat = make_unified_fcc(q_batch, lab)
        
        # Decode
        q_rec = run_experiment_3d(feat, lab, table, use_folding=True)
        
        # Find best symmetry match for each quaternion
        q_ex = q_batch.unsqueeze(1).expand(-1, 24, -1)
        s_ex = lab.fcc_syms.unsqueeze(0).expand(q_batch.shape[0], -1, -1)
        w1, x1, y1, z1 = q_ex[...,0], q_ex[...,1], q_ex[...,2], q_ex[...,3]
        w2, x2, y2, z2 = s_ex[...,0], s_ex[...,1], s_ex[...,2], s_ex[...,3]
        
        # q_input * each_symmetry -> targets
        targets = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)  # (batch, 24, 4)
        
        # Find best matching symmetry variant
        dots = torch.sum(targets * q_rec.unsqueeze(1), dim=-1).abs()  # (batch, 24)
        best_idx = torch.argmax(dots, dim=1)  # (batch,)
        best_overlap = dots[torch.arange(q_batch.shape[0]), best_idx]
        
        # Get closest variant (matching input's symmetry)
        closest_quats = targets[torch.arange(q_batch.shape[0]), best_idx]
        
        # Handle sign (q and -q are same rotation)
        direct_dots = torch.sum(closest_quats * q_rec, dim=-1)
        flip_mask = direct_dots < 0
        closest_quats[flip_mask] = -closest_quats[flip_mask]
        
        q_reconstructed_all.append(closest_quats)
        all_overlaps.extend(best_overlap.cpu().tolist())
        
        elapsed = time.time() - start_time
        progress = batch_end / num_quats
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"  Processed {batch_end}/{num_quats} ({progress*100:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    # Concatenate results
    q_reconstructed_all = torch.cat(q_reconstructed_all, dim=0)
    all_overlaps = np.array(all_overlaps)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s ({num_quats/total_time:.0f} quaternions/sec)")
    
    # 3. Statistics
    print("\n" + "="*70)
    print("RECONSTRUCTION STATISTICS")
    print("="*70)
    print(f"Symmetry Overlap:")
    print(f"  Minimum: {np.min(all_overlaps):.6f}")
    print(f"  Maximum: {np.max(all_overlaps):.6f}")
    print(f"  Mean:    {np.mean(all_overlaps):.6f}")
    print(f"  Median:  {np.median(all_overlaps):.6f}")
    print(f"  Std Dev: {np.std(all_overlaps):.6f}")
    
    # Misorientation angle from overlap
    misori_angles = 2 * np.arccos(np.clip(all_overlaps, 0, 1)) * 180 / np.pi
    print(f"\nMisorientation Angle:")
    print(f"  Maximum: {np.max(misori_angles):.4f}°")
    print(f"  Mean:    {np.mean(misori_angles):.4f}°")
    print(f"  Median:  {np.median(misori_angles):.4f}°")
    
    # 4. Render IPF comparison
    if is_image:
        print("\n" + "="*70)
        print("RENDERING IPF COMPARISON")
        print("="*70)
        
        output_png = "deterministic_autoencoder_ipf_comparison.png"
        
        # Reshape back to image format
        q_input_img = q_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        q_output_img = q_reconstructed_all.cpu().reshape(img_shape[0], img_shape[1], 4).numpy()
        
        # Define FCC symmetry
        fcc_sym = Phase(space_group=225).point_group
        
        print(f"Rendering IPF comparison to: {output_png}")
        render_input_output_comparison(
            q_input_img,
            q_output_img,
            fcc_sym,
            out_png=output_png,
            ref_dir="ALL",
            include_key=True,
            overwrite=True,
            format_input=False,
            dpi=300
        )
        print(f"✓ Saved comparison image: {output_png}")
    
    # 5. Final verdict
    success_rate = np.sum(all_overlaps > 0.99) / len(all_overlaps) * 100
    print(f"\n>> Success rate (overlap > 0.99): {success_rate:.2f}%")
    
    if np.min(all_overlaps) > 0.95:
        print(">> SUCCESS: Deterministic autoencoder works on real data!")
    else:
        print(f">> WARNING: {np.sum(all_overlaps < 0.95)} quaternions below 0.95 overlap")
    
    # ===========================================================================
    # DIAGNOSTIC: Analyze what's going wrong
    # ===========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # Find worst cases
    worst_indices = np.argsort(all_overlaps)[:10]  # 10 worst
    best_indices = np.argsort(all_overlaps)[-10:]  # 10 best
    
    print("\n--- 5 WORST CASES ---")
    for i, idx in enumerate(worst_indices[:5]):
        q_in = q_all[idx]
        q_out = q_reconstructed_all[idx]
        
        # Re-encode and analyze
        feat = make_unified_fcc(q_in.unsqueeze(0), lab)
        f4_in = feat[:, :9]; f6_in = feat[:, 9:]
        ref4 = lab.s4.view(1, 9)
        ref6 = lab.s6.view(1, 13)
        out4 = lab.tp4(f4_in, ref4)
        out6 = lab.tp6(f6_in, ref6)
        
        score_L4 = out4[0, 0].item()
        score_L6 = out6[0, 0].item()
        torque = out4[0, 1:4] + out6[0, 1:4]
        torque_norm = torque.norm().item()
        torque_dir = (torque / torque.norm()).numpy()
        
        # Get the angle from input quaternion
        w_in = q_in[0].item()
        angle_in = 2 * np.arccos(np.clip(abs(w_in), 0, 1)) * 180 / np.pi
        axis_in = q_in[1:].numpy()
        axis_in_norm = np.linalg.norm(axis_in)
        if axis_in_norm > 1e-6:
            axis_in = axis_in / axis_in_norm
        
        print(f"\nCase {i+1} (index {idx}, overlap={all_overlaps[idx]:.4f}):")
        print(f"  Input Q:     {q_in.numpy()}")
        print(f"  Input angle: {angle_in:.1f}°, axis: {axis_in}")
        print(f"  Output Q:    {q_out.numpy()}")
        print(f"  Score L4:    {score_L4:.6f}")
        print(f"  Score L6:    {score_L6:.6f}")
        print(f"  Torque norm: {torque_norm:.6f}")
        print(f"  Torque dir:  {torque_dir}")
    
    # Check if torque is near zero (identity rotation)
    print("\n--- TORQUE MAGNITUDE DISTRIBUTION ---")
    torque_norms = []
    for batch_start in range(0, min(10000, num_quats), 1000):
        batch_end = min(batch_start + 1000, num_quats)
        q_batch = q_all[batch_start:batch_end]
        feat = make_unified_fcc(q_batch, lab)
        f4_in = feat[:, :9]; f6_in = feat[:, 9:]
        ref4 = lab.s4.view(1, 9).expand(q_batch.shape[0], -1)
        ref6 = lab.s6.view(1, 13).expand(q_batch.shape[0], -1)
        out4 = lab.tp4(f4_in, ref4)
        out6 = lab.tp6(f6_in, ref6)
        torque = out4[:, 1:4] + out6[:, 1:4]
        torque_norms.extend(torque.norm(dim=1).tolist())
    
    torque_norms = np.array(torque_norms)
    print(f"  Min torque norm:    {torque_norms.min():.6f}")
    print(f"  Max torque norm:    {torque_norms.max():.6f}")
    print(f"  Mean torque norm:   {torque_norms.mean():.6f}")
    print(f"  Near-zero (<0.01):  {np.sum(torque_norms < 0.01)} / {len(torque_norms)}")
    
    # Check L4/L6 score ranges
    print("\n--- SCORE RANGES (first 10k samples) ---")
    scores_L4_samples = []
    scores_L6_samples = []
    for batch_start in range(0, min(10000, num_quats), 1000):
        batch_end = min(batch_start + 1000, num_quats)
        q_batch = q_all[batch_start:batch_end]
        feat = make_unified_fcc(q_batch, lab)
        f4_in = feat[:, :9]; f6_in = feat[:, 9:]
        ref4 = lab.s4.view(1, 9).expand(q_batch.shape[0], -1)
        ref6 = lab.s6.view(1, 13).expand(q_batch.shape[0], -1)
        out4 = lab.tp4(f4_in, ref4)
        out6 = lab.tp6(f6_in, ref6)
        scores_L4_samples.extend(out4[:, 0].tolist())
        scores_L6_samples.extend(out6[:, 0].tolist())
    
    scores_L4_samples = np.array(scores_L4_samples)
    scores_L6_samples = np.array(scores_L6_samples)
    
    print(f"  L4 score range: [{scores_L4_samples.min():.4f}, {scores_L4_samples.max():.4f}]")
    print(f"  L6 score range: [{scores_L6_samples.min():.4f}, {scores_L6_samples.max():.4f}]")
    print(f"  Table L4 range: [{table.scores_L4.min():.4f}, {table.scores_L4.max():.4f}]")
    print(f"  Table L6 range: [{table.scores_L6.min():.4f}, {table.scores_L6.max():.4f}]")
    
    # Check if data scores fall outside table range
    L4_out_of_range = np.sum((scores_L4_samples < table.scores_L4.min().item()) | 
                             (scores_L4_samples > table.scores_L4.max().item()))
    L6_out_of_range = np.sum((scores_L6_samples < table.scores_L6.min().item()) | 
                             (scores_L6_samples > table.scores_L6.max().item()))
    print(f"  L4 out of table range: {L4_out_of_range} / {len(scores_L4_samples)}")
    print(f"  L6 out of table range: {L6_out_of_range} / {len(scores_L6_samples)}")
    
    # KEY DIAGNOSTIC: Check if torque direction matches input axis
    print("\n--- TORQUE vs INPUT AXIS ALIGNMENT ---")
    axis_torque_dots = []
    for idx in range(min(1000, num_quats)):
        q_in = q_all[idx]
        feat = make_unified_fcc(q_in.unsqueeze(0), lab)
        f4_in = feat[:, :9]; f6_in = feat[:, 9:]
        ref4 = lab.s4.view(1, 9)
        ref6 = lab.s6.view(1, 13)
        out4 = lab.tp4(f4_in, ref4)
        out6 = lab.tp6(f6_in, ref6)
        
        torque = out4[0, 1:4] + out6[0, 1:4]
        torque_dir = torch.nn.functional.normalize(torque, dim=0)
        
        # Input axis (from quaternion)
        axis_in = q_in[1:4]
        axis_in_norm = axis_in.norm()
        if axis_in_norm > 1e-6:
            axis_in = axis_in / axis_in_norm
            dot = torch.abs(torch.dot(torque_dir, axis_in)).item()
            axis_torque_dots.append(dot)
    
    axis_torque_dots = np.array(axis_torque_dots)
    print(f"  Mean |dot(torque, input_axis)|: {axis_torque_dots.mean():.4f}")
    print(f"  Min:  {axis_torque_dots.min():.4f}")
    print(f"  Max:  {axis_torque_dots.max():.4f}")
    print(f"  Aligned (>0.9): {np.sum(axis_torque_dots > 0.9)} / {len(axis_torque_dots)}")
    print(f"  Misaligned (<0.5): {np.sum(axis_torque_dots < 0.5)} / {len(axis_torque_dots)}")
    
    # This is critical: if torque doesn't point along input axis, 
    # we can't recover the rotation axis from the torque!