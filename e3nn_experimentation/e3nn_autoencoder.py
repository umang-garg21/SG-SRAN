import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
import numpy as np
import math
import glob
import os
from orix.quaternion import symmetry as SYM
from visualization.visualize_sr_results import render_sr_hr_side_by_side
from visualization.ipf_render import render_ipf_image

# =============================================================================
# 1. HARD-CODED PHYSICS: THE CUBIC SEEDS
# =============================================================================
def get_cubic_seeds(device):
    """
    Fundamental FCC Invariant vectors for L=4 and L=6.
    These 'seeds' ensure we only populate the cubic manifold.
    """
    s4 = torch.zeros(9, device=device)
    # The L=4 cubic harmonic components in e3nn basis
    s4[0], s4[4], s4[8] = math.sqrt(7/12), math.sqrt(5/24), math.sqrt(5/24)
    
    s6 = torch.zeros(13, device=device)
    # The L=6 cubic harmonic components in e3nn basis
    s6[0], s6[4], s6[8] = math.sqrt(1/8), -math.sqrt(7/16), -math.sqrt(7/16)
    return s4, s6

# =============================================================================
# 2. THE EQUIVARIANT ENCODER (Lifting)
# =============================================================================
class EquivariantEncoder(nn.Module):
    def __init__(self, n_l0=16, n_l4=4, n_l6=2):
        super().__init__()
        self.n_l0, self.n_l4, self.n_l6 = n_l0, n_l4, n_l6
        # Features: 16 Scalars, 4 Cubic L4 tensors, 2 Cubic L6 tensors
        self.irreps_out = o3.Irreps(f"{n_l0}x0e + {n_l4}x4e + {n_l6}x6e")

    def forward(self, quats):
        B = quats.shape[0]
        R = o3.quaternion_to_matrix(quats)
        s4, s6 = get_cubic_seeds(quats.device)

        # Scalars (L=0)
        scalars = torch.zeros(B, 16, device=quats.device)
        scalars[:, 0] = 1.0 # Identity channel

        # Get Euler angles from rotation matrices for Wigner D
        from e3nn.o3 import matrix_to_angles
        alpha, beta, gamma = matrix_to_angles(R)
        
        # Compute Wigner D matrices on CPU then move to device (wigner_D has internal CPU tensors)
        alpha_cpu = alpha.cpu()
        beta_cpu = beta.cpu()
        gamma_cpu = gamma.cpu()
        
        D4 = o3.wigner_D(4, alpha_cpu, beta_cpu, gamma_cpu).to(quats.device)
        D6 = o3.wigner_D(6, alpha_cpu, beta_cpu, gamma_cpu).to(quats.device)
        
        f4 = torch.einsum("bij,j->bi", D4, s4).repeat(1, self.n_l4)
        f6 = torch.einsum("bij,j->bi", D6, s6).repeat(1, self.n_l6)
        
        return torch.cat([scalars, f4, f6], dim=-1)

# =============================================================================
# 3. THE EQUIVARIANT DECODER (Contraction)
# ============================================================================
=
class EquivariantDecoder(nn.Module):
    """
    Reconstructs L=0,1 from L=4,6 using Tensor Product contractions.
    This is the physically rigorous inverse of the lifting operation.
    """
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps("1x0e + 1x1o") # Target: Quaternion

        # Linear projection within the irrep space to mix channels
        self.linear = o3.Linear(self.irreps_in, self.irreps_in)
        
        # Tensor Product Contraction: (L_in x L_in) -> L_out
        # In group theory, 4x4 and 6x6 contain 0 and 1 subspaces.
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_in, self.irreps_out
        )

    def forward(self, x):
        # Mix the channels equivariantly
        x_mixed = self.linear(x)
        # Contract the features against themselves to find lower harmonics
        q_raw = self.tp(x_mixed, x_mixed)
        # Project back to the S3 Hypersphere
        return F.normalize(q_raw, p=2, dim=-1)

# =============================================================================
# 4. DIAGNOSTIC MIRROR MODEL & TEST
# =============================================================================
class MirrorSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EquivariantEncoder()
        self.decoder = EquivariantDecoder(self.encoder.irreps_out)

    def forward(self, q):
        latent = self.encoder(q)
        q_recon = self.decoder(latent)
        return q_recon

def run_mirror_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorSystem().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Executing Physics-Aware Mirror Test on {device}...")

    # Training Loop
    for epoch in range(501):
        # Sample random orientations
        q_gt = F.normalize(torch.randn(512, 4, device=device), p=2, dim=-1)
        
        q_recon = model(q_gt)
        
        # Misorientation Loss (1 - cos(theta))
        R_gt = o3.quaternion_to_matrix(q_gt)
        R_recon = o3.quaternion_to_matrix(q_recon)
        R_rel = torch.bmm(R_gt, R_recon.transpose(1, 2))
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        loss = (1.0 - (trace - 1.0) / 2.0).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            angle = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.1, 1.0)).mean())
            print(f"Epoch {epoch:04d} | Avg Misorientation Error: {angle.item():.6f}°")

    print("\n[Final Scientific Validation]")
    # Check Equivariance: Rotate input and see if output rotates identically
    q_test = F.normalize(torch.randn(1, 4, device=device), p=2, dim=-1)
    
    # 1. Recon the original
    q_recon_orig = model(q_test)
    
    # 2. Rotate the input by 45 deg around X
    rot = o3.matrix_to_quaternion(o3.axis_angle_to_matrix(torch.tensor([1.,0.,0.]), torch.tensor([math.pi/4]))).to(device)
    # Quaternion multiplication (Hamilton product)
    def q_mul(q, p):
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = p.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    q_rotated_in = q_mul(rot, q_test)
    q_recon_rot = model(q_rotated_in)
    
    # The expected output is the original reconstruction rotated
    q_expected_rot = q_mul(rot, q_recon_orig)
    
    equivariance_error = torch.abs(q_recon_rot - q_expected_rot).mean().item()
    print(f"Equivariance Deviation: {equivariance_error:.2e}")
    if equivariance_error < 1e-5:
        print("✅ PASS: The system is perfectly equivariant.")

# =============================================================================
# 5. DATASET PROCESSING & VISUALIZATION
# =============================================================================
def process_dataset_sample(data_path, model, device, sym_class, output_dir="outputs"):
    """
    Process a quaternion dataset sample and visualize input/output IPF maps.
    
    Parameters
    ----------
    data_path : str
        Path to .npy file containing quaternion data (H, W, 4)
    model : nn.Module
        The trained MirrorSystem model
    device : torch.device
        Device for computation
    sym_class : orix symmetry
        Symmetry class for IPF rendering (e.g., SYM.Oh for cubic)
    output_dir : str
        Directory to save visualization outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load quaternion data
    print(f"\nLoading data from: {data_path}")
    q_data = np.load(data_path)  # Shape: (H, W, 4)
    print(f"Data shape: {q_data.shape}")
    
    H, W = q_data.shape[:2]
    
    # Prepare input quaternions
    q_input = torch.from_numpy(q_data).float().to(device)
    q_input_flat = q_input.reshape(-1, 4)
    q_input_flat = F.normalize(q_input_flat, p=2, dim=-1)
    
    # Process through model
    print("Processing through equivariant model...")
    model.eval()
    with torch.no_grad():
        q_output_flat = model(q_input_flat)
    
    # Reshape back to spatial dimensions
    q_output = q_output_flat.reshape(H, W, 4).cpu().numpy()
    q_input_np = q_input_flat.reshape(H, W, 4).cpu().numpy()
    
    # Save outputs
    basename = os.path.splitext(os.path.basename(data_path))[0]
    
    # Render input IPF
    input_ipf_path = os.path.join(output_dir, f"{basename}_input_ipf.png")
    print(f"Rendering input IPF to: {input_ipf_path}")
    render_ipf_image(
        q_input_np,
        sym_class=sym_class,
        out_png=input_ipf_path,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True
    )
    
    # Render output IPF
    output_ipf_path = os.path.join(output_dir, f"{basename}_output_ipf.png")
    print(f"Rendering output IPF to: {output_ipf_path}")
    render_ipf_image(
        q_output,
        sym_class=sym_class,
        out_png=output_ipf_path,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True
    )
    
    # Render side-by-side comparison
    comparison_path = os.path.join(output_dir, f"{basename}_comparison.png")
    print(f"Rendering comparison to: {comparison_path}")
    render_sr_hr_side_by_side(
        sr_q_arr=q_output,
        hr_q_arr=q_input_np,
        sym_class=sym_class,
        out_png=comparison_path,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=300
    )
    
    # Compute reconstruction error
    R_input = o3.quaternion_to_matrix(q_input_flat)
    R_output = o3.quaternion_to_matrix(q_output_flat)
    R_rel = torch.bmm(R_input, R_output.transpose(1, 2))
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    angles = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))
    
    print(f"\n[Reconstruction Statistics]")
    print(f"Mean misorientation: {angles.mean().item():.4f}°")
    print(f"Median misorientation: {angles.median().item():.4f}°")
    print(f"Max misorientation: {angles.max().item():.4f}°")
    print(f"Min misorientation: {angles.min().item():.4f}°")
    
    return {
        "input_ipf": input_ipf_path,
        "output_ipf": output_ipf_path,
        "comparison": comparison_path,
        "mean_error": angles.mean().item(),
        "median_error": angles.median().item()
    }

def run_dataset_test(dataset_dir, num_samples=3, output_dir="outputs"):
    """
    Run the mirror test and then process dataset samples.
    
    Parameters
    ----------
    dataset_dir : str
        Path to directory containing .npy quaternion files
    num_samples : int
        Number of samples to process
    output_dir : str
        Directory to save outputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Run mirror test to train the model
    print("="*80)
    print("STEP 1: Training Mirror System")
    print("="*80)
    model = MirrorSystem().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(501):
        q_gt = F.normalize(torch.randn(512, 4, device=device), p=2, dim=-1)
        q_recon = model(q_gt)
        
        # Original Misorientation Loss: 1 - cos(theta)
        R_gt = o3.quaternion_to_matrix(q_gt)
        R_recon = o3.quaternion_to_matrix(q_recon)
        R_rel = torch.bmm(R_gt, R_recon.transpose(1, 2))
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        loss = (1.0 - (trace - 1.0) / 2.0).mean()

        loss.backward()
        # Add gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.9999, 0.9999)
            angle = torch.rad2deg(torch.acos(cos_angle).mean())
            print(f"Epoch {epoch:04d} | Avg Misorientation Error: {angle.item():.6f}° | Loss: {loss.item():.6f}")
    
    print("\n✅ Mirror system training complete!")
    
    # Step 2: Process dataset samples
    print("\n" + "="*80)
    print("STEP 2: Processing Dataset Samples")
    print("="*80)
    
    # Get list of .npy files
    file_pattern = os.path.join(dataset_dir, "*.npy")
    npy_files = sorted(glob.glob(file_pattern))
    
    if not npy_files:
        print(f"No .npy files found in {dataset_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    print(f"Processing first {num_samples} samples...")
    
    # Use Oh (cubic) symmetry for FCC materials
    sym_class = SYM.Oh
    
    results = []
    for i, npy_file in enumerate(npy_files[:num_samples]):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        result = process_dataset_sample(
            npy_file,
            model,
            device,
            sym_class,
            output_dir=output_dir
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(results)} samples")
    print(f"All outputs saved to: {output_dir}/")
    print("\nMean reconstruction errors:")
    for i, res in enumerate(results):
        print(f"  Sample {i+1}: {res['mean_error']:.4f}°")

if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718/Train/HR_Images"
    OUTPUT_DIR = "outputs/e3nn_gan_results"
    NUM_SAMPLES = 3
    
    # Run both mirror test and dataset processing
    run_dataset_test(DATASET_DIR, num_samples=NUM_SAMPLES, output_dir=OUTPUT_DIR)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from e3nn import o3
# from e3nn.nn import FullyConnectedNet
# import numpy as np
# import math
# import glob
# import os
# from orix.quaternion import symmetry as SYM
# from visualization.visualize_sr_results import render_sr_hr_side_by_side

# # =============================================================================
# # 1. HARD-CODED PHYSICS: THE CUBIC SEEDS & SYMMETRY
# # =============================================================================
# def get_cubic_seeds(device):
#     s4 = torch.zeros(9, device=device)
#     s4[0], s4[4], s4[8] = math.sqrt(7/12), math.sqrt(5/24), math.sqrt(5/24)
#     s6 = torch.zeros(13, device=device)
#     s6[0], s6[4], s6[8] = math.sqrt(1/8), -math.sqrt(7/16), -math.sqrt(7/16)
#     return s4, s6

# def get_fcc_symmetry_matrices(device):
#     """Returns the 48 rotation matrices of the Oh point group using orix."""
#     oh_sym = SYM.Oh
#     # Convert orix symmetry to 3x3 matrices
#     sym_mats = torch.from_numpy(oh_sym.to_matrix()).float().to(device)
#     return sym_mats

# # =============================================================
# # 1. RIGOROUS ENCODER (Fixed Physics)
# # =============================================================
# class EquivariantEncoder(nn.Module):
#     def __init__(self, n_0=16, n_l4=4, n_l6=2):
#         super().__init__()
#         self.irreps_out = o3.Irreps(f"{n_0}x0e + {n_l4}x4e + {n_l6}x6e")
#         self.n_l0, self.n_l4, self.n_l6 = n_0, n_l4, n_l6

#     def forward(self, quats):
#         B = quats.shape[0]
#         device = quats.device
#         R = o3.quaternion_to_matrix(quats)
#         s4, s6 = get_cubic_seeds(device)
        
#         # Wigner-D requires Euler angles
#         R_cpu = R.cpu()
#         alpha, beta, gamma = o3.matrix_to_angles(R_cpu)
#         D4 = o3.wigner_D(4, alpha, beta, gamma).to(device)
#         D6 = o3.wigner_D(6, alpha, beta, gamma).to(device)
        
#         f4 = torch.einsum("bij,j->bi", D4, s4).repeat(1, self.n_l4)
#         f6 = torch.einsum("bij,j->bi", D6, s6).repeat(1, self.n_l6)
        
#         # Identity scalar (Channel 0) and small noise for others to keep grads alive
#         scalars = torch.zeros(B, 16, device=device)
#         scalars[:, 0] = 1.0
        
#         return torch.cat([scalars, f4, f6], dim=-1)

# # =============================================================
# # 2. STABILIZED DECODER (With Residual Bypass)
# # =============================================================
# class EquivariantDecoder(nn.Module):
#     def __init__(self, irreps_in):
#         super().__init__()
#         self.irreps_in = o3.Irreps(irreps_in)
#         self.irreps_out = o3.Irreps("1x0e + 1x1o")
        
#         self.linear = o3.Linear(self.irreps_in, self.irreps_in)
#         self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_in, self.irreps_out)
#         self.scalar_bypass = o3.Linear("16x0e", self.irreps_out)

#         # REDUCED BOOST: 15.0 was too high, causing the 'Lock-in'
#         with torch.no_grad():
#             self.linear.weight.data.mul_(2.0) 
#             self.tp.weight.data.mul_(2.0)
#             # Initialize bypass to be near-zero so TP has to learn
#             self.scalar_bypass.weight.data.fill_(0.01)

#     def forward(self, x):
#         scalars = x[:, :16]
#         x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        
#         #Add a tiny bit of noise during training to prevent mode collapse
#         if self.training:
#            x = x + torch.randn_like(x) * 0.01
        
#         x_mixed = self.linear(x)
#         q_main = self.tp(x_mixed, x_mixed)
#         q_bypass = self.scalar_bypass(scalars)
        
#         return F.normalize(q_main + q_bypass, p=2, dim=-1)
#         #return F.normalize(q_main, p=2, dim=-1)

# # =============================================================
# # 3. STABLE SYMMETRY LOSS (The Anchor)
# # =============================-=========================x=======
# def stable_symmetry_aware_loss(q_pred, q_gt, sym_quats):
#     """
#     Physically rigorous loss for FCC materials.
    
#     Parameters:
#         q_pred: (B, 4) Predicted quaternions from the Decoder.
#         q_gt:   (B, 4) Ground truth quaternions from EBSD.
#         sym_quats: (48, 4) Oh symmetry group operators in quaternion form.
#     """
#     B = q_pred.shape[0]
#     device = q_pred.device

#     # 1. Expand GT into all 48 symmetric equivalents
#     # We use a batch quaternion multiplication (Hamilton product)
#     def hamilton_product(p, q):
#         # p: (1, 48, 4), q: (B, 1, 4) -> (B, 48, 4)
#         pw, px, py, pz = p.unbind(-1)
#         qw, qx, qy, qz = q.unbind(-1)
#         return torch.stack([
#             pw*qw - px*qx - py*qy - pz*qz,
#             pw*qx + px*qw + py*qz - pz*qy,
#             pw*qy - px*qz + py*qw + pz*qx,
#             pw*qz + px*qy - py*qx + pz*qw
#         ], dim=-1)

#     # Generate the 48 physical equivalents of the ground truth
#     q_gt_expanded = hamilton_product(sym_quats.unsqueeze(0), q_gt.unsqueeze(1)) 

#     # 2. Account for Double-Cover (q and -q are the same rotation)
#     # We find the distance to all 48 and all -48 versions
#     # dist shape: (B, 48)
#     dist_pos = torch.sum((q_pred.unsqueeze(1) - q_gt_expanded)**2, dim=-1)
#     dist_neg = torch.sum((q_pred.unsqueeze(1) + q_gt_expanded)**2, dim=-1)

#     # Find the absolute closest symmetry per sample
#     min_dist_pos, idx_pos = torch.min(dist_pos, dim=1)
#     min_dist_neg, idx_neg = torch.min(dist_neg, dim=1)

#     # Select the target that provides the shortest path for the gradient
#     # This prevents the 'kick' from symmetry jumps
#     use_neg = min_dist_neg < min_dist_pos
#     chosen_idx = torch.where(use_neg, idx_neg, idx_pos)

#     # Gather the best q_gt_sym
#     best_q_gt = torch.gather(q_gt_expanded, 1, chosen_idx.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
#     # Apply negative sign if -q was closer
#     best_q_gt = torch.where(use_neg.unsqueeze(-1), -best_q_gt, best_q_gt)

#     # 3. PHYSICAL LOSS (For Monitoring and Logs)
#     # We use the trace of the relative rotation matrix for the 'True' value
#     with torch.no_grad():
#         R_pred = o3.quaternion_to_matrix(q_pred)
#         R_gt_best = o3.quaternion_to_matrix(best_q_gt)
#         R_rel = torch.bmm(R_pred, R_gt_best.transpose(1, 2))
#         trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
#         # Loss value between 0 (matched) and 1 (180 deg)
#         loss_physics = (1.0 - (trace - 1.0) / 2.0).mean()

#     # 4. GRADIENT LOSS (For the Optimizer)
#     # SCALE: We multiply by 1000.0 to move the 'Dec grad' from 1e-10 to 1e-2.
#     # This provides the 'Driving Force' for training.
#     loss_gradient = torch.sum((q_pred - best_q_gt)**2, dim=-1).mean()

#     # 5. STRAIGHT-THROUGH ESTIMATOR (STE)
#     # Forward: returns actual physical misorientation metric
#     # Backward: flows stable, amplified gradients
#     #loss = loss_physics.detach() + loss_gradient - loss_gradient.detach()
#     loss = loss_gradient

#     return loss
# # =============================================================================
# # 4. SYMMETRY-INVARIANT LOSS (Change 2: Oh Invariance)
# # =============================================================================
# def symmetry_invariant_misorientation_loss(q_pred, q_gt, sym_mats, edge_width=0.1):
#     """
#     Calculates the minimum angular distance considering 48 FCC symmetries.
#     Uses linear approximation at extremes and in gradient computation to ensure stable gradients.
    
#     The key insight: we compute the loss value using full nonlinear operations,
#     but provide a linearized gradient path through straight-through estimator.
#     """
#     R_pred = o3.quaternion_to_matrix(q_pred)
#     R_gt = o3.quaternion_to_matrix(q_gt)
    
#     # Broadcast across 48 symmetries: R_rel = R_pred * (R_sym * R_gt)^T
#     # sym_mats: (48, 3, 3), R_gt: (B, 3, 3)
#     # First compute R_gt_sym = sym_mats @ R_gt for each symmetry
#     B = R_gt.shape[0]
#     R_gt_sym = torch.einsum("nij,bjk->bnik", sym_mats, R_gt)  # (B, 48, 3, 3)
    
#     # Compute R_rel[b,n] = R_pred[b] @ R_gt_sym[b,n].T
#     # Reshape for batch matrix multiply
#     R_pred_expanded = R_pred.unsqueeze(1).expand(B, 48, 3, 3).reshape(B * 48, 3, 3)
#     R_gt_sym_t = R_gt_sym.transpose(-2, -1).reshape(B * 48, 3, 3)
#     R_rel = torch.bmm(R_pred_expanded, R_gt_sym_t).reshape(B, 48, 3, 3)
    
#     # Compute traces for each (batch, symmetry) pair
#     traces = R_rel[:, :, 0, 0] + R_rel[:, :, 1, 1] + R_rel[:, :, 2, 2]  # (B, 48)
#     max_trace, max_idx = torch.max(traces, dim=1)  # Both (B,)
    
#     # Clamp and compute cos_theta
#     cos_theta = torch.clamp((max_trace - 1.0) / 2.0, -1.0, 1.0)
    
#     # Standard loss: 1 - cos_theta
#     # Use the actual loss value for both forward and backward
#     # The clamping above prevents gradient issues at extremes
#     loss = (1.0 - cos_theta).mean()
    
#     return loss

# def simple_misorientation_loss(q_pred, q_gt):
#     """Simple quaternion dot product loss without symmetry."""
#     dot = torch.abs((q_pred * q_gt).sum(dim=-1))
#     dot = torch.clamp(dot, 0.0, 1.0)
#     return (1.0 - dot).mean()

# # =============================================================================
# # 5. INTEGRATED MIRROR SYSTEM & DATASET PROCESSING (Change 3)
# # =============================================================================
# class MirrorSystem(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = EquivariantEncoder()
#         self.decoder = EquivariantDecoder(self.encoder.irreps_out)

#     def forward(self, q):
#         latent = self.encoder(q)
#         return self.decoder(latent)

# def process_ebsd_sample(data_path, model, device, sym_mats, output_dir=None, stage=""):
#     """Processes real NPY data, renders IPF maps, and returns reconstructed quaternions."""
#     q_data = np.load(data_path)
#     H, W, _ = q_data.shape
#     q_input = torch.from_numpy(q_data).float().to(device).reshape(-1, 4)
#     q_input = F.normalize(q_input, p=2, dim=-1)
    
#     model.eval()
#     with torch.no_grad():
#         q_output = model(q_input)
    
#     # Simple reconstruction quality metric
#     similarity = torch.abs((q_output * q_input).sum(dim=-1)).mean().item()
    
#     print(f"\nSample: {os.path.basename(data_path)}")
#     print(f"Reconstruction similarity: {similarity:.4f} (1.0 = perfect)")
    
#     # Reshape to spatial dimensions
#     q_input_np = q_input.reshape(H, W, 4).cpu().numpy()
#     q_output_np = q_output.reshape(H, W, 4).cpu().numpy()
    
#     # Render IPF comparison if output directory provided
#     if output_dir:
#         basename = os.path.splitext(os.path.basename(data_path))[0]
#         ipf_path = os.path.join(output_dir, f"{basename}_{stage}_comparison.png")
        
#         render_sr_hr_side_by_side(
#             sr_q_arr=q_output_np,
#             hr_q_arr=q_input_np,
#             sym_class=SYM.Oh,
#             out_png=ipf_path,
#             ref_dir="Z",
#             include_key=True,
#             overwrite=True,
#             format_input=True,
#             dpi=150
#         )
#         print(f"IPF map saved: {ipf_path}")
    
#     return q_output_np

# # =============================================================================
# # EXECUTION
# # =============================================================================
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sym_mats = get_fcc_symmetry_matrices(device)
    
#     model = MirrorSystem().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate
    
#     # Setup output directories and dataset paths
#     DATASET_DIR = "/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718/Train/HR_Images"
#     OUTPUT_DIR = "outputs/e3nn_gan_results"
#     OUTPUT_BEFORE = os.path.join(OUTPUT_DIR, "before_training")
#     OUTPUT_AFTER = os.path.join(OUTPUT_DIR, "after_training")
    
#     os.makedirs(OUTPUT_BEFORE, exist_ok=True)
#     os.makedirs(OUTPUT_AFTER, exist_ok=True)
    
#     # Get dataset files
#     import glob
#     npy_files = sorted(glob.glob(f"{DATASET_DIR}/*.npy"))[:3]  # First 3 samples
    
#     # Process BEFORE training
#     # if len(npy_files) > 0:
#     #     print("\n" + "="*80)
#     #     print("PROCESSING BEFORE TRAINING")
#     #     print("="*80)
#     #     for i, npy_file in enumerate(npy_files):
#     #         print(f"\nSample {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
#     #         q_output = process_ebsd_sample(npy_file, model, device, sym_mats, 
#     #                                       output_dir=OUTPUT_BEFORE, stage="before")
#     #         output_path = os.path.join(OUTPUT_BEFORE, f"reconstructed_{os.path.basename(npy_file)}")
#     #         np.save(output_path, q_output)
#     #         print(f"Quaternions saved: {output_path}")
    
    
#     # print("\n" + "="*80)
#     # print("STARTING TRAINING")
#     # print("="*80)
    
#     # Training with Symmetry-Invariant Loss
#     print("Testing encoder output...")
#     q_test = F.normalize(torch.randn(4, 4, device=device), p=2, dim=-1)
#     q_test.requires_grad_(True)
#     latent_test = model.encoder(q_test)
#     print(f"Latent shape: {latent_test.shape}")
#     print(f"Latent stats: min={latent_test.min():.3f}, max={latent_test.max():.3f}, mean={latent_test.mean():.3f}, std={latent_test.std():.3f}")
#     print(f"Has NaN: {torch.isnan(latent_test).any()}")
#     print(f"Has Inf: {torch.isinf(latent_test).any()}")
    
#     # Test if gradients flow through encoder
#     test_loss = latent_test.sum()
#     test_loss.backward()
#     if q_test.grad is not None:
#         grad_norm = q_test.grad.norm().item()
#         print(f"Gradient flows through encoder: YES (grad norm: {grad_norm:.6f})")
#     else:
#         print(f"Gradient flows through encoder: NO - THIS IS THE PROBLEM!")
#     print()
    
#     for epoch in range(501):
#         q_gt = F.normalize(torch.randn(512, 4, device=device), p=2, dim=-1)
#         q_recon = model(q_gt)
        
#         # Check reconstruction quality
#         if epoch % 100 == 0:
#             recon_similarity = torch.abs((q_recon * q_gt).sum(dim=-1)).mean().item()
#             print(f"\nReconstruction similarity (before loss): {recon_similarity:.4f}")
        
#         # Check for NaN in model output
#         if torch.isnan(q_recon).any():
#             print(f"NaN in model output at epoch {epoch}")
#             print(f"q_recon stats: min={q_recon.min():.3f}, max={q_recon.max():.3f}")
#             break
        
#         #loss = symmetry_invariant_misorientation_loss(q_recon, q_gt, sym_mats)
#         #loss = simple_misorientation_loss(q_recon, q_gt)
#         loss = stable_symmetry_aware_loss(q_recon, q_gt, o3.matrix_to_quaternion(sym_mats))
#         # Check for NaN in loss
#         if torch.isnan(loss):
#             print(f"NaN in loss at epoch {epoch}")
#             break
        
#         loss.backward()
        
#         # Check for NaN in gradients
#         has_nan_grad = False
#         for name, param in model.named_parameters():
#             if param.grad is not None and torch.isnan(param.grad).any():
#                 print(f"NaN gradient in {name} at epoch {epoch}")
#                 print(f"Param stats: min={param.min():.3f}, max={param.max():.3f}")
#                 print(f"Grad stats: min={param.grad.min():.3f}, max={param.grad.max():.3f}")
#                 has_nan_grad = True
#                 break
#         if has_nan_grad:
#             break

#         # NEW: Clip gradients before the step
#         #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         # Check gradient norms before optimizer step (for diagnostics)
#         if epoch % 100 == 0:
#             encoder_grad = sum(p.grad.norm().item()**2 for p in model.encoder.parameters() if p.grad is not None)**0.5
#             decoder_grad = sum(p.grad.norm().item()**2 for p in model.decoder.parameters() if p.grad is not None)**0.5
        
#         optimizer.step()
#         optimizer.zero_grad()
        
#         if epoch % 100 == 0:
#             # Physical angle error from loss value
#             # loss = 1 - cos(theta), so cos(theta) = 1 - loss
#             cos_theta = torch.clamp(1.0 - loss.detach(), -1.0, 1.0)
#             angle = torch.rad2deg(torch.acos(cos_theta))
            
#             print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | Physical Misorientation: {angle.item():.6f}° | "
#                   f"Enc grad: {encoder_grad:.2e} | Dec grad: {decoder_grad:.2e}")

#     print("\n" + "="*80)
#     print("TRAINING COMPLETE")
#     print("="*80)
    
#     # Process AFTER training
#     if len(npy_files) > 0:
#         print("\n" + "="*80)
#         print("PROCESSING AFTER TRAINING")
#         print("="*80)
#         for i, npy_file in enumerate(npy_files):
#             print(f"\nSample {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
#             q_output = process_ebsd_sample(npy_file, model, device, sym_mats,
#                                           output_dir=OUTPUT_AFTER, stage="after")
#             output_path = os.path.join(OUTPUT_AFTER, f"reconstructed_{os.path.basename(npy_file)}")
#             np.save(output_path, q_output)
#             print(f"Quaternions saved: {output_path}")
        
#         print(f"\n✓ Before training results: {OUTPUT_BEFORE}/")
#         print(f"✓ After training results:  {OUTPUT_AFTER}/")
#     else:
#         print(f"\nNo .npy files found in {DATASET_DIR}")
    
#     # Example Processing (legacy comment)
#     #process_ebsd_sample("path_to_your_data.npy", model, device, sym_mats)