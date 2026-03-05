"""
Example usage of the EBSDSuper class for EBSD super-resolution
"""
import torch
from simple_encoder_decoder import EBSDSuper

# ==============================================================================
# EXAMPLE 1: Simple forward pass with quaternions
# ==============================================================================
def example_simple_forward():
    print("="*70)
    print("EXAMPLE 1: Simple Forward Pass")
    print("="*70)
    
    # Initialize model
    model = EBSDSuper(device='cuda:1', grid_samples=10000, batch_size=1000)
    
    # Create some random quaternions (normalized)
    quats = torch.randn(100, 4, device='cuda:1')
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    
    # Process
    output_quats = model(quats)
    
    print(f"Input shape: {quats.shape}")
    print(f"Output shape: {output_quats.shape}")
    print(f"Sample input: {quats[0].cpu().numpy()}")
    print(f"Sample output: {output_quats[0].cpu().numpy()}")
    print()

# ==============================================================================
# EXAMPLE 2: Process an image file with automatic rendering
# ==============================================================================
def example_process_image():
    print("="*70)
    print("EXAMPLE 2: Process Image File")
    print("="*70)
    
    # Initialize model
    model = EBSDSuper(device='cuda:1', grid_samples=10000, batch_size=1000)
    
    # Process image with automatic comparison rendering
    input_file = "/data/home/umang/Materials/Materials_data_mount/EBSD/IN718_FZ_2D_SR_x4/Open718_QSR_x4/Train/HR_Data/Open718_QSR_x4_train_hr_x_block_0.npy"
    
    output_quats, stats = model.process_image(
        input_file,
        output_path="ebsd_reconstruction.png",
        render_comparison=True,
        dpi=300
    )
    
    print(f"\nOutput quaternion array shape: {output_quats.shape}")
    print(f"Statistics summary: {stats['summary']}")
    print()

# ==============================================================================
# EXAMPLE 3: Batch processing with statistics
# ==============================================================================
def example_batch_with_stats():
    print("="*70)
    print("EXAMPLE 3: Batch Processing with Statistics")
    print("="*70)
    
    # Initialize model
    model = EBSDSuper(device='cuda:1', grid_samples=10000, batch_size=500)
    
    # Create random quaternions
    quats = torch.randn(5000, 4, device='cuda:1')
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    
    # Process with statistics
    output_quats, stats = model.process_batch(quats, return_stats=True)
    
    print(f"Processed {quats.shape[0]} quaternions")
    print(f"\nStatistics:")
    print(f"  Mean error: {stats['summary']['error_mean']:.6e}")
    print(f"  Mean misorientation: {stats['summary']['misorientation_mean']:.4f}°")
    print()

if __name__ == "__main__":
    # Run examples
    print("\n" + "="*70)
    print("EBSD SUPER-RESOLUTION MODULE - USAGE EXAMPLES")
    print("="*70 + "\n")
    
    # Example 1: Simple forward pass
    example_simple_forward()
    
    # Example 2: Process image (this is the main use case)
    example_process_image()
    
    # Example 3: Batch processing
    # example_batch_with_stats()  # Uncomment to run