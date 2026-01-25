import torch
import math
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from e3nn import o3

def visualize_l4_seed():
    device = 'cpu'
    
    # 1. SETUP: The Exact Coefficients you calculated
    # -------------------------------------------------
    # L=4 has 2L+1 = 9 coefficients.
    # Indices: 0(m=-4) ... 4(m=0) ... 8(m=+4)
    seed_coeffs = torch.zeros(9, device=device)
    seed_coeffs[4] = 0.7638  # m=0  (Aligns with Z)
    seed_coeffs[8] = 0.6455  # m=+4 (Aligns with X/Y via Cosine)
    
    # 2. GENERATE GRID: Create a mesh on the sphere
    # -------------------------------------------------
    # We use a standard lat/lon grid for a smooth surface
    theta = torch.linspace(0, math.pi, 100)      # 0 to 180 deg
    phi = torch.linspace(0, 2 * math.pi, 200)    # 0 to 360 deg
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # Convert spherical grid to vectors (x, y, z)
    # These are the "directions" we will query
    x_grid = torch.sin(theta) * torch.cos(phi)
    y_grid = torch.sin(theta) * torch.sin(phi)
    z_grid = torch.cos(theta)
    
    # Flatten for e3nn processing (Batch of vectors)
    vectors = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
    
    # 3. COMPUTE SHAPE: e3nn calculates the signal magnitude
    # -------------------------------------------------
    # Calculate Y_lm for all points on grid
    Y4 = o3.spherical_harmonics(4, vectors, normalize=True)
    
    # The Signal = Dot Product(Y_lm, Coefficients)
    # This tells us "How strong is the L=4 shape in this direction?"
    signal = torch.einsum("bi,i->b", Y4, seed_coeffs)
    
    # 4. DEFORM THE SPHERE: Map magnitude to Radius
    # -------------------------------------------------
    # We want the radius of the plot to match the signal strength.
    # r(theta, phi) = |signal|
    # (Using abs() because lobes can be negative, but volume is positive)
    radius = signal.abs().reshape(x_grid.shape)
    
    # Convert back to Cartesian for plotting
    # New X = old_direction_X * radius
    X = radius * x_grid
    Y = radius * y_grid
    Z = radius * z_grid

    # 5. VISUALIZE: Plotly
    # -------------------------------------------------
    fig = go.Figure(data=[go.Surface(
        x=X.numpy(), 
        y=Y.numpy(), 
        z=Z.numpy(),
        surfacecolor=radius.numpy(), # Color by magnitude
        colorscale='Viridis',
        opacity=0.9
    )])

    # Add Axis Lines (To prove alignment)
    axis_len = 1.2 * radius.max()
    lines = [
        # X-axis (Red)
        go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name='X-Axis'),
        # Y-axis (Green)
        go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='Y-Axis'),
        # Z-axis (Blue)
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines', line=dict(color='blue', width=5), name='Z-Axis')
    ]
    fig.add_traces(lines)

    fig.update_layout(
        title='Diagram 1: The "Spiky Ball" (L=4 Invariant Seed)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' # Keep 1:1 aspect ratio so it doesn't look stretched
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    print("Generating interactive diagram...")
    # Save to HTML file
    output_html = "spiky_ball_l4.html"
    fig.write_html(output_html)
    print(f"Interactive visualization saved to: {output_html}")
    
    # Generate 2D matplotlib image
    print("Generating 2D matplotlib image...")
    fig_mpl = plt.figure(figsize=(12, 5))
    
    # 3D plot view 1
    ax1 = fig_mpl.add_subplot(121, projection='3d')
    ax1.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), 
                     facecolors=plt.cm.viridis(radius.numpy() / radius.max()),
                     shade=False, alpha=0.9)
    
    # Add axes
    axis_len_val = float(axis_len)
    ax1.plot([0, axis_len_val], [0, 0], [0, 0], 'r-', linewidth=3, label='X-axis')
    ax1.plot([0, 0], [0, axis_len_val], [0, 0], 'g-', linewidth=3, label='Y-axis')
    ax1.plot([0, 0], [0, 0], [0, axis_len_val], 'b-', linewidth=3, label='Z-axis')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Spiky Ball (L=4 Seed)')
    ax1.legend()
    
    # 2D heatmap view
    ax2 = fig_mpl.add_subplot(122)
    im = ax2.imshow(radius.numpy(), cmap='viridis', origin='lower')
    ax2.set_xlabel('Phi (azimuth)')
    ax2.set_ylabel('Theta (polar angle)')
    ax2.set_title('2D Heatmap: Signal Magnitude')
    plt.colorbar(im, ax=ax2, label='Magnitude')
    
    plt.tight_layout()
    output_png = "spiky_ball_l4.png"
    plt.savefig(output_png, dpi=100, bbox_inches='tight')
    print(f"2D image saved to: {output_png}")
    plt.close()
    
    # Also try to display (will fail in terminal but that's okay)
    try:
        fig.show()
    except:
        pass

if __name__ == "__main__":
    visualize_l4_seed()