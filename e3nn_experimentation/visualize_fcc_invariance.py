import torch
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from e3nn import o3
from itertools import product

# Use matplotlib backend that doesn't display
plt.switch_backend('Agg')

def generate_fcc_lattice(a=1.0, n_cells=2):
    """
    Generate FCC lattice points.
    FCC lattice: cube vertices + face centers
    
    Args:
        a: Lattice constant
        n_cells: Number of unit cells in each direction
    
    Returns:
        Array of FCC lattice points
    """
    points = []
    
    # Generate base lattice points
    for i, j, k in product(range(-n_cells, n_cells+1), repeat=3):
        # Cube vertices
        points.append([i*a, j*a, k*a])
        # Face centers
        points.append([(i+0.5)*a, (j+0.5)*a, k*a])
        points.append([(i+0.5)*a, j*a, (k+0.5)*a])
        points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    # Remove duplicates
    points = np.array(points)
    points = np.unique(np.round(points, 6), axis=0)
    
    # Normalize to unit sphere for visualization
    distances = np.linalg.norm(points, axis=1)
    distances = distances[distances > 0]  # Remove origin
    max_dist = distances.max()
    
    fcc_points = points / max_dist
    
    return fcc_points

def visualize_fcc_invariance():
    device = 'cpu'
    
    # FCC Detection Weights (from compute_fcc_seeds.py)
    w_l4 = -0.161595   # L=4 weight for FCC
    w_l6 = -0.584355   # L=6 weight for FCC
    
    print("FCC Invariance Visualization")
    print("=" * 60)
    print(f"Using weights from FCC lattice detection:")
    print(f"  L=4 Weight (w_l4) = {w_l4}")
    print(f"  L=6 Weight (w_l6) = {w_l6}")
    print("=" * 60)
    
    # 1. SETUP: Coefficients for L=4 and L=6
    # -------------------------------------------------
    # L=4 has 2L+1 = 9 coefficients
    seed_l4 = torch.zeros(9, device=device)
    seed_l4[4] = 0.7638   # m=0
    seed_l4[8] = 0.6455   # m=+4
    
    # L=6 has 2L+1 = 13 coefficients
    seed_l6 = torch.zeros(13, device=device)
    seed_l6[6] = 0.5      # m=0
    seed_l6[12] = 0.5     # m=+6
    
    # 2. GENERATE GRID: Create a mesh on the sphere
    # -------------------------------------------------
    theta = torch.linspace(0, math.pi, 100)      # 0 to 180 deg
    phi = torch.linspace(0, 2 * math.pi, 200)    # 0 to 360 deg
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # Convert spherical grid to vectors
    x_grid = torch.sin(theta) * torch.cos(phi)
    y_grid = torch.sin(theta) * torch.sin(phi)
    z_grid = torch.cos(theta)
    
    # Flatten for e3nn processing
    vectors = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
    
    # 3. COMPUTE FCC INVARIANCE SCORE
    # -------------------------------------------------
    # Calculate spherical harmonics for L=4 and L=6
    Y4 = o3.spherical_harmonics(4, vectors, normalize=True)
    Y6 = o3.spherical_harmonics(6, vectors, normalize=True)
    
    # Compute signal magnitude for each harmonic
    signal_l4 = torch.einsum("bi,i->b", Y4, seed_l4)
    signal_l6 = torch.einsum("bi,i->b", Y6, seed_l6)
    
    # FCC Invariance Score = w_l4 * |f4| + w_l6 * |f6|
    # (Following the formula from compute_fcc_seeds.py)
    fcc_score = w_l4 * signal_l4.abs() + w_l6 * signal_l6.abs()
    
    # Reshape for plotting
    fcc_score_reshaped = fcc_score.reshape(x_grid.shape)
    
    # Normalize to 0-1 range for visualization
    score_min = fcc_score_reshaped.min()
    score_max = fcc_score_reshaped.max()
    score_normalized = (fcc_score_reshaped - score_min) / (score_max - score_min + 1e-6)
    
    # Prepare contributions for separate views
    l4_abs = (w_l4 * signal_l4.abs()).reshape(x_grid.shape).abs().numpy()
    l6_abs = (w_l6 * signal_l6.abs()).reshape(x_grid.shape).abs().numpy()
    combined_abs = l4_abs + l6_abs
    
    # Normalize each for color scales
    def normalize(arr):
        amin = arr.min()
        amax = arr.max()
        return (arr - amin) / (amax - amin + 1e-6)
    norm_l4 = normalize(l4_abs)
    norm_l6 = normalize(l6_abs)
    norm_combined = normalize(combined_abs)
    
    # 4. DEFORM SPHERE: Map each contribution to its own radius
    # -------------------------------------------------
    radius_l4 = (l4_abs + 0.1)
    radius_l6 = (l6_abs + 0.1)
    radius_combined = (combined_abs + 0.1)
    
    # Cartesian coordinates for each
    X_l4 = radius_l4 * x_grid.numpy()
    Y_l4 = radius_l4 * y_grid.numpy()
    Z_l4 = radius_l4 * z_grid.numpy()
    
    X_l6 = radius_l6 * x_grid.numpy()
    Y_l6 = radius_l6 * y_grid.numpy()
    Z_l6 = radius_l6 * z_grid.numpy()
    
    X_comb = radius_combined * x_grid.numpy()
    Y_comb = radius_combined * y_grid.numpy()
    Z_comb = radius_combined * z_grid.numpy()
    
    # 5. VISUALIZE: Plotly (Interactive) - three plots
    # -------------------------------------------------
    fcc_lattice = generate_fcc_lattice(a=0.3, n_cells=1)
    
    def make_fig(Xv, Yv, Zv, color_vals, title, colorbar_title, opacity=0.7):
        fig = go.Figure(data=[go.Surface(
            x=Xv, y=Yv, z=Zv,
            surfacecolor=color_vals,
            colorscale='Plasma',
            opacity=opacity,
            colorbar=dict(title=colorbar_title)
        )])
        fig.add_trace(go.Scatter3d(
            x=fcc_lattice[:, 0], y=fcc_lattice[:, 1], z=fcc_lattice[:, 2],
            mode='markers', name='FCC Lattice Points', showlegend=True,
            marker=dict(size=5, color='red', opacity=0.85, line=dict(width=0.5, color='darkred'))
        ))
        axis_len_val = float(np.max([Xv.max(), Yv.max(), Zv.max()]))
        fig.add_traces([
            go.Scatter3d(x=[0, axis_len_val], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name='X-Axis', showlegend=False, hoverinfo='skip'),
            go.Scatter3d(x=[0, 0], y=[0, axis_len_val], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='Y-Axis', showlegend=False, hoverinfo='skip'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len_val], mode='lines', line=dict(color='blue', width=5), name='Z-Axis', showlegend=False, hoverinfo='skip')
        ])
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=60),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top', y=1.05,
                xanchor='center', x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)', borderwidth=1
            )
        )
        return fig
    
    fig_l4 = make_fig(X_l4, Y_l4, Z_l4, norm_l4, 'FCC: |L=4| Contribution Surface', '|w_l4 * f4| (normalized)')
    fig_l6 = make_fig(X_l6, Y_l6, Z_l6, norm_l6, 'FCC: |L=6| Contribution Surface', '|w_l6 * f6| (normalized)')
    fig_comb = make_fig(X_comb, Y_comb, Z_comb, norm_combined, 'FCC: Combined |L=4| + |L=6| Surface', 'Combined (normalized)')
    
    print("\nGenerating interactive diagrams...")
    output_html_l4 = "fcc_l4.html"
    output_html_l6 = "fcc_l6.html"
    output_html_comb = "fcc_combined.html"
    fig_l4.write_html(output_html_l4)
    fig_l6.write_html(output_html_l6)
    fig_comb.write_html(output_html_comb)
    print(f"✓ Interactive visualizations saved to: {output_html_l4}, {output_html_l6}, {output_html_comb}")

    # Build a single HTML with 3 interactive subplots (L4, L6, Combined)
    print("Generating combined interactive triptych (L4, L6, Combined)...")
    triptych = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("|L=4| Contribution", "|L=6| Contribution", "Combined |L=4| + |L=6|")
    )

    # Helper to add axes and FCC lattice points to a subplot
    def add_overlays(fig, scene_id, axis_len_val, lattice_pts, label_prefix):
        fig.add_trace(go.Scatter3d(x=[0, axis_len_val], y=[0, 0], z=[0, 0],
                                   mode='lines', line=dict(color='red', width=4), name=f'{label_prefix} X-Axis', showlegend=False, hoverinfo='skip'),
                      row=1, col=scene_id)
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len_val], z=[0, 0],
                                   mode='lines', line=dict(color='green', width=4), name=f'{label_prefix} Y-Axis', showlegend=False, hoverinfo='skip'),
                      row=1, col=scene_id)
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len_val],
                                   mode='lines', line=dict(color='blue', width=4), name=f'{label_prefix} Z-Axis', showlegend=False, hoverinfo='skip'),
                      row=1, col=scene_id)
        fig.add_trace(go.Scatter3d(x=lattice_pts[:, 0], y=lattice_pts[:, 1], z=lattice_pts[:, 2],
                                   mode='markers', name=f'{label_prefix} FCC Lattice', showlegend=True,
                                   marker=dict(size=5, color='red', opacity=0.85,
                                               line=dict(width=0.5, color='darkred'))),
                      row=1, col=scene_id)

    # L4 subplot
    triptych.add_trace(go.Surface(x=X_l4, y=Y_l4, z=Z_l4,
                                  surfacecolor=norm_l4, colorscale='Plasma', opacity=0.75,
                                  showscale=False), row=1, col=1)
    axis_len_l4 = float(np.max([X_l4.max(), Y_l4.max(), Z_l4.max()]))
    fcc_lattice = generate_fcc_lattice(a=0.3, n_cells=1)
    add_overlays(triptych, 1, axis_len_l4, fcc_lattice, 'L4')

    # L6 subplot
    triptych.add_trace(go.Surface(x=X_l6, y=Y_l6, z=Z_l6,
                                  surfacecolor=norm_l6, colorscale='Plasma', opacity=0.75,
                                  showscale=False), row=1, col=2)
    axis_len_l6 = float(np.max([X_l6.max(), Y_l6.max(), Z_l6.max()]))
    add_overlays(triptych, 2, axis_len_l6, fcc_lattice, 'L6')

    # Combined subplot
    triptych.add_trace(go.Surface(x=X_comb, y=Y_comb, z=Z_comb,
                                  surfacecolor=norm_combined, colorscale='Plasma', opacity=0.75,
                                  showscale=False), row=1, col=3)
    axis_len_comb = float(np.max([X_comb.max(), Y_comb.max(), Z_comb.max()]))
    add_overlays(triptych, 3, axis_len_comb, fcc_lattice, 'Combined')

    # Scene layout titles and aspect
    triptych.update_layout(
        margin=dict(l=0, r=0, b=0, t=80),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        scene3=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top', y=1.08,
            xanchor='center', x=0.5,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(0,0,0,0.2)', borderwidth=1
        )
    )

    triptych_html = "fcc_triptych.html"
    triptych.write_html(triptych_html)
    print(f"✓ Combined interactive HTML saved to: {triptych_html}")
    
    # 6. VISUALIZE: Matplotlib (Static 2D - side-by-side contributions)
    # -------------------------------------------------
    print("Generating 2D matplotlib image (L4, L6, Combined)...")
    fig_mpl = plt.figure(figsize=(15, 5))
    
    # Compute individual contributions (absolute magnitude)
    l4_contrib = (w_l4 * signal_l4.abs()).reshape(x_grid.shape).numpy()
    l6_contrib = (w_l6 * signal_l6.abs()).reshape(x_grid.shape).numpy()
    l4_abs = np.abs(l4_contrib)
    l6_abs = np.abs(l6_contrib)
    combined_abs = l4_abs + l6_abs
    
    # Panel 1: |L=4 Contribution|
    ax1 = fig_mpl.add_subplot(131)
    im_l4 = ax1.imshow(l4_abs, cmap='RdYlBu_r', origin='lower', aspect='auto')
    ax1.set_xlabel('Phi (azimuth)')
    ax1.set_ylabel('Theta (polar angle)')
    ax1.set_title('|L=4 Contribution|')
    cbar_l4 = plt.colorbar(im_l4, ax=ax1, label='|w_l4 * f4|')
    
    # Panel 2: |L=6 Contribution|
    ax2 = fig_mpl.add_subplot(132)
    im_l6 = ax2.imshow(l6_abs, cmap='RdYlBu_r', origin='lower', aspect='auto')
    ax2.set_xlabel('Phi (azimuth)')
    ax2.set_ylabel('Theta (polar angle)')
    ax2.set_title('|L=6 Contribution|')
    cbar_l6 = plt.colorbar(im_l6, ax=ax2, label='|w_l6 * f6|')
    
    # Panel 3: Combined |L=4| + |L=6|
    ax3 = fig_mpl.add_subplot(133)
    im_comb = ax3.imshow(combined_abs, cmap='hot', origin='lower', aspect='auto')
    ax3.set_xlabel('Phi (azimuth)')
    ax3.set_ylabel('Theta (polar angle)')
    ax3.set_title('Combined Contribution |L=4| + |L=6|')
    cbar_comb = plt.colorbar(im_comb, ax=ax3, label='Magnitude')
    
    plt.suptitle('Spherical Harmonic Feature Contributions for FCC', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_png = "fcc_contributions.png"
    plt.savefig(output_png, dpi=120, bbox_inches='tight')
    print(f"✓ 2D image saved to: {output_png}")
    plt.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("FCC Invariance Statistics:")
    print("=" * 60)
    print(f"FCC Score Range: [{score_min:.6f}, {score_max:.6f}]")
    print(f"FCC Score Mean:  {fcc_score_reshaped.mean():.6f}")
    print(f"FCC Score Std:   {fcc_score_reshaped.std():.6f}")
    print("\nFiles saved:")
    print(f"  - {output_html_l4}")
    print(f"  - {output_html_l6}")
    print(f"  - {output_html_comb}")
    print(f"  - {output_png}")
    print("=" * 60)

if __name__ == "__main__":
    visualize_fcc_invariance()
