"""
3D Cluster Boundary Visualization
Interactive 3D visualization of p-adic cluster structure with citrullination vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json

OUTPUT_DIR = Path(__file__).parent

# Check for plotly, provide fallback to matplotlib 3D
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


def generate_cluster_centers(n_clusters=21):
    """Generate 21 cluster centers representing amino acid groups."""
    np.random.seed(42)

    # Amino acid groupings (by property)
    aa_groups = {
        'hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'],
        'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
        'charged_pos': ['K', 'R', 'H'],
        'charged_neg': ['D', 'E'],
        'special': ['G'],
    }

    # Create structured cluster positions
    centers = {}

    # Hydrophobic cluster region
    for i, aa in enumerate(aa_groups['hydrophobic']):
        angle = (i / len(aa_groups['hydrophobic'])) * np.pi
        centers[aa] = np.array([
            2 * np.cos(angle),
            2 * np.sin(angle),
            -1 + np.random.randn() * 0.2
        ])

    # Polar cluster region
    for i, aa in enumerate(aa_groups['polar']):
        angle = (i / len(aa_groups['polar'])) * np.pi + np.pi/4
        centers[aa] = np.array([
            1.5 * np.cos(angle),
            1.5 * np.sin(angle),
            0.5 + np.random.randn() * 0.2
        ])

    # Charged positive (including R - arginine)
    for i, aa in enumerate(aa_groups['charged_pos']):
        angle = (i / len(aa_groups['charged_pos'])) * np.pi/2 + np.pi/3
        centers[aa] = np.array([
            1 * np.cos(angle) - 1,
            1 * np.sin(angle) + 1,
            1.5 + np.random.randn() * 0.2
        ])

    # Charged negative
    for i, aa in enumerate(aa_groups['charged_neg']):
        angle = (i / len(aa_groups['charged_neg'])) * np.pi/2
        centers[aa] = np.array([
            1 * np.cos(angle) + 1,
            1 * np.sin(angle) - 1,
            1.2 + np.random.randn() * 0.2
        ])

    # Glycine (special)
    centers['G'] = np.array([0, 0, 0])

    # Citrulline (Cit) - modified arginine position
    # Key: It's in a DIFFERENT cluster than R
    centers['Cit'] = np.array([0.5, 0.5, 0.8])

    return centers


def create_plotly_visualization():
    """Create interactive 3D cluster visualization with plotly."""
    centers = generate_cluster_centers()

    # Create figure
    fig = go.Figure()

    # Color mapping by amino acid type
    aa_colors = {
        'A': '#FF6B6B', 'V': '#FF6B6B', 'L': '#FF6B6B', 'I': '#FF6B6B',
        'M': '#FF6B6B', 'F': '#FF6B6B', 'W': '#FF6B6B', 'P': '#FF6B6B',  # Hydrophobic
        'S': '#4ECDC4', 'T': '#4ECDC4', 'N': '#4ECDC4', 'Q': '#4ECDC4',
        'Y': '#4ECDC4', 'C': '#4ECDC4',  # Polar
        'K': '#45B7D1', 'R': '#45B7D1', 'H': '#45B7D1',  # Charged+
        'D': '#96CEB4', 'E': '#96CEB4',  # Charged-
        'G': '#DDA0DD',  # Special
        'Cit': '#FFD700',  # Citrulline
    }

    # Add cluster centers as spheres
    for aa, pos in centers.items():
        size = 20 if aa in ['R', 'Cit'] else 15
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers+text',
            marker=dict(size=size, color=aa_colors.get(aa, '#999999'),
                       line=dict(width=2, color='white')),
            text=[aa],
            textposition='top center',
            name=aa,
            hovertemplate=f'{aa}<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})<extra></extra>'
        ))

    # Add R → Cit transition vector (citrullination)
    r_pos = centers['R']
    cit_pos = centers['Cit']

    fig.add_trace(go.Scatter3d(
        x=[r_pos[0], cit_pos[0]],
        y=[r_pos[1], cit_pos[1]],
        z=[r_pos[2], cit_pos[2]],
        mode='lines',
        line=dict(color='red', width=8),
        name='Citrullination (R→Cit)',
        hovertemplate='Citrullination transition<br>CROSSES cluster boundary<extra></extra>'
    ))

    # Add arrow head
    fig.add_trace(go.Cone(
        x=[cit_pos[0]], y=[cit_pos[1]], z=[cit_pos[2]],
        u=[cit_pos[0] - r_pos[0]], v=[cit_pos[1] - r_pos[1]], w=[cit_pos[2] - r_pos[2]],
        sizemode='absolute', sizeref=0.3,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        name='Direction'
    ))

    # Add cluster boundary surface (simplified as mesh)
    # Create a boundary plane between R and Cit clusters
    midpoint = (r_pos + cit_pos) / 2
    normal = cit_pos - r_pos
    normal = normal / np.linalg.norm(normal)

    # Create mesh for boundary plane
    u = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Grid for plane
    plane_size = 1.5
    plane_points = []
    for s in np.linspace(-plane_size, plane_size, 10):
        for t in np.linspace(-plane_size, plane_size, 10):
            point = midpoint + s * u + t * v
            plane_points.append(point)

    plane_points = np.array(plane_points)

    fig.add_trace(go.Mesh3d(
        x=plane_points[:, 0],
        y=plane_points[:, 1],
        z=plane_points[:, 2],
        opacity=0.3,
        color='rgba(255, 0, 0, 0.3)',
        name='Cluster Boundary',
        hovertemplate='Cluster Boundary<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='P-adic Cluster Structure: Citrullination Boundary Crossing',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        annotations=[
            dict(
                x=0.5, y=-0.1,
                xref='paper', yref='paper',
                text='Citrullination of Arginine (R→Cit) crosses the cluster boundary, triggering immune recognition',
                showarrow=False,
                font=dict(size=12, color='gray')
            )
        ],
        width=1000,
        height=800
    )

    return fig


def create_matplotlib_fallback():
    """Create 3D visualization with matplotlib as fallback."""
    from utils.plotting import setup_scientific_style, PALETTE, save_figure

    setup_scientific_style()
    centers = generate_cluster_centers()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color mapping
    aa_colors = {
        'A': '#FF6B6B', 'V': '#FF6B6B', 'L': '#FF6B6B', 'I': '#FF6B6B',
        'M': '#FF6B6B', 'F': '#FF6B6B', 'W': '#FF6B6B', 'P': '#FF6B6B',
        'S': '#4ECDC4', 'T': '#4ECDC4', 'N': '#4ECDC4', 'Q': '#4ECDC4',
        'Y': '#4ECDC4', 'C': '#4ECDC4',
        'K': '#45B7D1', 'R': '#45B7D1', 'H': '#45B7D1',
        'D': '#96CEB4', 'E': '#96CEB4',
        'G': '#DDA0DD',
        'Cit': '#FFD700',
    }

    # Plot clusters
    for aa, pos in centers.items():
        size = 200 if aa in ['R', 'Cit'] else 100
        ax.scatter(pos[0], pos[1], pos[2], c=aa_colors.get(aa, '#999'),
                  s=size, edgecolors='white', linewidths=1.5, alpha=0.8)
        ax.text(pos[0], pos[1], pos[2] + 0.2, aa, fontsize=10, ha='center')

    # R → Cit vector
    r_pos = centers['R']
    cit_pos = centers['Cit']
    ax.quiver(r_pos[0], r_pos[1], r_pos[2],
             cit_pos[0] - r_pos[0], cit_pos[1] - r_pos[1], cit_pos[2] - r_pos[2],
             color='red', linewidth=3, arrow_length_ratio=0.15)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('P-adic Cluster Structure\nCitrullination Boundary Crossing', fontweight='bold')

    return fig


def main():
    if HAS_PLOTLY:
        fig = create_plotly_visualization()
        output_path = OUTPUT_DIR / 'cluster_boundary_3d.html'
        fig.write_html(str(output_path))
        print(f"Saved interactive visualization to {output_path}")

        # Also save static image
        try:
            static_path = OUTPUT_DIR / 'cluster_boundary_3d.png'
            fig.write_image(str(static_path), width=1200, height=900, scale=2)
            print(f"Saved static image to {static_path}")
        except Exception as e:
            print(f"Could not save static image (kaleido may not be installed): {e}")
    else:
        from utils.plotting import save_figure
        fig = create_matplotlib_fallback()
        save_figure(fig, OUTPUT_DIR, 'cluster_boundary_3d')
        print(f"Saved matplotlib visualization to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
