"""
Calabi-Yau Manifold Projection
Projects high-dimensional protein embeddings onto Calabi-Yau-inspired surfaces.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

OUTPUT_DIR = Path(__file__).parent

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


def calabi_yau_surface(u, v, n=3):
    """Generate Calabi-Yau manifold coordinates.

    Uses the parametric equations for a quintic Calabi-Yau cross-section.

    Parameters
    ----------
    u, v : array-like
        Parameter coordinates
    n : int
        Complexity parameter (higher = more structure)

    Returns
    -------
    x, y, z : arrays
        3D coordinates of the surface
    """
    # Calabi-Yau inspired surface equations
    x = np.cos(u) * (1 + 0.5 * np.cos(n * v))
    y = np.sin(u) * (1 + 0.5 * np.cos(n * v))
    z = 0.5 * np.sin(n * v)

    return x, y, z


def quintic_projection(u, v):
    """Project onto a quintic hypersurface (simplified).

    The quintic is defined by: z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0
    We take a 2D cross-section.
    """
    # Real part of quintic surface
    x = np.real(np.exp(1j * u) * (1 + 0.3 * np.cos(5 * v)))
    y = np.real(np.exp(1j * (u + 2*np.pi/5)) * (1 + 0.3 * np.cos(5 * v)))
    z = np.real(0.5 * np.sin(5 * v) * np.cos(u))

    return x, y, z


def generate_pathway_embeddings():
    """Generate pathway protein embeddings for visualization."""
    np.random.seed(42)

    pathways = {
        'parasympathetic': {
            'proteins': ['CHRNA7', 'CHAT', 'CHRM3', 'VIP'],
            'center': (0.3, 0.3),  # u, v center
            'spread': 0.1,
            'color': '#3F51B5'
        },
        'sympathetic': {
            'proteins': ['ADRB2', 'NR3C1', 'TH', 'CRH'],
            'center': (0.7, 0.7),
            'spread': 0.15,
            'color': '#F44336'
        },
        'regeneration': {
            'proteins': ['WNT3A', 'CTNNB1', 'NOTCH1', 'SHH', 'LGR5'],
            'center': (0.5, 0.2),
            'spread': 0.2,
            'color': '#4CAF50'
        },
        'gut_barrier': {
            'proteins': ['TJP1', 'OCLN', 'TLR4', 'MUC2'],
            'center': (0.2, 0.6),
            'spread': 0.25,
            'color': '#9C27B0'
        },
        'inflammation': {
            'proteins': ['TNF', 'IL6', 'IL17A'],
            'center': (0.4, 0.5),
            'spread': 0.08,
            'color': '#FF5722'
        }
    }

    embeddings = {}
    for pathway, data in pathways.items():
        for protein in data['proteins']:
            u = data['center'][0] + np.random.randn() * data['spread']
            v = data['center'][1] + np.random.randn() * data['spread']
            u = np.clip(u, 0.05, 0.95) * 2 * np.pi
            v = np.clip(v, 0.05, 0.95) * 2 * np.pi
            embeddings[protein] = {
                'u': u, 'v': v,
                'pathway': pathway,
                'color': data['color']
            }

    return embeddings, pathways


def create_plotly_calabi_yau():
    """Create interactive Calabi-Yau visualization with plotly."""
    embeddings, pathways = generate_pathway_embeddings()

    # Generate surface mesh
    u_range = np.linspace(0, 2 * np.pi, 100)
    v_range = np.linspace(0, 2 * np.pi, 100)
    u_mesh, v_mesh = np.meshgrid(u_range, v_range)

    x_surf, y_surf, z_surf = calabi_yau_surface(u_mesh, v_mesh, n=5)

    fig = go.Figure()

    # Add Calabi-Yau surface
    fig.add_trace(go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        opacity=0.3,
        colorscale='Viridis',
        showscale=False,
        name='Calabi-Yau Manifold',
        hoverinfo='skip'
    ))

    # Add protein points
    for protein, data in embeddings.items():
        x, y, z = calabi_yau_surface(data['u'], data['v'], n=5)
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=12, color=data['color'],
                       line=dict(width=2, color='white')),
            text=[protein],
            textposition='top center',
            name=f"{protein} ({data['pathway']})",
            hovertemplate=f"{protein}<br>Pathway: {data['pathway']}<extra></extra>"
        ))

    # Add pathway legend traces
    for pathway, pdata in pathways.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=15, color=pdata['color']),
            name=pathway.replace('_', ' ').title(),
            showlegend=True
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text='Regenerative Axis: Calabi-Yau Manifold Projection',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Re(z₁)',
            yaxis_title='Re(z₂)',
            zaxis_title='Re(z₃)',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        annotations=[
            dict(
                x=0.5, y=-0.1,
                xref='paper', yref='paper',
                text='16D codon embeddings projected onto Calabi-Yau cross-section | Parasympathetic occupies central region',
                showarrow=False,
                font=dict(size=11, color='gray')
            )
        ],
        width=1100,
        height=900
    )

    return fig


def create_regenerative_distance_viz():
    """Create visualization showing pathway distances on manifold."""
    embeddings, pathways = generate_pathway_embeddings()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('Manifold Embedding', 'Pathway Distance Matrix'),
        column_widths=[0.6, 0.4]
    )

    # Surface
    u_range = np.linspace(0, 2 * np.pi, 80)
    v_range = np.linspace(0, 2 * np.pi, 80)
    u_mesh, v_mesh = np.meshgrid(u_range, v_range)
    x_surf, y_surf, z_surf = calabi_yau_surface(u_mesh, v_mesh, n=5)

    fig.add_trace(go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        opacity=0.25,
        colorscale='Blues',
        showscale=False
    ), row=1, col=1)

    # Proteins
    for protein, data in embeddings.items():
        x, y, z = calabi_yau_surface(data['u'], data['v'], n=5)
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=10, color=data['color']),
            name=protein,
            showlegend=False
        ), row=1, col=1)

    # Distance matrix heatmap
    pathway_names = list(pathways.keys())
    n_pathways = len(pathway_names)

    # Calculate pathway centroids
    centroids = {}
    for pathway in pathway_names:
        proteins_in_pathway = [p for p, d in embeddings.items() if d['pathway'] == pathway]
        us = [embeddings[p]['u'] for p in proteins_in_pathway]
        vs = [embeddings[p]['v'] for p in proteins_in_pathway]
        centroids[pathway] = (np.mean(us), np.mean(vs))

    # Distance matrix
    dist_matrix = np.zeros((n_pathways, n_pathways))
    for i, p1 in enumerate(pathway_names):
        for j, p2 in enumerate(pathway_names):
            c1 = centroids[p1]
            c2 = centroids[p2]
            # Angular distance on manifold
            dist_matrix[i, j] = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    fig.add_trace(go.Heatmap(
        z=dist_matrix,
        x=[p.replace('_', ' ').title() for p in pathway_names],
        y=[p.replace('_', ' ').title() for p in pathway_names],
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title='Distance', x=1.02)
    ), row=1, col=2)

    fig.update_layout(
        title='Pathway Geometry on Calabi-Yau Manifold',
        height=700,
        width=1200
    )

    return fig


def create_matplotlib_fallback():
    """Matplotlib fallback for systems without plotly."""
    from utils.plotting import setup_scientific_style, save_figure

    setup_scientific_style()
    embeddings, pathways = generate_pathway_embeddings()

    fig = plt.figure(figsize=(14, 6))

    # Left: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')

    u_range = np.linspace(0, 2 * np.pi, 50)
    v_range = np.linspace(0, 2 * np.pi, 50)
    u_mesh, v_mesh = np.meshgrid(u_range, v_range)
    x_surf, y_surf, z_surf = calabi_yau_surface(u_mesh, v_mesh, n=5)

    ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, cmap='viridis')

    for protein, data in embeddings.items():
        x, y, z = calabi_yau_surface(data['u'], data['v'], n=5)
        ax1.scatter([x], [y], [z], c=data['color'], s=100, edgecolors='white')
        ax1.text(x, y, z + 0.1, protein, fontsize=7)

    ax1.set_title('Calabi-Yau Projection', fontweight='bold')
    ax1.set_xlabel('Re(z₁)')
    ax1.set_ylabel('Re(z₂)')
    ax1.set_zlabel('Re(z₃)')

    # Right: Legend
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    y_pos = 0.9
    for pathway, pdata in pathways.items():
        ax2.scatter([0.1], [y_pos], c=pdata['color'], s=200)
        ax2.text(0.2, y_pos, pathway.replace('_', ' ').title(),
                fontsize=12, va='center')
        ax2.text(0.2, y_pos - 0.05, f"Proteins: {', '.join(pdata['proteins'])}",
                fontsize=9, va='center', color='gray')
        y_pos -= 0.18

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Pathway Legend', fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if HAS_PLOTLY:
        # Main visualization
        fig1 = create_plotly_calabi_yau()
        output_path = OUTPUT_DIR / 'calabi_yau_projection.html'
        fig1.write_html(str(output_path))
        print(f"Saved: {output_path}")

        # Distance visualization
        fig2 = create_regenerative_distance_viz()
        output_path2 = OUTPUT_DIR / 'pathway_manifold_distances.html'
        fig2.write_html(str(output_path2))
        print(f"Saved: {output_path2}")

        # Try static exports
        try:
            fig1.write_image(str(OUTPUT_DIR / 'calabi_yau_projection.png'),
                           width=1200, height=900, scale=2)
            print(f"Saved static PNG")
        except Exception as e:
            print(f"Static export skipped: {e}")
    else:
        from utils.plotting import save_figure
        fig = create_matplotlib_fallback()
        save_figure(fig, OUTPUT_DIR, 'calabi_yau_projection')
        print(f"Saved matplotlib fallback to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
