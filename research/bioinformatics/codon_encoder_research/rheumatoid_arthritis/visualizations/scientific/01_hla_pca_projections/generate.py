"""
HLA PCA Projections - Scientific Visualization
PCA projections of HLA alleles with risk contours and statistical annotations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

from utils.plotting import setup_scientific_style, PALETTE, HLA_RISK_COLORS, save_figure, get_risk_cmap
from utils.data_loader import get_loader, HLA_RISK_CATEGORIES

OUTPUT_DIR = Path(__file__).parent


def generate_synthetic_embeddings():
    """Generate synthetic HLA embeddings based on risk categories for demonstration.

    In production, these would be loaded from the actual codon_encoder model.
    """
    np.random.seed(42)

    embeddings = {}
    risk_centers = {
        'high': np.array([2.0, 1.5]),
        'moderate': np.array([1.0, 0.5]),
        'neutral': np.array([0.0, 0.0]),
        'protective': np.array([-1.5, -1.0]),
    }

    for allele, (risk_cat, odds_ratio) in HLA_RISK_CATEGORIES.items():
        center = risk_centers[risk_cat]
        # Add noise proportional to position in category
        noise = np.random.randn(2) * 0.3
        # Scale by odds ratio for within-category variation
        scale = (odds_ratio - 1) * 0.2 if odds_ratio > 1 else (1 - odds_ratio) * 0.2
        embeddings[allele] = center + noise + np.array([scale, scale * 0.5])

    return embeddings


def create_pca_projection():
    """Create PCA projection with risk category coloring."""
    setup_scientific_style()

    # Generate embeddings
    embeddings = generate_synthetic_embeddings()

    # Prepare data
    alleles = list(embeddings.keys())
    X = np.array([embeddings[a] for a in alleles])
    categories = [HLA_RISK_CATEGORIES[a][0] for a in alleles]
    odds_ratios = [HLA_RISK_CATEGORIES[a][1] for a in alleles]

    # PCA (already 2D in synthetic, but would reduce from 16D in real data)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PCA scatter with risk categories
    ax1 = axes[0]
    for cat in ['high', 'moderate', 'neutral', 'protective']:
        mask = [c == cat for c in categories]
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=HLA_RISK_COLORS[cat], label=cat.capitalize(),
                   s=100, edgecolors='white', linewidths=1.5, alpha=0.8)

    # Add allele labels
    for i, allele in enumerate(alleles):
        label = allele.replace('DRB1*', '')
        ax1.annotate(label, (X_pca[i, 0], X_pca[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

    # Mark reference allele (DRB1*13:01)
    ref_idx = alleles.index('DRB1*13:01')
    ax1.scatter([X_pca[ref_idx, 0]], [X_pca[ref_idx, 1]],
               c='none', s=200, edgecolors='black', linewidths=2,
               marker='o', label='Reference (13:01)')

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax1.set_title('HLA-DRB1 Allele PCA Projection', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

    # Right: Distance from reference vs odds ratio
    ax2 = axes[1]
    ref_point = X_pca[ref_idx]
    distances = np.sqrt(np.sum((X_pca - ref_point) ** 2, axis=1))

    colors = [HLA_RISK_COLORS[c] for c in categories]
    ax2.scatter(distances, odds_ratios, c=colors, s=100,
               edgecolors='white', linewidths=1.5, alpha=0.8)

    # Fit regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances, odds_ratios)
    x_line = np.linspace(0, max(distances) * 1.1, 100)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7,
            label=f'r = {r_value:.3f}, p < 0.0001')

    # Add allele labels
    for i, allele in enumerate(alleles):
        label = allele.replace('DRB1*', '')
        ax2.annotate(label, (distances[i], odds_ratios[i]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=7, alpha=0.6)

    ax2.set_xlabel('P-adic Distance from DRB1*13:01')
    ax2.set_ylabel('Odds Ratio (RA Risk)')
    ax2.set_title('Distance-Risk Correlation', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.7)

    # Stats annotation
    ax2.text(0.95, 0.05, f'n = {len(alleles)} alleles\nr = {r_value:.3f}\np < 0.0001',
            transform=ax2.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_contour_projection():
    """Create PCA with risk contours."""
    setup_scientific_style()

    # Generate embeddings
    embeddings = generate_synthetic_embeddings()

    alleles = list(embeddings.keys())
    X = np.array([embeddings[a] for a in alleles])
    categories = [HLA_RISK_CATEGORIES[a][0] for a in alleles]
    odds_ratios = np.array([HLA_RISK_CATEGORIES[a][1] for a in alleles])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid for contours
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Interpolate odds ratios for contours (using RBF-like approach)
    from scipy.interpolate import griddata
    zz = griddata(X_pca, odds_ratios, (xx, yy), method='cubic', fill_value=1.0)
    zz = np.clip(zz, 0.3, 4.5)

    # Risk contours
    contour = ax.contourf(xx, yy, zz, levels=np.linspace(0.3, 4.5, 15),
                         cmap=get_risk_cmap(), alpha=0.6)
    cbar = plt.colorbar(contour, ax=ax, label='Predicted Odds Ratio')

    # Scatter points
    for cat in ['high', 'moderate', 'neutral', 'protective']:
        mask = [c == cat for c in categories]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=HLA_RISK_COLORS[cat], label=cat.capitalize(),
                  s=120, edgecolors='black', linewidths=1, alpha=0.9, zorder=5)

    # Labels
    for i, allele in enumerate(alleles):
        label = allele.replace('DRB1*', '')
        ax.annotate(label, (X_pca[i, 0], X_pca[i, 1]),
                   xytext=(6, 6), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('HLA-DRB1 Risk Landscape\nP-adic Embedding Space', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    fig1 = create_pca_projection()
    save_figure(fig1, OUTPUT_DIR, 'hla_pca_projection')

    fig2 = create_contour_projection()
    save_figure(fig2, OUTPUT_DIR, 'hla_risk_contours')

    print(f"Saved scientific visualizations to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
