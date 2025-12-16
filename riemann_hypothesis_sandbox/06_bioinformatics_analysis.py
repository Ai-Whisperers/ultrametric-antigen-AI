"""
06_bioinformatics_analysis.py - Apply 3-adic embeddings to codon/protein analysis

Hypothesis: The 3-adic structure learned by the VAE may capture biological
patterns in genetic code, since:
1. Codons are triplets (3 nucleotide positions)
2. Amino acids cluster by chemical properties (~3 major groups)
3. Synonymous substitutions follow patterns related to codon position

This script tests whether 3-adic distance in embedding space correlates
with biological substitution rates (PAM/BLOSUM matrices).

Usage:
    python 06_bioinformatics_analysis.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Standard genetic code
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Amino acid properties (hydrophobicity scale: Kyte-Doolittle)
AA_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    '*': 0.0  # Stop codon
}

# Amino acid chemical classes
AA_CLASS = {
    'A': 'nonpolar', 'V': 'nonpolar', 'L': 'nonpolar', 'I': 'nonpolar',
    'M': 'nonpolar', 'F': 'nonpolar', 'W': 'nonpolar', 'P': 'nonpolar',
    'G': 'nonpolar',
    'S': 'polar', 'T': 'polar', 'C': 'polar', 'Y': 'polar',
    'N': 'polar', 'Q': 'polar',
    'K': 'charged', 'R': 'charged', 'H': 'charged',
    'D': 'charged', 'E': 'charged',
    '*': 'stop'
}

# BLOSUM62 matrix (simplified - just amino acid similarity scores)
BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'L'): -2,
    ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0, ('N', 'E'): 0,
    ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3,
    ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,
    ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,
    ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2,
    ('H', 'Y'): 2, ('H', 'V'): -3,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,
    ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,
    ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,
    ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
    ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
    ('Y', 'Y'): 7, ('Y', 'V'): -1,
    ('V', 'V'): 4,
}


def get_blosum_score(aa1, aa2):
    """Get BLOSUM62 score for amino acid pair."""
    if aa1 == '*' or aa2 == '*':
        return -4  # Stop codon penalty
    key = (aa1, aa2) if (aa1, aa2) in BLOSUM62 else (aa2, aa1)
    return BLOSUM62.get(key, 0)


def codon_to_ternary(codon):
    """Map codon to ternary representation.

    Encoding: Each nucleotide position uses chemical grouping:
    - Purines (A, G) vs Pyrimidines (C, T) → 2 groups
    - Strong (G, C) vs Weak (A, T) hydrogen bonding → 2 groups
    - Amino (A, C) vs Keto (G, T) → 2 groups

    We use a 3-value encoding based on nucleotide ring structure:
    - A (purine, amino): 0
    - G (purine, keto): 1
    - C (pyrimidine, amino): 0 (maps with A)
    - T (pyrimidine, keto): 2

    Alternative: Gray code-like encoding for smooth transitions.
    """
    # Chemical property encoding
    # Group by: purine/pyrimidine + hydrogen bonding
    nuc_to_ternary = {
        'A': 0,  # Purine, weak H-bond
        'G': 1,  # Purine, strong H-bond
        'C': 2,  # Pyrimidine, strong H-bond (complement of G)
        'T': 0,  # Pyrimidine, weak H-bond (complement of A, maps to same)
    }

    # Convert 3-position codon to base-3 number
    result = 0
    for i, nuc in enumerate(codon):
        result += nuc_to_ternary[nuc] * (3 ** (2 - i))
    return result


def codon_to_ternary_v2(codon):
    """Alternative encoding using all 4 nucleotides → 2 trits per position.

    This gives 2^6 = 64 possibilities, matching exactly 64 codons.
    But we want to map to our 3^9 space, so we extend.
    """
    # Map each nucleotide to 2 ternary digits
    nuc_encoding = {
        'A': (0, 0),
        'C': (0, 1),
        'G': (1, 0),
        'T': (1, 1),
    }

    trits = []
    for nuc in codon:
        trits.extend(nuc_encoding[nuc])

    # Pad to 9 trits (our model's input size)
    while len(trits) < 9:
        trits.append(0)

    # Convert to index in 3^9 space
    result = 0
    for i, t in enumerate(trits):
        result += t * (3 ** (8 - i))
    return result


def hamming_distance_codons(c1, c2):
    """Hamming distance between two codons."""
    return sum(a != b for a, b in zip(c1, c2))


def analyze_codon_structure(embeddings):
    """Analyze how codons map to embedding space."""
    print("\n" + "="*60)
    print("CODON STRUCTURE ANALYSIS")
    print("="*60)

    codons = list(GENETIC_CODE.keys())
    amino_acids = [GENETIC_CODE[c] for c in codons]

    # Map codons to ternary indices
    ternary_indices = [codon_to_ternary_v2(c) for c in codons]

    print(f"\nCodon mapping to ternary space:")
    print(f"  64 codons → indices in [0, {max(ternary_indices)}]")
    print(f"  Unique indices: {len(set(ternary_indices))}")

    # Get embeddings for codon indices
    z_B = embeddings['z_B']
    codon_embeddings = z_B[ternary_indices]

    # Compute pairwise distances in embedding space
    emb_distances = squareform(pdist(codon_embeddings, metric='euclidean'))

    # Compute BLOSUM similarity matrix
    n_codons = len(codons)
    blosum_sim = np.zeros((n_codons, n_codons))
    for i in range(n_codons):
        for j in range(n_codons):
            aa_i, aa_j = amino_acids[i], amino_acids[j]
            blosum_sim[i, j] = get_blosum_score(aa_i, aa_j)

    # Compute Hamming distance matrix
    hamming_dist = np.zeros((n_codons, n_codons))
    for i in range(n_codons):
        for j in range(n_codons):
            hamming_dist[i, j] = hamming_distance_codons(codons[i], codons[j])

    # Correlation: embedding distance vs BLOSUM similarity
    # (should be negative: similar AAs have high BLOSUM, should have small embedding distance)
    upper_tri = np.triu_indices(n_codons, k=1)
    emb_flat = emb_distances[upper_tri]
    blosum_flat = blosum_sim[upper_tri]
    hamming_flat = hamming_dist[upper_tri]

    corr_blosum, p_blosum = stats.spearmanr(emb_flat, blosum_flat)
    corr_hamming, p_hamming = stats.spearmanr(emb_flat, hamming_flat)

    print(f"\n  Embedding distance vs BLOSUM similarity:")
    print(f"    Spearman r = {corr_blosum:.4f} (p = {p_blosum:.2e})")
    print(f"    (negative = similar AAs cluster in embedding space)")

    print(f"\n  Embedding distance vs Hamming distance:")
    print(f"    Spearman r = {corr_hamming:.4f} (p = {p_hamming:.2e})")
    print(f"    (positive = sequence-similar codons cluster)")

    return {
        'n_codons': n_codons,
        'ternary_indices': ternary_indices,
        'corr_blosum': float(corr_blosum),
        'p_blosum': float(p_blosum),
        'corr_hamming': float(corr_hamming),
        'p_hamming': float(p_hamming),
        'emb_distances': emb_distances,
        'blosum_sim': blosum_sim,
        'codons': codons,
        'amino_acids': amino_acids
    }


def analyze_synonymous_codons(embeddings, codon_results):
    """Analyze if synonymous codons (same AA) cluster in embedding space."""
    print("\n" + "="*60)
    print("SYNONYMOUS CODON CLUSTERING")
    print("="*60)

    codons = codon_results['codons']
    amino_acids = codon_results['amino_acids']
    emb_distances = codon_results['emb_distances']

    # Group codons by amino acid
    aa_to_codons = {}
    for i, (c, aa) in enumerate(zip(codons, amino_acids)):
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        aa_to_codons[aa].append(i)

    # For each amino acid with multiple codons, compute within-group vs between-group distances
    within_distances = []
    between_distances = []

    results_by_aa = {}

    for aa, indices in aa_to_codons.items():
        if len(indices) < 2:
            continue

        # Within-group distances
        within = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                within.append(emb_distances[indices[i], indices[j]])

        # Between-group distances (to other AAs)
        between = []
        other_indices = [i for i in range(len(codons)) if amino_acids[i] != aa]
        for idx in indices:
            for other in other_indices:
                between.append(emb_distances[idx, other])

        within_mean = np.mean(within) if within else np.nan
        between_mean = np.mean(between) if between else np.nan

        results_by_aa[aa] = {
            'n_codons': len(indices),
            'within_mean': within_mean,
            'between_mean': between_mean,
            'ratio': within_mean / between_mean if between_mean > 0 else np.nan
        }

        within_distances.extend(within)
        between_distances.extend(between)

    # Overall statistics
    within_mean = np.mean(within_distances)
    between_mean = np.mean(between_distances)

    # Statistical test: within should be smaller than between
    stat, p_value = stats.mannwhitneyu(within_distances, between_distances, alternative='less')

    print(f"\n  Within-AA distance (synonymous codons): {within_mean:.4f}")
    print(f"  Between-AA distance (different AAs): {between_mean:.4f}")
    print(f"  Ratio (within/between): {within_mean/between_mean:.4f}")
    print(f"  Mann-Whitney U test (within < between): p = {p_value:.2e}")

    if p_value < 0.05:
        print(f"\n  SIGNIFICANT: Synonymous codons cluster together in embedding space!")
    else:
        print(f"\n  Not significant: Synonymous codons don't cluster specially.")

    # Per-AA breakdown
    print(f"\n  Per-amino-acid clustering (ratio < 1 = good clustering):")
    for aa in sorted(results_by_aa.keys()):
        r = results_by_aa[aa]
        if not np.isnan(r['ratio']):
            marker = "*" if r['ratio'] < 1 else " "
            print(f"    {aa} ({r['n_codons']} codons): ratio = {r['ratio']:.3f} {marker}")

    return {
        'within_mean': float(within_mean),
        'between_mean': float(between_mean),
        'ratio': float(within_mean / between_mean),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'by_aa': results_by_aa
    }


def analyze_chemical_properties(embeddings, codon_results):
    """Test if embedding captures amino acid chemical properties."""
    print("\n" + "="*60)
    print("CHEMICAL PROPERTY ENCODING")
    print("="*60)

    codons = codon_results['codons']
    amino_acids = codon_results['amino_acids']
    ternary_indices = codon_results['ternary_indices']

    z_B = embeddings['z_B']
    codon_embeddings = z_B[ternary_indices]
    radii = np.linalg.norm(codon_embeddings, axis=1)

    # Get hydrophobicity values
    hydrophobicity = np.array([AA_HYDROPHOBICITY[aa] for aa in amino_acids])

    # Get chemical class
    class_map = {'nonpolar': 0, 'polar': 1, 'charged': 2, 'stop': 3}
    chem_class = np.array([class_map[AA_CLASS[aa]] for aa in amino_acids])

    # Correlation: radius vs hydrophobicity
    corr_hydro, p_hydro = stats.spearmanr(radii, hydrophobicity)

    print(f"\n  Radius vs Hydrophobicity:")
    print(f"    Spearman r = {corr_hydro:.4f} (p = {p_hydro:.2e})")

    # ANOVA: radius by chemical class
    classes = ['nonpolar', 'polar', 'charged']
    class_radii = {}
    for cls in classes:
        mask = np.array([AA_CLASS[aa] == cls for aa in amino_acids])
        class_radii[cls] = radii[mask]

    f_stat, p_anova = stats.f_oneway(
        class_radii['nonpolar'],
        class_radii['polar'],
        class_radii['charged']
    )

    print(f"\n  Radius by chemical class (ANOVA):")
    print(f"    F-statistic = {f_stat:.4f}, p = {p_anova:.2e}")
    for cls in classes:
        r = class_radii[cls]
        print(f"    {cls}: radius = {r.mean():.4f} +/- {r.std():.4f} (n={len(r)})")

    return {
        'corr_hydrophobicity': float(corr_hydro),
        'p_hydrophobicity': float(p_hydro),
        'f_stat_class': float(f_stat),
        'p_anova_class': float(p_anova),
        'class_radii': {k: {'mean': float(v.mean()), 'std': float(v.std())}
                       for k, v in class_radii.items()}
    }


def visualize_codon_embedding(embeddings, codon_results, output_dir):
    """Visualize codon structure in embedding space."""
    codons = codon_results['codons']
    amino_acids = codon_results['amino_acids']
    ternary_indices = codon_results['ternary_indices']

    z_B = embeddings['z_B']
    codon_embeddings = z_B[ternary_indices]

    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(codon_embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Color by amino acid
    ax1 = axes[0]
    unique_aa = sorted(set(amino_acids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_aa)))
    aa_to_color = {aa: colors[i] for i, aa in enumerate(unique_aa)}

    for i, (x, y) in enumerate(coords_2d):
        aa = amino_acids[i]
        ax1.scatter(x, y, c=[aa_to_color[aa]], s=50, alpha=0.7)
        ax1.annotate(codons[i], (x, y), fontsize=6, alpha=0.5)

    ax1.set_title('Codons in Embedding Space (colored by AA)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    # 2. Color by chemical class
    ax2 = axes[1]
    class_colors = {'nonpolar': 'blue', 'polar': 'green', 'charged': 'red', 'stop': 'black'}

    for cls in ['nonpolar', 'polar', 'charged', 'stop']:
        mask = np.array([AA_CLASS[aa] == cls for aa in amino_acids])
        if mask.any():
            ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=class_colors[cls], label=cls, s=50, alpha=0.7)

    ax2.set_title('Codons by Chemical Class')
    ax2.set_xlabel(f'PC1')
    ax2.set_ylabel(f'PC2')
    ax2.legend()

    # 3. Dendrogram based on embedding distances
    ax3 = axes[2]
    emb_distances = codon_results['emb_distances']
    condensed = squareform(emb_distances)
    Z = linkage(condensed, method='average')

    dendrogram(Z, ax=ax3, labels=amino_acids, leaf_rotation=90, leaf_font_size=6)
    ax3.set_title('Hierarchical Clustering of Codons')
    ax3.set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig(output_dir / 'codon_embedding_analysis.png', dpi=150)
    plt.close()

    print(f"\n  Saved visualization to {output_dir}/codon_embedding_analysis.png")


def main():
    output_dir = PROJECT_ROOT / 'riemann_hypothesis_sandbox' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    data = torch.load(
        PROJECT_ROOT / 'riemann_hypothesis_sandbox' / 'embeddings' / 'embeddings.pt',
        weights_only=False
    )

    embeddings = {
        'z_A': data.get('z_A_hyp', data.get('z_hyperbolic')).numpy() if torch.is_tensor(data.get('z_A_hyp', data.get('z_hyperbolic'))) else data.get('z_A_hyp', data.get('z_hyperbolic')),
        'z_B': data.get('z_B_hyp', data.get('z_hyperbolic')).numpy() if torch.is_tensor(data.get('z_B_hyp', data.get('z_hyperbolic'))) else data.get('z_B_hyp', data.get('z_hyperbolic')),
    }

    print(f"Loaded embeddings: shape = {embeddings['z_B'].shape}")

    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'analysis': 'bioinformatics_codon_mapping'
    }

    # Analysis 1: Codon structure
    codon_results = analyze_codon_structure(embeddings)
    results['codon_structure'] = {
        'corr_blosum': codon_results['corr_blosum'],
        'p_blosum': codon_results['p_blosum'],
        'corr_hamming': codon_results['corr_hamming'],
        'p_hamming': codon_results['p_hamming']
    }

    # Analysis 2: Synonymous codon clustering
    syn_results = analyze_synonymous_codons(embeddings, codon_results)
    results['synonymous_clustering'] = syn_results

    # Analysis 3: Chemical properties
    chem_results = analyze_chemical_properties(embeddings, codon_results)
    results['chemical_properties'] = chem_results

    # Visualization
    visualize_codon_embedding(embeddings, codon_results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("BIOINFORMATICS ANALYSIS SUMMARY")
    print("="*60)

    syn_p_str = f"{syn_results['p_value']:.2e}"
    blosum_verdict = 'POSITIVE: Similar amino acids cluster in embedding space' if codon_results['corr_blosum'] < -0.1 else 'WEAK/NONE: No clear AA similarity structure'
    syn_verdict = f"POSITIVE: Synonymous codons cluster together (p={syn_p_str})" if syn_results['significant'] else 'NEGATIVE: No special clustering of synonymous codons'
    chem_verdict = 'POSITIVE: Embedding radius encodes chemical class' if chem_results['p_anova_class'] < 0.05 else 'NEGATIVE: No chemical class structure in radius'
    overall = 'DOES' if syn_results['significant'] or chem_results['p_anova_class'] < 0.05 else 'does NOT'

    print(f"""
Key Findings:

1. BLOSUM CORRELATION: r = {codon_results['corr_blosum']:.3f}
   {blosum_verdict}

2. SYNONYMOUS CLUSTERING: ratio = {syn_results['ratio']:.3f}
   {syn_verdict}

3. CHEMICAL PROPERTIES:
   - Hydrophobicity correlation: r = {chem_results['corr_hydrophobicity']:.3f}
   - Chemical class ANOVA: p = {chem_results['p_anova_class']:.2e}
   {chem_verdict}

INTERPRETATION:
The 3-adic embedding {overall} capture biologically meaningful codon structure.
""")

    # Save results
    results_file = output_dir / 'bioinformatics_results.json'

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == '__main__':
    main()
