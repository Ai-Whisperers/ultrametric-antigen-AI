#!/usr/bin/env python3
"""
Citrullination Boundary Analysis for RA Autoantigens

Tests the hypothesis that citrullination (R→Cit) moves peptide epitopes
across p-adic boundaries, converting "self" to "foreign" in immune recognition.

Key RA autoantigens analyzed:
- Fibrinogen α/β (FGA, FGB)
- Vimentin (VIM)
- Alpha-enolase (ENO1)
- Collagen type II (COL2A1)
- Filaggrin (FLG)

Citrullination converts Arginine (R) to Citrulline - a non-coded amino acid.
This experiment measures how this modification shifts epitope embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from scipy import stats

# ============================================================================
# RA AUTOANTIGENS WITH KNOWN CITRULLINATION SITES
# ============================================================================

# Amino acid to codon mapping (most common codons used)
AA_TO_CODON = {
    'A': 'GCT', 'R': 'CGG', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC',
    'E': 'GAG', 'Q': 'CAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
    'L': 'CTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCG',
    'S': 'TCG', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG',
    '*': 'TGA',
}

# Alternative codons for arginine (R) - important for analysis
ARGININE_CODONS = ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG']

# RA Autoantigens with epitope sequences around citrullination sites
# Format: protein name, epitope sequence (with R at citrullination site), position
RA_AUTOANTIGENS = {
    # FIBRINOGEN ALPHA (FGA)
    # Key citrullinated epitopes recognized by ACPA
    'FGA_R38': {
        'protein': 'Fibrinogen α',
        'gene': 'FGA',
        'epitope_aa': 'GPRVVERHQS',  # R at position 4 (0-indexed)
        'cit_position': 4,
        'immunodominant': True,
        'notes': 'Major ACPA target in RA',
    },
    'FGA_R42': {
        'protein': 'Fibrinogen α',
        'gene': 'FGA',
        'epitope_aa': 'ERHQSACKET',
        'cit_position': 1,
        'immunodominant': True,
        'notes': 'Adjacent to R38 epitope',
    },

    # FIBRINOGEN BETA (FGB)
    'FGB_R72': {
        'protein': 'Fibrinogen β',
        'gene': 'FGB',
        'epitope_aa': 'APPPISGGGY',  # Context around R72
        'cit_position': -1,  # No R in this window, control
        'immunodominant': False,
        'notes': 'Control - no citrullination',
    },
    'FGB_R74': {
        'protein': 'Fibrinogen β',
        'gene': 'FGB',
        'epitope_aa': 'HARPAKAATN',
        'cit_position': 2,
        'immunodominant': True,
        'notes': 'Key citrullination site',
    },

    # VIMENTIN (VIM)
    # Major RA autoantigen, MCV (mutated citrullinated vimentin) used in diagnosis
    'VIM_R71': {
        'protein': 'Vimentin',
        'gene': 'VIM',
        'epitope_aa': 'RLRSSVPGVR',
        'cit_position': 0,  # First R
        'immunodominant': True,
        'notes': 'Sa antigen epitope',
    },
    'VIM_R257': {
        'protein': 'Vimentin',
        'gene': 'VIM',
        'epitope_aa': 'SSLNLRETNL',
        'cit_position': 5,
        'immunodominant': True,
        'notes': 'MCV epitope',
    },
    'VIM_R450': {
        'protein': 'Vimentin',
        'gene': 'VIM',
        'epitope_aa': 'EEIAFLKKLH',
        'cit_position': -1,  # Control, no R
        'immunodominant': False,
        'notes': 'Control region',
    },

    # ALPHA-ENOLASE (ENO1)
    # CEP-1 (citrullinated enolase peptide 1) is highly specific for RA
    'ENO1_R9': {
        'protein': 'Alpha-enolase',
        'gene': 'ENO1',
        'epitope_aa': 'KIREEIFDSRGNP',
        'cit_position': 3,  # Main R
        'immunodominant': True,
        'notes': 'CEP-1 epitope, highly RA-specific',
    },
    'ENO1_R15': {
        'protein': 'Alpha-enolase',
        'gene': 'ENO1',
        'epitope_aa': 'SRGNPTVEVDLFT',
        'cit_position': 1,
        'immunodominant': True,
        'notes': 'CEP-1 extended epitope',
    },

    # COLLAGEN TYPE II (COL2A1)
    # Cartilage-specific, relevant to joint destruction
    'COL2_R124': {
        'protein': 'Collagen II',
        'gene': 'COL2A1',
        'epitope_aa': 'GARGLTGRPG',
        'cit_position': 1,  # First R
        'immunodominant': True,
        'notes': 'Cartilage autoantigen',
    },
    'COL2_R260': {
        'protein': 'Collagen II',
        'gene': 'COL2A1',
        'epitope_aa': 'GPRGDKGETG',
        'cit_position': 2,
        'immunodominant': False,
        'notes': 'Secondary epitope',
    },

    # FILAGGRIN (FLG)
    # Original ACPA target (anti-filaggrin antibodies)
    'FLG_R30': {
        'protein': 'Filaggrin',
        'gene': 'FLG',
        'epitope_aa': 'SHQESTRGRS',
        'cit_position': 6,
        'immunodominant': True,
        'notes': 'Original CCP epitope',
    },

    # SYNOVIAL PROTEINS (for regeneration analysis)
    'HYAL_R205': {
        'protein': 'Hyaluronan synthase 2',
        'gene': 'HAS2',
        'epitope_aa': 'LLVKRWQNHL',
        'cit_position': 3,
        'immunodominant': False,
        'notes': 'Synovial fluid component',
    },
    'PRG4_R100': {
        'protein': 'Lubricin (PRG4)',
        'gene': 'PRG4',
        'epitope_aa': 'EPAPTTPKRE',
        'cit_position': 8,
        'immunodominant': False,
        'notes': 'Joint lubrication protein',
    },
}

# ============================================================================
# CODON ENCODER
# ============================================================================

class CodonEncoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.cluster_head = nn.Linear(embed_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def encode(self, x):
        return self.encoder(x)


def codon_to_onehot(codon):
    """Convert codon string to one-hot encoding."""
    nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    onehot = np.zeros(12)
    for i, nuc in enumerate(codon.upper()):
        if nuc in nucleotides:
            onehot[i * 4 + nucleotides[nuc]] = 1
    return onehot


def aa_sequence_to_codons(aa_sequence):
    """Convert amino acid sequence to codons."""
    return [AA_TO_CODON.get(aa, 'NNN') for aa in aa_sequence.upper()]


# ============================================================================
# CITRULLINATION ANALYSIS
# ============================================================================

def encode_epitope(aa_sequence, encoder, device='cpu'):
    """Encode an epitope's codon sequence."""
    codons = aa_sequence_to_codons(aa_sequence)
    embeddings = []

    for codon in codons:
        if codon != 'NNN':
            onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder.encode(onehot).cpu().numpy().squeeze()
            embeddings.append(emb)

    return np.array(embeddings), codons


def simulate_citrullination(epitope_embeddings, cit_position, encoder, device='cpu'):
    """
    Simulate citrullination by removing arginine contribution.

    Citrulline is not encoded by any codon, so we model it as:
    1. Zero embedding (complete loss of codon signal)
    2. Average of neighboring codons (context-based)
    """
    if cit_position < 0 or cit_position >= len(epitope_embeddings):
        return epitope_embeddings, 'no_change'

    modified = epitope_embeddings.copy()

    # Strategy 1: Zero the citrullinated position (loss of information)
    modified_zero = modified.copy()
    modified_zero[cit_position] = np.zeros_like(modified_zero[cit_position])

    # Strategy 2: Average of neighbors (context preservation)
    modified_avg = modified.copy()
    neighbors = []
    if cit_position > 0:
        neighbors.append(modified[cit_position - 1])
    if cit_position < len(modified) - 1:
        neighbors.append(modified[cit_position + 1])
    if neighbors:
        modified_avg[cit_position] = np.mean(neighbors, axis=0)

    return modified_zero, modified_avg


def compute_embedding_shift(original, modified):
    """Compute the shift in embedding space due to modification."""
    # Sequence-level: mean embedding
    orig_mean = np.mean(original, axis=0)
    mod_mean = np.mean(modified, axis=0)

    euclidean_shift = np.linalg.norm(orig_mean - mod_mean)

    # Position-level shift at citrullination site
    # (captured in the mean, but we can also look at individual positions)

    # Angular shift
    cos_sim = np.dot(orig_mean, mod_mean) / (np.linalg.norm(orig_mean) * np.linalg.norm(mod_mean) + 1e-8)
    angular_shift = np.arccos(np.clip(cos_sim, -1, 1))

    # Radial shift (change in distance from origin)
    radial_shift = np.linalg.norm(mod_mean) - np.linalg.norm(orig_mean)

    return {
        'euclidean': euclidean_shift,
        'angular': angular_shift,
        'radial': radial_shift,
        'original_norm': np.linalg.norm(orig_mean),
        'modified_norm': np.linalg.norm(mod_mean),
    }


def analyze_cluster_boundary_crossing(original_emb, modified_emb, cluster_centers):
    """
    Check if citrullination causes cluster boundary crossing.
    """
    orig_mean = np.mean(original_emb, axis=0)
    mod_mean = np.mean(modified_emb, axis=0)

    # Find nearest cluster for each
    orig_dists = [np.linalg.norm(orig_mean - c) for c in cluster_centers]
    mod_dists = [np.linalg.norm(mod_mean - c) for c in cluster_centers]

    orig_cluster = np.argmin(orig_dists)
    mod_cluster = np.argmin(mod_dists)

    boundary_crossed = orig_cluster != mod_cluster

    # Distance to boundary (how close to crossing?)
    orig_dist_to_nearest = orig_dists[orig_cluster]
    orig_dist_to_second = sorted(orig_dists)[1]
    margin = orig_dist_to_second - orig_dist_to_nearest

    return {
        'boundary_crossed': boundary_crossed,
        'original_cluster': int(orig_cluster),
        'modified_cluster': int(mod_cluster),
        'margin_to_boundary': margin,
        'original_dist_to_center': orig_dist_to_nearest,
        'modified_dist_to_center': mod_dists[mod_cluster],
    }


def analyze_all_arginine_codons(encoder, device='cpu'):
    """
    Analyze how different arginine codons embed differently.
    This shows the "wobble" space available for R.
    """
    results = {}
    embeddings = []

    for codon in ARGININE_CODONS:
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode(onehot).cpu().numpy().squeeze()
        embeddings.append(emb)
        results[codon] = emb

    # Compute variance within arginine codons
    emb_array = np.array(embeddings)
    centroid = np.mean(emb_array, axis=0)
    variance = np.mean([np.linalg.norm(e - centroid)**2 for e in emb_array])
    max_spread = np.max([np.linalg.norm(emb_array[i] - emb_array[j])
                         for i in range(len(emb_array)) for j in range(i+1, len(emb_array))])

    return {
        'codons': results,
        'centroid': centroid,
        'variance': variance,
        'max_spread': max_spread,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(results, arginine_analysis, output_path):
    """Create comprehensive visualization."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Citrullination shift by protein
    ax1 = axes[0, 0]
    proteins = []
    shifts = []
    immunodominant = []

    for name, data in results.items():
        if data['has_cit_site']:
            proteins.append(name)
            shifts.append(data['shift_zero']['euclidean'])
            immunodominant.append(data['immunodominant'])

    colors = ['red' if im else 'blue' for im in immunodominant]
    bars = ax1.barh(range(len(proteins)), shifts, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(proteins)))
    ax1.set_yticklabels(proteins, fontsize=8)
    ax1.set_xlabel('Embedding Shift (Euclidean)')
    ax1.set_title('Citrullination-Induced Embedding Shift\n(Red=Immunodominant, Blue=Non-immunodominant)')
    ax1.invert_yaxis()

    # 2. Boundary crossing analysis
    ax2 = axes[0, 1]
    crossed = sum(1 for d in results.values() if d.get('boundary_crossed', False))
    not_crossed = sum(1 for d in results.values() if d['has_cit_site'] and not d.get('boundary_crossed', False))
    no_site = sum(1 for d in results.values() if not d['has_cit_site'])

    ax2.pie([crossed, not_crossed, no_site],
            labels=[f'Boundary Crossed\n({crossed})',
                   f'Same Cluster\n({not_crossed})',
                   f'No Cit Site\n({no_site})'],
            colors=['red', 'green', 'gray'],
            autopct='%1.0f%%')
    ax2.set_title('Cluster Boundary Crossing After Citrullination')

    # 3. Immunodominant vs non-immunodominant shifts
    ax3 = axes[0, 2]
    immuno_shifts = [d['shift_zero']['euclidean'] for d in results.values()
                     if d['has_cit_site'] and d['immunodominant']]
    non_immuno_shifts = [d['shift_zero']['euclidean'] for d in results.values()
                         if d['has_cit_site'] and not d['immunodominant']]

    ax3.boxplot([immuno_shifts, non_immuno_shifts],
                labels=['Immunodominant', 'Non-immunodominant'])
    ax3.set_ylabel('Embedding Shift')
    ax3.set_title('Shift by Immunodominance Status')

    # Statistical test
    if len(immuno_shifts) > 1 and len(non_immuno_shifts) > 1:
        stat, pval = stats.mannwhitneyu(immuno_shifts, non_immuno_shifts, alternative='greater')
        ax3.text(0.5, 0.95, f'Mann-Whitney p={pval:.4f}',
                transform=ax3.transAxes, ha='center', fontsize=10)

    # 4. Arginine codon space
    ax4 = axes[1, 0]
    arg_embeddings = np.array(list(arginine_analysis['codons'].values()))
    arg_labels = list(arginine_analysis['codons'].keys())

    if arg_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        arg_2d = pca.fit_transform(arg_embeddings)
    else:
        arg_2d = arg_embeddings

    ax4.scatter(arg_2d[:, 0], arg_2d[:, 1], s=150, c='red', alpha=0.7)
    for i, label in enumerate(arg_labels):
        ax4.annotate(label, (arg_2d[i, 0], arg_2d[i, 1]), fontsize=10, ha='center')

    # Add centroid
    centroid_2d = np.mean(arg_2d, axis=0)
    ax4.scatter([centroid_2d[0]], [centroid_2d[1]], s=200, c='black', marker='x', linewidths=3)

    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title(f'Arginine (R) Codon Space\nMax spread: {arginine_analysis["max_spread"]:.3f}')

    # 5. Shift vs margin to boundary
    ax5 = axes[1, 1]
    shifts_plot = []
    margins_plot = []
    immuno_plot = []

    for name, data in results.items():
        if data['has_cit_site'] and 'margin_to_boundary' in data:
            shifts_plot.append(data['shift_zero']['euclidean'])
            margins_plot.append(data['margin_to_boundary'])
            immuno_plot.append(data['immunodominant'])

    colors_scatter = ['red' if im else 'blue' for im in immuno_plot]
    ax5.scatter(margins_plot, shifts_plot, c=colors_scatter, s=100, alpha=0.7)
    ax5.set_xlabel('Margin to Cluster Boundary (before cit)')
    ax5.set_ylabel('Embedding Shift (after cit)')
    ax5.set_title('Shift vs Boundary Margin\n(Red=Immunodominant)')

    # Add trend line
    if len(shifts_plot) > 2:
        z = np.polyfit(margins_plot, shifts_plot, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(margins_plot), max(margins_plot), 100)
        ax5.plot(x_line, p(x_line), 'k--', alpha=0.5)

        corr, pval = stats.spearmanr(margins_plot, shifts_plot)
        ax5.text(0.05, 0.95, f'r={corr:.3f}, p={pval:.3f}',
                transform=ax5.transAxes, fontsize=10)

    # 6. Protein-level summary
    ax6 = axes[1, 2]
    protein_summary = defaultdict(list)
    for name, data in results.items():
        if data['has_cit_site']:
            protein_summary[data['protein']].append(data['shift_zero']['euclidean'])

    proteins_agg = list(protein_summary.keys())
    mean_shifts = [np.mean(v) for v in protein_summary.values()]
    std_shifts = [np.std(v) if len(v) > 1 else 0 for v in protein_summary.values()]

    ax6.barh(range(len(proteins_agg)), mean_shifts, xerr=std_shifts, color='purple', alpha=0.7, capsize=3)
    ax6.set_yticks(range(len(proteins_agg)))
    ax6.set_yticklabels(proteins_agg, fontsize=9)
    ax6.set_xlabel('Mean Embedding Shift')
    ax6.set_title('Citrullination Sensitivity by Protein')
    ax6.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CITRULLINATION BOUNDARY ANALYSIS")
    print("Testing P-Adic Geometry for RA Autoantigen Recognition")
    print("=" * 70)

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load encoder
    print("\nLoading codon encoder...")
    encoder_path = results_dir / 'codon_encoder.pt'
    encoder = CodonEncoder()
    checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['model_state'])
    encoder.eval()

    # Get cluster centers from encoder
    cluster_centers = encoder.cluster_centers.detach().numpy()
    print(f"  Loaded encoder with {len(cluster_centers)} cluster centers")

    # Analyze arginine codon space first
    print("\nAnalyzing arginine (R) codon space...")
    arg_analysis = analyze_all_arginine_codons(encoder)
    print(f"  Arginine codons: {list(arg_analysis['codons'].keys())}")
    print(f"  Within-R variance: {arg_analysis['variance']:.4f}")
    print(f"  Max spread between R codons: {arg_analysis['max_spread']:.4f}")

    # Analyze each autoantigen
    print("\nAnalyzing RA autoantigens...")
    results = {}

    for name, antigen in RA_AUTOANTIGENS.items():
        print(f"\n  {name}: {antigen['protein']}")
        print(f"    Epitope: {antigen['epitope_aa']}")
        print(f"    Cit position: {antigen['cit_position']}")

        # Encode epitope
        embeddings, codons = encode_epitope(antigen['epitope_aa'], encoder)
        print(f"    Codons: {codons}")

        has_cit_site = antigen['cit_position'] >= 0

        if has_cit_site:
            # Simulate citrullination
            mod_zero, mod_avg = simulate_citrullination(embeddings, antigen['cit_position'], encoder)

            # Compute shifts
            shift_zero = compute_embedding_shift(embeddings, mod_zero)
            shift_avg = compute_embedding_shift(embeddings, mod_avg)

            # Check boundary crossing
            boundary_result = analyze_cluster_boundary_crossing(embeddings, mod_zero, cluster_centers)

            print(f"    Embedding shift (zero): {shift_zero['euclidean']:.4f}")
            print(f"    Embedding shift (avg):  {shift_avg['euclidean']:.4f}")
            print(f"    Boundary crossed: {boundary_result['boundary_crossed']}")
            print(f"    Cluster: {boundary_result['original_cluster']} → {boundary_result['modified_cluster']}")

            results[name] = {
                'protein': antigen['protein'],
                'gene': antigen['gene'],
                'epitope': antigen['epitope_aa'],
                'cit_position': antigen['cit_position'],
                'immunodominant': antigen['immunodominant'],
                'has_cit_site': True,
                'shift_zero': shift_zero,
                'shift_avg': shift_avg,
                'boundary_crossed': boundary_result['boundary_crossed'],
                'original_cluster': boundary_result['original_cluster'],
                'modified_cluster': boundary_result['modified_cluster'],
                'margin_to_boundary': boundary_result['margin_to_boundary'],
            }
        else:
            print(f"    No citrullination site (control)")
            results[name] = {
                'protein': antigen['protein'],
                'gene': antigen['gene'],
                'epitope': antigen['epitope_aa'],
                'cit_position': -1,
                'immunodominant': antigen['immunodominant'],
                'has_cit_site': False,
            }

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    cit_sites = [r for r in results.values() if r['has_cit_site']]
    immuno_sites = [r for r in cit_sites if r['immunodominant']]
    non_immuno_sites = [r for r in cit_sites if not r['immunodominant']]

    print(f"\n  Total epitopes analyzed: {len(results)}")
    print(f"  With citrullination sites: {len(cit_sites)}")
    print(f"  Immunodominant: {len(immuno_sites)}")
    print(f"  Non-immunodominant: {len(non_immuno_sites)}")

    # Boundary crossing statistics
    boundary_crossed = sum(1 for r in cit_sites if r['boundary_crossed'])
    print(f"\n  Boundary crossings: {boundary_crossed}/{len(cit_sites)} ({100*boundary_crossed/len(cit_sites):.1f}%)")

    # Compare immunodominant vs non-immunodominant
    if immuno_sites and non_immuno_sites:
        immuno_shifts = [r['shift_zero']['euclidean'] for r in immuno_sites]
        non_immuno_shifts = [r['shift_zero']['euclidean'] for r in non_immuno_sites]

        print(f"\n  Mean shift (immunodominant): {np.mean(immuno_shifts):.4f} ± {np.std(immuno_shifts):.4f}")
        print(f"  Mean shift (non-immunodominant): {np.mean(non_immuno_shifts):.4f} ± {np.std(non_immuno_shifts):.4f}")

        # Statistical test
        if len(immuno_shifts) > 1 and len(non_immuno_shifts) > 1:
            stat, pval = stats.mannwhitneyu(immuno_shifts, non_immuno_shifts, alternative='greater')
            print(f"  Mann-Whitney U test (immuno > non-immuno): p = {pval:.4f}")

    # Create visualization
    print("\nGenerating visualization...")
    vis_path = results_dir / 'citrullination_analysis.png'
    create_visualization(results, arg_analysis, vis_path)

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if boundary_crossed > len(cit_sites) / 2:
        print("""
    *** KEY FINDING: Majority of citrullination events cross cluster boundaries ***

    This supports the hypothesis that citrullination converts "self" peptides
    to "foreign" by moving them across p-adic boundaries in embedding space.
    The immune system may use a distance-based metric similar to our p-adic
    geometry to distinguish self from non-self.
        """)
    else:
        print("""
    Citrullination does not consistently cross cluster boundaries.
    The immunogenicity may depend on other factors (HLA binding, T cell repertoire).
    However, the SHIFT magnitude may still predict immunodominance.
        """)

    if immuno_sites and non_immuno_sites:
        immuno_mean = np.mean([r['shift_zero']['euclidean'] for r in immuno_sites])
        non_immuno_mean = np.mean([r['shift_zero']['euclidean'] for r in non_immuno_sites])

        if immuno_mean > non_immuno_mean:
            print(f"""
    *** IMMUNODOMINANCE CORRELATION ***

    Immunodominant epitopes show LARGER embedding shifts ({immuno_mean:.4f}) than
    non-immunodominant epitopes ({non_immuno_mean:.4f}).

    This suggests: epitopes that shift MORE in p-adic space after citrullination
    are more likely to trigger autoimmune response. The magnitude of geometric
    displacement may predict immunogenicity.
            """)

    # Regeneration implications
    print("\n" + "-" * 70)
    print("IMPLICATIONS FOR SYNOVIAL REGENERATION")
    print("-" * 70)
    print("""
    For regenerating healthy synovium without triggering autoimmunity:

    1. CODON OPTIMIZATION: Select synonymous codons that place epitopes
       deep inside p-adic clusters (far from boundaries)

    2. ARGININE CODON CHOICE: Different R codons create spread of {:.4f} in
       embedding space. Choosing "central" R codons may reduce immunogenicity.

    3. AVOID HIGH-SHIFT SEQUENCES: Screen synthetic synoviocyte genes for
       epitopes that would shift significantly upon citrullination.

    4. TOLERANCE INDUCTION: Design altered peptide ligands that occupy the
       same p-adic region but don't trigger T cell activation.
    """.format(arg_analysis['max_spread']))

    # Save results (convert numpy/bool types to native Python)
    output_data = {
        'n_epitopes': len(results),
        'n_with_cit_site': len(cit_sites),
        'n_boundary_crossed': int(boundary_crossed),
        'arginine_codon_spread': float(arg_analysis['max_spread']),
        'arginine_variance': float(arg_analysis['variance']),
        'epitopes': {
            name: {
                'protein': r['protein'],
                'immunodominant': bool(r['immunodominant']),
                'shift': float(r['shift_zero']['euclidean']) if r['has_cit_site'] else None,
                'boundary_crossed': bool(r['boundary_crossed']) if r.get('boundary_crossed') is not None else None,
            }
            for name, r in results.items()
        }
    }

    output_path = results_dir / 'citrullination_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved results to {output_path}")


if __name__ == '__main__':
    main()
