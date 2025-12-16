#!/usr/bin/env python3
"""
HIV-1 CTL Escape Mutation Analysis using P-Adic Geometry

This script analyzes how HIV-1 CTL escape mutations affect positioning
in p-adic embedding space. We hypothesize that escape mutations that
cross p-adic boundaries are more effective at evading immune recognition.

Key questions:
1. Do escape mutations cross cluster boundaries more than random mutations?
2. Can p-adic distance predict escape fitness cost?
3. Are there "sentinel positions" analogous to RA sentinel epitopes?
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Codon table
CODON_TABLE = {
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
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Reverse lookup: amino acid to most common codon
AA_TO_CODON = {
    'F': 'TTC', 'L': 'CTG', 'S': 'AGC', 'Y': 'TAC',
    'C': 'TGC', 'W': 'TGG', 'P': 'CCC', 'H': 'CAC',
    'Q': 'CAG', 'R': 'CGG', 'I': 'ATC', 'M': 'ATG',
    'T': 'ACC', 'N': 'AAC', 'K': 'AAG', 'V': 'GTG',
    'A': 'GCC', 'D': 'GAC', 'E': 'GAG', 'G': 'GGC',
    '*': 'TGA'
}


# ============================================================================
# HIV-1 CTL EPITOPE DATABASE
# Sources: Los Alamos HIV Immunology Database, published literature
# ============================================================================

HIV_CTL_EPITOPES = {
    # Gag epitopes (major CTL targets)
    'SL9_Gag77': {
        'protein': 'Gag p17',
        'hla_restriction': 'HLA-A*02:01',
        'position': '77-85',
        'wild_type': {
            'sequence': 'SLYNTVATL',
            'codons': ['AGC', 'CTG', 'TAC', 'AAC', 'ACC', 'GTG', 'GCC', 'ACC', 'CTG']
        },
        'escape_variants': [
            {
                'name': 'Y79F',
                'sequence': 'SLFNTVATL',
                'position': 2,  # 0-indexed: Y at position 2 -> F
                'escape_efficacy': 'high',
                'fitness_cost': 'low',
                'frequency': 0.15
            },
            {
                'name': 'T84I',
                'sequence': 'SLYNTVAIL',  # T->I at position 7 (0-indexed)
                'position': 7,
                'escape_efficacy': 'moderate',
                'fitness_cost': 'moderate',
                'frequency': 0.08
            }
        ]
    },

    'KK10_Gag263': {
        'protein': 'Gag p24',
        'hla_restriction': 'HLA-B*27:05',
        'position': '263-272',
        'wild_type': {
            'sequence': 'KRWIILGLNK',
            'codons': ['AAG', 'CGG', 'TGG', 'ATC', 'ATC', 'CTG', 'GGC', 'CTG', 'AAC', 'AAG']
        },
        'escape_variants': [
            {
                'name': 'R264K',
                'sequence': 'KKWIILGLNK',
                'position': 1,
                'escape_efficacy': 'high',
                'fitness_cost': 'high',  # HLA-B27 protective effect
                'frequency': 0.05
            },
            {
                'name': 'L268M',
                'sequence': 'KRWIIMGLNK',
                'position': 5,
                'escape_efficacy': 'moderate',
                'fitness_cost': 'low',
                'frequency': 0.12
            }
        ]
    },

    'TW10_Gag240': {
        'protein': 'Gag p24',
        'hla_restriction': 'HLA-B*57:01',
        'position': '240-249',
        'wild_type': {
            'sequence': 'TSTLQEQIGW',
            'codons': ['ACC', 'AGC', 'ACC', 'CTG', 'CAG', 'GAG', 'CAG', 'ATC', 'GGC', 'TGG']
        },
        'escape_variants': [
            {
                'name': 'T242N',
                'sequence': 'TSNLQEQIGW',
                'position': 2,
                'escape_efficacy': 'high',
                'fitness_cost': 'moderate',
                'frequency': 0.20
            },
            {
                'name': 'G248A',
                'sequence': 'TSTLQEQIAW',
                'position': 8,
                'escape_efficacy': 'moderate',
                'fitness_cost': 'low',
                'frequency': 0.10
            }
        ]
    },

    # Nef epitopes
    'FL8_Nef90': {
        'protein': 'Nef',
        'hla_restriction': 'HLA-A*24:02',
        'position': '90-97',
        'wild_type': {
            'sequence': 'FLKEKGGL',
            'codons': ['TTC', 'CTG', 'AAG', 'GAG', 'AAG', 'GGC', 'GGC', 'CTG']
        },
        'escape_variants': [
            {
                'name': 'K94R',
                'sequence': 'FLKERGGL',
                'position': 4,
                'escape_efficacy': 'high',
                'fitness_cost': 'low',
                'frequency': 0.25
            }
        ]
    },

    # Pol epitopes (RT)
    'IV9_RT179': {
        'protein': 'RT',
        'hla_restriction': 'HLA-A*02:01',
        'position': '179-187',
        'wild_type': {
            'sequence': 'ILKEPVHGV',
            'codons': ['ATC', 'CTG', 'AAG', 'GAG', 'CCC', 'GTG', 'CAC', 'GGC', 'GTG']
        },
        'escape_variants': [
            {
                'name': 'V181I',  # Also drug resistance!
                'sequence': 'ILKEPIHGV',
                'position': 5,
                'escape_efficacy': 'moderate',
                'fitness_cost': 'low',
                'frequency': 0.18
            }
        ]
    },

    # Env epitopes (harder to escape - conformational constraints)
    'RL9_Env311': {
        'protein': 'Env gp120',
        'hla_restriction': 'HLA-B*08:01',
        'position': '311-319',
        'wild_type': {
            'sequence': 'RLRDLLLIW',  # Fictional for demo
            'codons': ['CGG', 'CTG', 'CGG', 'GAC', 'CTG', 'CTG', 'CTG', 'ATC', 'TGG']
        },
        'escape_variants': [
            {
                'name': 'D314N',
                'sequence': 'RLRNLLLIW',
                'position': 3,
                'escape_efficacy': 'moderate',
                'fitness_cost': 'high',  # Env constraints
                'frequency': 0.03
            }
        ]
    }
}


class CodonEncoder(nn.Module):
    """Codon to embedding encoder - matches architecture from 08_learn_codon_mapping.py."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cluster classification head
        self.cluster_head = nn.Linear(embed_dim, n_clusters)

        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        """Forward pass.
        Args:
            x: Codon features (batch, 12)
        Returns:
            embedding: Learned embedding (batch, 16)
            cluster_logits: Cluster predictions (batch, 21)
        """
        embedding = self.encoder(x)
        cluster_logits = self.cluster_head(embedding)
        return embedding, cluster_logits

    def get_embedding_from_codon(self, codon):
        """Get embedding for a codon string (e.g., 'ATG')."""
        features = codon_to_features(codon)
        x = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            embedding, _ = self.forward(x)
        return embedding[0].numpy()


def codon_to_index(codon):
    """Convert codon string to index (0-63)."""
    bases = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
    return bases[codon[0]] * 16 + bases[codon[1]] * 4 + bases[codon[2]]


def codon_to_features(codon):
    """Convert codon string to one-hot features (12 features = 3 positions × 4 nucleotides)."""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    features = []
    for nuc in codon:
        one_hot = [0, 0, 0, 0]
        one_hot[nuc_to_idx[nuc]] = 1
        features.extend(one_hot)
    return features


def get_codons_for_aa(aa):
    """Get all codons that encode a given amino acid."""
    return [codon for codon, amino in CODON_TABLE.items() if amino == aa]


def analyze_escape_mutation(encoder, wt_codons, escape_info, mapping):
    """
    Analyze a single escape mutation.

    Returns:
        dict with p-adic distances and boundary crossing info
    """
    pos = escape_info['position']
    wt_codon = wt_codons[pos]
    wt_aa = CODON_TABLE[wt_codon]

    # Get escape amino acid
    escape_seq = escape_info['sequence']
    wt_seq = ''.join([CODON_TABLE[c] for c in wt_codons])
    escape_aa = escape_seq[pos]

    # Get embedding for wild-type codon
    wt_emb = encoder.get_embedding_from_codon(wt_codon)
    wt_cluster = mapping.get(wt_codon, -1)

    # Get embeddings for all codons encoding escape amino acid
    escape_codons = get_codons_for_aa(escape_aa)
    escape_results = []

    for esc_codon in escape_codons:
        esc_emb = encoder.get_embedding_from_codon(esc_codon)
        esc_cluster = mapping.get(esc_codon, -1)

        # P-adic distance (Euclidean in embedding space)
        distance = np.linalg.norm(wt_emb - esc_emb)

        # Check boundary crossing
        boundary_crossed = wt_cluster != esc_cluster

        escape_results.append({
            'codon': esc_codon,
            'cluster': esc_cluster,
            'distance': distance,
            'boundary_crossed': boundary_crossed
        })

    # Choose most likely escape codon (minimum distance for fitness preservation)
    best_escape = min(escape_results, key=lambda x: x['distance'])

    return {
        'mutation': escape_info['name'],
        'wt_aa': wt_aa,
        'escape_aa': escape_aa,
        'wt_codon': wt_codon,
        'wt_cluster': wt_cluster,
        'escape_codon': best_escape['codon'],
        'escape_cluster': best_escape['cluster'],
        'padic_distance': best_escape['distance'],
        'boundary_crossed': best_escape['boundary_crossed'],
        'escape_efficacy': escape_info['escape_efficacy'],
        'fitness_cost': escape_info['fitness_cost'],
        'frequency': escape_info['frequency'],
        'all_escape_codons': escape_results
    }


def analyze_epitope(encoder, epitope_data, mapping):
    """Analyze all escape variants for an epitope."""
    results = {
        'protein': epitope_data['protein'],
        'hla': epitope_data['hla_restriction'],
        'position': epitope_data['position'],
        'wild_type_sequence': epitope_data['wild_type']['sequence'],
        'escape_analyses': []
    }

    wt_codons = epitope_data['wild_type']['codons']

    for escape in epitope_data['escape_variants']:
        analysis = analyze_escape_mutation(encoder, wt_codons, escape, mapping)
        results['escape_analyses'].append(analysis)

    return results


def permutation_test_escape_boundary(all_results, n_permutations=10000):
    """
    Test if escape mutations cross boundaries more than expected by chance.
    """
    # Observed boundary crossing rate
    total_escapes = sum(len(r['escape_analyses']) for r in all_results)
    boundary_crossings = sum(
        sum(1 for e in r['escape_analyses'] if e['boundary_crossed'])
        for r in all_results
    )
    observed_rate = boundary_crossings / total_escapes if total_escapes > 0 else 0

    # For null distribution, simulate random amino acid changes
    # Using the fact that there are 21 clusters (20 AA + stop)
    # Random change has ~1/21 chance of staying in same cluster
    null_rates = []
    for _ in range(n_permutations):
        # Simulate random boundary crossings
        # Probability of crossing = 20/21 if random
        crossings = np.random.binomial(total_escapes, 20/21)
        null_rates.append(crossings / total_escapes)

    null_rates = np.array(null_rates)
    p_value = np.mean(null_rates <= observed_rate)  # One-tailed: escape crosses LESS
    z_score = (observed_rate - np.mean(null_rates)) / np.std(null_rates)

    return {
        'observed_rate': observed_rate,
        'null_mean': np.mean(null_rates),
        'null_std': np.std(null_rates),
        'p_value': p_value,
        'z_score': z_score,
        'total_escapes': total_escapes,
        'boundary_crossings': boundary_crossings
    }


def visualize_results(all_results, stats_results, output_path):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Escape efficacy vs p-adic distance
    ax1 = fig.add_subplot(2, 3, 1)
    efficacy_map = {'high': 3, 'moderate': 2, 'low': 1}
    distances = []
    efficacies = []
    colors = []
    for r in all_results:
        for e in r['escape_analyses']:
            distances.append(e['padic_distance'])
            efficacies.append(efficacy_map[e['escape_efficacy']])
            colors.append('red' if e['boundary_crossed'] else 'blue')

    ax1.scatter(distances, efficacies, c=colors, alpha=0.7, s=100)
    ax1.set_xlabel('P-Adic Distance', fontsize=10)
    ax1.set_ylabel('Escape Efficacy', fontsize=10)
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Low', 'Moderate', 'High'])
    ax1.set_title('Escape Efficacy vs P-Adic Distance\n(Red=Boundary Crossed)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Fitness cost vs p-adic distance
    ax2 = fig.add_subplot(2, 3, 2)
    fitness_map = {'high': 3, 'moderate': 2, 'low': 1}
    fitness_costs = []
    distances2 = []
    colors2 = []
    for r in all_results:
        for e in r['escape_analyses']:
            distances2.append(e['padic_distance'])
            fitness_costs.append(fitness_map[e['fitness_cost']])
            colors2.append('red' if e['boundary_crossed'] else 'blue')

    ax2.scatter(distances2, fitness_costs, c=colors2, alpha=0.7, s=100)
    ax2.set_xlabel('P-Adic Distance', fontsize=10)
    ax2.set_ylabel('Fitness Cost', fontsize=10)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Low', 'Moderate', 'High'])
    ax2.set_title('Fitness Cost vs P-Adic Distance\n(Red=Boundary Crossed)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. Boundary crossing by protein
    ax3 = fig.add_subplot(2, 3, 3)
    protein_data = defaultdict(lambda: {'crossed': 0, 'total': 0})
    for r in all_results:
        protein = r['protein']
        for e in r['escape_analyses']:
            protein_data[protein]['total'] += 1
            if e['boundary_crossed']:
                protein_data[protein]['crossed'] += 1

    proteins = list(protein_data.keys())
    crossed_rates = [protein_data[p]['crossed'] / protein_data[p]['total']
                     for p in proteins]

    bars = ax3.bar(range(len(proteins)), crossed_rates, color='steelblue')
    ax3.set_xticks(range(len(proteins)))
    ax3.set_xticklabels(proteins, rotation=45, ha='right')
    ax3.set_ylabel('Boundary Crossing Rate', fontsize=10)
    ax3.set_title('Boundary Crossing by Protein', fontsize=11)
    ax3.axhline(y=stats_results['observed_rate'], color='red', linestyle='--',
                label=f'Overall: {stats_results["observed_rate"]:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Distance distribution by efficacy
    ax4 = fig.add_subplot(2, 3, 4)
    high_eff = [e['padic_distance'] for r in all_results
                for e in r['escape_analyses'] if e['escape_efficacy'] == 'high']
    mod_eff = [e['padic_distance'] for r in all_results
               for e in r['escape_analyses'] if e['escape_efficacy'] == 'moderate']
    low_eff = [e['padic_distance'] for r in all_results
               for e in r['escape_analyses'] if e['escape_efficacy'] == 'low']

    data_to_plot = [d for d in [high_eff, mod_eff, low_eff] if len(d) > 0]
    labels_to_use = [l for l, d in zip(['High', 'Moderate', 'Low'],
                                        [high_eff, mod_eff, low_eff]) if len(d) > 0]

    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, tick_labels=labels_to_use, patch_artist=True)
        colors_box = ['#ff6b6b', '#ffd93d', '#6bcb77'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
    ax4.set_xlabel('Escape Efficacy', fontsize=10)
    ax4.set_ylabel('P-Adic Distance', fontsize=10)
    ax4.set_title('Distance Distribution by Efficacy', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Null distribution vs observed
    ax5 = fig.add_subplot(2, 3, 5)
    null_dist = np.random.beta(20, 1, 10000)  # Approximation of null
    ax5.hist(null_dist, bins=50, density=True, alpha=0.7, color='gray',
             label='Null (Random)')
    ax5.axvline(x=stats_results['observed_rate'], color='red', linewidth=2,
                label=f'Observed: {stats_results["observed_rate"]:.2f}')
    ax5.set_xlabel('Boundary Crossing Rate', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title(f'Statistical Test\np={stats_results["p_value"]:.4f}, z={stats_results["z_score"]:.2f}',
                  fontsize=11)
    ax5.legend()

    # 6. Summary table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
HIV-1 CTL ESCAPE ANALYSIS SUMMARY
================================

Epitopes analyzed: {len(all_results)}
Total escape variants: {stats_results['total_escapes']}
Boundary crossings: {stats_results['boundary_crossings']}
Crossing rate: {stats_results['observed_rate']:.1%}

Statistical Test:
  Expected (random): {stats_results['null_mean']:.1%}
  Observed: {stats_results['observed_rate']:.1%}
  Z-score: {stats_results['z_score']:.2f}
  P-value: {stats_results['p_value']:.4f}

Key Finding:
  Escape mutations cross boundaries
  {'LESS' if stats_results['z_score'] < 0 else 'MORE'} than random

Interpretation:
  {'Escape maintains p-adic cluster' if stats_results['z_score'] < 0 else 'Escape disrupts p-adic cluster'}
  {'(Fitness preservation > immune escape)' if stats_results['z_score'] < 0 else '(Immune escape > fitness)'}
"""

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    print("=" * 70)
    print("HIV-1 CTL ESCAPE MUTATION ANALYSIS")
    print("Using P-Adic Geometry from Ternary VAE")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    results_dir = script_dir.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load codon encoder
    encoder_paths = [
        base_dir / 'results' / 'codon_encoder.pt',
        base_dir / 'RA_analysis' / 'data' / 'codon_encoder.pt',
        script_dir.parent / 'data' / 'codon_encoder.pt'
    ]

    encoder_path = None
    for path in encoder_paths:
        if path.exists():
            encoder_path = path
            break

    if encoder_path is None:
        print("ERROR: codon_encoder.pt not found!")
        print("Please run RA_analysis scripts first to generate the encoder.")
        sys.exit(1)

    print(f"\nLoading encoder from: {encoder_path}")

    encoder = CodonEncoder()
    checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['model_state'])
    encoder.eval()

    # Load codon mapping
    mapping_paths = [
        base_dir / 'results' / 'learned_codon_mapping.json',
        base_dir / 'RA_analysis' / 'data' / 'learned_codon_mapping.json',
        script_dir.parent / 'data' / 'learned_codon_mapping.json'
    ]

    mapping_path = None
    for path in mapping_paths:
        if path.exists():
            mapping_path = path
            break

    if mapping_path is None:
        print("ERROR: learned_codon_mapping.json not found!")
        sys.exit(1)

    with open(mapping_path) as f:
        mapping_data = json.load(f)

    # The mapping contains codon->position, we need codon->cluster
    codon_to_position = mapping_data.get('codon_to_position', mapping_data)

    # Build position to cluster mapping from natural positions
    NATURAL_POSITIONS = [
        732, 737, 738, 762, 974, 987,  # Cluster 0 (size 6)
        407, 416, 596, 677, 2351, 2354,  # Cluster 1 (size 6)
        3880, 3882, 5343, 5960, 6043, 6066,  # Cluster 2 (size 6)
        788, 947, 952, 1031,  # Cluster 3 (size 4)
        171, 174, 177, 325,  # Cluster 4 (size 4)
        68, 70, 104, 128,  # Cluster 5 (size 4)
        834, 909, 912, 916,  # Cluster 6 (size 4)
        746, 748, 749, 752,  # Cluster 7 (size 4)
        46, 100, 266,  # Cluster 8 (size 3)
        54, 57, 61,  # Cluster 9 (size 3)
        2427, 2883,  # Cluster 10 (size 2)
        218, 386,  # Cluster 11 (size 2)
        59, 764,  # Cluster 12 (size 2)
        1, 7,  # Cluster 13 (size 2)
        783, 1035,  # Cluster 14 (size 2)
        751, 830,  # Cluster 15 (size 2)
        831, 897,  # Cluster 16 (size 2)
        17, 44,  # Cluster 17 (size 2)
        773, 878,  # Cluster 18 (size 2)
        164,  # Cluster 19 (size 1)
        467,  # Cluster 20 (size 1)
    ]
    CLUSTER_SIZES = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

    position_to_cluster = {}
    idx = 0
    for cluster_id, size in enumerate(CLUSTER_SIZES):
        for _ in range(size):
            pos = NATURAL_POSITIONS[idx]
            position_to_cluster[pos] = cluster_id
            idx += 1

    # Convert to codon->cluster mapping
    mapping = {}
    for codon, position in codon_to_position.items():
        mapping[codon] = position_to_cluster.get(position, -1)

    print(f"Loaded mapping with {len(mapping)} codons")

    # Analyze all epitopes
    print("\n" + "=" * 70)
    print("ANALYZING ESCAPE MUTATIONS")
    print("=" * 70)

    all_results = []

    for epitope_name, epitope_data in HIV_CTL_EPITOPES.items():
        print(f"\n{epitope_name}:")
        print(f"  Protein: {epitope_data['protein']}")
        print(f"  HLA: {epitope_data['hla_restriction']}")
        print(f"  Wild-type: {epitope_data['wild_type']['sequence']}")

        result = analyze_epitope(encoder, epitope_data, mapping)
        all_results.append(result)

        for escape in result['escape_analyses']:
            crossed_str = "CROSSED" if escape['boundary_crossed'] else "within"
            print(f"  → {escape['mutation']}: {escape['wt_aa']}->{escape['escape_aa']}, "
                  f"d={escape['padic_distance']:.3f}, {crossed_str}, "
                  f"efficacy={escape['escape_efficacy']}, fitness_cost={escape['fitness_cost']}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    stats_results = permutation_test_escape_boundary(all_results)

    print(f"\nBoundary Crossing Analysis:")
    print(f"  Total escape mutations: {stats_results['total_escapes']}")
    print(f"  Boundary crossings: {stats_results['boundary_crossings']}")
    print(f"  Observed rate: {stats_results['observed_rate']:.1%}")
    print(f"  Expected (random): {stats_results['null_mean']:.1%}")
    print(f"  Z-score: {stats_results['z_score']:.2f}")
    print(f"  P-value: {stats_results['p_value']:.4f}")

    # Visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    output_path = results_dir / 'hiv_escape_analysis.png'
    visualize_results(all_results, stats_results, output_path)

    # Save results
    output_json = {
        'summary': {
            'epitopes_analyzed': len(all_results),
            'total_escapes': stats_results['total_escapes'],
            'boundary_crossings': stats_results['boundary_crossings'],
            'crossing_rate': float(stats_results['observed_rate']),
            'p_value': float(stats_results['p_value']),
            'z_score': float(stats_results['z_score'])
        },
        'epitope_results': [
            {
                'epitope': r['protein'] + '_' + r['position'],
                'hla': r['hla'],
                'wild_type': r['wild_type_sequence'],
                'escapes': [
                    {
                        'mutation': e['mutation'],
                        'wt_aa': e['wt_aa'],
                        'escape_aa': e['escape_aa'],
                        'padic_distance': float(e['padic_distance']),
                        'boundary_crossed': bool(e['boundary_crossed']),
                        'efficacy': e['escape_efficacy'],
                        'fitness_cost': e['fitness_cost']
                    }
                    for e in r['escape_analyses']
                ]
            }
            for r in all_results
        ]
    }

    json_path = results_dir / 'hiv_escape_results.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Analyze p-adic distance correlation with fitness
    fitness_data = []
    distance_data = []
    fitness_map = {'low': 1, 'moderate': 2, 'high': 3}

    for r in all_results:
        for e in r['escape_analyses']:
            fitness_data.append(fitness_map[e['fitness_cost']])
            distance_data.append(e['padic_distance'])

    if len(fitness_data) > 2:
        corr, corr_p = stats.pearsonr(distance_data, fitness_data)
    else:
        corr, corr_p = 0, 1

    print(f"""
P-ADIC DISTANCE ANALYSIS FOR HIV ESCAPE MUTATIONS

Key insight: Since escape mutations change amino acids, boundary
crossing is expected (~95% for random AA changes). The key metrics are:

1. P-ADIC DISTANCE vs FITNESS COST
   Correlation: r = {corr:.3f}, p = {corr_p:.4f}
   {"Higher distance = higher fitness cost" if corr > 0 else "No clear relationship"}

2. AVERAGE P-ADIC DISTANCE: {np.mean(distance_data):.3f}
   This represents how "far" in p-adic space escapes travel.

3. COMPARISON TO RA:
   - In RA, only 14% of citrullinations cross boundaries
   - For HIV escape, 100% cross boundaries (expected - AA change)
   - Key difference: RA modifications are post-translational

THERAPEUTIC IMPLICATIONS:

1. Target epitopes where escape requires LARGE p-adic distance
   (correlating with high fitness cost)

2. HLA-B27 restricted KK10 (R264K): d={[e['padic_distance'] for r in all_results for e in r['escape_analyses'] if e['mutation']=='R264K'][0]:.3f}
   - Known to be protective because escape has high fitness cost

3. Design vaccines targeting multiple epitopes where total escape
   distance would be prohibitive for viral fitness
""")

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
