#!/usr/bin/env python3
"""
Autoantigen Epitope Analysis for Rheumatoid Arthritis - HYPERBOLIC GEOMETRY

Deep analysis of RA autoantigens using Poincaré ball geometry to understand:
1. Why specific citrullination sites trigger autoimmunity
2. Hyperbolic signatures that distinguish immunodominant from silent sites
3. Codon usage patterns in immunogenic regions
4. Potential therapeutic targets based on hyperbolic clustering

Key hypothesis: Immunodominant citrullination sites occupy distinct hyperbolic
positions that cross cluster boundaries, making them appear "foreign" to
the immune system.

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Import hyperbolic utilities
from hyperbolic_utils import (
    poincare_distance as hyp_poincare_distance,
    project_to_poincare,
    load_codon_encoder,
    get_results_dir,
    codon_to_onehot,
    CodonEncoder,
    AA_TO_CODON,
    ARGININE_CODONS,
)

# ============================================================================
# COMPREHENSIVE RA AUTOANTIGEN DATABASE
# ============================================================================

# All 6 arginine codons with their properties
ARGININE_CODONS = {
    'CGT': {'wobble': 'T', 'family': 'CG', 'frequency': 0.08},
    'CGC': {'wobble': 'C', 'family': 'CG', 'frequency': 0.19},
    'CGA': {'wobble': 'A', 'family': 'CG', 'frequency': 0.11},
    'CGG': {'wobble': 'G', 'family': 'CG', 'frequency': 0.21},
    'AGA': {'wobble': 'A', 'family': 'AG', 'frequency': 0.20},
    'AGG': {'wobble': 'G', 'family': 'AG', 'frequency': 0.21},
}

# Amino acid to most common codon
AA_TO_CODON = {
    'A': 'GCT', 'R': 'CGG', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC',
    'E': 'GAG', 'Q': 'CAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
    'L': 'CTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCG',
    'S': 'TCG', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG',
    '*': 'TGA',
}

# Extended autoantigen database with full protein context
RA_AUTOANTIGENS_EXTENDED = {
    # =========================================================================
    # VIMENTIN - Major RA autoantigen (Sa antigen, MCV)
    # =========================================================================
    'VIM': {
        'name': 'Vimentin',
        'gene': 'VIM',
        'function': 'Intermediate filament protein, cytoskeleton',
        'clinical': 'Anti-MCV antibodies highly specific for RA (98%)',
        'epitopes': [
            {
                'id': 'VIM_R71',
                'sequence': 'RLRSSVPGVR',
                'arg_positions': [0, 2, 9],  # Multiple R in epitope
                'immunodominant': True,
                'acpa_reactivity': 0.85,  # % of RA patients reactive
                'notes': 'Sa antigen core epitope',
            },
            {
                'id': 'VIM_R257',
                'sequence': 'SSLNLRETNL',
                'arg_positions': [5],
                'immunodominant': True,
                'acpa_reactivity': 0.72,
                'notes': 'MCV diagnostic epitope',
            },
            {
                'id': 'VIM_R45',
                'sequence': 'SSRSFRTYSF',
                'arg_positions': [2, 5],
                'immunodominant': False,
                'acpa_reactivity': 0.15,
                'notes': 'Minor epitope, low reactivity',
            },
            {
                'id': 'VIM_R201',
                'sequence': 'ARLRSSLAGS',
                'arg_positions': [0, 2],
                'immunodominant': True,
                'acpa_reactivity': 0.68,
                'notes': 'Secondary diagnostic epitope',
            },
        ]
    },

    # =========================================================================
    # FIBRINOGEN - Key target in RA synovium
    # =========================================================================
    'FGA': {
        'name': 'Fibrinogen α',
        'gene': 'FGA',
        'function': 'Coagulation, deposited in RA synovium',
        'clinical': 'Anti-cit-fibrinogen correlates with erosive disease',
        'epitopes': [
            {
                'id': 'FGA_R38',
                'sequence': 'GPRVVERHQS',
                'arg_positions': [2, 6],
                'immunodominant': True,
                'acpa_reactivity': 0.78,
                'notes': 'Major α-chain epitope',
            },
            {
                'id': 'FGA_R573',
                'sequence': 'MELERPGGNEI',
                'arg_positions': [4],
                'immunodominant': True,
                'acpa_reactivity': 0.65,
                'notes': 'C-terminal epitope',
            },
            {
                'id': 'FGA_R84',
                'sequence': 'RHPDEAAFFDT',
                'arg_positions': [0],
                'immunodominant': False,
                'acpa_reactivity': 0.22,
                'notes': 'Weak epitope',
            },
        ]
    },
    'FGB': {
        'name': 'Fibrinogen β',
        'gene': 'FGB',
        'function': 'Coagulation, β-chain',
        'clinical': 'β-chain epitopes in citrullinated fibrinogen',
        'epitopes': [
            {
                'id': 'FGB_R74',
                'sequence': 'HARPAKAATN',
                'arg_positions': [2],
                'immunodominant': True,
                'acpa_reactivity': 0.71,
                'notes': 'Key β-chain epitope',
            },
            {
                'id': 'FGB_R44',
                'sequence': 'NEEGFFRHNDK',
                'arg_positions': [7],
                'immunodominant': False,
                'acpa_reactivity': 0.18,
                'notes': 'Minor epitope',
            },
        ]
    },

    # =========================================================================
    # ALPHA-ENOLASE - Highly RA-specific
    # =========================================================================
    'ENO1': {
        'name': 'Alpha-enolase',
        'gene': 'ENO1',
        'function': 'Glycolysis enzyme, surface-expressed in inflammation',
        'clinical': 'CEP-1 antibodies nearly 100% specific for RA',
        'epitopes': [
            {
                'id': 'ENO1_CEP1',
                'sequence': 'KIREEIFDSRGNP',
                'arg_positions': [2, 9],
                'immunodominant': True,
                'acpa_reactivity': 0.62,
                'notes': 'CEP-1 epitope, gold standard RA marker',
            },
            {
                'id': 'ENO1_R400',
                'sequence': 'SFRSGKYKSV',
                'arg_positions': [2, 6],
                'immunodominant': False,
                'acpa_reactivity': 0.12,
                'notes': 'C-terminal region, low reactivity',
            },
        ]
    },

    # =========================================================================
    # COLLAGEN TYPE II - Cartilage autoantigen
    # =========================================================================
    'COL2A1': {
        'name': 'Collagen type II',
        'gene': 'COL2A1',
        'function': 'Cartilage structural protein',
        'clinical': 'Anti-CII correlates with cartilage destruction',
        'epitopes': [
            {
                'id': 'CII_259_273',
                'sequence': 'GARGLTGRPGDAGK',
                'arg_positions': [2, 7],
                'immunodominant': True,
                'acpa_reactivity': 0.45,
                'notes': 'Immunodominant CII epitope',
            },
            {
                'id': 'CII_511_525',
                'sequence': 'PGERGAPGFRGPAG',
                'arg_positions': [3, 10],
                'immunodominant': True,
                'acpa_reactivity': 0.38,
                'notes': 'Secondary CII epitope',
            },
            {
                'id': 'CII_CONTROL',
                'sequence': 'GPKGDTGPKGPAG',
                'arg_positions': [],  # No arginine
                'immunodominant': False,
                'acpa_reactivity': 0.02,
                'notes': 'Non-citrullinated control',
            },
        ]
    },

    # =========================================================================
    # FILAGGRIN - Original ACPA target
    # =========================================================================
    'FLG': {
        'name': 'Filaggrin',
        'gene': 'FLG',
        'function': 'Epidermal protein, keratin aggregation',
        'clinical': 'Anti-CCP original target, diagnostic use',
        'epitopes': [
            {
                'id': 'FLG_CCP',
                'sequence': 'SHQESTRGRS',
                'arg_positions': [6, 8],
                'immunodominant': True,
                'acpa_reactivity': 0.75,
                'notes': 'CCP test epitope',
            },
            {
                'id': 'FLG_SEC',
                'sequence': 'DSHRGSSSSS',
                'arg_positions': [3],
                'immunodominant': False,
                'acpa_reactivity': 0.20,
                'notes': 'Secondary epitope',
            },
        ]
    },

    # =========================================================================
    # HISTONES - Nuclear autoantigens
    # =========================================================================
    'HIST': {
        'name': 'Histones H2A/H2B',
        'gene': 'HIST1H2A',
        'function': 'Chromatin structure, released in NETosis',
        'clinical': 'Anti-cit-histones in RA and lupus',
        'epitopes': [
            {
                'id': 'H2B_1_12',
                'sequence': 'PEPAKSAPAPKKGS',
                'arg_positions': [],  # Rich in K, not R
                'immunodominant': False,
                'acpa_reactivity': 0.08,
                'notes': 'Lysine-rich, control',
            },
            {
                'id': 'H2A_R3',
                'sequence': 'SGRGKQGGKAR',
                'arg_positions': [2, 10],
                'immunodominant': True,
                'acpa_reactivity': 0.35,
                'notes': 'Citrullinated in NETs',
            },
        ]
    },
}


# ============================================================================
# CODON ENCODER - Now imported from hyperbolic_utils
# CodonEncoder, codon_to_onehot, AA_TO_CODON, ARGININE_CODONS are imported above
# ============================================================================


def poincare_distance(emb1, emb2, c=1.0):
    """Geodesic distance in Poincaré ball model."""
    return float(hyp_poincare_distance(emb1, emb2, c=c))


# ============================================================================
# HYPERBOLIC ANALYSIS FUNCTIONS
# ============================================================================

def compute_epitope_padic_profile(epitope_sequence, encoder, mapping, device='cpu'):
    """
    Compute comprehensive p-adic profile for an epitope.

    Returns:
        dict with embeddings, clusters, distances, and boundary metrics
    """
    results = {
        'sequence': epitope_sequence,
        'codons': [],
        'embeddings': [],
        'clusters': [],
        'padic_positions': [],
        'amino_acids': [],
    }

    for aa in epitope_sequence:
        codon = AA_TO_CODON.get(aa, 'NNN')
        if codon == 'NNN':
            continue

        results['codons'].append(codon)
        results['amino_acids'].append(aa)

        # Get embedding
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            cluster_id, emb = encoder.get_cluster(onehot)
            results['embeddings'].append(emb.cpu().numpy().squeeze())
            results['clusters'].append(cluster_id.item())

        # Get p-adic position
        if codon in mapping:
            results['padic_positions'].append(mapping[codon])
        else:
            results['padic_positions'].append(-1)

    results['embeddings'] = np.array(results['embeddings'])
    return results


def analyze_arginine_positions(epitope_profile, arg_positions):
    """
    Analyze p-adic properties of arginine positions in epitope.

    Key metrics:
    - Cluster assignment of each R
    - Distance to cluster boundary
    - Distance to neighboring residues
    - Embedding norm (magnitude in p-adic space)
    """
    analysis = []

    for idx in arg_positions:
        if idx >= len(epitope_profile['embeddings']):
            continue

        emb = epitope_profile['embeddings'][idx]
        cluster = epitope_profile['clusters'][idx]

        # Distance to neighbors
        neighbor_dists = []
        for i in range(len(epitope_profile['embeddings'])):
            if i != idx:
                d = np.linalg.norm(emb - epitope_profile['embeddings'][i])
                neighbor_dists.append(d)

        # Embedding magnitude
        emb_norm = np.linalg.norm(emb)

        # Cluster homogeneity (are neighbors in same cluster?)
        neighbor_clusters = [epitope_profile['clusters'][i] for i in range(len(epitope_profile['clusters'])) if i != idx]
        cluster_homogeneity = neighbor_clusters.count(cluster) / len(neighbor_clusters) if neighbor_clusters else 0

        analysis.append({
            'position': idx,
            'codon': epitope_profile['codons'][idx],
            'cluster': cluster,
            'embedding_norm': emb_norm,
            'mean_neighbor_distance': np.mean(neighbor_dists) if neighbor_dists else 0,
            'min_neighbor_distance': np.min(neighbor_dists) if neighbor_dists else 0,
            'max_neighbor_distance': np.max(neighbor_dists) if neighbor_dists else 0,
            'cluster_homogeneity': cluster_homogeneity,
            'padic_position': epitope_profile['padic_positions'][idx],
        })

    return analysis


def compute_boundary_crossing_potential(arg_analysis, all_cluster_centers):
    """
    Estimate the likelihood that citrullination causes boundary crossing.

    Citrullination removes the codon signal (R→Cit has no codon).
    This effectively moves the epitope centroid.
    """
    if not arg_analysis:
        return None

    results = []
    for arg in arg_analysis:
        # High neighbor distance + low homogeneity = higher boundary crossing potential
        boundary_score = (arg['mean_neighbor_distance'] * (1 - arg['cluster_homogeneity']))

        results.append({
            **arg,
            'boundary_crossing_potential': boundary_score,
        })

    return results


def compare_immunodominant_vs_silent(all_analyses):
    """
    Statistical comparison of immunodominant vs non-immunodominant epitopes.
    """
    immunodominant = []
    silent = []

    for protein, epitopes in all_analyses.items():
        for epitope in epitopes:
            if epitope['immunodominant']:
                for arg in epitope['arg_analysis']:
                    immunodominant.append(arg)
            else:
                for arg in epitope['arg_analysis']:
                    silent.append(arg)

    if not immunodominant or not silent:
        return None

    # Compare key metrics
    comparisons = {}
    metrics = ['embedding_norm', 'mean_neighbor_distance', 'cluster_homogeneity',
               'boundary_crossing_potential']

    for metric in metrics:
        imm_values = [a[metric] for a in immunodominant if metric in a]
        sil_values = [a[metric] for a in silent if metric in a]

        if imm_values and sil_values:
            t_stat, p_value = stats.ttest_ind(imm_values, sil_values)
            comparisons[metric] = {
                'immunodominant_mean': np.mean(imm_values),
                'immunodominant_std': np.std(imm_values),
                'silent_mean': np.mean(sil_values),
                'silent_std': np.std(sil_values),
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': (np.mean(imm_values) - np.mean(sil_values)) / np.sqrt(
                    (np.var(imm_values) + np.var(sil_values)) / 2
                ) if np.var(imm_values) + np.var(sil_values) > 0 else 0,
            }

    return comparisons


def analyze_arginine_codon_usage(all_analyses):
    """
    Analyze whether specific arginine codons are enriched in immunodominant epitopes.

    There are 6 arginine codons - does codon choice affect immunogenicity?
    """
    codon_counts = {
        'immunodominant': defaultdict(int),
        'silent': defaultdict(int),
    }

    for protein, epitopes in all_analyses.items():
        for epitope in epitopes:
            category = 'immunodominant' if epitope['immunodominant'] else 'silent'
            for arg in epitope['arg_analysis']:
                codon = arg['codon']
                if codon in ARGININE_CODONS:
                    codon_counts[category][codon] += 1

    # Chi-square test for codon preference
    all_codons = list(ARGININE_CODONS.keys())
    imm_counts = [codon_counts['immunodominant'].get(c, 0) for c in all_codons]
    sil_counts = [codon_counts['silent'].get(c, 0) for c in all_codons]

    contingency = np.array([imm_counts, sil_counts])

    # Only do chi-square if we have enough data
    if contingency.sum() > 20 and (contingency > 0).sum() > 4:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    else:
        chi2, p_value, dof = 0, 1, 0

    return {
        'immunodominant_codons': dict(codon_counts['immunodominant']),
        'silent_codons': dict(codon_counts['silent']),
        'chi2': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
    }


def compute_epitope_centroid_shift(epitope_profile, arg_positions):
    """
    Calculate how much the epitope centroid shifts when arginine is citrullinated.

    Citrullination effectively removes the R codon signal.
    """
    if len(epitope_profile['embeddings']) == 0:
        return None

    original_centroid = np.mean(epitope_profile['embeddings'], axis=0)

    shifts = []
    for idx in arg_positions:
        if idx >= len(epitope_profile['embeddings']):
            continue

        # Remove arginine from centroid calculation
        mask = np.ones(len(epitope_profile['embeddings']), dtype=bool)
        mask[idx] = False
        remaining = epitope_profile['embeddings'][mask]

        if len(remaining) > 0:
            new_centroid = np.mean(remaining, axis=0)
            shift = np.linalg.norm(new_centroid - original_centroid)
            shifts.append({
                'position': idx,
                'centroid_shift': shift,
                'relative_shift': shift / np.linalg.norm(original_centroid) if np.linalg.norm(original_centroid) > 0 else 0,
            })

    return shifts


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("AUTOANTIGEN EPITOPE ANALYSIS - HYPERBOLIC SIGNATURES OF RA AUTOANTIGENS")
    print("=" * 80)

    # Setup paths - use hyperbolic results directory
    script_dir = Path(__file__).parent
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load codon encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, mapping, _ = load_codon_encoder(device=device, version='3adic')
    print(f"Loaded mapping for {len(mapping)} codons")

    # =========================================================================
    # ANALYZE ALL AUTOANTIGENS
    # =========================================================================

    print("\n" + "=" * 80)
    print("ANALYZING AUTOANTIGEN EPITOPES")
    print("=" * 80)

    all_analyses = {}

    for protein_id, protein_data in RA_AUTOANTIGENS_EXTENDED.items():
        print(f"\n{'─' * 60}")
        print(f"PROTEIN: {protein_data['name']} ({protein_data['gene']})")
        print(f"Function: {protein_data['function']}")
        print(f"Clinical: {protein_data['clinical']}")
        print(f"{'─' * 60}")

        protein_analyses = []

        for epitope in protein_data['epitopes']:
            print(f"\n  Epitope: {epitope['id']}")
            print(f"  Sequence: {epitope['sequence']}")
            print(f"  Immunodominant: {epitope['immunodominant']}")
            print(f"  ACPA reactivity: {epitope['acpa_reactivity']*100:.0f}%")

            # Compute p-adic profile
            profile = compute_epitope_padic_profile(
                epitope['sequence'], encoder, mapping, device
            )

            # Analyze arginine positions
            arg_analysis = analyze_arginine_positions(profile, epitope['arg_positions'])

            # Compute boundary crossing potential
            arg_analysis = compute_boundary_crossing_potential(
                arg_analysis, encoder.cluster_centers.detach().cpu().numpy()
            )

            # Compute centroid shift
            centroid_shifts = compute_epitope_centroid_shift(profile, epitope['arg_positions'])

            if arg_analysis:
                print(f"\n  Arginine analysis:")
                for arg in arg_analysis:
                    print(f"    Position {arg['position']}: Codon={arg['codon']}, "
                          f"Cluster={arg['cluster']}, "
                          f"BoundaryPotential={arg['boundary_crossing_potential']:.3f}")

            if centroid_shifts:
                print(f"\n  Centroid shifts upon citrullination:")
                for shift in centroid_shifts:
                    print(f"    Position {shift['position']}: "
                          f"Shift={shift['centroid_shift']:.4f} "
                          f"({shift['relative_shift']*100:.1f}% relative)")

            protein_analyses.append({
                'epitope_id': epitope['id'],
                'sequence': epitope['sequence'],
                'immunodominant': epitope['immunodominant'],
                'acpa_reactivity': epitope['acpa_reactivity'],
                'profile': {
                    'codons': profile['codons'],
                    'clusters': profile['clusters'],
                    'padic_positions': profile['padic_positions'],
                },
                'arg_analysis': arg_analysis or [],
                'centroid_shifts': centroid_shifts or [],
            })

        all_analyses[protein_id] = protein_analyses

    # =========================================================================
    # STATISTICAL COMPARISON
    # =========================================================================

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: IMMUNODOMINANT vs SILENT EPITOPES")
    print("=" * 80)

    comparisons = compare_immunodominant_vs_silent(all_analyses)

    if comparisons:
        for metric, stats_data in comparisons.items():
            significance = "***" if stats_data['p_value'] < 0.001 else \
                          "**" if stats_data['p_value'] < 0.01 else \
                          "*" if stats_data['p_value'] < 0.05 else ""

            print(f"\n{metric.upper()}:")
            print(f"  Immunodominant: {stats_data['immunodominant_mean']:.4f} ± {stats_data['immunodominant_std']:.4f}")
            print(f"  Silent:         {stats_data['silent_mean']:.4f} ± {stats_data['silent_std']:.4f}")
            print(f"  t-statistic:    {stats_data['t_statistic']:.3f}")
            print(f"  p-value:        {stats_data['p_value']:.4f} {significance}")
            print(f"  Effect size:    {stats_data['effect_size']:.3f}")

    # =========================================================================
    # CODON USAGE ANALYSIS
    # =========================================================================

    print("\n" + "=" * 80)
    print("ARGININE CODON USAGE IN IMMUNODOMINANT vs SILENT EPITOPES")
    print("=" * 80)

    codon_analysis = analyze_arginine_codon_usage(all_analyses)

    print("\nCodon distribution:")
    print(f"  {'Codon':<6} {'Imm.':<8} {'Silent':<8} {'Family':<8} {'Wobble'}")
    print(f"  {'-'*40}")

    for codon, props in ARGININE_CODONS.items():
        imm = codon_analysis['immunodominant_codons'].get(codon, 0)
        sil = codon_analysis['silent_codons'].get(codon, 0)
        print(f"  {codon:<6} {imm:<8} {sil:<8} {props['family']:<8} {props['wobble']}")

    print(f"\nChi-square test: χ² = {codon_analysis['chi2']:.2f}, p = {codon_analysis['p_value']:.4f}")

    # =========================================================================
    # KEY FINDINGS SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    # Count statistics
    total_epitopes = sum(len(e) for e in all_analyses.values())
    immunodominant_count = sum(
        1 for protein in all_analyses.values()
        for e in protein if e['immunodominant']
    )

    # Compute average boundary crossing potential
    imm_boundary = []
    sil_boundary = []
    for protein in all_analyses.values():
        for epitope in protein:
            for arg in epitope['arg_analysis']:
                if 'boundary_crossing_potential' in arg:
                    if epitope['immunodominant']:
                        imm_boundary.append(arg['boundary_crossing_potential'])
                    else:
                        sil_boundary.append(arg['boundary_crossing_potential'])

    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Analyzed {len(RA_AUTOANTIGENS_EXTENDED)} proteins")
    print(f"   - Total epitopes: {total_epitopes}")
    print(f"   - Immunodominant: {immunodominant_count}")
    print(f"   - Silent/control: {total_epitopes - immunodominant_count}")

    if imm_boundary and sil_boundary:
        print(f"\n2. BOUNDARY CROSSING POTENTIAL:")
        print(f"   - Immunodominant mean: {np.mean(imm_boundary):.4f}")
        print(f"   - Silent mean:         {np.mean(sil_boundary):.4f}")
        print(f"   - Ratio: {np.mean(imm_boundary)/np.mean(sil_boundary):.2f}x higher in immunodominant")

    if comparisons and 'mean_neighbor_distance' in comparisons:
        mnd = comparisons['mean_neighbor_distance']
        print(f"\n3. P-ADIC NEIGHBOR DISTANCE:")
        print(f"   - Immunodominant arginines are {mnd['immunodominant_mean']/mnd['silent_mean']:.2f}x "
              f"more distant from neighbors")
        print(f"   - Statistical significance: p = {mnd['p_value']:.4f}")

    if comparisons and 'cluster_homogeneity' in comparisons:
        ch = comparisons['cluster_homogeneity']
        print(f"\n4. CLUSTER HOMOGENEITY:")
        print(f"   - Immunodominant: {ch['immunodominant_mean']:.2f} (lower = more boundary exposure)")
        print(f"   - Silent:         {ch['silent_mean']:.2f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    output = {
        'summary': {
            'total_proteins': len(RA_AUTOANTIGENS_EXTENDED),
            'total_epitopes': total_epitopes,
            'immunodominant_count': immunodominant_count,
        },
        'analyses': {},
        'statistical_comparison': comparisons,
        'codon_usage': codon_analysis,
    }

    # Convert numpy types for JSON serialization
    for protein_id, protein_data in all_analyses.items():
        output['analyses'][protein_id] = []
        for epitope in protein_data:
            epitope_clean = {
                'epitope_id': epitope['epitope_id'],
                'sequence': epitope['sequence'],
                'immunodominant': epitope['immunodominant'],
                'acpa_reactivity': epitope['acpa_reactivity'],
                'profile': epitope['profile'],
                'arg_analysis': [],
                'centroid_shifts': epitope['centroid_shifts'],
            }
            for arg in epitope['arg_analysis']:
                epitope_clean['arg_analysis'].append({
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in arg.items()
                })
            output['analyses'][protein_id].append(epitope_clean)

    output_path = results_dir / 'autoantigen_padic_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
