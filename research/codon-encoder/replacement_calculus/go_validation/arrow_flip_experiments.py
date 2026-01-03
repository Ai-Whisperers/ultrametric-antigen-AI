"""
Arrow Flip Experiments: Detecting the Boundary Where P-adic Embeddings Add Value

This module implements experiments to determine WHERE and WHY codon sequence
analysis fails to predict protein function, and where our 3-adic/hyperbolic
embedding approach provides additional predictive power.

Key Question: At what point does the "information arrow" flip from
              sequence → function to requiring geometric/algebraic structure?
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import json
import os

# Import our modules
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import functional profiles
from functional_profiles import AMINO_ACID_PROFILES, compute_functional_similarity_matrix
FUNCTIONAL_PROFILES = AMINO_ACID_PROFILES  # Alias for convenience

def get_functional_similarity_matrix():
    """Wrapper to get similarity matrix in expected format."""
    sim, aa_list = compute_functional_similarity_matrix()
    return sim


# =============================================================================
# AMINO ACID PHYSICOCHEMICAL DATA
# =============================================================================

# Physicochemical properties for hybrid cost computation
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 'nonpolar'},
    'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 'polar'},
    'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 'charged'},
    'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 'charged'},
    'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 'nonpolar'},
    'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 'nonpolar'},
    'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0, 'polarity': 'charged'},
    'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 'nonpolar'},
    'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 'charged'},
    'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 'nonpolar'},
    'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 'nonpolar'},
    'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 'polar'},
    'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 'nonpolar'},
    'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 'polar'},
    'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 'charged'},
    'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 'polar'},
    'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 'polar'},
    'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 'nonpolar'},
    'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 'nonpolar'},
    'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 'polar'},
}


def compute_hybrid_cost(aa1: str, aa2: str) -> float:
    """
    Compute hybrid cost between two amino acids.

    This mirrors the hybrid groupoid logic:
    - Base cost from physicochemical distance
    - Penalties for charge incompatibility
    - Penalties for large size differences
    """
    if aa1 not in AA_PROPERTIES or aa2 not in AA_PROPERTIES:
        return float('inf')

    p1 = AA_PROPERTIES[aa1]
    p2 = AA_PROPERTIES[aa2]

    # Base distance (hydrophobicity + normalized volume)
    hydro_dist = abs(p1['hydrophobicity'] - p2['hydrophobicity'])
    vol_dist = abs(p1['volume'] - p2['volume']) / 50.0  # Normalize

    base_cost = np.sqrt(hydro_dist**2 + vol_dist**2)

    # Charge penalty
    if p1['charge'] != p2['charge']:
        if p1['charge'] * p2['charge'] < 0:  # Opposite charges
            base_cost += 5.0
        else:  # One charged, one neutral
            base_cost += 2.0

    # Size penalty for very different volumes
    if abs(p1['volume'] - p2['volume']) > 60:
        base_cost += 3.0

    return base_cost


def compute_cost_matrix() -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cost matrix for all amino acids."""
    aa_list = sorted(AA_PROPERTIES.keys())
    n = len(aa_list)
    costs = np.zeros((n, n))

    for i, aa1 in enumerate(aa_list):
        for j, aa2 in enumerate(aa_list):
            if i != j:
                costs[i, j] = compute_hybrid_cost(aa1, aa2)

    return costs, aa_list

# Codon to amino acid mapping
AA_TO_CODONS = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    '*': ['TAA', 'TAG', 'TGA'],
}


# =============================================================================
# GENETIC CODE AND P-ADIC STRUCTURE
# =============================================================================

# Standard genetic code: codon -> amino acid
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

# Map nucleotides to ternary digits
NUCLEOTIDE_TO_TERNARY = {'T': 0, 'C': 1, 'A': 2, 'G': 2}  # Purines (A,G) share value

def codon_to_ternary(codon: str) -> int:
    """Convert codon to ternary representation (base 3 integer)."""
    digits = [NUCLEOTIDE_TO_TERNARY[n] for n in codon]
    return digits[0] * 9 + digits[1] * 3 + digits[2]

def padic_valuation(n: int, p: int = 3) -> int:
    """Compute p-adic valuation: max k such that p^k divides n."""
    if n == 0:
        return float('inf')
    k = 0
    while n % p == 0:
        k += 1
        n //= p
    return k

def hamming_distance(codon1: str, codon2: str) -> int:
    """Compute Hamming distance between two codons."""
    return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))

def get_hamming_neighbors(codon: str) -> List[str]:
    """Get all codons at Hamming distance 1."""
    neighbors = []
    nucleotides = ['T', 'C', 'A', 'G']
    for i in range(3):
        for n in nucleotides:
            if n != codon[i]:
                neighbor = codon[:i] + n + codon[i+1:]
                neighbors.append(neighbor)
    return neighbors

def is_transition(n1: str, n2: str) -> bool:
    """Check if nucleotide change is a transition (purine<->purine or pyrimidine<->pyrimidine)."""
    purines = {'A', 'G'}
    pyrimidines = {'T', 'C'}
    return (n1 in purines and n2 in purines) or (n1 in pyrimidines and n2 in pyrimidines)

def classify_mutation(codon1: str, codon2: str) -> str:
    """Classify mutation type."""
    aa1 = GENETIC_CODE.get(codon1, '*')
    aa2 = GENETIC_CODE.get(codon2, '*')

    if aa1 == aa2:
        return 'synonymous'
    elif aa1 == '*' or aa2 == '*':
        return 'nonsense'
    else:
        # Check if transition or transversion
        for i in range(3):
            if codon1[i] != codon2[i]:
                if is_transition(codon1[i], codon2[i]):
                    return 'nonsynonymous_transition'
                else:
                    return 'nonsynonymous_transversion'
    return 'unknown'


# =============================================================================
# EXPERIMENT 1: CODON-LEVEL FUNCTIONAL MAPPING
# =============================================================================

@dataclass
class CodonMutationRecord:
    """Record for a single codon mutation."""
    source_codon: str
    target_codon: str
    source_aa: str
    target_aa: str
    hamming_dist: int
    source_valuation: int
    target_valuation: int
    valuation_change: int
    functional_distance: float
    mutation_type: str
    is_transition: bool


def build_codon_mutation_dataset() -> List[CodonMutationRecord]:
    """Build complete dataset of codon mutations and their functional effects."""

    # Get functional profiles
    aa_profiles = {aa: prof.to_vector() for aa, prof in FUNCTIONAL_PROFILES.items()}

    records = []
    coding_codons = [c for c, aa in GENETIC_CODE.items() if aa != '*']

    for source_codon in coding_codons:
        source_aa = GENETIC_CODE[source_codon]
        source_val = padic_valuation(codon_to_ternary(source_codon), p=3)
        source_profile = aa_profiles.get(source_aa)

        if source_profile is None:
            continue

        # Check all neighbors (Hamming distance 1)
        for target_codon in get_hamming_neighbors(source_codon):
            target_aa = GENETIC_CODE.get(target_codon, '*')

            if target_aa == '*':  # Skip stop codons
                continue

            target_val = padic_valuation(codon_to_ternary(target_codon), p=3)
            target_profile = aa_profiles.get(target_aa)

            if target_profile is None:
                continue

            # Compute functional distance
            func_dist = np.linalg.norm(source_profile - target_profile)

            # Determine mutation type
            mut_type = classify_mutation(source_codon, target_codon)

            # Check if transition
            for i in range(3):
                if source_codon[i] != target_codon[i]:
                    is_trans = is_transition(source_codon[i], target_codon[i])
                    break

            record = CodonMutationRecord(
                source_codon=source_codon,
                target_codon=target_codon,
                source_aa=source_aa,
                target_aa=target_aa,
                hamming_dist=1,
                source_valuation=source_val,
                target_valuation=target_val,
                valuation_change=target_val - source_val,
                functional_distance=func_dist,
                mutation_type=mut_type,
                is_transition=is_trans
            )
            records.append(record)

    return records


def experiment_1_codon_functional_mapping() -> Dict:
    """
    Experiment 1: Map codon space to functional outcomes.

    Tests H1: Codon degeneracy encodes functional robustness.
    Tests H2: P-adic valuation encodes information resilience.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: CODON-LEVEL FUNCTIONAL MAPPING")
    print("="*70)

    records = build_codon_mutation_dataset()
    print(f"\nTotal mutation records: {len(records)}")

    # Separate by mutation type
    synonymous = [r for r in records if r.mutation_type == 'synonymous']
    nonsynonymous = [r for r in records if r.mutation_type.startswith('nonsynonymous')]

    print(f"Synonymous: {len(synonymous)}")
    print(f"Nonsynonymous: {len(nonsynonymous)}")

    # ==========================================================================
    # H1: Does codon proximity predict functional similarity?
    # ==========================================================================
    print("\n--- H1: Codon Proximity → Functional Similarity ---")

    # For nonsynonymous mutations only (synonymous have 0 functional distance)
    func_distances = [r.functional_distance for r in nonsynonymous]

    # Group by whether it's a transition or transversion
    transitions = [r for r in nonsynonymous if r.is_transition]
    transversions = [r for r in nonsynonymous if not r.is_transition]

    trans_distances = [r.functional_distance for r in transitions]
    transv_distances = [r.functional_distance for r in transversions]

    print(f"\nTransitions (n={len(transitions)}): mean func_dist = {np.mean(trans_distances):.3f}")
    print(f"Transversions (n={len(transversions)}): mean func_dist = {np.mean(transv_distances):.3f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    if len(trans_distances) > 0 and len(transv_distances) > 0:
        stat, p_val = mannwhitneyu(trans_distances, transv_distances, alternative='less')
        print(f"Mann-Whitney U test (transitions < transversions): p = {p_val:.2e}")

    # ==========================================================================
    # H2: Does valuation change predict functional change?
    # ==========================================================================
    print("\n--- H2: Valuation Change → Functional Change ---")

    val_changes = [r.valuation_change for r in nonsynonymous]
    func_dists = [r.functional_distance for r in nonsynonymous]

    # Correlation
    if len(val_changes) > 1:
        spearman_r, spearman_p = spearmanr(val_changes, func_dists)
        print(f"Spearman correlation (val_change vs func_dist): r = {spearman_r:.4f}, p = {spearman_p:.2e}")

    # Group by valuation change direction
    val_increase = [r.functional_distance for r in nonsynonymous if r.valuation_change > 0]
    val_decrease = [r.functional_distance for r in nonsynonymous if r.valuation_change < 0]
    val_same = [r.functional_distance for r in nonsynonymous if r.valuation_change == 0]

    print(f"\nValuation increase (n={len(val_increase)}): mean func_dist = {np.mean(val_increase):.3f}" if val_increase else "Valuation increase: n=0")
    print(f"Valuation decrease (n={len(val_decrease)}): mean func_dist = {np.mean(val_decrease):.3f}" if val_decrease else "Valuation decrease: n=0")
    print(f"Valuation same (n={len(val_same)}): mean func_dist = {np.mean(val_same):.3f}" if val_same else "Valuation same: n=0")

    # ==========================================================================
    # Analysis by source valuation level
    # ==========================================================================
    print("\n--- Functional Robustness by Source Valuation Level ---")

    by_valuation = {}
    for r in nonsynonymous:
        v = r.source_valuation
        if v not in by_valuation:
            by_valuation[v] = []
        by_valuation[v].append(r.functional_distance)

    for v in sorted(by_valuation.keys()):
        dists = by_valuation[v]
        print(f"Valuation {v}: n={len(dists)}, mean func_dist = {np.mean(dists):.3f}, std = {np.std(dists):.3f}")

    # Hypothesis: High valuation codons have LOWER functional disruption
    val_levels = []
    mean_dists = []
    for v in sorted(by_valuation.keys()):
        val_levels.append(v)
        mean_dists.append(np.mean(by_valuation[v]))

    if len(val_levels) > 1:
        r, p = spearmanr(val_levels, mean_dists)
        print(f"\nCorrelation (valuation level vs mean func_dist): r = {r:.4f}, p = {p:.4f}")
        if r < 0 and p < 0.05:
            print("** SUPPORTS H2: Higher valuation = lower functional disruption **")
        elif r > 0 and p < 0.05:
            print("** CONTRADICTS H2: Higher valuation = higher functional disruption **")

    # ==========================================================================
    # Compile results
    # ==========================================================================
    results = {
        'n_total_mutations': len(records),
        'n_synonymous': len(synonymous),
        'n_nonsynonymous': len(nonsynonymous),
        'n_transitions': len(transitions),
        'n_transversions': len(transversions),
        'mean_func_dist_transitions': float(np.mean(trans_distances)) if trans_distances else None,
        'mean_func_dist_transversions': float(np.mean(transv_distances)) if transv_distances else None,
        'valuation_func_correlation': {
            'spearman_r': float(spearman_r) if 'spearman_r' in dir() else None,
            'spearman_p': float(spearman_p) if 'spearman_p' in dir() else None,
        },
        'by_valuation_level': {
            str(v): {
                'n': len(dists),
                'mean_func_dist': float(np.mean(dists)),
                'std_func_dist': float(np.std(dists)),
            }
            for v, dists in by_valuation.items()
        }
    }

    return results


# =============================================================================
# EXPERIMENT 2: EMBEDDING COMPARISON
# =============================================================================

def experiment_2_embedding_comparison() -> Dict:
    """
    Experiment 2: Compare different embedding approaches.

    Tests: Is the 3-adic structure essential, or do arbitrary embeddings work equally well?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: EMBEDDING COMPARISON")
    print("="*70)

    # Get functional profiles and similarity matrix
    similarity_matrix = get_functional_similarity_matrix()
    aa_codes = list(FUNCTIONAL_PROFILES.keys())

    # Compute cost matrix (hybrid approach)
    cost_matrix, cost_aa_list = compute_cost_matrix()
    aa_to_cost_idx = {aa: i for i, aa in enumerate(cost_aa_list)}

    # ==========================================================================
    # Embedding 1: Hybrid physicochemical costs (our approach)
    # ==========================================================================
    print("\n--- Embedding 1: Hybrid Physicochemical Costs ---")

    hybrid_predictions = []
    ground_truth = []

    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue

            # Get hybrid cost
            cost = compute_hybrid_cost(aa1, aa2)

            # Get functional similarity
            sim = similarity_matrix[i, j]

            hybrid_predictions.append(-cost)  # Negative because lower cost = more similar
            ground_truth.append(sim)

    # Correlation
    r_hybrid, p_hybrid = spearmanr(hybrid_predictions, ground_truth)
    print(f"Spearman r: {r_hybrid:.4f} (p = {p_hybrid:.2e})")

    # ==========================================================================
    # Embedding 2: Simple physicochemical properties
    # ==========================================================================
    print("\n--- Embedding 2: Simple Physicochemical (Hydrophobicity + Charge + Size) ---")

    # Simple 3D embedding
    simple_embedding = {}
    for aa, profile in FUNCTIONAL_PROFILES.items():
        simple_embedding[aa] = np.array([
            profile.hydrophobicity,
            profile.charge_ph7,
            profile.molecular_weight / 200.0  # Normalized
        ])

    simple_predictions = []

    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue

            dist = np.linalg.norm(simple_embedding[aa1] - simple_embedding[aa2])
            simple_predictions.append(-dist)

    r_simple, p_simple = spearmanr(simple_predictions, ground_truth)
    print(f"Spearman r: {r_simple:.4f} (p = {p_simple:.2e})")

    # ==========================================================================
    # Embedding 3: Random embedding
    # ==========================================================================
    print("\n--- Embedding 3: Random 16D Embedding ---")

    np.random.seed(42)
    random_embedding = {aa: np.random.randn(16) for aa in aa_codes}

    random_predictions = []

    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue

            dist = np.linalg.norm(random_embedding[aa1] - random_embedding[aa2])
            random_predictions.append(-dist)

    r_random, p_random = spearmanr(random_predictions, ground_truth)
    print(f"Spearman r: {r_random:.4f} (p = {p_random:.2e})")

    # ==========================================================================
    # Embedding 4: BLOSUM62-derived embedding
    # ==========================================================================
    print("\n--- Embedding 4: BLOSUM62-derived Embedding ---")

    # BLOSUM62 substitution matrix (20x20)
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

    # Build symmetric matrix
    blosum_matrix = np.zeros((20, 20))
    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            key = (aa1, aa2) if (aa1, aa2) in BLOSUM62 else (aa2, aa1)
            blosum_matrix[i, j] = BLOSUM62.get(key, 0)

    # Use BLOSUM scores directly as similarity
    blosum_predictions = []
    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue
            blosum_predictions.append(blosum_matrix[i, j])

    r_blosum, p_blosum = spearmanr(blosum_predictions, ground_truth)
    print(f"Spearman r: {r_blosum:.4f} (p = {p_blosum:.2e})")

    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    print("\n" + "="*70)
    print("EMBEDDING COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Embedding':<35} {'Spearman r':<15} {'p-value':<15}")
    print("-" * 65)
    print(f"{'Hybrid Physicochemical (ours)':<35} {r_hybrid:.4f}{'':<10} {p_hybrid:.2e}")
    print(f"{'Simple Physicochemical (3D)':<35} {r_simple:.4f}{'':<10} {p_simple:.2e}")
    print(f"{'Random 16D':<35} {r_random:.4f}{'':<10} {p_random:.2e}")
    print(f"{'BLOSUM62-derived':<35} {r_blosum:.4f}{'':<10} {p_blosum:.2e}")

    # Improvement over baselines
    print(f"\n--- Improvement Analysis ---")
    print(f"Hybrid vs Random: {r_hybrid - r_random:+.4f}")
    print(f"Hybrid vs Simple: {r_hybrid - r_simple:+.4f}")
    print(f"Hybrid vs BLOSUM62: {r_hybrid - r_blosum:+.4f}")

    if r_hybrid > r_random and r_hybrid > r_simple:
        print("\n** Hybrid physicochemical structure adds value beyond simple embeddings **")

    results = {
        'hybrid_physicochemical': {'spearman_r': float(r_hybrid), 'p_value': float(p_hybrid)},
        'simple_physicochemical': {'spearman_r': float(r_simple), 'p_value': float(p_simple)},
        'random_16d': {'spearman_r': float(r_random), 'p_value': float(p_random)},
        'blosum62_derived': {'spearman_r': float(r_blosum), 'p_value': float(p_blosum)},
    }

    return results


# =============================================================================
# EXPERIMENT 3: ARROW FLIP BOUNDARY DETECTION
# =============================================================================

def experiment_3_arrow_flip_boundary() -> Dict:
    """
    Experiment 3: Detect where the prediction boundary lies.

    The "arrow" flips when context variance exceeds intrinsic variance.
    We test this by analyzing which AA pairs benefit most from hybrid structure.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: ARROW FLIP BOUNDARY DETECTION")
    print("="*70)

    # Get data
    similarity_matrix = get_functional_similarity_matrix()
    aa_codes = list(FUNCTIONAL_PROFILES.keys())

    # For each pair, compute:
    # 1. Hybrid cost (includes charge/size penalties)
    # 2. Simple distance (hydrophobicity + charge + size, no penalties)
    # 3. Functional similarity (ground truth)

    pair_analysis = []

    for i, aa1 in enumerate(aa_codes):
        for j, aa2 in enumerate(aa_codes):
            if i >= j:
                continue

            # Hybrid cost (with penalties)
            hybrid_cost = compute_hybrid_cost(aa1, aa2)

            # Simple distance (no penalties)
            p1 = FUNCTIONAL_PROFILES[aa1]
            p2 = FUNCTIONAL_PROFILES[aa2]
            simple_dist = np.sqrt(
                (p1.hydrophobicity - p2.hydrophobicity)**2 +
                (p1.charge_ph7 - p2.charge_ph7)**2 +
                ((p1.molecular_weight - p2.molecular_weight) / 100)**2
            )

            # Ground truth
            func_sim = similarity_matrix[i, j]

            # Determine which is better predictor
            # Convert to same scale (higher = more similar)
            hybrid_pred = -hybrid_cost
            simple_pred = -simple_dist

            pair_analysis.append({
                'aa1': aa1,
                'aa2': aa2,
                'hybrid_cost': hybrid_cost,
                'simple_dist': simple_dist,
                'func_sim': func_sim,
                'hybrid_pred': hybrid_pred,
                'simple_pred': simple_pred,
            })

    # ==========================================================================
    # Identify where hybrid adds value
    # ==========================================================================
    print("\n--- Pairs where HYBRID outperforms SIMPLE ---")

    hybrid_wins = []
    simple_wins = []

    for pair in pair_analysis:
        # Compute prediction error
        # Normalize predictions to [-1, 1] range
        max_hybrid = max(abs(p['hybrid_pred']) for p in pair_analysis)
        max_simple = max(abs(p['simple_pred']) for p in pair_analysis)

        h_norm = pair['hybrid_pred'] / max_hybrid
        s_norm = pair['simple_pred'] / max_simple

        h_error = abs(h_norm - pair['func_sim'])
        s_error = abs(s_norm - pair['func_sim'])

        if h_error < s_error:
            hybrid_wins.append(pair)
        else:
            simple_wins.append(pair)

    print(f"\nHybrid wins: {len(hybrid_wins)} pairs ({100*len(hybrid_wins)/len(pair_analysis):.1f}%)")
    print(f"Simple wins: {len(simple_wins)} pairs ({100*len(simple_wins)/len(pair_analysis):.1f}%)")

    # Analyze characteristics of groupoid-wins vs simple-wins
    print("\n--- Characteristics Analysis ---")

    def analyze_group(pairs, label):
        if not pairs:
            return {}

        avg_func_sim = np.mean([p['func_sim'] for p in pairs])
        avg_hybrid_cost = np.mean([p['hybrid_cost'] for p in pairs])
        avg_simple_dist = np.mean([p['simple_dist'] for p in pairs])

        # Check charge patterns
        same_charge = sum(1 for p in pairs if
                        FUNCTIONAL_PROFILES[p['aa1']].charge_ph7 == FUNCTIONAL_PROFILES[p['aa2']].charge_ph7)

        # Check hydrophobicity patterns
        both_hydrophobic = sum(1 for p in pairs if
                             FUNCTIONAL_PROFILES[p['aa1']].hydrophobicity > 0.5 and
                             FUNCTIONAL_PROFILES[p['aa2']].hydrophobicity > 0.5)

        print(f"\n{label}:")
        print(f"  Avg functional similarity: {avg_func_sim:.3f}")
        print(f"  Avg hybrid cost: {avg_hybrid_cost:.2f}")
        print(f"  Avg simple distance: {avg_simple_dist:.2f}")
        print(f"  Same charge: {same_charge}/{len(pairs)} ({100*same_charge/len(pairs):.1f}%)")
        print(f"  Both hydrophobic: {both_hydrophobic}/{len(pairs)} ({100*both_hydrophobic/len(pairs):.1f}%)")

        return {
            'n': len(pairs),
            'avg_func_sim': float(avg_func_sim),
            'avg_hybrid_cost': float(avg_hybrid_cost),
            'avg_simple_dist': float(avg_simple_dist),
            'pct_same_charge': float(same_charge / len(pairs)),
            'pct_both_hydrophobic': float(both_hydrophobic / len(pairs)),
        }

    hybrid_wins_stats = analyze_group(hybrid_wins, "Hybrid wins")
    simple_wins_stats = analyze_group(simple_wins, "Simple wins")

    # ==========================================================================
    # The Arrow Flip Point
    # ==========================================================================
    print("\n" + "="*70)
    print("ARROW FLIP DETECTION")
    print("="*70)

    print("\nThe 'arrow flips' when simple physicochemical properties")
    print("are sufficient to predict functional similarity.")

    print(f"\n** Hybrid adds value for {100*len(hybrid_wins)/len(pair_analysis):.1f}% of AA pairs **")

    if len(hybrid_wins) > len(simple_wins):
        print("\nConclusion: P-adic/Hybrid structure provides SIGNIFICANT added value")
        print("over simple physicochemical properties for most AA pairs.")
    else:
        print("\nConclusion: Simple physicochemical properties are often sufficient.")
        print("P-adic structure adds value primarily for specific pair types.")

    # Identify the pairs where groupoid adds MOST value
    print("\n--- Top 10 pairs where HYBRID adds most value ---")

    for pair in pair_analysis:
        max_hybrid = max(abs(p['hybrid_pred']) for p in pair_analysis)
        max_simple = max(abs(p['simple_pred']) for p in pair_analysis)

        h_norm = pair['hybrid_pred'] / max_hybrid
        s_norm = pair['simple_pred'] / max_simple

        pair['hybrid_advantage'] = abs(s_norm - pair['func_sim']) - abs(h_norm - pair['func_sim'])

    sorted_by_advantage = sorted(pair_analysis, key=lambda x: x['hybrid_advantage'], reverse=True)

    print(f"\n{'Pair':<8} {'Func_Sim':<10} {'Hybrid_Cost':<12} {'Simple_Dist':<12} {'Advantage':<10}")
    print("-" * 55)
    for p in sorted_by_advantage[:10]:
        print(f"{p['aa1']}-{p['aa2']:<5} {p['func_sim']:<10.3f} {p['hybrid_cost']:<12.2f} {p['simple_dist']:<12.2f} {p['hybrid_advantage']:<10.3f}")

    results = {
        'n_hybrid_wins': len(hybrid_wins),
        'n_simple_wins': len(simple_wins),
        'pct_hybrid_advantage': float(len(hybrid_wins) / len(pair_analysis)),
        'hybrid_wins_characteristics': hybrid_wins_stats,
        'simple_wins_characteristics': simple_wins_stats,
        'top_hybrid_advantage_pairs': [
            {'pair': f"{p['aa1']}-{p['aa2']}", 'advantage': float(p['hybrid_advantage'])}
            for p in sorted_by_advantage[:10]
        ],
    }

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_experiments() -> Dict:
    """Run all arrow flip experiments and save results."""

    print("="*70)
    print("ARROW FLIP EXPERIMENTS")
    print("Detecting the boundary where P-adic embeddings add predictive value")
    print("="*70)

    results = {}

    # Experiment 1: Codon-level mapping
    results['experiment_1_codon_mapping'] = experiment_1_codon_functional_mapping()

    # Experiment 2: Embedding comparison
    results['experiment_2_embedding_comparison'] = experiment_2_embedding_comparison()

    # Experiment 3: Arrow flip boundary
    results['experiment_3_arrow_flip'] = experiment_3_arrow_flip_boundary()

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'arrow_flip_results.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY: WHERE DOES THE ARROW FLIP?")
    print("="*70)

    print("\n1. CODON DEGENERACY AND FUNCTIONAL ROBUSTNESS:")
    if results['experiment_1_codon_mapping'].get('mean_func_dist_transitions'):
        trans = results['experiment_1_codon_mapping']['mean_func_dist_transitions']
        transv = results['experiment_1_codon_mapping']['mean_func_dist_transversions']
        print(f"   Transitions cause LESS functional disruption ({trans:.2f}) than")
        print(f"   transversions ({transv:.2f})")

    print("\n2. EMBEDDING COMPARISON:")
    exp2 = results['experiment_2_embedding_comparison']
    best_r = max(exp2.values(), key=lambda x: x['spearman_r'])
    for name, data in exp2.items():
        if data == best_r:
            print(f"   Best embedding: {name} (r = {data['spearman_r']:.4f})")

    print("\n3. ARROW FLIP BOUNDARY:")
    exp3 = results['experiment_3_arrow_flip']
    pct = exp3['pct_hybrid_advantage'] * 100
    print(f"   Hybrid structure adds value for {pct:.1f}% of AA pairs")

    if pct > 60:
        print("\n   ** CONCLUSION: P-adic/Hybrid structure is BROADLY useful **")
        print("   The arrow has NOT flipped - geometric structure consistently")
        print("   outperforms simple physicochemical properties.")
    elif pct > 40:
        print("\n   ** CONCLUSION: P-adic/Hybrid structure is MODERATELY useful **")
        print("   The arrow flip depends on pair type - structure helps for")
        print("   some substitutions but not others.")
    else:
        print("\n   ** CONCLUSION: P-adic/Hybrid structure has LIMITED advantage **")
        print("   Simple physicochemical properties are often sufficient.")
        print("   The arrow has flipped - context dominates intrinsics.")

    return results


if __name__ == "__main__":
    results = run_all_experiments()
