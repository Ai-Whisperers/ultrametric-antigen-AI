#!/usr/bin/env python3
"""
Minimum Viable Experiment: Does p-adic geometry encode contact preferences?

Hypothesis: Hydrophobic residues contact each other in protein cores.
If p-adic pairwise distances reflect this, hydrophobic-hydrophobic pairs
should have systematically different distances than hydrophobic-hydrophilic pairs.

This takes <1 minute and requires NO external data.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from itertools import combinations

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance

# Amino acid classifications
HYDROPHOBIC = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'}  # Core-preferring
HYDROPHILIC = {'R', 'K', 'D', 'E', 'N', 'Q', 'H'}       # Surface-preferring
NEUTRAL = {'G', 'S', 'T', 'C', 'Y'}                      # Context-dependent

# Genetic code
CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
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


def load_embeddings():
    """Load pre-extracted embeddings."""
    emb_path = Path(__file__).parent.parent / 'embeddings' / 'v5_11_3_embeddings.pt'
    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'

    import json
    with open(map_path) as f:
        mapping = json.load(f)

    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    z_hyp = emb_data['z_B_hyp']  # VAE-B for hierarchy

    return z_hyp, mapping['codon_to_position']


def classify_aa(aa):
    """Classify amino acid by hydrophobicity."""
    if aa in HYDROPHOBIC:
        return 'hydrophobic'
    elif aa in HYDROPHILIC:
        return 'hydrophilic'
    else:
        return 'neutral'


def main():
    print("=" * 70)
    print("SIGNAL VALIDATION: P-adic Geometry → Contact Preferences")
    print("=" * 70)
    print()

    # Load embeddings
    print("Loading embeddings...")
    z_hyp, codon_to_pos = load_embeddings()
    print(f"  Loaded {len(z_hyp)} embeddings, {len(codon_to_pos)} codons mapped")

    # Build AA → mean embedding mapping
    aa_embeddings = {}
    for codon, pos in codon_to_pos.items():
        aa = CODON_TO_AA.get(codon)
        if aa:
            if aa not in aa_embeddings:
                aa_embeddings[aa] = []
            aa_embeddings[aa].append(z_hyp[pos])

    # Average over synonymous codons
    for aa in aa_embeddings:
        aa_embeddings[aa] = torch.stack(aa_embeddings[aa]).mean(dim=0)

    print(f"  Computed mean embeddings for {len(aa_embeddings)} amino acids")

    # Compute ALL pairwise distances
    print()
    print("Computing pairwise hyperbolic distances...")

    aa_list = sorted(aa_embeddings.keys())
    n = len(aa_list)

    dist_matrix = np.zeros((n, n))
    for i, aa1 in enumerate(aa_list):
        for j, aa2 in enumerate(aa_list):
            if i < j:
                d = poincare_distance(
                    aa_embeddings[aa1].unsqueeze(0),
                    aa_embeddings[aa2].unsqueeze(0),
                    c=1.0
                ).item()
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    # Group distances by interaction type
    distances_by_type = {
        'hydrophobic-hydrophobic': [],
        'hydrophilic-hydrophilic': [],
        'hydrophobic-hydrophilic': [],
        'neutral-any': [],
    }

    for i, aa1 in enumerate(aa_list):
        for j, aa2 in enumerate(aa_list):
            if i < j:
                c1, c2 = classify_aa(aa1), classify_aa(aa2)
                d = dist_matrix[i, j]

                if c1 == 'hydrophobic' and c2 == 'hydrophobic':
                    distances_by_type['hydrophobic-hydrophobic'].append(d)
                elif c1 == 'hydrophilic' and c2 == 'hydrophilic':
                    distances_by_type['hydrophilic-hydrophilic'].append(d)
                elif (c1 == 'hydrophobic' and c2 == 'hydrophilic') or \
                     (c1 == 'hydrophilic' and c2 == 'hydrophobic'):
                    distances_by_type['hydrophobic-hydrophilic'].append(d)
                else:
                    distances_by_type['neutral-any'].append(d)

    # Report statistics
    print()
    print("=" * 70)
    print("RESULTS: Pairwise Distance Distribution by Interaction Type")
    print("=" * 70)
    print()
    print(f"{'Interaction Type':<30} {'N':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for itype, dists in distances_by_type.items():
        if dists:
            arr = np.array(dists)
            print(f"{itype:<30} {len(arr):>5} {arr.mean():>8.4f} {arr.std():>8.4f} "
                  f"{arr.min():>8.4f} {arr.max():>8.4f}")

    # Statistical tests
    print()
    print("=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    hh = np.array(distances_by_type['hydrophobic-hydrophobic'])
    hl = np.array(distances_by_type['hydrophobic-hydrophilic'])
    ll = np.array(distances_by_type['hydrophilic-hydrophilic'])

    # Key test: Are hydrophobic-hydrophobic distances different from cross-type?
    t_stat, p_value = stats.ttest_ind(hh, hl)
    print()
    print(f"Test 1: Hydrophobic-Hydrophobic vs Hydrophobic-Hydrophilic")
    print(f"  H0: No difference in mean pairwise distance")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Mean HH: {hh.mean():.4f}, Mean HL: {hl.mean():.4f}")
    print(f"  Difference: {hh.mean() - hl.mean():+.4f}")

    if p_value < 0.05:
        direction = "CLOSER" if hh.mean() < hl.mean() else "FARTHER"
        print(f"  >>> SIGNIFICANT! Hydrophobic pairs are {direction} in p-adic space")
    else:
        print(f"  >>> Not significant at α=0.05")

    # Test 2: Hydrophilic-Hydrophilic vs cross-type
    t_stat2, p_value2 = stats.ttest_ind(ll, hl)
    print()
    print(f"Test 2: Hydrophilic-Hydrophilic vs Hydrophobic-Hydrophilic")
    print(f"  t-statistic: {t_stat2:.4f}")
    print(f"  p-value: {p_value2:.2e}")
    print(f"  Mean LL: {ll.mean():.4f}, Mean HL: {hl.mean():.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((hh.std()**2 + hl.std()**2) / 2)
    cohens_d = (hh.mean() - hl.mean()) / pooled_std
    print()
    print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) > 0.8:
        print("  >>> LARGE effect size")
    elif abs(cohens_d) > 0.5:
        print("  >>> MEDIUM effect size")
    elif abs(cohens_d) > 0.2:
        print("  >>> SMALL effect size")
    else:
        print("  >>> Negligible effect size")

    # Conclusion
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if p_value < 0.05 and hh.mean() < hl.mean():
        print("""
    SIGNAL DETECTED!

    Hydrophobic amino acid pairs are CLOSER in p-adic hyperbolic space
    than hydrophobic-hydrophilic pairs. This means:

    1. P-adic geometry encodes interaction preferences
    2. Pairwise distances could predict hydrophobic core contacts
    3. Contact prediction from codon embeddings is FEASIBLE

    NEXT STEP: Test on real protein contact maps.
        """)
    elif p_value < 0.05 and hh.mean() > hl.mean():
        print("""
    UNEXPECTED SIGNAL!

    Hydrophobic pairs are FARTHER apart in p-adic space.
    This is opposite to contact prediction hypothesis but still
    shows p-adic geometry encodes interaction type information.

    NEXT STEP: Investigate if this inverse relationship is useful.
        """)
    else:
        print("""
    NO SIGNIFICANT SIGNAL

    P-adic pairwise distances do not discriminate interaction types
    at the amino acid level. Possible reasons:

    1. Signal may exist at codon level (synonymous codons differ)
    2. Need to consider sequence context, not just AA pairs
    3. Contact prediction may require different features

    NEXT STEP: Test codon-level (not AA-averaged) distances.
        """)

    return distances_by_type


if __name__ == '__main__':
    distances = main()
