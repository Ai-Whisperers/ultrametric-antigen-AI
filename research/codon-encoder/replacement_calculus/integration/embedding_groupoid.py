#!/usr/bin/env python3
"""Embedding-Based Groupoid Builder.

This version uses embedding distance as the morphism validity criterion
instead of p-adic valuation. This tests whether the LEARNED embeddings
capture biological relationships better than the p-adic structure.

Key difference from vae_groupoid_builder.py:
- Morphism validity: based on embedding distance, not valuation
- Cost function: uses hyperbolic distance, not invariant deltas
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import Dict, List, Tuple, Optional
import numpy as np

from replacement_calculus.invariants import InvariantTuple, valuation
from replacement_calculus.groups import LocalMinimum
from replacement_calculus.morphisms import Morphism, MorphismType
from replacement_calculus.groupoids import Groupoid, find_escape_path, analyze_groupoid_structure


# =============================================================================
# Genetic Code Constants
# =============================================================================

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
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in CODON_TABLE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

def codon_to_index(codon: str) -> int:
    nucleotides = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
    return nucleotides[codon[0]] * 16 + nucleotides[codon[1]] * 4 + nucleotides[codon[2]]

CODON_TO_INDEX = {c: codon_to_index(c) for c in CODON_TABLE.keys()}


# =============================================================================
# BLOSUM62 Substitution Matrix (Simplified)
# =============================================================================

# Positive = conservative (allowed), Negative = radical (disallowed)
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

def get_blosum_score(aa1: str, aa2: str) -> int:
    """Get BLOSUM62 score for amino acid pair."""
    if aa1 == aa2:
        return BLOSUM62.get((aa1, aa2), 0)
    return BLOSUM62.get((aa1, aa2), BLOSUM62.get((aa2, aa1), -4))


# =============================================================================
# Embedding-Based Morphism Validity
# =============================================================================

def embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Euclidean distance between embeddings."""
    return np.linalg.norm(emb1 - emb2)


def is_valid_embedding_morphism(
    source: LocalMinimum,
    target: LocalMinimum,
    max_distance: float = 2.0,
) -> Tuple[bool, str, float]:
    """Check if morphism is valid based on embedding distance.

    A morphism is valid if the centroids are within max_distance.

    Returns:
        (is_valid, reason, distance)
    """
    if source.center is None or target.center is None:
        return False, "Missing center", float('inf')

    dist = embedding_distance(source.center, target.center)

    if dist <= max_distance:
        return True, f"Distance {dist:.3f} <= {max_distance}", dist
    else:
        return False, f"Distance {dist:.3f} > {max_distance}", dist


# =============================================================================
# Groupoid Construction
# =============================================================================

def load_codon_embeddings() -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load codon embeddings from JSON."""
    json_path = Path(__file__).parent.parent.parent / 'extraction' / 'results' / 'codon_embeddings_v5_12_3.json'

    if not json_path.exists():
        raise FileNotFoundError(f"Codon embeddings not found at {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    embeddings = {}
    for codon, info in data['codons'].items():
        embeddings[codon] = np.array(info['embedding'])

    return embeddings, data.get('metadata', {})


def build_embedding_groupoid(
    embeddings: Dict[str, np.ndarray],
    max_distance: float = 2.0,
) -> Groupoid:
    """Build groupoid with embedding-based morphism validity.

    Args:
        embeddings: Dictionary mapping codon -> embedding vector
        max_distance: Maximum centroid distance for valid morphism

    Returns:
        Groupoid with amino acids as objects
    """
    groupoid = Groupoid(name="embedding_groupoid")
    aa_to_idx: Dict[str, int] = {}

    # Create LocalMinima for each amino acid
    for aa, codons in AA_TO_CODONS.items():
        if aa == '*':
            continue

        # Get member embeddings
        members = [embeddings[c] for c in codons if c in embeddings]
        if not members:
            continue

        generators = [CODON_TO_INDEX[c] for c in codons if c in embeddings]
        center = np.mean(members, axis=0)

        minimum = LocalMinimum(
            name=f"AA_{aa}",
            generators=generators,
            relations=[],
            center=center,
            members=members,
            metadata={'amino_acid': aa, 'codons': codons, 'degeneracy': len(codons)},
        )

        idx = groupoid.add_object(minimum)
        aa_to_idx[aa] = idx

    # Find valid morphisms based on embedding distance
    n = groupoid.n_objects()

    for i in range(n):
        source = groupoid.objects[i]
        for j in range(n):
            if i == j:
                continue

            target = groupoid.objects[j]

            is_valid, reason, dist = is_valid_embedding_morphism(
                source, target, max_distance
            )

            if is_valid:
                # Create morphism with distance as cost
                morphism = Morphism(
                    source=source,
                    target=target,
                    map_function=lambda x: x,  # Simple identity
                    morphism_type=MorphismType.HOMOMORPHISM,
                    cost=dist,
                )
                groupoid.morphisms[(i, j)].append(morphism)

    return groupoid


def validate_with_blosum(groupoid: Groupoid) -> Dict:
    """Validate groupoid structure against BLOSUM62.

    Hypothesis: Valid morphisms (close embeddings) should have positive BLOSUM scores.
    """
    results = {
        'true_positives': 0,   # Morphism exists AND BLOSUM positive
        'false_positives': 0,  # Morphism exists BUT BLOSUM negative
        'true_negatives': 0,   # No morphism AND BLOSUM negative
        'false_negatives': 0,  # No morphism BUT BLOSUM positive
        'pairs': [],
    }

    n = groupoid.n_objects()

    for i in range(n):
        aa1 = groupoid.objects[i].metadata.get('amino_acid')
        if not aa1:
            continue

        for j in range(n):
            if i == j:
                continue

            aa2 = groupoid.objects[j].metadata.get('amino_acid')
            if not aa2:
                continue

            # Check if morphism exists
            has_morphism = groupoid.has_morphism(i, j)

            # Get BLOSUM score
            blosum = get_blosum_score(aa1, aa2)
            is_conservative = blosum >= 0

            pair_info = {
                'aa1': aa1,
                'aa2': aa2,
                'morphism_exists': has_morphism,
                'blosum_score': blosum,
                'is_conservative': is_conservative,
            }

            if has_morphism and is_conservative:
                results['true_positives'] += 1
                pair_info['classification'] = 'TP'
            elif has_morphism and not is_conservative:
                results['false_positives'] += 1
                pair_info['classification'] = 'FP'
            elif not has_morphism and not is_conservative:
                results['true_negatives'] += 1
                pair_info['classification'] = 'TN'
            else:  # not has_morphism and is_conservative
                results['false_negatives'] += 1
                pair_info['classification'] = 'FN'

            results['pairs'].append(pair_info)

    # Compute metrics
    tp, fp, tn, fn = (results['true_positives'], results['false_positives'],
                      results['true_negatives'], results['false_negatives'])

    results['accuracy'] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return results


def find_optimal_threshold(embeddings: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
    """Find optimal distance threshold by maximizing F1 against BLOSUM62."""
    best_f1 = 0
    best_threshold = 2.0
    best_results = None

    for threshold in np.arange(0.5, 5.0, 0.25):
        groupoid = build_embedding_groupoid(embeddings, max_distance=threshold)
        results = validate_with_blosum(groupoid)

        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_threshold = threshold
            best_results = results

    return best_threshold, best_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EMBEDDING-BASED GROUPOID ANALYSIS")
    print("=" * 60)

    # Load embeddings
    print("\n1. Loading codon embeddings...")
    try:
        embeddings, metadata = load_codon_embeddings()
        print(f"   Loaded {len(embeddings)} codon embeddings")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return

    # Find optimal threshold
    print("\n2. Finding optimal distance threshold...")
    best_threshold, best_results = find_optimal_threshold(embeddings)
    print(f"   Optimal threshold: {best_threshold:.2f}")
    print(f"   Best F1 score: {best_results['f1']:.4f}")

    # Build groupoid with optimal threshold
    print(f"\n3. Building groupoid with threshold={best_threshold:.2f}...")
    groupoid = build_embedding_groupoid(embeddings, max_distance=best_threshold)

    analysis = analyze_groupoid_structure(groupoid)
    print(f"   Objects: {analysis['n_objects']}")
    print(f"   Morphisms: {analysis['n_morphisms']}")
    print(f"   Connected: {analysis['is_connected']}")

    # Validate against BLOSUM62
    print("\n4. Validation against BLOSUM62:")
    print(f"   True Positives: {best_results['true_positives']}")
    print(f"   False Positives: {best_results['false_positives']}")
    print(f"   True Negatives: {best_results['true_negatives']}")
    print(f"   False Negatives: {best_results['false_negatives']}")
    print(f"   Accuracy: {best_results['accuracy']:.2%}")
    print(f"   Precision: {best_results['precision']:.2%}")
    print(f"   Recall: {best_results['recall']:.2%}")
    print(f"   F1 Score: {best_results['f1']:.4f}")

    # Test escape paths with optimal groupoid
    print("\n5. Testing escape paths:")
    test_pairs = [
        ('L', 'I'), ('L', 'M'), ('K', 'R'), ('D', 'E'), ('L', 'D'),
        ('F', 'Y'), ('S', 'T'), ('V', 'A'),
    ]

    for aa1, aa2 in test_pairs:
        idx1 = groupoid._object_index.get(f"AA_{aa1}")
        idx2 = groupoid._object_index.get(f"AA_{aa2}")

        if idx1 is None or idx2 is None:
            continue

        path = find_escape_path(groupoid, idx1, idx2)
        blosum = get_blosum_score(aa1, aa2)

        if path:
            total_cost = sum(m.cost for m in path)
            print(f"   {aa1} → {aa2}: PATH (cost={total_cost:.2f}, BLOSUM={blosum})")
        else:
            print(f"   {aa1} → {aa2}: NO PATH (BLOSUM={blosum})")

    # Save results
    output_path = Path(__file__).parent / 'embedding_groupoid_analysis.json'
    output_data = {
        'optimal_threshold': float(best_threshold),
        'groupoid_analysis': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in analysis.items()},
        'validation': {k: v for k, v in best_results.items() if k != 'pairs'},
        'n_pairs_analyzed': len(best_results['pairs']),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n6. Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
