#!/usr/bin/env python3
"""VAE → Groupoid Integration.

This script bridges the TernaryVAE embeddings to the Replacement Calculus framework:
1. Load trained VAE model
2. Extract codon embeddings
3. Group by amino acid (natural LocalMinima)
4. Build Groupoid with valid morphisms
5. Analyze escape paths

Usage:
    python vae_groupoid_builder.py --checkpoint path/to/best.pt
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from typing import Dict, List, Tuple
import numpy as np
import torch

from replacement_calculus.invariants import InvariantTuple, valuation
from replacement_calculus.groups import LocalMinimum, Constraint, create_codon_local_minimum
from replacement_calculus.morphisms import (
    Morphism, MorphismType, is_valid_morphism, compute_morphism_cost
)
from replacement_calculus.groupoids import (
    Groupoid, find_escape_path, analyze_groupoid_structure
)


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

# Amino acid to codons mapping
AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in CODON_TABLE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

# Codon to index (ternary encoding)
NUCLEOTIDE_TO_TERNARY = {'T': 0, 'C': 1, 'A': 2, 'G': 2}  # Simplified

def codon_to_index(codon: str) -> int:
    """Convert codon to integer index (0-63)."""
    nucleotides = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
    return nucleotides[codon[0]] * 16 + nucleotides[codon[1]] * 4 + nucleotides[codon[2]]

CODON_TO_INDEX = {c: codon_to_index(c) for c in CODON_TABLE.keys()}
INDEX_TO_CODON = {v: k for k, v in CODON_TO_INDEX.items()}


# =============================================================================
# VAE Loading
# =============================================================================

def load_vae_embeddings(checkpoint_path: str) -> Tuple[np.ndarray, Dict]:
    """Load VAE codon embeddings.

    Tries to load from pre-extracted JSON first, then falls back to synthetic.

    Returns:
        embeddings: (64, latent_dim) array of codon embeddings
        metadata: checkpoint metadata
    """
    import json

    # Try loading pre-extracted codon embeddings
    json_paths = [
        Path(__file__).parent.parent.parent / 'extraction' / 'results' / 'codon_embeddings_v5_12_3.json',
        Path(__file__).parent.parent.parent.parent.parent / 'research' / 'codon-encoder' / 'extraction' / 'results' / 'codon_embeddings_v5_12_3.json',
    ]

    for json_path in json_paths:
        if json_path.exists():
            print(f"   Loading pre-extracted embeddings from: {json_path.name}")
            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Extract embeddings in codon order
                embeddings = []
                for codon in sorted(CODON_TABLE.keys()):
                    if codon in data['codons']:
                        emb = np.array(data['codons'][codon]['embedding'])
                        embeddings.append(emb)
                    else:
                        print(f"   Warning: Missing codon {codon}")
                        embeddings.append(np.zeros(16))

                embeddings = np.array(embeddings)

                metadata = {
                    'checkpoint': data.get('metadata', {}).get('checkpoint', 'v5_12_3'),
                    'latent_dim': embeddings.shape[1] if len(embeddings) > 0 else 16,
                    'n_codons': len(embeddings),
                    'source': str(json_path),
                }

                return embeddings, metadata

            except Exception as e:
                print(f"   Warning: Failed to load JSON ({e})")

    print("   No pre-extracted embeddings found, using synthetic embeddings...")

    # Generate synthetic embeddings where radius ~ 1/valuation
    latent_dim = 16
    embeddings = []

    for codon in sorted(CODON_TABLE.keys()):
        idx = CODON_TO_INDEX[codon]
        v = valuation(idx, 3)

        # Radius should be ~0.9 for v=0, ~0.1 for v=9
        target_radius = 0.85 - v * 0.07

        # Random direction, fixed radius
        direction = np.random.randn(latent_dim)
        direction = direction / np.linalg.norm(direction)
        emb = direction * target_radius
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    metadata = {
        'checkpoint': 'synthetic',
        'latent_dim': latent_dim,
        'n_codons': len(embeddings),
    }

    return embeddings, metadata


# =============================================================================
# Groupoid Construction
# =============================================================================

def build_amino_acid_groupoid(
    embeddings: np.ndarray,
    p: int = 3,
) -> Groupoid:
    """Build a Groupoid where each amino acid is a LocalMinimum.

    Args:
        embeddings: (64, latent_dim) codon embeddings
        p: Prime for valuation

    Returns:
        Groupoid with amino acids as objects and valid morphisms
    """
    groupoid = Groupoid(name="amino_acid_groupoid")

    # Create LocalMinima for each amino acid
    aa_to_idx: Dict[str, int] = {}

    for aa, codons in AA_TO_CODONS.items():
        if aa == '*':  # Skip stop codons
            continue

        # Get generators (codon indices)
        generators = [CODON_TO_INDEX[c] for c in codons]

        # Get member embeddings
        members = [embeddings[CODON_TO_INDEX[c]] for c in codons]

        # Create local minimum
        minimum = LocalMinimum(
            name=f"AA_{aa}",
            generators=generators,
            relations=[],  # Simple case: no explicit constraints
            center=np.mean(members, axis=0) if members else None,
            members=members,
            metadata={
                'amino_acid': aa,
                'codons': codons,
                'degeneracy': len(codons),
            }
        )

        idx = groupoid.add_object(minimum)
        aa_to_idx[aa] = idx

    # Find valid morphisms between amino acids
    # A morphism is valid if valuation doesn't decrease
    n_objects = groupoid.n_objects()

    for i in range(n_objects):
        source = groupoid.objects[i]
        I_source = source.invariant_tuple(p)

        for j in range(n_objects):
            if i == j:
                continue

            target = groupoid.objects[j]
            I_target = target.invariant_tuple(p)

            # Check if morphism could be valid (target dominates source)
            if I_target >= I_source:
                # Create a simple morphism (identity on overlapping generators)
                def make_map(src=source, tgt=target):
                    # Map each source generator to closest target generator
                    src_embs = {g: src.members[k] for k, g in enumerate(src.generators)}
                    tgt_embs = {g: tgt.members[k] for k, g in enumerate(tgt.generators)}

                    mapping = {}
                    for sg, se in src_embs.items():
                        min_dist = float('inf')
                        closest = tgt.generators[0]
                        for tg, te in tgt_embs.items():
                            d = np.linalg.norm(se - te)
                            if d < min_dist:
                                min_dist = d
                                closest = tg
                        mapping[sg] = closest

                    return lambda x, m=mapping: m.get(x, x)

                morphism = Morphism(
                    source=source,
                    target=target,
                    map_function=make_map(),
                    morphism_type=MorphismType.HOMOMORPHISM,
                )

                # Validate and compute cost
                is_valid, reason = is_valid_morphism(morphism, p)
                if is_valid:
                    morphism.cost = compute_morphism_cost(morphism, p)
                    groupoid.add_morphism(i, j, morphism)

    return groupoid


def analyze_escape_paths(
    groupoid: Groupoid,
    source_aa: str,
    target_aa: str,
) -> Dict:
    """Analyze escape path between two amino acids.

    Args:
        groupoid: The amino acid groupoid
        source_aa: Source amino acid (e.g., 'L' for Leucine)
        target_aa: Target amino acid (e.g., 'M' for Methionine)

    Returns:
        Analysis dict with path info
    """
    source_name = f"AA_{source_aa}"
    target_name = f"AA_{target_aa}"

    # Find indices
    source_idx = groupoid._object_index.get(source_name)
    target_idx = groupoid._object_index.get(target_name)

    if source_idx is None or target_idx is None:
        return {'error': f"Amino acid not found: {source_aa} or {target_aa}"}

    # Find escape path
    path = find_escape_path(groupoid, source_idx, target_idx)

    if path is None:
        return {
            'source': source_aa,
            'target': target_aa,
            'path_exists': False,
            'reason': 'No valid morphism chain exists',
        }

    # Analyze path
    path_info = []
    total_cost = 0.0

    for morphism in path:
        path_info.append({
            'from': morphism.source.metadata.get('amino_acid', morphism.source.name),
            'to': morphism.target.metadata.get('amino_acid', morphism.target.name),
            'cost': morphism.cost,
            'type': morphism.morphism_type.value,
        })
        total_cost += morphism.cost

    return {
        'source': source_aa,
        'target': target_aa,
        'path_exists': True,
        'path_length': len(path),
        'total_cost': total_cost,
        'path': path_info,
    }


# =============================================================================
# Gene Ontology Validation
# =============================================================================

# Amino acid property groups (simplified GO-like categories)
AA_PROPERTY_GROUPS = {
    'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'],
    'polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
    'charged_positive': ['K', 'R', 'H'],
    'charged_negative': ['D', 'E'],
    'special': ['G', 'P'],
}

# Known substitution matrices (simplified BLOSUM-like)
SUBSTITUTION_SCORES = {
    ('L', 'I'): 2, ('L', 'V'): 1, ('L', 'M'): 2,  # Similar
    ('K', 'R'): 2, ('D', 'E'): 2,                  # Same charge
    ('L', 'D'): -3, ('K', 'E'): -3,                # Opposite properties
}

def validate_with_substitution_data(
    groupoid: Groupoid,
) -> Dict:
    """Validate groupoid structure against known substitution patterns.

    Hypothesis: Valid morphisms should correspond to conservative substitutions.
    """
    results = {
        'conservative_matches': 0,
        'conservative_misses': 0,
        'radical_prevented': 0,
        'radical_allowed': 0,
    }

    # Test all amino acid pairs
    for aa1 in AA_TO_CODONS.keys():
        if aa1 == '*':
            continue
        for aa2 in AA_TO_CODONS.keys():
            if aa2 == '*' or aa1 == aa2:
                continue

            # Check if morphism exists
            path_info = analyze_escape_paths(groupoid, aa1, aa2)
            morphism_exists = path_info.get('path_exists', False)

            # Get substitution score
            score = SUBSTITUTION_SCORES.get((aa1, aa2), 0)
            if score == 0:
                score = SUBSTITUTION_SCORES.get((aa2, aa1), 0)

            # Conservative = positive score, Radical = negative score
            if score > 0:  # Conservative substitution
                if morphism_exists:
                    results['conservative_matches'] += 1
                else:
                    results['conservative_misses'] += 1
            elif score < 0:  # Radical substitution
                if not morphism_exists:
                    results['radical_prevented'] += 1
                else:
                    results['radical_allowed'] += 1

    # Compute accuracy
    total_tests = sum(results.values())
    if total_tests > 0:
        correct = results['conservative_matches'] + results['radical_prevented']
        results['accuracy'] = correct / total_tests
    else:
        results['accuracy'] = 0.0

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build Groupoid from VAE embeddings')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to VAE checkpoint')
    parser.add_argument('--output', type=str, default='groupoid_analysis.json',
                        help='Output file for analysis')
    args = parser.parse_args()

    print("=" * 60)
    print("REPLACEMENT CALCULUS: VAE → GROUPOID INTEGRATION")
    print("=" * 60)

    # Load embeddings
    print("\n1. Loading embeddings...")
    if args.checkpoint:
        embeddings, metadata = load_vae_embeddings(args.checkpoint)
    else:
        print("   No checkpoint provided, using synthetic embeddings")
        embeddings, metadata = load_vae_embeddings(None)

    print(f"   Loaded {metadata['n_codons']} codon embeddings")
    print(f"   Latent dimension: {metadata['latent_dim']}")

    # Build groupoid
    print("\n2. Building amino acid groupoid...")
    groupoid = build_amino_acid_groupoid(embeddings)

    analysis = analyze_groupoid_structure(groupoid)
    print(f"   Objects (amino acids): {analysis['n_objects']}")
    print(f"   Morphisms: {analysis['n_morphisms']}")
    print(f"   Connected: {analysis['is_connected']}")
    print(f"   Maximal elements: {[groupoid.objects[i].metadata.get('amino_acid', '?') for i in analysis['maximal_indices']]}")
    print(f"   Minimal elements: {[groupoid.objects[i].metadata.get('amino_acid', '?') for i in analysis['minimal_indices']]}")

    # Test escape paths
    print("\n3. Testing escape paths...")
    test_pairs = [
        ('L', 'I'),  # Similar (Leucine → Isoleucine)
        ('L', 'M'),  # Similar (Leucine → Methionine)
        ('K', 'R'),  # Both positive (Lysine → Arginine)
        ('D', 'E'),  # Both negative (Aspartate → Glutamate)
        ('L', 'D'),  # Very different (Leucine → Aspartate)
    ]

    escape_results = []
    for source, target in test_pairs:
        result = analyze_escape_paths(groupoid, source, target)
        escape_results.append(result)

        if result.get('path_exists'):
            print(f"   {source} → {target}: PATH EXISTS (cost={result['total_cost']:.2f}, length={result['path_length']})")
        else:
            print(f"   {source} → {target}: NO PATH")

    # Validate with substitution data
    print("\n4. Validating against substitution data...")
    validation = validate_with_substitution_data(groupoid)
    print(f"   Conservative matches: {validation['conservative_matches']}")
    print(f"   Conservative misses: {validation['conservative_misses']}")
    print(f"   Radical prevented: {validation['radical_prevented']}")
    print(f"   Radical allowed: {validation['radical_allowed']}")
    print(f"   Accuracy: {validation['accuracy']:.2%}")

    # Save results
    output_data = {
        'metadata': metadata,
        'groupoid_analysis': {k: v if not isinstance(v, np.floating) else float(v)
                             for k, v in analysis.items()},
        'escape_paths': escape_results,
        'substitution_validation': validation,
    }

    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n5. Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
