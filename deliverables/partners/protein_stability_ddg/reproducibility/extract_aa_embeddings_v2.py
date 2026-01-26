#!/usr/bin/env python3
"""Extract Amino Acid Embeddings - V2 (Proper Codon Mapping)

This script extracts amino acid embeddings using PROPER codon-to-index mapping
from the genetic code, aggregating over synonymous codons.

Key improvements over V1:
1. Uses src.biology.codons for proper codon→index mapping (0-63)
2. Aggregates codon embeddings per amino acid (1-6 synonymous codons)
3. Uses CodonEncoder p-adic distances as ground truth

Usage:
    python extract_aa_embeddings_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.biology.codons import (
    GENETIC_CODE,
    CODON_TO_INDEX,
    AMINO_ACID_TO_CODONS,
    codon_index_to_triplet,
)
from src.encoders.codon_encoder import (
    compute_padic_distance_between_codons,
    compute_amino_acid_distance,
    AA_PROPERTIES,
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def poincare_distance_np(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance in the Poincaré ball."""
    sqrt_c = c ** 0.5
    diff = u - v
    diff_norm_sq = np.sum(diff * diff)
    u_norm_sq = np.sum(u * u)
    v_norm_sq = np.sum(v * v)
    denominator = (1 - c * u_norm_sq) * (1 - c * v_norm_sq)
    denominator = max(denominator, 1e-10)
    x = 1 + 2 * c * diff_norm_sq / denominator
    x = max(x, 1.0 + 1e-10)
    return (1 / sqrt_c) * np.arccosh(x)


def compute_aa_padic_distance(aa1: str, aa2: str) -> float:
    """Compute p-adic distance between amino acids.

    Uses minimum p-adic distance over all codon pairs encoding these AAs.
    This captures the "closest" evolutionary relationship.
    """
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0  # Maximum distance

    min_dist = 1.0
    for c1 in codons1:
        idx1 = CODON_TO_INDEX[c1]
        for c2 in codons2:
            idx2 = CODON_TO_INDEX[c2]
            dist = compute_padic_distance_between_codons(idx1, idx2)
            min_dist = min(min_dist, dist)

    return min_dist


def compute_aa_property_distance(aa1: str, aa2: str) -> float:
    """Compute property-based distance between amino acids."""
    props1 = np.array(AA_PROPERTIES.get(aa1, (0, 0, 0, 0)))
    props2 = np.array(AA_PROPERTIES.get(aa2, (0, 0, 0, 0)))
    return float(np.linalg.norm(props1 - props2))


def main():
    parser = argparse.ArgumentParser(
        description="Extract AA embeddings with proper codon mapping (V2)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/aa_embeddings_v2.json",
        help="Output path for embeddings"
    )
    parser.add_argument(
        "--padic-weight", type=float, default=0.5,
        help="Weight for p-adic distance component"
    )
    parser.add_argument(
        "--property-weight", type=float, default=0.5,
        help="Weight for property distance component"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extracting AA Embeddings V2 (Proper Codon Mapping)")
    print("=" * 70)

    # Build amino acid data
    print("\nBuilding amino acid embeddings from codon structure...")

    aa_data = {}
    for aa in AMINO_ACIDS:
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        codon_indices = [CODON_TO_INDEX[c] for c in codons]

        # Compute mean codon index as proxy for "position" in genetic code
        mean_idx = np.mean(codon_indices) if codon_indices else 32

        # Properties from AA_PROPERTIES
        props = AA_PROPERTIES.get(aa, (0, 0, 0, 0))

        aa_data[aa] = {
            'codons': codons,
            'codon_indices': codon_indices,
            'n_codons': len(codons),
            'mean_codon_index': float(mean_idx),
            'hydrophobicity': props[0],
            'charge': props[1],
            'size': props[2],
            'polarity': props[3],
            # Use properties as pseudo-embedding
            'embedding': list(props),
        }

    print(f"  Built data for {len(aa_data)} amino acids")

    # Compute pairwise distances
    print("\nComputing pairwise distances...")

    padic_distances = {}
    property_distances = {}
    combined_distances = {}

    for i, aa1 in enumerate(AMINO_ACIDS):
        for aa2 in AMINO_ACIDS[i+1:]:
            key = f"{aa1}-{aa2}"

            padic_dist = compute_aa_padic_distance(aa1, aa2)
            prop_dist = compute_aa_property_distance(aa1, aa2)

            # Normalize property distance to [0, 1]
            prop_dist_norm = prop_dist / 4.0  # Max possible ~4.0

            combined = (args.padic_weight * padic_dist +
                       args.property_weight * prop_dist_norm)

            padic_distances[key] = padic_dist
            property_distances[key] = prop_dist
            combined_distances[key] = combined

    # Save results
    output_data = {
        "metadata": {
            "version": "v2",
            "method": "proper_codon_mapping",
            "padic_weight": args.padic_weight,
            "property_weight": args.property_weight,
            "timestamp": datetime.now().isoformat(),
            "source": "src.biology.codons + src.encoders.codon_encoder"
        },
        "amino_acids": aa_data,
        "padic_distances": padic_distances,
        "property_distances": property_distances,
        "combined_distances": combined_distances
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nEmbeddings saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nAmino Acids by Degeneracy (# synonymous codons):")
    by_degeneracy = sorted(aa_data.items(), key=lambda x: x[1]['n_codons'], reverse=True)
    for aa, data in by_degeneracy[:10]:
        print(f"  {aa}: {data['n_codons']} codons {data['codons']}")

    print("\nTop 10 Closest AA Pairs (p-adic):")
    sorted_padic = sorted(padic_distances.items(), key=lambda x: x[1])
    for pair, dist in sorted_padic[:10]:
        print(f"  {pair}: {dist:.4f}")

    print("\nTop 10 Most Distant AA Pairs (p-adic):")
    for pair, dist in sorted_padic[-10:]:
        print(f"  {pair}: {dist:.4f}")

    print("\nTop 10 Closest AA Pairs (properties):")
    sorted_props = sorted(property_distances.items(), key=lambda x: x[1])
    for pair, dist in sorted_props[:10]:
        print(f"  {pair}: {dist:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
