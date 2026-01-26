# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Evolutionary Divergence Analysis for Dengue Serotypes.

This script investigates the evolutionary context of DENV-4's divergence:
1. When did DENV-4 split from other serotypes?
2. How divergent are DENV-4 strains from each other vs other serotypes?
3. Is the high variability due to ancient divergence or high mutation rate?

Key question: Is DENV-4 variability caused by:
  A) Higher mutation rate (polymerase fidelity) - would show recent divergence
  B) Ancient divergence - would show deeper phylogenetic split

Usage:
    python validation/test_evolutionary_divergence.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import datetime

import numpy as np
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_pairwise_identity(seq1: str, seq2: str) -> float:
    """Compute pairwise sequence identity between two sequences."""
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b and a in "ACGT")
    valid = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a in "ACGT" and b in "ACGT")

    return matches / valid if valid > 0 else 0.0


def compute_within_serotype_divergence(sequences: list[tuple[str, str]]) -> dict:
    """Compute pairwise divergence within a serotype."""
    if len(sequences) < 2:
        return {"mean_identity": 0, "min_identity": 0, "max_identity": 0, "n_pairs": 0}

    identities = []

    # Sample pairs to avoid O(n^2) for large datasets
    n = len(sequences)
    max_pairs = min(100, n * (n - 1) // 2)

    import random
    random.seed(42)

    pairs_checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pairs_checked >= max_pairs:
                break
            identity = compute_pairwise_identity(sequences[i][1], sequences[j][1])
            identities.append(identity)
            pairs_checked += 1
        if pairs_checked >= max_pairs:
            break

    return {
        "mean_identity": np.mean(identities) if identities else 0,
        "min_identity": min(identities) if identities else 0,
        "max_identity": max(identities) if identities else 0,
        "std_identity": np.std(identities) if identities else 0,
        "n_pairs": len(identities),
    }


def compute_between_serotype_divergence(
    seqs1: list[tuple[str, str]],
    seqs2: list[tuple[str, str]],
    max_pairs: int = 50,
) -> dict:
    """Compute pairwise divergence between two serotypes."""
    if not seqs1 or not seqs2:
        return {"mean_identity": 0, "min_identity": 0, "max_identity": 0, "n_pairs": 0}

    identities = []

    import random
    random.seed(42)

    # Sample pairs
    sample1 = random.sample(seqs1, min(len(seqs1), max_pairs))
    sample2 = random.sample(seqs2, min(len(seqs2), max_pairs))

    for acc1, seq1 in sample1:
        for acc2, seq2 in sample2:
            identity = compute_pairwise_identity(seq1, seq2)
            identities.append(identity)

    return {
        "mean_identity": np.mean(identities) if identities else 0,
        "min_identity": min(identities) if identities else 0,
        "max_identity": max(identities) if identities else 0,
        "std_identity": np.std(identities) if identities else 0,
        "n_pairs": len(identities),
    }


def estimate_relative_divergence_time(identity: float) -> str:
    """Estimate relative divergence time from sequence identity.

    Based on typical flavivirus mutation rates (~10^-3 to 10^-4 per site per year).
    This is a rough approximation for comparative purposes only.
    """
    if identity > 0.99:
        return "Very recent (<10 years)"
    elif identity > 0.95:
        return "Recent (10-50 years)"
    elif identity > 0.85:
        return "Moderate (50-200 years)"
    elif identity > 0.70:
        return "Ancient (200-500 years)"
    else:
        return "Very ancient (>500 years)"


def analyze_geographic_clade_structure(
    metadata_cache: Path,
    strain_cache: Path,
) -> dict:
    """Check if DENV-4 strains come from more diverse geographic clades."""

    if not metadata_cache.exists():
        return {}

    with open(metadata_cache) as f:
        metadata = json.load(f)

    results = {}

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        data = metadata.get(serotype, {})
        countries = data.get("countries", [])

        # Count unique countries
        n_countries = len(set(countries))

        # Geographic entropy
        if countries:
            counts = Counter(countries)
            total = len(countries)
            entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
        else:
            entropy = 0

        results[serotype] = {
            "n_countries": n_countries,
            "geographic_entropy": entropy,
            "countries": list(set(countries))[:10],  # First 10 unique
        }

    return results


def run_divergence_analysis(cache_dir: Path) -> dict:
    """Run complete evolutionary divergence analysis."""

    print("=" * 70)
    print("EVOLUTIONARY DIVERGENCE ANALYSIS")
    print("=" * 70)
    print()
    print("Investigating whether DENV-4 variability is due to:")
    print("  A) Higher mutation rate (would show similar within-serotype divergence)")
    print("  B) Ancient divergence (would show deeper phylogenetic split)")
    print()

    strain_cache = cache_dir / "dengue_strains.json"
    metadata_cache = cache_dir / "dengue_metadata.json"

    if not strain_cache.exists():
        return {"error": "Missing strain data"}

    with open(strain_cache) as f:
        all_sequences = json.load(f)

    # Convert sequences
    sequences = {}
    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        seqs = all_sequences.get(serotype, [])
        if seqs and isinstance(seqs[0], list):
            sequences[serotype] = [(s[0], s[1]) for s in seqs]
        else:
            sequences[serotype] = seqs

    # Within-serotype divergence
    print("-" * 70)
    print("WITHIN-SEROTYPE DIVERGENCE")
    print("-" * 70)
    print()
    print("How similar are strains WITHIN each serotype?")
    print()

    within_divergence = {}
    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        result = compute_within_serotype_divergence(sequences.get(serotype, []))
        within_divergence[serotype] = result

        time_estimate = estimate_relative_divergence_time(result["mean_identity"])

        print(f"{serotype}:")
        print(f"  Mean identity: {result['mean_identity']:.1%}")
        print(f"  Range: {result['min_identity']:.1%} - {result['max_identity']:.1%}")
        print(f"  Divergence estimate: {time_estimate}")
        print()

    # Between-serotype divergence
    print("-" * 70)
    print("BETWEEN-SEROTYPE DIVERGENCE")
    print("-" * 70)
    print()
    print("How similar are strains BETWEEN different serotypes?")
    print()

    between_divergence = {}
    serotypes = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]

    for i, s1 in enumerate(serotypes):
        for s2 in serotypes[i+1:]:
            key = f"{s1}_vs_{s2}"
            result = compute_between_serotype_divergence(
                sequences.get(s1, []),
                sequences.get(s2, []),
            )
            between_divergence[key] = result

            print(f"{s1} vs {s2}: {result['mean_identity']:.1%} identity")

    print()

    # Geographic analysis
    print("-" * 70)
    print("GEOGRAPHIC CLADE STRUCTURE")
    print("-" * 70)
    print()

    geographic = analyze_geographic_clade_structure(metadata_cache, strain_cache)

    for serotype, data in geographic.items():
        print(f"{serotype}: {data['n_countries']} countries, entropy={data['geographic_entropy']:.2f}")

    print()

    # Key comparisons
    print("-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    print()

    denv4_within = within_divergence.get("DENV-4", {}).get("mean_identity", 0)
    other_within = np.mean([
        within_divergence.get(s, {}).get("mean_identity", 0)
        for s in ["DENV-1", "DENV-2", "DENV-3"]
    ])

    print(f"1. WITHIN-SEROTYPE IDENTITY:")
    print(f"   DENV-4: {denv4_within:.1%}")
    print(f"   Others: {other_within:.1%}")

    if denv4_within < other_within * 0.9:  # DENV-4 is <90% as similar
        print(f"   → DENV-4 strains are MORE DIVERGENT from each other")
        print(f"   → Suggests ANCIENT DIVERSIFICATION, not just high mutation rate")
    else:
        print(f"   → DENV-4 strains are similarly divergent")
        print(f"   → High variability may be due to mutation rate")

    print()

    # Check DENV-4 vs others
    denv4_vs_others = [
        between_divergence.get(f"DENV-{i}_vs_DENV-4", {}).get("mean_identity", 0)
        for i in [1, 2, 3]
    ]
    denv123_pairs = [
        between_divergence.get("DENV-1_vs_DENV-2", {}).get("mean_identity", 0),
        between_divergence.get("DENV-1_vs_DENV-3", {}).get("mean_identity", 0),
        between_divergence.get("DENV-2_vs_DENV-3", {}).get("mean_identity", 0),
    ]

    print(f"2. BETWEEN-SEROTYPE IDENTITY:")
    print(f"   DENV-4 vs others: {np.mean(denv4_vs_others):.1%}")
    print(f"   DENV-1/2/3 pairs: {np.mean(denv123_pairs):.1%}")

    if np.mean(denv4_vs_others) < np.mean(denv123_pairs) * 0.95:
        print(f"   → DENV-4 diverged EARLIER than DENV-1/2/3 split")
        print(f"   → DENV-4 is phylogenetically more distant")
    else:
        print(f"   → All serotypes equally distant from each other")

    print()

    # Synthesis
    print("=" * 70)
    print("SYNTHESIS: Why is DENV-4 so variable?")
    print("=" * 70)
    print()

    if denv4_within < 0.70:  # Very low within-serotype identity
        print("EVIDENCE POINTS TO: ANCIENT DIVERSIFICATION")
        print()
        print("DENV-4 strains in our sample are NOT closely related.")
        print("They represent DIVERGENT LINEAGES that split long ago.")
        print()
        print("Implications:")
        print("  1. DENV-4 may have diversified into multiple clades early")
        print("  2. Geographic isolation maintained separate lineages")
        print("  3. No single 'DENV-4 polymerase' - multiple variants exist")
        print("  4. Primer design must account for clade-specific variation")
        mechanism = "ancient_diversification"
    else:
        print("EVIDENCE POINTS TO: HIGH MUTATION RATE")
        print()
        print("DENV-4 strains show similar within-serotype divergence")
        print("but accumulate more mutations overall.")
        print()
        print("Implications:")
        print("  1. DENV-4 polymerase likely has lower fidelity")
        print("  2. Mutations accumulate faster genome-wide")
        print("  3. All DENV-4 strains evolve rapidly from common ancestor")
        mechanism = "high_mutation_rate"

    print()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "within_serotype_divergence": within_divergence,
        "between_serotype_divergence": between_divergence,
        "geographic_structure": geographic,
        "inferred_mechanism": mechanism,
        "denv4_within_identity": denv4_within,
        "other_within_identity_mean": other_within,
    }


def main():
    """Main entry point."""
    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    results = run_divergence_analysis(cache_dir)

    # Save results
    output_path = validation_dir / "evolutionary_divergence_results.json"

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
