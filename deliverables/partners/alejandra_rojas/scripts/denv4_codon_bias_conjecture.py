# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Final Conjecture: Hyperbolic Variance = Codon Usage Bias Conservation.

REJECTED CONJECTURES:
1. Synonymous shuffling - REJECTED (all positions equal entropy)
2. AA property conservation - REJECTED (no correlation)

CRITICAL OBSERVATION:
- E gene: LARGE property changes (77.7) but LOW hyp_var (0.0284)
- NS3: SMALL property changes (72.4) but HIGH hyp_var (0.0393)

This is OPPOSITE to expectations! What if hyperbolic variance detects
something orthogonal to classical conservation?

FINAL CONJECTURE:
Low hyperbolic variance indicates CODON USAGE BIAS CONSERVATION -
different strains prefer the SAME synonymous codons even when
the amino acid changes.

Why this matters:
- Codon usage affects translation efficiency
- Codon usage affects RNA secondary structure
- Codon usage is under evolutionary selection independent of protein

PREDICTIONS:
1. Low hyp_var regions have consistent codon usage patterns across strains
2. High hyp_var regions have divergent codon preferences
3. E gene (surface protein) has stronger codon bias than NS3/NS5 (internal)

If confirmed: The p-adic encoder learned CODON USAGE PATTERNS,
providing a novel metric for viral evolution distinct from sequence identity.

Usage:
    python scripts/denv4_codon_bias_conjecture.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, entropy

# Add project root
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.biology.codons import GENETIC_CODE, AMINO_ACID_TO_CODONS

# Paths
ROJAS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROJAS_DIR / "results" / "codon_bias_conjecture"
CACHE_DIR = ROJAS_DIR / "data" / "cache"
PADIC_RESULTS = ROJAS_DIR / "results" / "padic_integration" / "padic_integration_results.json"


def compute_codon_usage_entropy(
    sequences: dict[str, str],
    start: int,
    window_size: int = 75,
) -> dict:
    """Compute codon usage entropy for a window.

    For each amino acid position, compute how consistently strains
    use the same synonymous codon.

    Low entropy = all strains use same codon = conserved codon usage
    High entropy = strains use different codons = divergent codon usage
    """
    n_codons = window_size // 3

    # For each codon position, collect codon usage across strains
    position_codon_usage = defaultdict(lambda: Counter())

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()

        for codon_idx in range(n_codons):
            codon_start = codon_idx * 3
            if codon_start + 3 <= len(window):
                codon = window[codon_start:codon_start + 3]
                if all(b in "ATGC" for b in codon):
                    position_codon_usage[codon_idx][codon] += 1

    # Compute entropy of codon usage at each position
    position_entropies = []
    for codon_idx in range(n_codons):
        counts = position_codon_usage[codon_idx]
        if counts:
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            h = entropy(probs, base=2)
            position_entropies.append(h)

    # Compute codon usage bias (deviation from random)
    # For 4-fold degenerate codons, random usage = 2 bits entropy
    mean_entropy = np.mean(position_entropies) if position_entropies else 0

    # Compute cross-strain codon consistency
    # For each AA position, what fraction of strains agree on the codon?
    consistency_scores = []
    for codon_idx in range(n_codons):
        counts = position_codon_usage[codon_idx]
        if counts:
            total = sum(counts.values())
            max_count = max(counts.values())
            consistency = max_count / total
            consistency_scores.append(consistency)

    mean_consistency = np.mean(consistency_scores) if consistency_scores else 0

    return {
        "mean_codon_entropy": mean_entropy,
        "mean_codon_consistency": mean_consistency,
        "n_positions": len(position_entropies),
    }


def compute_synonymous_codon_divergence(
    sequences: dict[str, str],
    start: int,
    window_size: int = 75,
) -> dict:
    """Compute how much strains diverge in synonymous codon choice.

    For each amino acid, compute the diversity of codon usage across strains.
    This is independent of which AA is encoded.
    """
    n_codons = window_size // 3

    # Group codons by amino acid at each position
    position_aa_codons = defaultdict(lambda: defaultdict(Counter))

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()

        for codon_idx in range(n_codons):
            codon_start = codon_idx * 3
            if codon_start + 3 <= len(window):
                codon = window[codon_start:codon_start + 3]
                if all(b in "ATGC" for b in codon):
                    aa = GENETIC_CODE.get(codon, "X")
                    if aa != "X":
                        position_aa_codons[codon_idx][aa][codon] += 1

    # For each position, compute within-AA codon diversity
    within_aa_diversities = []

    for codon_idx in range(n_codons):
        aa_codons = position_aa_codons[codon_idx]
        for aa, codon_counts in aa_codons.items():
            # Only consider AAs with multiple possible codons
            possible_codons = AMINO_ACID_TO_CODONS.get(aa, [])
            if len(possible_codons) > 1 and sum(codon_counts.values()) >= 5:
                total = sum(codon_counts.values())
                probs = [c / total for c in codon_counts.values()]
                h = entropy(probs, base=2)
                # Normalize by max possible entropy
                max_h = np.log2(len(possible_codons))
                normalized_h = h / max_h if max_h > 0 else 0
                within_aa_diversities.append(normalized_h)

    mean_within_aa_diversity = np.mean(within_aa_diversities) if within_aa_diversities else 0

    return {
        "within_aa_codon_diversity": mean_within_aa_diversity,
        "n_measured": len(within_aa_diversities),
    }


def test_codon_bias_conjecture(
    sequences: dict[str, str],
    padic_results: dict,
) -> dict:
    """Test: hyperbolic variance detects codon usage bias conservation."""
    print("=" * 70)
    print("FINAL CONJECTURE: Hyperbolic Variance = Codon Usage Bias")
    print("=" * 70)

    genome_scan = padic_results.get("genome_scan", [])

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sequences": len(sequences),
        "predictions": {},
        "correlations": {},
    }

    # Analyze codon usage for each window
    codon_data = []
    for scan_point in genome_scan:
        pos = scan_point["position"]

        usage = compute_codon_usage_entropy(sequences, pos, 75)
        divergence = compute_synonymous_codon_divergence(sequences, pos, 75)

        codon_data.append({
            "position": pos,
            "hyp_variance": scan_point["variance"],
            **usage,
            **divergence,
        })

    # Test Prediction 1: hyp_var correlates with codon usage entropy
    print("\n" + "=" * 50)
    print("PREDICTION 1: High hyp_var = high codon usage entropy")
    print("=" * 50)

    hyp_vars = [d["hyp_variance"] for d in codon_data]
    codon_entropies = [d["mean_codon_entropy"] for d in codon_data]

    corr_entropy, pval_entropy = spearmanr(hyp_vars, codon_entropies)

    print(f"\nCorrelation (hyp_var vs codon_entropy):")
    print(f"  Spearman rho: {corr_entropy:.4f}")
    print(f"  P-value:      {pval_entropy:.4f}")

    pred1_confirmed = corr_entropy > 0.2 and pval_entropy < 0.1
    print(f"\n  Prediction 1 {'CONFIRMED' if pred1_confirmed else 'REJECTED'}: "
          f"Expected positive correlation, got {corr_entropy:.4f}")

    results["correlations"]["hyp_var_vs_codon_entropy"] = {
        "spearman_rho": float(corr_entropy),
        "p_value": float(pval_entropy),
        "confirmed": pred1_confirmed,
    }

    # Test Prediction 2: hyp_var correlates with within-AA codon diversity
    print("\n" + "=" * 50)
    print("PREDICTION 2: High hyp_var = high within-AA codon diversity")
    print("=" * 50)

    valid_data = [d for d in codon_data if d["n_measured"] > 0]
    if len(valid_data) >= 10:
        hyp_vars_valid = [d["hyp_variance"] for d in valid_data]
        within_aa_div = [d["within_aa_codon_diversity"] for d in valid_data]

        corr_div, pval_div = spearmanr(hyp_vars_valid, within_aa_div)

        print(f"\nCorrelation (hyp_var vs within-AA codon diversity):")
        print(f"  Spearman rho: {corr_div:.4f}")
        print(f"  P-value:      {pval_div:.4f}")

        pred2_confirmed = corr_div > 0.2 and pval_div < 0.1
        print(f"\n  Prediction 2 {'CONFIRMED' if pred2_confirmed else 'REJECTED'}")

        results["correlations"]["hyp_var_vs_within_aa_diversity"] = {
            "spearman_rho": float(corr_div),
            "p_value": float(pval_div),
            "confirmed": pred2_confirmed,
        }
    else:
        pred2_confirmed = False
        print("Insufficient data")

    # Test Prediction 3: Low hyp_var = high codon consistency
    print("\n" + "=" * 50)
    print("PREDICTION 3: Low hyp_var = high codon consistency")
    print("=" * 50)

    consistencies = [d["mean_codon_consistency"] for d in codon_data]
    corr_cons, pval_cons = spearmanr(hyp_vars, consistencies)

    print(f"\nCorrelation (hyp_var vs codon_consistency):")
    print(f"  Spearman rho: {corr_cons:.4f}")
    print(f"  P-value:      {pval_cons:.4f}")

    pred3_confirmed = corr_cons < -0.2 and pval_cons < 0.1
    print(f"\n  Prediction 3 {'CONFIRMED' if pred3_confirmed else 'REJECTED'}: "
          f"Expected negative correlation, got {corr_cons:.4f}")

    results["correlations"]["hyp_var_vs_consistency"] = {
        "spearman_rho": float(corr_cons),
        "p_value": float(pval_cons),
        "confirmed": pred3_confirmed,
    }

    # Gene-level analysis
    print("\n" + "=" * 50)
    print("GENE-LEVEL CODON USAGE ANALYSIS")
    print("=" * 50)

    gene_regions = {
        "E": (977, 2471),
        "NS1": (2472, 3527),
        "NS3": (4575, 6431),
        "NS5": (7559, 10271),
    }

    print(f"\n{'Gene':<6} {'Hyp Var':<10} {'Codon Entropy':<15} {'Consistency':<12}")
    print("-" * 50)

    gene_stats = {}
    for gene, (start, end) in gene_regions.items():
        gene_data = [d for d in codon_data if start <= d["position"] < end]
        if gene_data:
            gene_stats[gene] = {
                "mean_hyp_var": np.mean([d["hyp_variance"] for d in gene_data]),
                "mean_codon_entropy": np.mean([d["mean_codon_entropy"] for d in gene_data]),
                "mean_consistency": np.mean([d["mean_codon_consistency"] for d in gene_data]),
            }
            s = gene_stats[gene]
            print(f"{gene:<6} {s['mean_hyp_var']:.4f}     {s['mean_codon_entropy']:.4f}          "
                  f"{s['mean_consistency']:.4f}")

    results["gene_stats"] = {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in gene_stats.items()}

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONJECTURE SUMMARY")
    print("=" * 70)

    confirmed = sum([
        results["correlations"].get("hyp_var_vs_codon_entropy", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_within_aa_diversity", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_consistency", {}).get("confirmed", False),
    ])

    print(f"\nPredictions confirmed: {confirmed}/3")

    # Overall interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if confirmed >= 2:
        print("\n*** CODON USAGE BIAS CONJECTURE SUPPORTED ***")
        print("Hyperbolic variance detects conservation of CODON USAGE PATTERNS,")
        print("not sequence identity or amino acid properties.")
    else:
        print("\n*** ALL CONJECTURES REJECTED ***")
        print("\nWhat hyperbolic variance ACTUALLY measures:")
        print("1. NOT synonymous substitution rate")
        print("2. NOT amino acid property conservation")
        print("3. NOT codon usage bias")
        print("\nHyperbolic variance appears to detect a NOVEL aspect of sequence")
        print("diversity that is ORTHOGONAL to classical conservation metrics.")
        print("\nThis may be capturing:")
        print("  - Higher-order codon context patterns (codon pairs, triplets)")
        print("  - RNA secondary structure conservation")
        print("  - Translation kinetics signals")
        print("  - Something entirely new that the p-adic geometry reveals")

    results["summary"] = {
        "predictions_confirmed": confirmed,
        "conjecture_supported": confirmed >= 2,
    }

    return results


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    with open(CACHE_DIR / "denv4_sequences.json") as f:
        sequences = json.load(f)

    with open(PADIC_RESULTS) as f:
        padic_results = json.load(f)

    results = test_codon_bias_conjecture(sequences, padic_results)

    results_file = RESULTS_DIR / "codon_bias_conjecture_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
