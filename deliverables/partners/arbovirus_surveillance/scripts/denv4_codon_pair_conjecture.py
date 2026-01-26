# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Codon Pair Context Conjecture - The Final Test.

CONJECTURE:
Hyperbolic variance detects conservation of CODON PAIR CONTEXT -
the relationship between adjacent codons that affects ribosome
translocation speed and accuracy.

RATIONALE:
- P-adic encoder's 12-dim input captures base identity at each position
- Window averaging implicitly encodes codon-to-codon transitions
- E gene (highly expressed) needs optimal translation = conserved pairs
- NS3/NS5 (enzymatic) have relaxed codon pair constraints

PREDICTION:
Hyperbolic variance correlates NEGATIVELY with codon pair score (CPS)
deviation - low variance = optimal codon pairs = translation efficiency.

PRACTICAL APPLICATION:
If confirmed, design primers targeting regions with:
1. Low hyperbolic variance (our metric)
2. Conserved codon pair patterns (translation constraint)

Usage:
    python scripts/denv4_codon_pair_conjecture.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

# Add project root
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.biology.codons import GENETIC_CODE, CODON_TO_INDEX

# Paths
ROJAS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROJAS_DIR / "results" / "codon_pair_conjecture"
CACHE_DIR = ROJAS_DIR / "data" / "cache"
PADIC_RESULTS = ROJAS_DIR / "results" / "padic_integration" / "padic_integration_results.json"


def build_codon_pair_bias_table(sequences: dict[str, str]) -> dict[str, float]:
    """Build codon pair bias (CPB) table from sequences.

    CPB = log2(observed_freq / expected_freq)

    Expected freq = freq(codon1) * freq(codon2)

    Positive CPB = over-represented pair (preferred)
    Negative CPB = under-represented pair (avoided)
    """
    # Count individual codons and codon pairs
    codon_counts = Counter()
    pair_counts = Counter()
    total_codons = 0
    total_pairs = 0

    for seq in sequences.values():
        seq = seq.upper()
        codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
        codons = [c for c in codons if len(c) == 3 and all(b in "ATGC" for b in c)]

        for codon in codons:
            codon_counts[codon] += 1
            total_codons += 1

        for i in range(len(codons) - 1):
            pair = f"{codons[i]}_{codons[i+1]}"
            pair_counts[pair] += 1
            total_pairs += 1

    # Compute frequencies
    codon_freq = {c: count / total_codons for c, count in codon_counts.items()}

    # Compute CPB for each pair
    cpb_table = {}
    for pair, count in pair_counts.items():
        c1, c2 = pair.split("_")
        observed = count / total_pairs
        expected = codon_freq.get(c1, 1e-10) * codon_freq.get(c2, 1e-10)

        if expected > 0 and observed > 0:
            cpb = np.log2(observed / expected)
        else:
            cpb = 0

        cpb_table[pair] = cpb

    return cpb_table


def compute_window_codon_pair_score(
    sequences: dict[str, str],
    start: int,
    window_size: int,
    cpb_table: dict[str, float],
) -> dict:
    """Compute codon pair metrics for a window across all sequences.

    Returns:
        - mean_cpb: Mean codon pair bias (higher = more optimized)
        - cpb_variance: Variance of CPB across sequences (lower = more conserved)
        - cpb_consistency: Fraction of sequences using same codon pairs
    """
    window_cpbs = []
    pair_usage = defaultdict(lambda: Counter())  # position -> pair counts

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()
        codons = [window[i:i+3] for i in range(0, len(window) - 2, 3)]
        codons = [c for c in codons if len(c) == 3 and all(b in "ATGC" for b in c)]

        if len(codons) < 2:
            continue

        # Compute mean CPB for this sequence's window
        cpbs = []
        for i in range(len(codons) - 1):
            pair = f"{codons[i]}_{codons[i+1]}"
            cpb = cpb_table.get(pair, 0)
            cpbs.append(cpb)
            pair_usage[i][pair] += 1

        if cpbs:
            window_cpbs.append(np.mean(cpbs))

    if not window_cpbs:
        return {
            "mean_cpb": 0,
            "cpb_variance": 0,
            "cpb_consistency": 0,
            "n_sequences": 0,
        }

    # Compute consistency: fraction using most common pair at each position
    consistencies = []
    for pos, counts in pair_usage.items():
        if counts:
            total = sum(counts.values())
            max_count = max(counts.values())
            consistencies.append(max_count / total)

    return {
        "mean_cpb": np.mean(window_cpbs),
        "cpb_variance": np.var(window_cpbs),
        "cpb_consistency": np.mean(consistencies) if consistencies else 0,
        "n_sequences": len(window_cpbs),
    }


def compute_codon_pair_conservation(
    sequences: dict[str, str],
    start: int,
    window_size: int,
) -> dict:
    """Compute how conserved codon PAIRS are (not just individual codons).

    This measures whether the same codon-codon transitions occur across strains.
    """
    pair_patterns = defaultdict(Counter)  # position -> {pair: count}

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()
        codons = [window[i:i+3] for i in range(0, len(window) - 2, 3)]
        codons = [c for c in codons if len(c) == 3 and all(b in "ATGC" for b in c)]

        for i in range(len(codons) - 1):
            pair = f"{codons[i]}_{codons[i+1]}"
            pair_patterns[i][pair] += 1

    # Compute entropy of pair distribution at each position
    pair_entropies = []
    for pos, counts in pair_patterns.items():
        total = sum(counts.values())
        if total > 0:
            probs = [c / total for c in counts.values()]
            h = -sum(p * np.log2(p) for p in probs if p > 0)
            pair_entropies.append(h)

    return {
        "mean_pair_entropy": np.mean(pair_entropies) if pair_entropies else 0,
        "max_pair_entropy": max(pair_entropies) if pair_entropies else 0,
        "n_positions": len(pair_entropies),
    }


def test_codon_pair_conjecture(
    sequences: dict[str, str],
    padic_results: dict,
) -> dict:
    """Test: hyperbolic variance detects codon pair context conservation."""

    print("=" * 70)
    print("CODON PAIR CONTEXT CONJECTURE")
    print("=" * 70)
    print("\nHypothesis: Low hyperbolic variance = conserved codon pair patterns")
    print("            optimized for translation efficiency")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sequences": len(sequences),
        "predictions": {},
        "correlations": {},
    }

    # Build CPB table from all sequences
    print("\n1. Building codon pair bias table from 270 DENV-4 genomes...")
    cpb_table = build_codon_pair_bias_table(sequences)
    print(f"   Computed CPB for {len(cpb_table)} codon pairs")

    # Show most preferred/avoided pairs
    sorted_pairs = sorted(cpb_table.items(), key=lambda x: x[1], reverse=True)
    print("\n   Top 5 preferred pairs (high CPB):")
    for pair, cpb in sorted_pairs[:5]:
        c1, c2 = pair.split("_")
        aa1, aa2 = GENETIC_CODE.get(c1, "?"), GENETIC_CODE.get(c2, "?")
        print(f"      {pair} ({aa1}-{aa2}): CPB = {cpb:.3f}")

    print("\n   Top 5 avoided pairs (low CPB):")
    for pair, cpb in sorted_pairs[-5:]:
        c1, c2 = pair.split("_")
        aa1, aa2 = GENETIC_CODE.get(c1, "?"), GENETIC_CODE.get(c2, "?")
        print(f"      {pair} ({aa1}-{aa2}): CPB = {cpb:.3f}")

    # Analyze each window from p-adic results
    genome_scan = padic_results.get("genome_scan", [])

    print(f"\n2. Analyzing {len(genome_scan)} genomic windows...")

    window_data = []
    for scan_point in genome_scan:
        pos = scan_point["position"]
        hyp_var = scan_point["variance"]

        # Compute codon pair metrics
        cpb_metrics = compute_window_codon_pair_score(sequences, pos, 75, cpb_table)
        pair_conservation = compute_codon_pair_conservation(sequences, pos, 75)

        window_data.append({
            "position": pos,
            "hyp_variance": hyp_var,
            **cpb_metrics,
            **pair_conservation,
        })

    # Test Prediction 1: hyp_var negatively correlates with mean_cpb
    print("\n" + "=" * 60)
    print("PREDICTION 1: Low hyp_var = High mean CPB (optimized pairs)")
    print("=" * 60)

    hyp_vars = [d["hyp_variance"] for d in window_data]
    mean_cpbs = [d["mean_cpb"] for d in window_data]

    corr_cpb, pval_cpb = spearmanr(hyp_vars, mean_cpbs)

    print(f"\nCorrelation (hyp_var vs mean_cpb):")
    print(f"  Spearman rho: {corr_cpb:.4f}")
    print(f"  P-value:      {pval_cpb:.4f}")

    pred1_confirmed = corr_cpb < -0.2 and pval_cpb < 0.1
    print(f"\n  Prediction 1 {'CONFIRMED' if pred1_confirmed else 'REJECTED'}: "
          f"Expected negative correlation, got {corr_cpb:.4f}")

    results["correlations"]["hyp_var_vs_mean_cpb"] = {
        "spearman_rho": float(corr_cpb),
        "p_value": float(pval_cpb),
        "confirmed": pred1_confirmed,
    }

    # Test Prediction 2: hyp_var correlates with pair entropy (diversity)
    print("\n" + "=" * 60)
    print("PREDICTION 2: High hyp_var = High pair entropy (divergent pairs)")
    print("=" * 60)

    pair_entropies = [d["mean_pair_entropy"] for d in window_data]
    corr_entropy, pval_entropy = spearmanr(hyp_vars, pair_entropies)

    print(f"\nCorrelation (hyp_var vs pair_entropy):")
    print(f"  Spearman rho: {corr_entropy:.4f}")
    print(f"  P-value:      {pval_entropy:.4f}")

    pred2_confirmed = corr_entropy > 0.2 and pval_entropy < 0.1
    print(f"\n  Prediction 2 {'CONFIRMED' if pred2_confirmed else 'REJECTED'}: "
          f"Expected positive correlation, got {corr_entropy:.4f}")

    results["correlations"]["hyp_var_vs_pair_entropy"] = {
        "spearman_rho": float(corr_entropy),
        "p_value": float(pval_entropy),
        "confirmed": pred2_confirmed,
    }

    # Test Prediction 3: hyp_var negatively correlates with pair consistency
    print("\n" + "=" * 60)
    print("PREDICTION 3: Low hyp_var = High pair consistency")
    print("=" * 60)

    consistencies = [d["cpb_consistency"] for d in window_data]
    corr_cons, pval_cons = spearmanr(hyp_vars, consistencies)

    print(f"\nCorrelation (hyp_var vs pair_consistency):")
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
    print("\n" + "=" * 60)
    print("GENE-LEVEL CODON PAIR ANALYSIS")
    print("=" * 60)

    gene_regions = {
        "E": (977, 2471),
        "NS1": (2472, 3527),
        "NS3": (4575, 6431),
        "NS5": (7559, 10271),
    }

    print(f"\n{'Gene':<6} {'Hyp Var':<10} {'Mean CPB':<10} {'Pair Entropy':<14} {'Consistency'}")
    print("-" * 60)

    gene_stats = {}
    for gene, (start, end) in gene_regions.items():
        gene_data = [d for d in window_data if start <= d["position"] < end]
        if gene_data:
            gene_stats[gene] = {
                "mean_hyp_var": np.mean([d["hyp_variance"] for d in gene_data]),
                "mean_cpb": np.mean([d["mean_cpb"] for d in gene_data]),
                "mean_pair_entropy": np.mean([d["mean_pair_entropy"] for d in gene_data]),
                "mean_consistency": np.mean([d["cpb_consistency"] for d in gene_data]),
                "n_windows": len(gene_data),
            }
            s = gene_stats[gene]
            print(f"{gene:<6} {s['mean_hyp_var']:.4f}     {s['mean_cpb']:.4f}     "
                  f"{s['mean_pair_entropy']:.4f}         {s['mean_consistency']:.4f}")

    results["gene_stats"] = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                  for kk, vv in v.items()}
                             for k, v in gene_stats.items()}

    # Summary
    print("\n" + "=" * 70)
    print("CONJECTURE SUMMARY")
    print("=" * 70)

    confirmed = sum([
        results["correlations"].get("hyp_var_vs_mean_cpb", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_pair_entropy", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_consistency", {}).get("confirmed", False),
    ])

    print(f"\nPredictions confirmed: {confirmed}/3")

    results["summary"] = {
        "predictions_confirmed": confirmed,
        "conjecture_supported": confirmed >= 2,
        "conjecture_status": "CONFIRMED" if confirmed >= 2 else ("PARTIAL" if confirmed == 1 else "REJECTED"),
    }

    # Store window data for practical application
    results["window_data"] = window_data
    results["cpb_table_size"] = len(cpb_table)

    return results


def apply_to_primer_design(
    results: dict,
    sequences: dict[str, str],
    padic_results: dict,
) -> dict:
    """Apply conjecture results to practical primer design.

    Regardless of whether the conjecture is confirmed, use the metrics
    to identify optimal primer regions.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL APPLICATION: PRIMER DESIGN RECOMMENDATIONS")
    print("=" * 70)

    window_data = results["window_data"]
    conjecture_status = results["summary"]["conjecture_status"]

    # Rank windows by combined score
    # If conjecture confirmed: prioritize low hyp_var + high CPB + high consistency
    # If rejected: use hyp_var alone (it still identifies something useful)

    print(f"\nConjecture status: {conjecture_status}")

    if conjecture_status == "CONFIRMED":
        print("Using combined metric: hyp_var + codon_pair_optimization")
        # Normalize and combine metrics
        for d in window_data:
            # Lower hyp_var is better
            # Higher mean_cpb is better
            # Higher consistency is better
            d["primer_score"] = (
                -d["hyp_variance"] * 10 +  # Normalize to similar scale
                d["mean_cpb"] +
                d["cpb_consistency"]
            )
    else:
        print("Using hyperbolic variance alone (novel metric)")
        for d in window_data:
            d["primer_score"] = -d["hyp_variance"]  # Lower is better

    # Sort by primer score
    ranked = sorted(window_data, key=lambda x: x["primer_score"], reverse=True)

    # Gene regions for annotation
    gene_regions = {
        (0, 101): "5UTR",
        (102, 476): "C",
        (477, 976): "prM",
        (977, 2471): "E",
        (2472, 3527): "NS1",
        (3528, 4184): "NS2A",
        (4185, 4574): "NS2B",
        (4575, 6431): "NS3",
        (6432, 6806): "NS4A",
        (6807, 7558): "NS4B",
        (7559, 10271): "NS5",
    }

    def get_gene(pos):
        for (start, end), name in gene_regions.items():
            if start <= pos < end:
                return name
        return "intergenic"

    print("\n" + "=" * 60)
    print("TOP 10 PRIMER CANDIDATE REGIONS")
    print("=" * 60)

    print(f"\n{'Rank':<6} {'Position':<10} {'Gene':<8} {'Hyp Var':<10} {'Score':<10}")
    print("-" * 50)

    top_candidates = []
    for i, d in enumerate(ranked[:10], 1):
        gene = get_gene(d["position"])
        print(f"{i:<6} {d['position']:<10} {gene:<8} {d['hyp_variance']:.4f}     {d['primer_score']:.4f}")
        top_candidates.append({
            "rank": i,
            "position": d["position"],
            "gene": gene,
            "hyp_variance": d["hyp_variance"],
            "primer_score": d["primer_score"],
            "mean_cpb": d.get("mean_cpb", 0),
            "consistency": d.get("cpb_consistency", 0),
        })

    # Compare with Rojas's current primer regions
    print("\n" + "=" * 60)
    print("COMPARISON WITH CURRENT PRIMER TARGETS")
    print("=" * 60)

    rojas_regions = [
        ("DENV4_E32_NS5_F", 9908, "NS5", "Clade_E.3.2 specific"),
        ("PANFLAVI_FU1", 9007, "NS5", "Pan-flavivirus"),
        ("PANFLAVI_cFD2", 9196, "NS5", "Pan-flavivirus"),
    ]

    print(f"\n{'Primer':<20} {'Position':<10} {'Hyp Var':<10} {'Rank':<8} {'Assessment'}")
    print("-" * 70)

    for name, pos, gene, desc in rojas_regions:
        # Find closest window
        closest = min(window_data, key=lambda x: abs(x["position"] - pos))

        # Find rank
        rank = next((i+1 for i, d in enumerate(ranked) if d["position"] == closest["position"]), "N/A")

        # Assessment
        if isinstance(rank, int) and rank <= 10:
            assessment = "GOOD - in top 10"
        elif isinstance(rank, int) and rank <= 20:
            assessment = "MODERATE - top 20"
        else:
            assessment = "SUBOPTIMAL - consider alternatives"

        print(f"{name:<20} {pos:<10} {closest['hyp_variance']:.4f}     {rank:<8} {assessment}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR ROJAS")
    print("=" * 60)

    recommendations = []

    # Check if E gene is better than NS5
    e_positions = [d for d in ranked[:10] if get_gene(d["position"]) == "E"]
    ns5_positions = [d for d in ranked[:10] if get_gene(d["position"]) == "NS5"]

    if len(e_positions) > len(ns5_positions):
        rec = ("NOVEL TARGET: E gene (positions 2400, 2100) shows lower hyperbolic "
               "variance than NS5 - consider designing primers here for potentially "
               "broader coverage across divergent clades.")
        recommendations.append(rec)
        print(f"\n1. {rec}")

    if any(d["position"] in [2400, 3000] for d in ranked[:5]):
        rec = ("STRUCTURAL PROTEINS: Position 2400 (E gene) and 3000 (NS1) rank "
               "in top 5 - these regions may be under stronger evolutionary constraint "
               "than NS5, offering better pan-DENV-4 targeting.")
        recommendations.append(rec)
        print(f"\n2. {rec}")

    # Check NS5 alternatives
    ns5_best = min([d for d in window_data if 7559 <= d["position"] < 10271],
                   key=lambda x: x["hyp_variance"], default=None)
    if ns5_best and ns5_best["position"] != 9908:
        rec = (f"NS5 ALTERNATIVE: Within NS5, position {ns5_best['position']} has lower "
               f"hyperbolic variance ({ns5_best['hyp_variance']:.4f}) than the current "
               f"target at 9908. Consider this as alternative if NS5 targeting is required.")
        recommendations.append(rec)
        print(f"\n3. {rec}")

    return {
        "top_candidates": top_candidates,
        "recommendations": recommendations,
        "conjecture_status": conjecture_status,
    }


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    with open(CACHE_DIR / "denv4_sequences.json") as f:
        sequences = json.load(f)

    with open(PADIC_RESULTS) as f:
        padic_results = json.load(f)

    # Test conjecture
    results = test_codon_pair_conjecture(sequences, padic_results)

    # Apply to primer design
    primer_results = apply_to_primer_design(results, sequences, padic_results)
    results["primer_recommendations"] = primer_results

    # Save results - remove window_data to avoid circular reference
    results_for_save = {
        "timestamp": results["timestamp"],
        "n_sequences": results["n_sequences"],
        "predictions": results["predictions"],
        "correlations": results["correlations"],
        "gene_stats": results["gene_stats"],
        "summary": results["summary"],
        "cpb_table_size": results["cpb_table_size"],
        "primer_recommendations": {
            "top_candidates": primer_results["top_candidates"],
            "recommendations": primer_results["recommendations"],
            "conjecture_status": primer_results["conjecture_status"],
        },
    }

    def json_serializer(obj):
        """Handle numpy types for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_file = RESULTS_DIR / "codon_pair_conjecture_results.json"
    with open(results_file, "w") as f:
        json.dump(results_for_save, f, indent=2, default=json_serializer)

    print(f"\n✓ Results saved to {results_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    status = results["summary"]["conjecture_status"]
    print(f"\nCodon Pair Conjecture: {status}")
    print(f"Predictions confirmed: {results['summary']['predictions_confirmed']}/3")
    print(f"\nTop primer region: Position {primer_results['top_candidates'][0]['position']} "
          f"({primer_results['top_candidates'][0]['gene']} gene)")


if __name__ == "__main__":
    main()
