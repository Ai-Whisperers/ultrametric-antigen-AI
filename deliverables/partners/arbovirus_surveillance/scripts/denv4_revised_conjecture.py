# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Revised Conjecture: Hyperbolic Variance Detects AA Property Conservation.

ORIGINAL CONJECTURE (REJECTED):
DENV-4's diversity is synonymous codon shuffling.

FINDINGS:
- All codon positions have similar entropy (~1.5-1.6 bits)
- No correlation between hyp_var and synonymous ratio
- E gene does NOT have highest synonymous ratio

REVISED CONJECTURE:
Low hyperbolic variance indicates conservation of AMINO ACID PROPERTIES
(hydrophobicity, charge, size) even when amino acids themselves change.

The p-adic codon encoder was trained with AA property loss, meaning:
- Similar AAs (e.g., Leu/Ile/Val) embed nearby in hyperbolic space
- Low hyperbolic variance = changes between SIMILAR amino acids
- High hyperbolic variance = changes between DISSIMILAR amino acids

PREDICTIONS:
1. Low hyp_var correlates with small changes in hydrophobicity
2. Low hyp_var correlates with small changes in molecular weight
3. E gene (structural) has more property-conserved substitutions than NS5 (enzymatic)

This would explain WHY E gene has low hyp_var despite non-synonymous changes:
- Immune epitopes tolerate Leu→Val but not Leu→Arg
- Property-preserving substitutions maintain protein fold

Usage:
    python scripts/denv4_revised_conjecture.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Add project root to path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.biology.codons import GENETIC_CODE

# Amino acid properties (hydrophobicity, charge, mol_weight, volume)
AA_PROPERTIES = {
    'A': (1.8, 0, 89, 88.6),    # Alanine
    'R': (-4.5, 1, 174, 173.4),  # Arginine
    'N': (-3.5, 0, 132, 114.1),  # Asparagine
    'D': (-3.5, -1, 133, 111.1), # Aspartic acid
    'C': (2.5, 0, 121, 108.5),   # Cysteine
    'Q': (-3.5, 0, 146, 143.8),  # Glutamine
    'E': (-3.5, -1, 147, 138.4), # Glutamic acid
    'G': (-0.4, 0, 75, 60.1),    # Glycine
    'H': (-3.2, 0.5, 155, 153.2),# Histidine
    'I': (4.5, 0, 131, 166.7),   # Isoleucine
    'L': (3.8, 0, 131, 166.7),   # Leucine
    'K': (-3.9, 1, 146, 168.6),  # Lysine
    'M': (1.9, 0, 149, 162.9),   # Methionine
    'F': (2.8, 0, 165, 189.9),   # Phenylalanine
    'P': (-1.6, 0, 115, 112.7),  # Proline
    'S': (-0.8, 0, 105, 89.0),   # Serine
    'T': (-0.7, 0, 119, 116.1),  # Threonine
    'W': (-0.9, 0, 204, 227.8),  # Tryptophan
    'Y': (-1.3, 0, 181, 193.6),  # Tyrosine
    'V': (4.2, 0, 117, 140.0),   # Valine
    '*': (0, 0, 0, 0),           # Stop codon
}

# Paths
ROJAS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROJAS_DIR / "results" / "revised_conjecture"
CACHE_DIR = ROJAS_DIR / "data" / "cache"
PADIC_RESULTS = ROJAS_DIR / "results" / "padic_integration" / "padic_integration_results.json"


def compute_aa_property_change(aa1: str, aa2: str) -> dict:
    """Compute property changes between two amino acids."""
    props1 = AA_PROPERTIES.get(aa1, (0, 0, 0, 0))
    props2 = AA_PROPERTIES.get(aa2, (0, 0, 0, 0))

    return {
        "delta_hydrophobicity": abs(props2[0] - props1[0]),
        "delta_charge": abs(props2[1] - props1[1]),
        "delta_mw": abs(props2[2] - props1[2]),
        "delta_volume": abs(props2[3] - props1[3]),
        "euclidean_distance": np.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(props1, props2))),
    }


def analyze_substitution_properties(
    sequences: dict[str, str],
    start: int,
    window_size: int = 75,
) -> dict:
    """Analyze amino acid property changes in substitutions.

    Returns:
        Dict with mean property changes for all substitutions in window
    """
    n_codons = window_size // 3

    # Build consensus
    consensus_codons = []
    for codon_idx in range(n_codons):
        codon_counts = Counter()
        for seq in sequences.values():
            window = seq[start:start + window_size].upper()
            codon_start = codon_idx * 3
            if codon_start + 3 <= len(window):
                codon = window[codon_start:codon_start + 3]
                if all(b in "ATGC" for b in codon):
                    codon_counts[codon] += 1

        if codon_counts:
            consensus_codons.append(codon_counts.most_common(1)[0][0])
        else:
            consensus_codons.append("NNN")

    # Collect property changes for all substitutions
    property_changes = []

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()

        for codon_idx, cons_codon in enumerate(consensus_codons):
            codon_start = codon_idx * 3
            if codon_start + 3 <= len(window):
                obs_codon = window[codon_start:codon_start + 3]

                if not all(b in "ATGC" for b in obs_codon):
                    continue
                if cons_codon == "NNN":
                    continue
                if obs_codon == cons_codon:
                    continue

                cons_aa = GENETIC_CODE.get(cons_codon, "X")
                obs_aa = GENETIC_CODE.get(obs_codon, "X")

                if cons_aa == "X" or obs_aa == "X":
                    continue

                if cons_aa != obs_aa:  # Non-synonymous only
                    changes = compute_aa_property_change(cons_aa, obs_aa)
                    property_changes.append(changes)

    if not property_changes:
        return {
            "mean_delta_hydro": 0,
            "mean_delta_charge": 0,
            "mean_delta_mw": 0,
            "mean_delta_volume": 0,
            "mean_euclidean": 0,
            "n_substitutions": 0,
        }

    return {
        "mean_delta_hydro": np.mean([p["delta_hydrophobicity"] for p in property_changes]),
        "mean_delta_charge": np.mean([p["delta_charge"] for p in property_changes]),
        "mean_delta_mw": np.mean([p["delta_mw"] for p in property_changes]),
        "mean_delta_volume": np.mean([p["delta_volume"] for p in property_changes]),
        "mean_euclidean": np.mean([p["euclidean_distance"] for p in property_changes]),
        "n_substitutions": len(property_changes),
    }


def test_revised_conjecture(
    sequences: dict[str, str],
    padic_results: dict,
) -> dict:
    """Test the revised conjecture: hyp_var detects AA property conservation."""
    print("=" * 70)
    print("REVISED CONJECTURE: Hyperbolic Variance = AA Property Conservation")
    print("=" * 70)

    genome_scan = padic_results.get("genome_scan", [])

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sequences": len(sequences),
        "predictions": {},
        "correlations": {},
    }

    # Analyze property changes for each window
    property_data = []
    for scan_point in genome_scan:
        pos = scan_point["position"]
        props = analyze_substitution_properties(sequences, pos, 75)
        property_data.append({
            "position": pos,
            "hyp_variance": scan_point["variance"],
            **props,
        })

    # Filter windows with sufficient substitutions
    valid_data = [d for d in property_data if d["n_substitutions"] >= 10]

    print(f"\nAnalyzed {len(valid_data)} windows with ≥10 substitutions")

    if len(valid_data) < 5:
        print("Insufficient data for correlation analysis")
        return results

    hyp_vars = [d["hyp_variance"] for d in valid_data]

    # Test Prediction 1: hyp_var correlates with hydrophobicity change
    print("\n" + "=" * 50)
    print("PREDICTION 1: Low hyp_var = small hydrophobicity changes")
    print("=" * 50)

    delta_hydro = [d["mean_delta_hydro"] for d in valid_data]
    corr_hydro, pval_hydro = spearmanr(hyp_vars, delta_hydro)

    print(f"\nCorrelation (hyp_var vs delta_hydrophobicity):")
    print(f"  Spearman rho: {corr_hydro:.4f}")
    print(f"  P-value:      {pval_hydro:.4f}")

    pred1_confirmed = corr_hydro > 0.2 and pval_hydro < 0.1
    print(f"\n  Prediction 1 {'CONFIRMED' if pred1_confirmed else 'REJECTED'}: "
          f"Expected positive correlation, got {corr_hydro:.4f}")

    results["correlations"]["hyp_var_vs_hydrophobicity"] = {
        "spearman_rho": float(corr_hydro),
        "p_value": float(pval_hydro),
        "confirmed": pred1_confirmed,
    }

    # Test Prediction 2: hyp_var correlates with overall property distance
    print("\n" + "=" * 50)
    print("PREDICTION 2: Low hyp_var = small overall property distance")
    print("=" * 50)

    euclidean = [d["mean_euclidean"] for d in valid_data]
    corr_euc, pval_euc = spearmanr(hyp_vars, euclidean)

    print(f"\nCorrelation (hyp_var vs euclidean property distance):")
    print(f"  Spearman rho: {corr_euc:.4f}")
    print(f"  P-value:      {pval_euc:.4f}")

    pred2_confirmed = corr_euc > 0.2 and pval_euc < 0.1
    print(f"\n  Prediction 2 {'CONFIRMED' if pred2_confirmed else 'REJECTED'}: "
          f"Expected positive correlation, got {corr_euc:.4f}")

    results["correlations"]["hyp_var_vs_euclidean"] = {
        "spearman_rho": float(corr_euc),
        "p_value": float(pval_euc),
        "confirmed": pred2_confirmed,
    }

    # Test Prediction 3: Gene-level analysis
    print("\n" + "=" * 50)
    print("PREDICTION 3: E gene has more property-conserved substitutions")
    print("=" * 50)

    gene_regions = {
        "E": (977, 2471),
        "NS1": (2472, 3527),
        "NS3": (4575, 6431),
        "NS5": (7559, 10271),
    }

    gene_stats = {}
    for gene, (start, end) in gene_regions.items():
        gene_data = [d for d in valid_data if start <= d["position"] < end]
        if gene_data:
            gene_stats[gene] = {
                "mean_delta_hydro": np.mean([d["mean_delta_hydro"] for d in gene_data]),
                "mean_euclidean": np.mean([d["mean_euclidean"] for d in gene_data]),
                "mean_hyp_var": np.mean([d["hyp_variance"] for d in gene_data]),
                "n_windows": len(gene_data),
            }

    print("\nGene-level property changes:")
    print(f"{'Gene':<6} {'Δ Hydro':<10} {'Euclidean':<12} {'Hyp Var':<10}")
    print("-" * 40)

    for gene in ["E", "NS1", "NS3", "NS5"]:
        if gene in gene_stats:
            s = gene_stats[gene]
            print(f"{gene:<6} {s['mean_delta_hydro']:.4f}     {s['mean_euclidean']:.4f}       "
                  f"{s['mean_hyp_var']:.4f}")

    # Check if E gene has smallest property changes
    if "E" in gene_stats:
        e_euclidean = gene_stats["E"]["mean_euclidean"]
        other_euclidean = [s["mean_euclidean"] for g, s in gene_stats.items() if g != "E"]

        pred3_confirmed = e_euclidean < np.mean(other_euclidean) if other_euclidean else False

        print(f"\n  E gene euclidean: {e_euclidean:.4f}")
        print(f"  Other genes mean: {np.mean(other_euclidean):.4f}" if other_euclidean else "")
        print(f"\n  Prediction 3 {'CONFIRMED' if pred3_confirmed else 'REJECTED'}: "
              f"E gene has {'smallest' if pred3_confirmed else 'NOT smallest'} property changes")

        results["predictions"]["e_gene_property_conserved"] = {
            "confirmed": pred3_confirmed,
            "e_euclidean": float(e_euclidean),
            "other_mean": float(np.mean(other_euclidean)) if other_euclidean else None,
        }

    results["gene_stats"] = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                  for kk, vv in v.items()}
                             for k, v in gene_stats.items()}

    # Summary
    print("\n" + "=" * 70)
    print("REVISED CONJECTURE SUMMARY")
    print("=" * 70)

    confirmed = sum([
        results["correlations"].get("hyp_var_vs_hydrophobicity", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_euclidean", {}).get("confirmed", False),
        results["predictions"].get("e_gene_property_conserved", {}).get("confirmed", False),
    ])

    print(f"\nPredictions confirmed: {confirmed}/3")

    if confirmed >= 2:
        print("\n*** REVISED CONJECTURE SUPPORTED ***")
        print("Hyperbolic variance detects amino acid PROPERTY conservation.")
        print("E gene maintains biochemical properties despite AA changes.")
        print("\nImplication: Design primers targeting property-conserved epitopes.")
    elif confirmed == 1:
        print("\n*** CONJECTURE PARTIALLY SUPPORTED ***")
    else:
        print("\n*** CONJECTURE NOT SUPPORTED ***")
        print("Hyperbolic variance does not correlate with AA property conservation.")
        print("DENV-4 diversity may be truly random at the molecular level.")

    results["summary"] = {
        "predictions_confirmed": confirmed,
        "conjecture_supported": confirmed >= 2,
    }

    return results


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sequences...")
    with open(CACHE_DIR / "denv4_sequences.json") as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} sequences")

    print("Loading p-adic results...")
    with open(PADIC_RESULTS) as f:
        padic_results = json.load(f)

    # Test revised conjecture
    results = test_revised_conjecture(sequences, padic_results)

    # Save results
    results_file = RESULTS_DIR / "revised_conjecture_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
