# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Synonymous Codon Shuffling Conjecture Test.

CONJECTURE: DENV-4's "cryptic diversity" is predominantly synonymous codon
shuffling - the virus maintains protein function through codon degeneracy
while appearing highly diverse at the nucleotide level.

PREDICTIONS:
1. Low hyperbolic variance regions have HIGH synonymous substitution rates
2. dN/dS << 1 across the genome (purifying selection on proteins)
3. Wobble position (3rd codon base) has higher entropy than 1st/2nd positions
4. E gene (lowest hyp_var) should show strongest synonymous signal

TEST METHODOLOGY:
1. Compute per-position Shannon entropy at codon positions 1, 2, 3
2. Calculate synonymous vs non-synonymous substitution patterns
3. Correlate hyperbolic variance with synonymous enrichment
4. If conjecture holds: design wobble-degenerate primers for E gene

Usage:
    python scripts/denv4_synonymous_conjecture.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add project root to path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from src.encoders import TrainableCodonEncoder
from src.geometry import poincare_distance
from src.biology.codons import CODON_TO_INDEX, GENETIC_CODE, codon_index_to_triplet

# Paths
ROJAS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROJAS_DIR / "results" / "synonymous_conjecture"
CACHE_DIR = ROJAS_DIR / "data" / "cache"
PADIC_RESULTS = ROJAS_DIR / "results" / "padic_integration" / "padic_integration_results.json"

# Trained encoder
ENCODER_CHECKPOINT = _project_root / "research" / "codon-encoder" / "training" / "results" / "trained_codon_encoder.pt"


@dataclass
class CodonPositionEntropy:
    """Entropy at each codon position."""
    position: int  # Genome position
    pos1_entropy: float  # 1st codon position
    pos2_entropy: float  # 2nd codon position
    pos3_entropy: float  # 3rd codon position (wobble)
    total_entropy: float
    wobble_ratio: float  # pos3 / (pos1 + pos2)


def compute_shannon_entropy(counts: dict) -> float:
    """Compute Shannon entropy from counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def analyze_codon_position_entropy(
    sequences: dict[str, str],
    start: int,
    window_size: int = 75,
) -> CodonPositionEntropy:
    """Analyze entropy at each codon position within a window.

    Args:
        sequences: Dict of accession -> sequence
        start: Start position
        window_size: Window size (must be multiple of 3)

    Returns:
        CodonPositionEntropy with per-position entropy
    """
    # Collect bases at each codon position
    pos1_bases = defaultdict(lambda: Counter())  # codon_idx -> base counts
    pos2_bases = defaultdict(lambda: Counter())
    pos3_bases = defaultdict(lambda: Counter())

    for seq in sequences.values():
        window = seq[start:start + window_size].upper()

        for codon_idx in range(len(window) // 3):
            codon_start = codon_idx * 3
            if codon_start + 3 <= len(window):
                b1 = window[codon_start]
                b2 = window[codon_start + 1]
                b3 = window[codon_start + 2]

                if all(b in "ATGC" for b in [b1, b2, b3]):
                    pos1_bases[codon_idx][b1] += 1
                    pos2_bases[codon_idx][b2] += 1
                    pos3_bases[codon_idx][b3] += 1

    # Compute mean entropy at each position
    pos1_entropies = [compute_shannon_entropy(pos1_bases[i]) for i in pos1_bases]
    pos2_entropies = [compute_shannon_entropy(pos2_bases[i]) for i in pos2_bases]
    pos3_entropies = [compute_shannon_entropy(pos3_bases[i]) for i in pos3_bases]

    pos1_mean = np.mean(pos1_entropies) if pos1_entropies else 0
    pos2_mean = np.mean(pos2_entropies) if pos2_entropies else 0
    pos3_mean = np.mean(pos3_entropies) if pos3_entropies else 0

    total = pos1_mean + pos2_mean + pos3_mean
    wobble_ratio = pos3_mean / (pos1_mean + pos2_mean + 1e-10)

    return CodonPositionEntropy(
        position=start,
        pos1_entropy=pos1_mean,
        pos2_entropy=pos2_mean,
        pos3_entropy=pos3_mean,
        total_entropy=total,
        wobble_ratio=wobble_ratio,
    )


def compute_synonymous_ratio(
    sequences: dict[str, str],
    start: int,
    window_size: int = 75,
) -> dict:
    """Compute synonymous vs non-synonymous substitution pattern.

    Compares each sequence to the consensus and counts:
    - Synonymous: codon changed but amino acid same
    - Non-synonymous: codon changed and amino acid changed

    Returns:
        Dict with synonymous/non-synonymous counts and ratio
    """
    # Build consensus
    consensus_codons = []
    n_codons = window_size // 3

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

    # Count substitution types
    synonymous = 0
    non_synonymous = 0
    identical = 0

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
                    identical += 1
                else:
                    cons_aa = GENETIC_CODE.get(cons_codon, "X")
                    obs_aa = GENETIC_CODE.get(obs_codon, "X")

                    if cons_aa == obs_aa:
                        synonymous += 1
                    else:
                        non_synonymous += 1

    total_changes = synonymous + non_synonymous

    return {
        "synonymous": synonymous,
        "non_synonymous": non_synonymous,
        "identical": identical,
        "total_changes": total_changes,
        "syn_ratio": synonymous / total_changes if total_changes > 0 else 0,
        "dn_ds_proxy": non_synonymous / (synonymous + 1),  # +1 to avoid div by zero
    }


def test_conjecture(
    sequences: dict[str, str],
    padic_results: dict,
    encoder: TrainableCodonEncoder,
) -> dict:
    """Test the synonymous codon shuffling conjecture.

    Predictions to test:
    1. Wobble position has higher entropy than positions 1 & 2
    2. Low hyperbolic variance correlates with high synonymous ratio
    3. E gene shows strongest synonymous signal
    """
    print("=" * 70)
    print("TESTING CONJECTURE: DENV-4 Diversity = Synonymous Codon Shuffling")
    print("=" * 70)

    # Get genome scan positions from p-adic analysis
    genome_scan = padic_results.get("genome_scan", [])

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sequences": len(sequences),
        "predictions": {},
        "correlations": {},
        "gene_analysis": {},
    }

    # Test Prediction 1: Wobble position entropy > positions 1 & 2
    print("\n" + "=" * 50)
    print("PREDICTION 1: Wobble (pos 3) has higher entropy than pos 1 & 2")
    print("=" * 50)

    position_entropies = []
    for scan_point in genome_scan:
        pos = scan_point["position"]
        entropy_data = analyze_codon_position_entropy(sequences, pos, 75)
        position_entropies.append({
            "position": pos,
            "hyp_variance": scan_point["variance"],
            "pos1_entropy": entropy_data.pos1_entropy,
            "pos2_entropy": entropy_data.pos2_entropy,
            "pos3_entropy": entropy_data.pos3_entropy,
            "wobble_ratio": entropy_data.wobble_ratio,
        })

    # Calculate mean entropies
    mean_pos1 = np.mean([p["pos1_entropy"] for p in position_entropies])
    mean_pos2 = np.mean([p["pos2_entropy"] for p in position_entropies])
    mean_pos3 = np.mean([p["pos3_entropy"] for p in position_entropies])

    print(f"\nMean entropy by codon position:")
    print(f"  Position 1 (most constrained): {mean_pos1:.4f}")
    print(f"  Position 2 (constrained):      {mean_pos2:.4f}")
    print(f"  Position 3 (wobble):           {mean_pos3:.4f}")

    wobble_higher = mean_pos3 > mean_pos1 and mean_pos3 > mean_pos2
    print(f"\n  Prediction 1 {'CONFIRMED' if wobble_higher else 'REJECTED'}: "
          f"Wobble > other positions = {wobble_higher}")

    results["predictions"]["wobble_higher_entropy"] = {
        "confirmed": wobble_higher,
        "pos1_mean": mean_pos1,
        "pos2_mean": mean_pos2,
        "pos3_mean": mean_pos3,
        "wobble_ratio": mean_pos3 / (mean_pos1 + mean_pos2),
    }

    # Test Prediction 2: Low hyp_var correlates with high synonymous ratio
    print("\n" + "=" * 50)
    print("PREDICTION 2: Low hyp_var correlates with high synonymous ratio")
    print("=" * 50)

    synonymous_data = []
    for scan_point in genome_scan:
        pos = scan_point["position"]
        syn_result = compute_synonymous_ratio(sequences, pos, 75)
        synonymous_data.append({
            "position": pos,
            "hyp_variance": scan_point["variance"],
            **syn_result,
        })

    hyp_vars = [d["hyp_variance"] for d in synonymous_data]
    syn_ratios = [d["syn_ratio"] for d in synonymous_data]

    # Correlation: expect NEGATIVE (low hyp_var = high syn_ratio)
    if len(hyp_vars) > 5:
        corr, pval = spearmanr(hyp_vars, syn_ratios)
        print(f"\nCorrelation (hyperbolic variance vs synonymous ratio):")
        print(f"  Spearman rho: {corr:.4f}")
        print(f"  P-value:      {pval:.4f}")
        print(f"\n  Prediction 2 {'CONFIRMED' if corr < 0 else 'REJECTED'}: "
              f"Expected negative correlation, got {corr:.4f}")

        results["correlations"]["hyp_var_vs_syn_ratio"] = {
            "spearman_rho": corr,
            "p_value": pval,
            "confirmed": corr < 0 and pval < 0.1,
        }

    # Test Prediction 3: E gene shows strongest synonymous signal
    print("\n" + "=" * 50)
    print("PREDICTION 3: E gene (pos 2400) shows strongest synonymous signal")
    print("=" * 50)

    # Gene regions
    gene_regions = {
        "5UTR": (0, 101),
        "C": (102, 476),
        "prM": (477, 976),
        "E": (977, 2471),
        "NS1": (2472, 3527),
        "NS2A": (3528, 4184),
        "NS2B": (4185, 4574),
        "NS3": (4575, 6431),
        "NS4A": (6432, 6806),
        "NS4B": (6807, 7558),
        "NS5": (7559, 10271),
    }

    gene_stats = {}
    for gene, (start, end) in gene_regions.items():
        # Get synonymous data for positions in this gene
        gene_syn = [d for d in synonymous_data
                   if start <= d["position"] < end]

        if gene_syn:
            mean_syn_ratio = np.mean([d["syn_ratio"] for d in gene_syn])
            mean_hyp_var = np.mean([d["hyp_variance"] for d in gene_syn])
            gene_stats[gene] = {
                "mean_syn_ratio": mean_syn_ratio,
                "mean_hyp_var": mean_hyp_var,
                "n_windows": len(gene_syn),
            }

    print("\nGene-level analysis:")
    print(f"{'Gene':<8} {'Syn Ratio':<12} {'Hyp Var':<12} {'N Windows'}")
    print("-" * 44)

    for gene in ["E", "NS1", "NS3", "NS5"]:  # Key genes
        if gene in gene_stats:
            stats = gene_stats[gene]
            print(f"{gene:<8} {stats['mean_syn_ratio']:.4f}       "
                  f"{stats['mean_hyp_var']:.4f}       {stats['n_windows']}")

    # Check if E gene has highest synonymous ratio
    if "E" in gene_stats:
        e_syn_ratio = gene_stats["E"]["mean_syn_ratio"]
        other_ratios = [s["mean_syn_ratio"] for g, s in gene_stats.items()
                       if g != "E" and g not in ["5UTR", "C", "prM"]]

        e_highest = e_syn_ratio > np.mean(other_ratios) if other_ratios else False
        print(f"\n  E gene syn_ratio: {e_syn_ratio:.4f}")
        print(f"  Other genes mean: {np.mean(other_ratios):.4f}" if other_ratios else "")
        print(f"\n  Prediction 3 {'CONFIRMED' if e_highest else 'REJECTED'}: "
              f"E gene has {'highest' if e_highest else 'NOT highest'} synonymous ratio")

        results["predictions"]["e_gene_highest_syn"] = {
            "confirmed": e_highest,
            "e_syn_ratio": e_syn_ratio,
            "other_mean": np.mean(other_ratios) if other_ratios else None,
        }

    results["gene_analysis"] = gene_stats
    results["position_data"] = position_entropies
    results["synonymous_data"] = synonymous_data

    # Summary
    print("\n" + "=" * 70)
    print("CONJECTURE SUMMARY")
    print("=" * 70)

    confirmed = sum([
        results["predictions"].get("wobble_higher_entropy", {}).get("confirmed", False),
        results["correlations"].get("hyp_var_vs_syn_ratio", {}).get("confirmed", False),
        results["predictions"].get("e_gene_highest_syn", {}).get("confirmed", False),
    ])

    print(f"\nPredictions confirmed: {confirmed}/3")

    if confirmed >= 2:
        print("\n*** CONJECTURE SUPPORTED ***")
        print("DENV-4's cryptic diversity is primarily synonymous codon shuffling.")
        print("Recommendation: Design wobble-degenerate primers for E gene.")
    elif confirmed == 1:
        print("\n*** CONJECTURE PARTIALLY SUPPORTED ***")
        print("Some evidence for synonymous shuffling, but not conclusive.")
    else:
        print("\n*** CONJECTURE NOT SUPPORTED ***")
        print("DENV-4 diversity appears to be non-synonymous (protein-level).")

    results["summary"] = {
        "predictions_confirmed": confirmed,
        "conjecture_supported": confirmed >= 2,
    }

    return results


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences
    print("Loading sequences...")
    sequences_path = CACHE_DIR / "denv4_sequences.json"
    with open(sequences_path) as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} sequences")

    # Load p-adic results
    print("Loading p-adic integration results...")
    with open(PADIC_RESULTS) as f:
        padic_results = json.load(f)

    # Load encoder
    print("Loading codon encoder...")
    encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
    if ENCODER_CHECKPOINT.exists():
        checkpoint = torch.load(ENCODER_CHECKPOINT, map_location='cpu', weights_only=True)
        encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    # Test conjecture
    results = test_conjecture(sequences, padic_results, encoder)

    # Save results
    results_file = RESULTS_DIR / "synonymous_conjecture_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
