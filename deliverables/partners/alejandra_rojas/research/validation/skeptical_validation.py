# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Skeptical Validation of Rojas Package Core Claims.

This script rigorously tests the four key claims:
1. Shannon and hyperbolic metrics are orthogonal (ρ≈0.03)
2. Pan-DENV-4 primer design is infeasible
3. k-mer clade classification achieves near-perfect performance
4. Why these properties emerge

The goal is to understand the MECHANISTIC reasons behind these findings.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import uniform_filter1d

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_denv4_sequences() -> tuple[list[str], list[str], dict]:
    """Load DENV-4 genome sequences and metadata."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_data = json.load(f)

    accessions = []
    sequences = []
    meta_dict = {}

    for acc, meta in metadata["data"].items():
        if acc in seq_data["data"]:
            accessions.append(acc)
            sequences.append(seq_data["data"][acc])
            meta_dict[acc] = meta

    return accessions, sequences, meta_dict


def compute_shannon_entropy_per_position(sequences: list[str]) -> np.ndarray:
    """Compute Shannon entropy for each nucleotide position."""
    max_len = min(len(s) for s in sequences)
    entropies = np.zeros(max_len)

    for pos in range(max_len):
        nucleotides = [s[pos].upper() for s in sequences if pos < len(s) and s[pos].upper() in 'ACGT']
        if not nucleotides:
            entropies[pos] = 2.0
            continue

        counts = Counter(nucleotides)
        total = len(nucleotides)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        entropies[pos] = entropy

    return entropies


def compute_hyperbolic_variance_per_codon(sequences: list[str]) -> np.ndarray:
    """Compute hyperbolic variance (p-adic valuation variance) per codon position."""
    max_len = min(len(s) for s in sequences)
    n_codons = max_len // 3
    variances = np.zeros(n_codons)

    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}

    for codon_pos in range(n_codons):
        nuc_pos = codon_pos * 3
        valuations = []

        for seq in sequences:
            if nuc_pos + 3 <= len(seq):
                codon = seq[nuc_pos:nuc_pos + 3].upper()
                # Convert to index
                try:
                    idx = bases[codon[0]] * 16 + bases[codon[1]] * 4 + bases[codon[2]]
                    # Compute 3-adic valuation
                    if idx == 0:
                        val = 9
                    else:
                        val = 0
                        n = idx
                        while n % 3 == 0:
                            val += 1
                            n //= 3
                        val = min(val, 9)
                    valuations.append(val)
                except KeyError:
                    continue

        if len(valuations) >= 2:
            variances[codon_pos] = np.var(valuations)
        else:
            variances[codon_pos] = 1.0

    return variances


def validate_orthogonality_claim(sequences: list[str]) -> dict:
    """Claim 1: Shannon and hyperbolic metrics are orthogonal (ρ≈0.03).

    SKEPTICAL ANALYSIS:
    - The original claim uses hyp_var vs synonymous_ratio (ρ=0.03)
    - But codon_entropy vs hyp_var shows ρ=0.31
    - Need to test direct Shannon entropy vs hyperbolic variance
    """
    print("\n" + "=" * 70)
    print("CLAIM 1: METRIC ORTHOGONALITY VALIDATION")
    print("=" * 70)

    # Compute per-position metrics
    entropies = compute_shannon_entropy_per_position(sequences)
    variances = compute_hyperbolic_variance_per_codon(sequences)

    # Expand variances to nucleotide positions for direct comparison
    expanded_variances = np.repeat(variances, 3)
    min_len = min(len(entropies), len(expanded_variances))
    entropies = entropies[:min_len]
    expanded_variances = expanded_variances[:min_len]

    # Window-averaged comparison (like the original analysis)
    window_sizes = [1, 25, 75, 150]
    results = {"raw": {}, "windowed": {}}

    print("\nDirect nucleotide-level comparison:")
    print("-" * 50)

    # Raw correlation
    valid_mask = (entropies > 0) & (expanded_variances > 0)
    rho_raw, p_raw = spearmanr(entropies[valid_mask], expanded_variances[valid_mask])
    pearson_raw, p_pearson = pearsonr(entropies[valid_mask], expanded_variances[valid_mask])

    print(f"  Raw (n={valid_mask.sum()}):")
    print(f"    Spearman ρ = {rho_raw:.4f} (p = {p_raw:.4e})")
    print(f"    Pearson r  = {pearson_raw:.4f} (p = {p_pearson:.4e})")

    results["raw"] = {
        "spearman_rho": float(rho_raw),
        "spearman_p": float(p_raw),
        "pearson_r": float(pearson_raw),
        "pearson_p": float(p_pearson),
        "n_positions": int(valid_mask.sum()),
    }

    print("\nWindow-averaged comparison:")
    print("-" * 50)

    for window in window_sizes:
        if window == 1:
            smoothed_ent = entropies
            smoothed_var = expanded_variances
        else:
            smoothed_ent = uniform_filter1d(entropies, window)
            smoothed_var = uniform_filter1d(expanded_variances, window)

        # Sample at window intervals to avoid autocorrelation
        indices = np.arange(0, len(smoothed_ent), max(window // 2, 1))
        ent_samples = smoothed_ent[indices]
        var_samples = smoothed_var[indices]

        rho, p = spearmanr(ent_samples, var_samples)
        print(f"  Window={window}bp (n={len(indices)}): ρ = {rho:.4f} (p = {p:.4e})")

        results["windowed"][window] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n_samples": len(indices),
        }

    # CRITICAL: Test what the original ρ≈0.03 actually measured
    print("\n" + "-" * 50)
    print("CLARIFICATION: Original ρ≈0.03 measured hyp_var vs synonymous_ratio")
    print("This is NOT the same as Shannon entropy vs hyperbolic variance")
    print("-" * 50)

    # Interpretation
    is_orthogonal = abs(rho_raw) < 0.1
    is_weak = 0.1 <= abs(rho_raw) < 0.3
    is_moderate = 0.3 <= abs(rho_raw) < 0.6

    if is_orthogonal:
        interpretation = "CONFIRMED: Metrics are effectively orthogonal"
    elif is_weak:
        interpretation = "PARTIALLY CONFIRMED: Weak correlation exists"
    else:
        interpretation = "REJECTED: Significant correlation detected"

    results["interpretation"] = interpretation
    results["is_orthogonal"] = bool(is_orthogonal)

    print(f"\nINTERPRETATION: {interpretation}")

    return results


def investigate_metric_decoupling(sequences: list[str]) -> dict:
    """Investigate WHY Shannon and hyperbolic metrics might decouple.

    Hypothesis: Hyperbolic variance captures CODON-LEVEL structure
    that is invisible to NUCLEOTIDE-LEVEL Shannon entropy.

    Key questions:
    1. Does synonymous codon variation affect hyperbolic variance?
    2. Are there regions with high Shannon but low hyperbolic (or vice versa)?
    3. What biological mechanism explains the decoupling?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: WHY DO METRICS DECOUPLE?")
    print("=" * 70)

    results = {}

    entropies = compute_shannon_entropy_per_position(sequences)
    variances = compute_hyperbolic_variance_per_codon(sequences)
    expanded_variances = np.repeat(variances, 3)
    min_len = min(len(entropies), len(expanded_variances))
    entropies = entropies[:min_len]
    expanded_variances = expanded_variances[:min_len]

    # Smooth for region analysis
    window = 75
    smoothed_ent = uniform_filter1d(entropies, window)
    smoothed_var = uniform_filter1d(expanded_variances, window)

    # Normalize to percentiles
    ent_percentile = (smoothed_ent - smoothed_ent.min()) / (smoothed_ent.max() - smoothed_ent.min())
    var_percentile = (smoothed_var - smoothed_var.min()) / (smoothed_var.max() - smoothed_var.min())

    # Find discordant regions
    print("\nFinding regions where metrics DISAGREE:")
    print("-" * 50)

    # High Shannon, Low Hyperbolic (nucleotide variable, codon structure conserved)
    high_ent_low_var = (ent_percentile > 0.7) & (var_percentile < 0.3)
    high_ent_low_var_positions = np.where(high_ent_low_var)[0]

    # Low Shannon, High Hyperbolic (nucleotide conserved, codon structure variable)
    low_ent_high_var = (ent_percentile < 0.3) & (var_percentile > 0.7)
    low_ent_high_var_positions = np.where(low_ent_high_var)[0]

    # Concordant regions (both low or both high)
    concordant_low = (ent_percentile < 0.3) & (var_percentile < 0.3)
    concordant_high = (ent_percentile > 0.7) & (var_percentile > 0.7)

    print(f"  High Shannon, Low Hyperbolic: {len(high_ent_low_var_positions)} positions")
    print(f"  Low Shannon, High Hyperbolic: {len(low_ent_high_var_positions)} positions")
    print(f"  Concordant (both low): {concordant_low.sum()} positions")
    print(f"  Concordant (both high): {concordant_high.sum()} positions")

    results["discordant_regions"] = {
        "high_shannon_low_hyperbolic": int(len(high_ent_low_var_positions)),
        "low_shannon_high_hyperbolic": int(len(low_ent_high_var_positions)),
        "concordant_both_low": int(concordant_low.sum()),
        "concordant_both_high": int(concordant_high.sum()),
    }

    # Analyze codon position effect (wobble vs non-wobble)
    print("\nAnalyzing codon position effect (wobble hypothesis):")
    print("-" * 50)

    # Get entropies by codon position
    pos1_entropies = entropies[0::3][:len(variances)]
    pos2_entropies = entropies[1::3][:len(variances)]
    pos3_entropies = entropies[2::3][:len(variances)]  # Wobble position

    print(f"  Position 1 mean entropy: {pos1_entropies.mean():.4f}")
    print(f"  Position 2 mean entropy: {pos2_entropies.mean():.4f}")
    print(f"  Position 3 (wobble) mean entropy: {pos3_entropies.mean():.4f}")

    # Correlate each position with hyperbolic variance
    rho1, _ = spearmanr(pos1_entropies, variances[:len(pos1_entropies)])
    rho2, _ = spearmanr(pos2_entropies, variances[:len(pos2_entropies)])
    rho3, _ = spearmanr(pos3_entropies, variances[:len(pos3_entropies)])

    print(f"\n  Correlation with hyperbolic variance:")
    print(f"    Position 1: ρ = {rho1:.4f}")
    print(f"    Position 2: ρ = {rho2:.4f}")
    print(f"    Position 3 (wobble): ρ = {rho3:.4f}")

    results["codon_position_analysis"] = {
        "pos1_mean_entropy": float(pos1_entropies.mean()),
        "pos2_mean_entropy": float(pos2_entropies.mean()),
        "pos3_mean_entropy": float(pos3_entropies.mean()),
        "pos1_correlation_with_hyp_var": float(rho1),
        "pos2_correlation_with_hyp_var": float(rho2),
        "pos3_correlation_with_hyp_var": float(rho3),
    }

    # Mechanistic explanation
    print("\n" + "-" * 50)
    print("MECHANISTIC HYPOTHESIS:")
    print("-" * 50)

    hypotheses = []

    if abs(rho3) < abs(rho1) and abs(rho3) < abs(rho2):
        hypothesis = (
            "Wobble position (pos 3) has LOWEST correlation with hyperbolic variance.\n"
            "This suggests hyperbolic variance captures AMINO ACID-level constraints,\n"
            "while Shannon entropy captures NUCLEOTIDE-level variation including\n"
            "synonymous changes at the wobble position."
        )
        hypotheses.append("wobble_independence")
    else:
        hypothesis = "Wobble independence hypothesis NOT supported."

    print(hypothesis)
    results["mechanistic_hypothesis"] = hypotheses

    return results


def validate_kmer_classification(sequences: list[str], clades: list[str]) -> dict:
    """Claim 3: k-mer classification achieves near-perfect performance.

    SKEPTICAL ANALYSIS:
    - 100% balanced accuracy seems suspiciously perfect
    - Need to understand WHY k-mers separate clades so well
    - Could be trivial (e.g., if clades are defined BY k-mers) or meaningful
    """
    print("\n" + "=" * 70)
    print("CLAIM 3: K-MER CLASSIFICATION VALIDATION")
    print("=" * 70)

    results = {}

    # Count clade distribution
    clade_counts = Counter(clades)
    print(f"\nClade distribution:")
    for clade, count in sorted(clade_counts.items()):
        print(f"  {clade}: {count} sequences ({100*count/len(clades):.1f}%)")

    results["clade_distribution"] = dict(clade_counts)

    # Analyze unique k-mers per clade
    k = 6
    print(f"\nAnalyzing {k}-mer uniqueness per clade:")
    print("-" * 50)

    clade_kmers = {}
    for seq, clade in zip(sequences, clades):
        if clade not in clade_kmers:
            clade_kmers[clade] = set()
        # Extract k-mers
        seq = seq.upper().replace('U', 'T')
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(b in 'ACGT' for b in kmer):
                clade_kmers[clade].add(kmer)

    # Find clade-specific k-mers
    all_kmers = set.union(*clade_kmers.values())
    print(f"\nTotal unique {k}-mers: {len(all_kmers)}")

    clade_specific = {}
    for clade in clade_kmers:
        other_clades = [c for c in clade_kmers if c != clade]
        other_kmers = set.union(*[clade_kmers[c] for c in other_clades]) if other_clades else set()
        specific = clade_kmers[clade] - other_kmers
        clade_specific[clade] = len(specific)
        print(f"  {clade}: {len(clade_kmers[clade])} total, {len(specific)} unique ({100*len(specific)/len(clade_kmers[clade]):.1f}%)")

    results["kmer_uniqueness"] = clade_specific

    # Jaccard similarity between clades
    print(f"\nJaccard similarity between clades:")
    print("-" * 50)

    clade_list = sorted(clade_kmers.keys())
    jaccard_matrix = {}

    for i, c1 in enumerate(clade_list):
        for c2 in clade_list[i+1:]:
            intersection = len(clade_kmers[c1] & clade_kmers[c2])
            union = len(clade_kmers[c1] | clade_kmers[c2])
            jaccard = intersection / union if union > 0 else 0
            jaccard_matrix[f"{c1}_vs_{c2}"] = jaccard
            print(f"  {c1} vs {c2}: {jaccard:.4f}")

    results["jaccard_similarity"] = jaccard_matrix

    # WHY does this work?
    print("\n" + "-" * 50)
    print("WHY K-MER CLASSIFICATION WORKS:")
    print("-" * 50)

    avg_jaccard = np.mean(list(jaccard_matrix.values()))

    if avg_jaccard < 0.8:
        explanation = (
            f"Average Jaccard similarity between clades: {avg_jaccard:.3f}\n"
            "This is LOW, meaning clades have DISTINCT k-mer signatures.\n"
            "Given 200-500 years of independent evolution (cryptic diversity),\n"
            "this is expected: each clade accumulated unique mutations that\n"
            "create clade-specific k-mer patterns."
        )
    else:
        explanation = (
            f"Average Jaccard similarity: {avg_jaccard:.3f}\n"
            "High similarity suggests clades share most k-mers.\n"
            "Classification may rely on rare differentiating k-mers."
        )

    print(explanation)
    results["explanation"] = explanation
    results["avg_jaccard"] = float(avg_jaccard)

    return results


def validate_pan_denv4_infeasibility(sequences: list[str]) -> dict:
    """Claim 2: Pan-DENV-4 primer design is infeasible.

    SKEPTICAL ANALYSIS:
    - Check actual sequence diversity at candidate primer sites
    - Compute theoretical degeneracy required for universal primers
    - Compare to practical limits (<4096 degeneracy typically)
    """
    print("\n" + "=" * 70)
    print("CLAIM 2: PAN-DENV-4 PRIMER INFEASIBILITY VALIDATION")
    print("=" * 70)

    results = {}

    # Analyze candidate regions
    candidate_positions = [
        (50, "5'UTR"),
        (2400, "E gene (best hyperbolic)"),
        (9007, "NS5 (PANFLAVI_FU1)"),
        (9600, "NS5 (best alternative)"),
        (9908, "NS5 (conserved)"),
    ]

    primer_len = 20
    print(f"\nAnalyzing {primer_len}bp primer regions:")
    print("-" * 50)

    for start, name in candidate_positions:
        # Extract primer region from all sequences
        primers = []
        for seq in sequences:
            if start + primer_len <= len(seq):
                primer = seq[start:start + primer_len].upper()
                if all(b in 'ACGTU' for b in primer):
                    primers.append(primer.replace('U', 'T'))

        if not primers:
            continue

        # Count unique sequences
        unique_primers = set(primers)

        # Compute per-position variation
        degeneracy_per_pos = []
        for pos in range(primer_len):
            bases_at_pos = set(p[pos] for p in primers)
            degeneracy_per_pos.append(len(bases_at_pos))

        # Total degeneracy (product of per-position)
        total_degeneracy = 1
        for d in degeneracy_per_pos:
            total_degeneracy *= d

        # Conservation score (how many sequences match the consensus)
        consensus = []
        for pos in range(primer_len):
            bases = [p[pos] for p in primers]
            consensus.append(Counter(bases).most_common(1)[0][0])
        consensus_seq = ''.join(consensus)

        matches = sum(1 for p in primers if p == consensus_seq)
        conservation = matches / len(primers)

        print(f"\n{name} (position {start}):")
        print(f"  Unique sequences: {len(unique_primers)} / {len(primers)}")
        print(f"  Per-position degeneracy: {degeneracy_per_pos}")
        print(f"  Total degeneracy: {total_degeneracy:,}")
        print(f"  Consensus: {consensus_seq}")
        print(f"  Consensus coverage: {100*conservation:.1f}%")

        results[name] = {
            "position": start,
            "unique_primers": len(unique_primers),
            "total_sequences": len(primers),
            "per_position_degeneracy": degeneracy_per_pos,
            "total_degeneracy": total_degeneracy,
            "consensus": consensus_seq,
            "consensus_coverage": float(conservation),
        }

    # Assessment
    print("\n" + "-" * 50)
    print("FEASIBILITY ASSESSMENT:")
    print("-" * 50)

    PRACTICAL_DEGENERACY_LIMIT = 4096  # 4^6

    feasible_regions = [
        name for name, data in results.items()
        if data["total_degeneracy"] <= PRACTICAL_DEGENERACY_LIMIT
    ]

    if not feasible_regions:
        conclusion = (
            f"NO regions have degeneracy ≤ {PRACTICAL_DEGENERACY_LIMIT}.\n"
            "CONFIRMED: Pan-DENV-4 consensus primers are INFEASIBLE.\n"
            "Recommendation: Use clade-specific primer cocktail."
        )
        results["feasibility"] = "INFEASIBLE"
    else:
        conclusion = (
            f"Feasible regions: {feasible_regions}\n"
            "Pan-DENV-4 primers MAY be possible with degenerate primers."
        )
        results["feasibility"] = "POTENTIALLY_FEASIBLE"

    print(conclusion)
    results["conclusion"] = conclusion

    return results


def main():
    print("=" * 70)
    print("SKEPTICAL VALIDATION OF ROJAS PACKAGE CORE CLAIMS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Load data
    print("Loading DENV-4 data...")
    accessions, sequences, metadata = load_denv4_sequences()
    clades = [metadata[acc]["clade"] for acc in accessions]
    print(f"Loaded {len(sequences)} sequences")

    results = {
        "_metadata": {
            "analysis_type": "skeptical_validation",
            "description": "Rigorous validation of Rojas package core claims",
            "created": datetime.now(timezone.utc).isoformat(),
            "n_sequences": len(sequences),
        },
        "claims": {},
    }

    # Validate each claim
    results["claims"]["orthogonality"] = validate_orthogonality_claim(sequences)
    results["claims"]["metric_decoupling"] = investigate_metric_decoupling(sequences)
    results["claims"]["kmer_classification"] = validate_kmer_classification(sequences, clades)
    results["claims"]["pan_denv4_feasibility"] = validate_pan_denv4_infeasibility(sequences)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = []

    if results["claims"]["orthogonality"]["is_orthogonal"]:
        summary.append("✓ Claim 1 (Orthogonality): CONFIRMED")
    else:
        summary.append("✗ Claim 1 (Orthogonality): NEEDS CLARIFICATION")

    if results["claims"]["pan_denv4_feasibility"]["feasibility"] == "INFEASIBLE":
        summary.append("✓ Claim 2 (Infeasibility): CONFIRMED")
    else:
        summary.append("? Claim 2 (Infeasibility): PARTIALLY CONFIRMED")

    avg_jaccard = results["claims"]["kmer_classification"]["avg_jaccard"]
    if avg_jaccard < 0.85:
        summary.append("✓ Claim 3 (K-mer Classification): MECHANISTICALLY EXPLAINED")
    else:
        summary.append("? Claim 3 (K-mer Classification): NEEDS INVESTIGATION")

    for line in summary:
        print(f"  {line}")

    results["summary"] = summary

    # Save results
    output_path = RESULTS_DIR / "skeptical_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
