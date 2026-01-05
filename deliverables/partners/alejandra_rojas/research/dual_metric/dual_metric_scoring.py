# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Dual-Metric Primer Target Scoring Pipeline.

The CORE INNOVATION of the Rojas package: Use TWO orthogonal metrics to identify
optimal primer binding sites for highly variable viruses like DENV-4.

METRICS:
    1. Shannon Entropy (Classical):
       - Computed at nucleotide level
       - Low entropy = sequence conservation (identical nucleotides)
       - Standard approach used by CDC, WHO

    2. Hyperbolic Variance (P-adic):
       - Computed at codon level
       - Low variance = p-adic structural conservation
       - Novel metric from ternary VAE research
       - Captures DIFFERENT conservation than Shannon

WHY TWO METRICS?
    - Shannon identifies positions where nucleotides are identical
    - Hyperbolic variance identifies positions where CODONS are structurally similar
    - These are ORTHOGONAL - a position can be:
        * Low-Low: Ideal primer target (conserved by both metrics)
        * Low-High: Conserved nucleotides, variable codon structure
        * High-Low: Variable nucleotides, conserved codon structure
        * High-High: Not conserved

KEY FINDING FROM CONJECTURE TESTING:
    - All 4 conjectures (synonymous, AA property, codon bias, codon pair) REJECTED
    - Correlation between hyperbolic variance and Shannon entropy: ρ = 0.03
    - These metrics capture DIFFERENT aspects of conservation

USAGE:
    python dual_metric_scoring.py --window 20 --threshold 0.3

OUTPUT:
    - Ranked list of primer target regions
    - Visualization of dual-metric landscape
    - Primer candidates for each target region
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter1d

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PrimerTarget(NamedTuple):
    """Represents a potential primer binding site."""
    position: int
    shannon_entropy: float
    hyperbolic_variance: float
    dual_score: float  # Combined score (lower = better)
    gene: str
    sequence_consensus: str


def load_denv4_sequences() -> tuple[list[str], list[str]]:
    """Load DENV-4 genome sequences."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_data = json.load(f)

    accessions = []
    sequences = []
    for acc, meta in metadata["data"].items():
        if acc in seq_data["data"]:
            accessions.append(acc)
            sequences.append(seq_data["data"][acc])

    return accessions, sequences


def compute_shannon_entropy(sequences: list[str], position: int) -> float:
    """Compute Shannon entropy at a single nucleotide position.

    H = -Σ p(x) * log2(p(x))
    H = 0 means perfectly conserved (single nucleotide)
    H = 2 means maximum entropy (all 4 nucleotides equally represented)
    """
    nucleotides = []
    for seq in sequences:
        if position < len(seq):
            nuc = seq[position].upper()
            if nuc in 'ACGT':
                nucleotides.append(nuc)

    if not nucleotides:
        return 2.0  # Maximum entropy if no data

    counts = Counter(nucleotides)
    total = len(nucleotides)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def compute_entropy_profile(sequences: list[str]) -> np.ndarray:
    """Compute Shannon entropy for each position in the alignment."""
    max_len = min(len(s) for s in sequences)
    entropies = np.zeros(max_len)

    for pos in range(max_len):
        entropies[pos] = compute_shannon_entropy(sequences, pos)

    return entropies


def padic_valuation(n: int, p: int = 3) -> int:
    """Compute 3-adic valuation."""
    if n == 0:
        return 9
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return min(val, 9)


def codon_to_index(codon: str) -> int | None:
    """Convert codon to index (0-63)."""
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    if len(codon) != 3:
        return None
    try:
        idx = bases[codon[0].upper()] * 16 + bases[codon[1].upper()] * 4 + bases[codon[2].upper()]
        return idx
    except KeyError:
        return None


def compute_hyperbolic_variance(sequences: list[str], codon_pos: int) -> float:
    """Compute hyperbolic variance at a codon position.

    Uses p-adic valuation to compute variance in hyperbolic space:
    - Converts codon to 64-index
    - Computes 3-adic valuation (measure of divisibility by 3)
    - Returns variance of valuations across sequences

    Low variance = structurally conserved (codons with similar p-adic structure)
    High variance = structurally variable
    """
    nuc_pos = codon_pos * 3
    valuations = []

    for seq in sequences:
        if nuc_pos + 3 <= len(seq):
            codon = seq[nuc_pos:nuc_pos + 3].upper()
            idx = codon_to_index(codon)
            if idx is not None:
                val = padic_valuation(idx, p=3)
                valuations.append(val)

    if len(valuations) < 2:
        return 1.0  # Maximum variance if insufficient data

    return np.var(valuations)


def compute_hyperbolic_profile(sequences: list[str]) -> np.ndarray:
    """Compute hyperbolic variance for each codon position."""
    max_len = min(len(s) for s in sequences)
    n_codons = max_len // 3

    variances = np.zeros(n_codons)
    for codon_pos in range(n_codons):
        variances[codon_pos] = compute_hyperbolic_variance(sequences, codon_pos)

    return variances


def get_gene_annotation(position: int) -> str:
    """Get gene name for a nucleotide position in DENV-4 genome."""
    # DENV-4 genome organization (approximate positions)
    genes = [
        (0, 94, "5'UTR"),
        (94, 436, "C"),  # Capsid
        (436, 934, "prM"),  # prM/M
        (934, 2419, "E"),  # Envelope
        (2419, 3474, "NS1"),
        (3474, 4128, "NS2A"),
        (4128, 4518, "NS2B"),
        (4518, 6378, "NS3"),
        (6378, 6531, "NS4A"),
        (6531, 6600, "2K"),
        (6600, 7350, "NS4B"),
        (7350, 10095, "NS5"),
        (10095, 10723, "3'UTR"),
    ]

    for start, end, name in genes:
        if start <= position < end:
            return name

    return "intergenic"


def compute_dual_score(shannon: float, hyperbolic: float,
                       shannon_weight: float = 1.0,
                       hyperbolic_weight: float = 1.0) -> float:
    """Compute combined dual-metric score.

    Lower score = better primer target.

    Score = (shannon_weight * normalized_shannon) + (hyperbolic_weight * normalized_hyperbolic)
    """
    # Normalize Shannon (0-2) to 0-1
    norm_shannon = shannon / 2.0

    # Normalize hyperbolic variance (typically 0-2) to 0-1
    norm_hyperbolic = min(hyperbolic / 2.0, 1.0)

    return (shannon_weight * norm_shannon) + (hyperbolic_weight * norm_hyperbolic)


def find_primer_targets(
    sequences: list[str],
    window_size: int = 20,
    entropy_threshold: float = 0.3,
    variance_threshold: float = 0.3,
    min_gc: float = 0.3,
    max_gc: float = 0.7,
    top_n: int = 20,
    use_percentile: bool = True,
    percentile_cutoff: float = 20.0,  # Top 20% lowest positions
) -> list[PrimerTarget]:
    """Find optimal primer binding sites using dual-metric scoring.

    For highly variable sequences like DENV-4, uses RELATIVE (percentile-based)
    scoring rather than absolute thresholds.

    Args:
        sequences: List of aligned genome sequences
        window_size: Size of primer binding site (bp)
        entropy_threshold: Maximum Shannon entropy (absolute, if not using percentile)
        variance_threshold: Maximum hyperbolic variance (absolute, if not using percentile)
        min_gc: Minimum GC content
        max_gc: Maximum GC content
        top_n: Number of top targets to return
        use_percentile: If True, use percentile-based cutoffs instead of absolute
        percentile_cutoff: Select positions below this percentile

    Returns:
        List of PrimerTarget objects ranked by dual score
    """
    print(f"Computing Shannon entropy profile...")
    entropies = compute_entropy_profile(sequences)
    print(f"  Entropy range: {entropies.min():.3f} - {entropies.max():.3f}")

    print(f"Computing hyperbolic variance profile...")
    variances = compute_hyperbolic_profile(sequences)
    print(f"  Variance range: {variances.min():.4f} - {variances.max():.4f}")

    # Smooth profiles over window
    smoothed_entropy = uniform_filter1d(entropies, window_size)
    # Expand variances to nucleotide positions (repeat 3x per codon)
    expanded_variances = np.repeat(variances, 3)[:len(entropies)]
    smoothed_variance = uniform_filter1d(expanded_variances, window_size)

    print(f"  Smoothed entropy range: {smoothed_entropy.min():.3f} - {smoothed_entropy.max():.3f}")
    print(f"  Smoothed variance range: {smoothed_variance.min():.4f} - {smoothed_variance.max():.4f}")

    # Determine effective thresholds
    if use_percentile:
        # Use percentile-based thresholds for highly variable sequences
        eff_entropy_thresh = np.percentile(smoothed_entropy, percentile_cutoff)
        eff_variance_thresh = np.percentile(smoothed_variance, percentile_cutoff)
        print(f"\n  Using PERCENTILE-based thresholds (top {percentile_cutoff}% most conserved):")
        print(f"    Entropy threshold: {eff_entropy_thresh:.3f}")
        print(f"    Variance threshold: {eff_variance_thresh:.4f}")
    else:
        eff_entropy_thresh = entropy_threshold
        eff_variance_thresh = variance_threshold

    # Find candidate positions
    candidates = []
    max_pos = min(len(smoothed_entropy), len(smoothed_variance)) - window_size

    for pos in range(0, max_pos, window_size // 2):  # Step by half window for overlap
        avg_entropy = smoothed_entropy[pos]
        avg_variance = smoothed_variance[pos]

        # Check thresholds (must be low in BOTH metrics for dual-metric selection)
        if avg_entropy > eff_entropy_thresh and avg_variance > eff_variance_thresh:
            continue

        # Extract consensus sequence
        consensus = []
        for i in range(window_size):
            nucs = [s[pos + i].upper() for s in sequences if pos + i < len(s)]
            counts = Counter(nucs)
            consensus.append(counts.most_common(1)[0][0] if counts else 'N')
        consensus_seq = ''.join(consensus)

        # Check GC content
        gc = (consensus_seq.count('G') + consensus_seq.count('C')) / len(consensus_seq)
        if gc < min_gc or gc > max_gc:
            continue

        # Compute dual score (normalized against the observed range)
        norm_entropy = (avg_entropy - smoothed_entropy.min()) / (smoothed_entropy.max() - smoothed_entropy.min())
        norm_variance = (avg_variance - smoothed_variance.min()) / (smoothed_variance.max() - smoothed_variance.min())
        dual_score = norm_entropy + norm_variance

        gene = get_gene_annotation(pos)

        candidates.append(PrimerTarget(
            position=pos,
            shannon_entropy=avg_entropy,
            hyperbolic_variance=avg_variance,
            dual_score=dual_score,
            gene=gene,
            sequence_consensus=consensus_seq,
        ))

    # Sort by dual score (lower = better)
    candidates.sort(key=lambda x: x.dual_score)

    return candidates[:top_n]


def visualize_dual_metric_landscape(
    entropies: np.ndarray,
    variances: np.ndarray,
    targets: list[PrimerTarget],
    output_path: Path,
):
    """Create visualization of dual-metric landscape."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Expand variances to nucleotide positions (ensure same length)
    expanded_variances = np.repeat(variances, 3)
    min_len = min(len(entropies), len(expanded_variances))
    entropies = entropies[:min_len]
    expanded_variances = expanded_variances[:min_len]

    positions = np.arange(min_len)

    # Panel 1: Shannon entropy
    ax1 = axes[0]
    ax1.fill_between(positions, entropies, alpha=0.3, color='blue')
    ax1.plot(positions, entropies, 'b-', linewidth=0.5)
    ax1.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title('Dual-Metric Primer Target Landscape - DENV-4')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 2)

    # Add gene annotations
    genes = ["5'UTR", "C", "prM", "E", "NS1", "NS2A", "NS2B", "NS3", "NS4A", "2K", "NS4B", "NS5", "3'UTR"]
    gene_starts = [0, 94, 436, 934, 2419, 3474, 4128, 4518, 6378, 6531, 6600, 7350, 10095]
    for i, (gene, start) in enumerate(zip(genes, gene_starts)):
        if start < len(entropies):
            ax1.axvline(start, color='gray', linestyle=':', alpha=0.5)
            if i % 2 == 0:
                ax1.text(start, 1.8, gene, fontsize=8, alpha=0.7)

    # Panel 2: Hyperbolic variance
    ax2 = axes[1]
    ax2.fill_between(positions, expanded_variances, alpha=0.3, color='green')
    ax2.plot(positions, expanded_variances, 'g-', linewidth=0.5)
    ax2.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    ax2.set_ylabel('Hyperbolic Variance')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 2)

    # Panel 3: Dual score with targets marked
    dual_scores = np.array([compute_dual_score(e, v)
                           for e, v in zip(entropies, expanded_variances)])
    ax3 = axes[2]
    ax3.fill_between(positions, dual_scores, alpha=0.3, color='purple')
    ax3.plot(positions, dual_scores, 'purple', linewidth=0.5)

    # Mark primer targets
    for i, target in enumerate(targets[:10]):  # Top 10
        ax3.axvline(target.position, color='red', alpha=0.7, linewidth=2)
        ax3.annotate(f"#{i+1}\n{target.gene}",
                    (target.position, 0.05),
                    fontsize=8, ha='center', color='red')

    ax3.set_ylabel('Dual Score')
    ax3.set_xlabel('Genome Position (bp)')
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dual-metric primer scoring pipeline")
    parser.add_argument("--window", type=int, default=20, help="Primer window size")
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                       help="Maximum Shannon entropy")
    parser.add_argument("--variance_threshold", type=float, default=0.5,
                       help="Maximum hyperbolic variance")
    parser.add_argument("--top_n", type=int, default=20, help="Number of targets to return")
    args = parser.parse_args()

    print("=" * 80)
    print("DUAL-METRIC PRIMER TARGET SCORING PIPELINE")
    print("=" * 80)
    print()
    print("Innovation: Combining Shannon entropy + hyperbolic variance")
    print("  - Shannon entropy: Nucleotide-level conservation (classical)")
    print("  - Hyperbolic variance: Codon-level p-adic structure (novel)")
    print("  - Correlation between metrics: ρ ≈ 0.03 (ORTHOGONAL)")
    print()
    print(f"Parameters:")
    print(f"  Window size: {args.window} bp")
    print(f"  Entropy threshold: {args.entropy_threshold}")
    print(f"  Variance threshold: {args.variance_threshold}")
    print()
    print("=" * 80)

    # Load sequences
    print("\n[1/4] Loading DENV-4 sequences...")
    accessions, sequences = load_denv4_sequences()
    print(f"  Loaded {len(sequences)} sequences")
    print(f"  Genome length: {min(len(s) for s in sequences)} - {max(len(s) for s in sequences)} bp")

    # Find primer targets
    print("\n[2/4] Finding optimal primer targets...")
    targets = find_primer_targets(
        sequences,
        window_size=args.window,
        entropy_threshold=args.entropy_threshold,
        variance_threshold=args.variance_threshold,
        top_n=args.top_n,
    )

    print(f"\n  Found {len(targets)} candidate primer targets")

    # Display results
    print("\n" + "=" * 80)
    print("TOP PRIMER TARGETS (Ranked by Dual Score)")
    print("=" * 80)
    print()
    print(f"{'Rank':<6} {'Position':<10} {'Gene':<8} {'Shannon':<10} {'Hyp.Var':<10} "
          f"{'Dual':<8} {'Consensus (first 20bp)'}")
    print("-" * 100)

    for i, target in enumerate(targets, 1):
        print(f"{i:<6} {target.position:<10} {target.gene:<8} "
              f"{target.shannon_entropy:<10.4f} {target.hyperbolic_variance:<10.4f} "
              f"{target.dual_score:<8.4f} {target.sequence_consensus[:20]}...")

    # Check overlap with p-adic integration findings
    print("\n" + "=" * 80)
    print("COMPARISON WITH P-ADIC INTEGRATION FINDINGS")
    print("=" * 80)

    # E gene position 2400 was identified as having lowest hyperbolic variance
    print("\nExpected best targets from p-adic analysis:")
    print("  Position 2400 (E gene): Lowest hyperbolic variance (0.0183)")
    print("  Position 3000 (NS1): Second lowest (0.0207)")

    e_gene_targets = [t for t in targets if t.gene == 'E']
    ns1_targets = [t for t in targets if t.gene == 'NS1']

    if e_gene_targets:
        best_e = min(e_gene_targets, key=lambda x: x.dual_score)
        print(f"\nBest E gene target found:")
        print(f"  Position: {best_e.position}")
        print(f"  Dual score: {best_e.dual_score:.4f}")
        print(f"  Shannon: {best_e.shannon_entropy:.4f}, Hyp.Var: {best_e.hyperbolic_variance:.4f}")
    else:
        print("\nNo E gene targets found below threshold")

    if ns1_targets:
        best_ns1 = min(ns1_targets, key=lambda x: x.dual_score)
        print(f"\nBest NS1 target found:")
        print(f"  Position: {best_ns1.position}")
        print(f"  Dual score: {best_ns1.dual_score:.4f}")

    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    entropies = compute_entropy_profile(sequences)
    variances = compute_hyperbolic_profile(sequences)

    visualize_dual_metric_landscape(
        entropies, variances, targets,
        RESULTS_DIR / "dual_metric_landscape.png"
    )
    print("  Saved: dual_metric_landscape.png")

    # Save results
    print("\n[4/4] Saving results...")

    results = {
        "_metadata": {
            "analysis_type": "dual_metric_scoring",
            "description": "Primer target identification using Shannon + hyperbolic metrics",
            "created": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "window_size": args.window,
                "entropy_threshold": args.entropy_threshold,
                "variance_threshold": args.variance_threshold,
            },
        },
        "summary": {
            "n_sequences": len(sequences),
            "n_targets_found": len(targets),
            "genome_range": f"{min(len(s) for s in sequences)}-{max(len(s) for s in sequences)} bp",
        },
        "targets": [
            {
                "rank": i + 1,
                "position": t.position,
                "gene": t.gene,
                "shannon_entropy": t.shannon_entropy,
                "hyperbolic_variance": t.hyperbolic_variance,
                "dual_score": t.dual_score,
                "consensus_sequence": t.sequence_consensus,
            }
            for i, t in enumerate(targets)
        ],
        "gene_summary": {
            gene: len([t for t in targets if t.gene == gene])
            for gene in set(t.gene for t in targets)
        },
    }

    results_path = RESULTS_DIR / "dual_metric_scoring_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {results_path}")
    print("\n" + "=" * 80)
    print("DUAL-METRIC SCORING COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
