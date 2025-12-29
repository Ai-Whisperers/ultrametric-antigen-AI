#!/usr/bin/env python3
"""
Codon Optimizer for Boundary-Safe Regenerative Sequences - HYPERBOLIC GEOMETRY

Optimizes DNA sequences to place all epitopes deep inside p-adic clusters,
maximizing distance from cluster boundaries to reduce immunogenicity.

Uses Poincaré ball (hyperbolic) geometry for boundary calculations.

Applications:
- Synoviocyte gene design for RA regeneration
- mRNA vaccine optimization
- Synthetic biology constructs

Algorithm:
1. Sliding window analysis of epitope contexts (9-15 AA)
2. Score each synonymous codon by boundary margin (Poincaré distance)
3. Greedy optimization with context awareness
4. Citrullination sensitivity screening for R positions

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
# Import hyperbolic utilities
from hyperbolic_utils import (CodonEncoder, codon_to_onehot, get_results_dir,
                              load_codon_encoder)
from hyperbolic_utils import poincare_distance as hyp_poincare_distance
from hyperbolic_utils import project_to_poincare

# ============================================================================
# GENETIC CODE
# ============================================================================

# Complete codon table
CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGA": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Reverse mapping: amino acid to list of codons
AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TABLE.items():
    AA_TO_CODONS[aa].append(codon)

# Human codon usage frequencies (per 1000 codons)
# Source: Kazusa Codon Usage Database
HUMAN_CODON_USAGE = {
    "TTT": 17.6,
    "TTC": 20.3,
    "TTA": 7.7,
    "TTG": 12.9,
    "CTT": 13.2,
    "CTC": 19.6,
    "CTA": 7.2,
    "CTG": 39.6,
    "ATT": 16.0,
    "ATC": 20.8,
    "ATA": 7.5,
    "ATG": 22.0,
    "GTT": 11.0,
    "GTC": 14.5,
    "GTA": 7.1,
    "GTG": 28.1,
    "TCT": 15.2,
    "TCC": 17.7,
    "TCA": 12.2,
    "TCG": 4.4,
    "CCT": 17.5,
    "CCC": 19.8,
    "CCA": 16.9,
    "CCG": 6.9,
    "ACT": 13.1,
    "ACC": 18.9,
    "ACA": 15.1,
    "ACG": 6.1,
    "GCT": 18.4,
    "GCC": 27.7,
    "GCA": 15.8,
    "GCG": 7.4,
    "TAT": 12.2,
    "TAC": 15.3,
    "TAA": 1.0,
    "TAG": 0.8,
    "CAT": 10.9,
    "CAC": 15.1,
    "CAA": 12.3,
    "CAG": 34.2,
    "AAT": 17.0,
    "AAC": 19.1,
    "AAA": 24.4,
    "AAG": 31.9,
    "GAT": 21.8,
    "GAC": 25.1,
    "GAA": 29.0,
    "GAG": 39.6,
    "TGT": 10.6,
    "TGC": 12.6,
    "TGA": 1.6,
    "TGG": 13.2,
    "CGT": 4.5,
    "CGC": 10.4,
    "CGA": 6.2,
    "CGG": 11.4,
    "AGT": 12.1,
    "AGC": 19.5,
    "AGA": 12.2,
    "AGG": 12.0,
    "GGT": 10.8,
    "GGC": 22.2,
    "GGA": 16.5,
    "GGG": 16.5,
}

# ============================================================================
# CODON ENCODER - Now imported from hyperbolic_utils
# CodonEncoder and codon_to_onehot are imported above
# ============================================================================


def poincare_distance(emb1, emb2, c=1.0):
    """Geodesic distance in Poincaré ball model."""
    return float(hyp_poincare_distance(emb1, emb2, c=c))


# ============================================================================
# BOUNDARY ANALYSIS - Updated for Hyperbolic Geometry
# ============================================================================


class BoundaryAnalyzer:
    """Analyzes codon embeddings relative to cluster boundaries in Poincaré ball."""

    def __init__(self, encoder: CodonEncoder, use_hyperbolic: bool = True):
        self.encoder = encoder
        self.encoder.eval()
        self.use_hyperbolic = use_hyperbolic

        # Project cluster centers to Poincaré ball
        raw_centers = encoder.cluster_centers.detach().numpy()
        if use_hyperbolic:
            self.cluster_centers = np.array([project_to_poincare(c, max_radius=0.95).squeeze() for c in raw_centers])
        else:
            self.cluster_centers = raw_centers
        self.n_clusters = len(self.cluster_centers)

        # Pre-compute all codon embeddings
        self.codon_embeddings = {}
        self.codon_clusters = {}
        self.codon_margins = {}

        for codon in CODON_TABLE.keys():
            emb = self._encode_codon(codon)
            self.codon_embeddings[codon] = emb

            # Find nearest cluster and margin (using Poincaré distance if hyperbolic)
            if use_hyperbolic:
                dists = [poincare_distance(emb, c) for c in self.cluster_centers]
            else:
                dists = [np.linalg.norm(emb - c) for c in self.cluster_centers]
            nearest = np.argmin(dists)
            second = np.partition(dists, 1)[1]

            self.codon_clusters[codon] = nearest
            self.codon_margins[codon] = second - dists[nearest]  # Margin to boundary

    def _encode_codon(self, codon: str) -> np.ndarray:
        """Encode a single codon and project to Poincaré ball if hyperbolic."""
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode(onehot).cpu().numpy().squeeze()
            if self.use_hyperbolic:
                emb = project_to_poincare(emb, max_radius=0.95).squeeze()
        return emb

    def get_codon_score(self, codon: str, weight_margin: float = 1.0, weight_usage: float = 0.3) -> float:
        """
        Score a codon for boundary safety.

        Higher score = better (safer, more central).
        """
        margin = self.codon_margins[codon]
        usage = HUMAN_CODON_USAGE.get(codon, 1.0) / 40.0  # Normalize to ~1

        # Combined score: margin (safety) + usage (expression)
        score = weight_margin * margin + weight_usage * usage
        return score

    def analyze_epitope_window(self, codons: List[str]) -> Dict:
        """
        Analyze an epitope window (typically 9-15 codons).

        Returns embedding centroid, cluster, and margin (using Poincaré distance).
        """
        embeddings = [self.codon_embeddings[c] for c in codons if c in self.codon_embeddings]
        if not embeddings:
            return {"valid": False}

        centroid = np.mean(embeddings, axis=0)

        # Find cluster and margin for centroid (using Poincaré distance if hyperbolic)
        if self.use_hyperbolic:
            dists = [poincare_distance(centroid, c) for c in self.cluster_centers]
        else:
            dists = [np.linalg.norm(centroid - c) for c in self.cluster_centers]
        nearest = np.argmin(dists)
        second = np.partition(dists, 1)[1]

        return {
            "valid": True,
            "centroid": centroid,
            "cluster": int(nearest),
            "margin": second - dists[nearest],
            "dist_to_center": dists[nearest],
        }

    def simulate_citrullination(self, codons: List[str], r_position: int) -> Dict:
        """
        Simulate citrullination at an R position.

        Returns the embedding shift and boundary crossing status (using Poincaré distance).
        """
        if r_position >= len(codons):
            return {"valid": False}

        # Original analysis
        original = self.analyze_epitope_window(codons)
        if not original["valid"]:
            return {"valid": False}

        # Modified: zero out the R position
        modified_codons = codons.copy()
        # Use neighbor average for citrullinated position
        neighbors = []
        if r_position > 0:
            neighbors.append(self.codon_embeddings[codons[r_position - 1]])
        if r_position < len(codons) - 1:
            neighbors.append(self.codon_embeddings[codons[r_position + 1]])

        if neighbors:
            modified_emb = np.mean(neighbors, axis=0)
        else:
            modified_emb = np.zeros_like(self.codon_embeddings[codons[r_position]])

        # Compute modified centroid
        embeddings = [self.codon_embeddings[c] for i, c in enumerate(codons) if i != r_position and c in self.codon_embeddings]
        embeddings.append(modified_emb)
        mod_centroid = np.mean(embeddings, axis=0)

        # Find cluster for modified (using Poincaré distance if hyperbolic)
        if self.use_hyperbolic:
            mod_dists = [poincare_distance(mod_centroid, c) for c in self.cluster_centers]
            shift = poincare_distance(original["centroid"], mod_centroid)
        else:
            mod_dists = [np.linalg.norm(mod_centroid - c) for c in self.cluster_centers]
            shift = np.linalg.norm(original["centroid"] - mod_centroid)

        mod_nearest = np.argmin(mod_dists)
        boundary_crossed = original["cluster"] != mod_nearest

        return {
            "valid": True,
            "shift": shift,
            "boundary_crossed": boundary_crossed,
            "original_cluster": original["cluster"],
            "modified_cluster": int(mod_nearest),
        }


# ============================================================================
# CODON OPTIMIZER
# ============================================================================


class CodonOptimizer:
    """
    Optimizes DNA sequences for boundary safety.

    Strategy:
    1. For each position, evaluate all synonymous codons
    2. Score by: boundary margin + codon usage + context
    3. Select optimal codons with greedy or beam search
    """

    def __init__(self, analyzer: BoundaryAnalyzer, epitope_window: int = 9):
        self.analyzer = analyzer
        self.epitope_window = epitope_window

    def optimize(
        self,
        protein_sequence: str,
        weight_margin: float = 1.0,
        weight_usage: float = 0.3,
        weight_cit_safety: float = 0.5,
        verbose: bool = False,
    ) -> Dict:
        """
        Optimize codon selection for a protein sequence.

        Args:
            protein_sequence: Amino acid sequence
            weight_margin: Weight for boundary margin
            weight_usage: Weight for human codon usage
            weight_cit_safety: Weight for citrullination resistance (R positions)
            verbose: Print progress

        Returns:
            Dictionary with optimized sequence and metrics
        """
        protein_sequence = protein_sequence.upper().replace(" ", "")
        n = len(protein_sequence)

        if verbose:
            print(f"Optimizing {n} amino acids...")

        # Initialize with most common codons
        current_codons = []
        for aa in protein_sequence:
            if aa in AA_TO_CODONS:
                # Start with highest usage codon
                codons = AA_TO_CODONS[aa]
                best = max(codons, key=lambda c: HUMAN_CODON_USAGE.get(c, 0))
                current_codons.append(best)
            else:
                current_codons.append("NNN")

        # Greedy optimization with context
        improved = True
        iteration = 0
        max_iterations = 3  # Multiple passes

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(n):
                aa = protein_sequence[i]
                if aa not in AA_TO_CODONS or aa == "*":
                    continue

                synonymous = AA_TO_CODONS[aa]
                if len(synonymous) == 1:
                    continue  # No alternatives (M, W)

                best_codon = current_codons[i]
                best_score = self._score_position(
                    current_codons,
                    i,
                    weight_margin,
                    weight_usage,
                    weight_cit_safety,
                )

                for codon in synonymous:
                    if codon == current_codons[i]:
                        continue

                    # Try this codon
                    test_codons = current_codons.copy()
                    test_codons[i] = codon

                    score = self._score_position(
                        test_codons,
                        i,
                        weight_margin,
                        weight_usage,
                        weight_cit_safety,
                    )

                    if score > best_score:
                        best_score = score
                        best_codon = codon
                        improved = True

                current_codons[i] = best_codon

            if verbose:
                print(f"  Iteration {iteration}: improved={improved}")

        # Compute final metrics
        optimized_dna = "".join(current_codons)

        # Analyze all epitope windows
        window_analyses = []
        for i in range(max(0, n - self.epitope_window + 1)):
            window = current_codons[i : i + self.epitope_window]
            analysis = self.analyzer.analyze_epitope_window(window)
            if analysis["valid"]:
                window_analyses.append(
                    {
                        "start": i,
                        "end": i + len(window),
                        "margin": analysis["margin"],
                        "cluster": analysis["cluster"],
                    }
                )

        # Find R positions and check citrullination safety
        r_positions = [i for i, aa in enumerate(protein_sequence) if aa == "R"]
        cit_analyses = []
        for r_pos in r_positions:
            # Get window around R
            start = max(0, r_pos - self.epitope_window // 2)
            end = min(n, start + self.epitope_window)
            window = current_codons[start:end]
            local_r_pos = r_pos - start

            cit = self.analyzer.simulate_citrullination(window, local_r_pos)
            if cit["valid"]:
                cit_analyses.append(
                    {
                        "position": r_pos,
                        "codon": current_codons[r_pos],
                        "shift": cit["shift"],
                        "boundary_crossed": cit["boundary_crossed"],
                    }
                )

        # Compute overall metrics
        margins = [w["margin"] for w in window_analyses]
        mean_margin = np.mean(margins) if margins else 0
        min_margin = np.min(margins) if margins else 0

        n_boundary_crossings = sum(1 for c in cit_analyses if c["boundary_crossed"])

        return {
            "protein_sequence": protein_sequence,
            "optimized_dna": optimized_dna,
            "codons": current_codons,
            "length_aa": n,
            "length_nt": len(optimized_dna),
            "metrics": {
                "mean_margin": float(mean_margin),
                "min_margin": float(min_margin),
                "n_epitope_windows": len(window_analyses),
                "n_r_positions": len(r_positions),
                "n_cit_boundary_crossings": n_boundary_crossings,
                "cit_safety_rate": (1 - (n_boundary_crossings / len(r_positions)) if r_positions else 1.0),
            },
            "epitope_windows": window_analyses,
            "citrullination_analysis": cit_analyses,
        }

    def _score_position(
        self,
        codons: List[str],
        position: int,
        weight_margin: float,
        weight_usage: float,
        weight_cit_safety: float,
    ) -> float:
        """Score a position considering context."""
        codon = codons[position]
        n = len(codons)

        # Base score: individual codon
        base_score = self.analyzer.get_codon_score(codon, weight_margin, weight_usage)

        # Context score: epitope windows containing this position
        context_score = 0
        n_windows = 0

        for start in range(
            max(0, position - self.epitope_window + 1),
            min(position + 1, n - self.epitope_window + 1),
        ):
            window = codons[start : start + self.epitope_window]
            analysis = self.analyzer.analyze_epitope_window(window)
            if analysis["valid"]:
                context_score += analysis["margin"]
                n_windows += 1

        if n_windows > 0:
            context_score /= n_windows

        # Citrullination safety for R positions
        cit_score = 0
        aa = CODON_TABLE.get(codon, "")
        if aa == "R" and weight_cit_safety > 0:
            # Check if this R position would cross boundary
            start = max(0, position - self.epitope_window // 2)
            end = min(n, start + self.epitope_window)
            window = codons[start:end]
            local_pos = position - start

            cit = self.analyzer.simulate_citrullination(window, local_pos)
            if cit["valid"]:
                # Penalize boundary crossing
                if cit["boundary_crossed"]:
                    cit_score = -1.0
                else:
                    cit_score = 0.5  # Bonus for safety

        total = base_score + context_score + weight_cit_safety * cit_score
        return total

    def compare_to_naive(self, protein_sequence: str) -> Dict:
        """
        Compare optimized sequence to naive (most common codons).
        """
        # Naive: just use most common codon for each AA
        naive_codons = []
        for aa in protein_sequence.upper():
            if aa in AA_TO_CODONS:
                codons = AA_TO_CODONS[aa]
                best = max(codons, key=lambda c: HUMAN_CODON_USAGE.get(c, 0))
                naive_codons.append(best)
            else:
                naive_codons.append("NNN")

        naive_dna = "".join(naive_codons)

        # Analyze naive
        naive_windows = []
        n = len(protein_sequence)
        for i in range(max(0, n - self.epitope_window + 1)):
            window = naive_codons[i : i + self.epitope_window]
            analysis = self.analyzer.analyze_epitope_window(window)
            if analysis["valid"]:
                naive_windows.append(analysis["margin"])

        # Get optimized
        optimized = self.optimize(protein_sequence)

        return {
            "naive_dna": naive_dna,
            "optimized_dna": optimized["optimized_dna"],
            "naive_mean_margin": (float(np.mean(naive_windows)) if naive_windows else 0),
            "optimized_mean_margin": optimized["metrics"]["mean_margin"],
            "improvement": optimized["metrics"]["mean_margin"] - (np.mean(naive_windows) if naive_windows else 0),
            "naive_cit_crossings": None,  # Would need full analysis
            "optimized_cit_crossings": optimized["metrics"]["n_cit_boundary_crossings"],
        }


# ============================================================================
# EXAMPLE SEQUENCES
# ============================================================================

# Key synovial proteins for regeneration
SYNOVIAL_PROTEINS = {
    "PRG4_fragment": {
        "name": "Lubricin (PRG4) - lubricating domain fragment",
        "sequence": "KEPAPTTTKPEPAPTTTKSPPVPAPT",
        "function": "Joint lubrication",
    },
    "HAS2_fragment": {
        "name": "Hyaluronan Synthase 2 - catalytic fragment",
        "sequence": "LLVKRWQNHLDIGFQVLDPRR",
        "function": "Hyaluronic acid synthesis",
    },
    "COL2A1_fragment": {
        "name": "Collagen II - immunogenic region",
        "sequence": "GARGLTGRPGDAGPQGKVGPS",
        "function": "Cartilage structure",
    },
    "FN1_fragment": {
        "name": "Fibronectin - RGD domain",
        "sequence": "GRGDSPKQGTWRPY",
        "function": "Cell adhesion",
    },
}


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualization(results: Dict, output_path: Path):
    """Create visualization of optimization results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Margin distribution across epitope windows
    ax1 = axes[0, 0]
    margins = [w["margin"] for w in results["epitope_windows"]]
    positions = [w["start"] for w in results["epitope_windows"]]

    ax1.bar(positions, margins, color="steelblue", alpha=0.7)
    ax1.axhline(
        y=results["metrics"]["mean_margin"],
        color="red",
        linestyle="--",
        label=f'Mean: {results["metrics"]["mean_margin"]:.3f}',
    )
    ax1.axhline(
        y=results["metrics"]["min_margin"],
        color="orange",
        linestyle=":",
        label=f'Min: {results["metrics"]["min_margin"]:.3f}',
    )
    ax1.set_xlabel("Epitope Window Start Position")
    ax1.set_ylabel("Boundary Margin")
    ax1.set_title("Boundary Safety Across Sequence")
    ax1.legend()

    # 2. Citrullination analysis
    ax2 = axes[0, 1]
    if results["citrullination_analysis"]:
        r_positions = [c["position"] for c in results["citrullination_analysis"]]
        shifts = [c["shift"] for c in results["citrullination_analysis"]]
        crossed = [c["boundary_crossed"] for c in results["citrullination_analysis"]]

        colors = ["red" if c else "green" for c in crossed]
        ax2.bar(r_positions, shifts, color=colors, alpha=0.7)
        ax2.set_xlabel("R Position")
        ax2.set_ylabel("Citrullination Shift")
        ax2.set_title("Citrullination Sensitivity at R Positions\n" "(Red=Boundary Crossed, Green=Safe)")

        # Add safety rate
        safety_rate = results["metrics"]["cit_safety_rate"]
        ax2.text(
            0.5,
            0.95,
            f"Safety Rate: {safety_rate:.1%}",
            transform=ax2.transAxes,
            ha="center",
            fontsize=12,
        )
    else:
        ax2.text(0.5, 0.5, "No R positions", ha="center", va="center")
        ax2.set_title("Citrullination Analysis")

    # 3. Codon usage vs margin tradeoff
    ax3 = axes[1, 0]
    codon_data = []
    for i, codon in enumerate(results["codons"]):
        if codon in HUMAN_CODON_USAGE:
            usage = HUMAN_CODON_USAGE[codon]
            # Find windows containing this position
            margins_at_pos = [w["margin"] for w in results["epitope_windows"] if w["start"] <= i < w["end"]]
            if margins_at_pos:
                codon_data.append((usage, np.mean(margins_at_pos)))

    if codon_data:
        usages, pos_margins = zip(*codon_data)
        ax3.scatter(usages, pos_margins, alpha=0.5, s=30)
        ax3.set_xlabel("Codon Usage Frequency (per 1000)")
        ax3.set_ylabel("Mean Margin at Position")
        ax3.set_title("Expression vs Safety Tradeoff")

    # 4. Summary metrics
    ax4 = axes[1, 1]
    ax4.axis("off")

    metrics_text = f"""
    OPTIMIZATION SUMMARY
    {'='*40}

    Sequence Length: {results['length_aa']} amino acids
                    {results['length_nt']} nucleotides

    Boundary Safety:
      Mean Margin:  {results['metrics']['mean_margin']:.4f}
      Min Margin:   {results['metrics']['min_margin']:.4f}
      Epitope Windows: {results['metrics']['n_epitope_windows']}

    Citrullination Safety:
      R Positions:  {results['metrics']['n_r_positions']}
      Boundary Crossings: {results['metrics']['n_cit_boundary_crossings']}
      Safety Rate:  {results['metrics']['cit_safety_rate']:.1%}

    DNA Sequence (first 60 nt):
    {results['optimized_dna'][:60]}...
    """

    ax4.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax4.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("CODON OPTIMIZER FOR BOUNDARY-SAFE REGENERATIVE SEQUENCES")
    print("USING HYPERBOLIC (POINCARÉ BALL) GEOMETRY")
    print("=" * 70)

    # Paths - use hyperbolic results directory
    script_dir = Path(__file__).parent
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    encoder, codon_mapping, _ = load_codon_encoder(device="cpu", version="3adic")

    # Create analyzer and optimizer (with hyperbolic geometry)
    print("Initializing boundary analyzer with Poincaré geometry...")
    analyzer = BoundaryAnalyzer(encoder, use_hyperbolic=True)
    optimizer = CodonOptimizer(analyzer, epitope_window=9)

    # Analyze pre-computed codon margins
    print("\nCodon Margin Analysis:")
    print("-" * 50)

    # Group by amino acid
    aa_margins = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        if aa != "*":
            aa_margins[aa].append((codon, analyzer.codon_margins[codon]))

    # Show amino acids with most optimization potential
    print("\nAmino acids with largest codon margin variation:")
    aa_variations = []
    for aa, codons in aa_margins.items():
        if len(codons) > 1:
            margins = [m for _, m in codons]
            variation = max(margins) - min(margins)
            aa_variations.append((aa, variation, codons))

    aa_variations.sort(key=lambda x: x[1], reverse=True)
    for aa, var, codons in aa_variations[:10]:
        best_codon = max(codons, key=lambda x: x[1])
        worst_codon = min(codons, key=lambda x: x[1])
        print(f"  {aa}: variation={var:.4f}")
        print(f"      Best: {best_codon[0]} (margin={best_codon[1]:.4f})")
        print(f"      Worst: {worst_codon[0]} (margin={worst_codon[1]:.4f})")

    # Optimize synovial proteins
    print("\n" + "=" * 70)
    print("OPTIMIZING SYNOVIAL PROTEINS")
    print("=" * 70)

    all_results = {}

    for key, protein in SYNOVIAL_PROTEINS.items():
        print(f"\n{protein['name']}")
        print(f"  Function: {protein['function']}")
        print(f"  Sequence: {protein['sequence']}")

        # Optimize
        result = optimizer.optimize(protein["sequence"], verbose=False)
        all_results[key] = result

        # Compare to naive
        comparison = optimizer.compare_to_naive(protein["sequence"])

        print("\n  Results:")
        print(f"    Length: {result['length_aa']} AA, {result['length_nt']} nt")
        print(f"    Mean Margin: {result['metrics']['mean_margin']:.4f} " f"(naive: {comparison['naive_mean_margin']:.4f})")
        print(f"    Improvement: {comparison['improvement']:.4f}")
        print(f"    R positions: {result['metrics']['n_r_positions']}")
        print(f"    Cit boundary crossings: {result['metrics']['n_cit_boundary_crossings']}")
        print(f"    Cit safety rate: {result['metrics']['cit_safety_rate']:.1%}")

        print("\n  Optimized DNA:")
        dna = result["optimized_dna"]
        for i in range(0, len(dna), 60):
            print(f"    {dna[i:i+60]}")

        # Create visualization
        vis_path = results_dir / f"codon_optimization_{key}.png"
        create_visualization(result, vis_path)
        print(f"\n  Saved visualization: {vis_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"""
    Optimized {len(SYNOVIAL_PROTEINS)} synovial proteins for boundary safety.

    Key Findings:
    """
    )

    for key, result in all_results.items():
        protein = SYNOVIAL_PROTEINS[key]
        print(
            f"    {protein['name'][:30]:30} | "
            f"Margin: {result['metrics']['mean_margin']:.3f} | "
            f"Cit Safety: {result['metrics']['cit_safety_rate']:.0%}"
        )

    print(
        """
    Applications:
    1. Use optimized DNA sequences for synthetic synoviocyte constructs
    2. These sequences minimize immunogenicity by:
       - Placing epitopes deep inside p-adic clusters
       - Reducing citrullination-induced boundary crossings
    3. Validate in HLA-transgenic mouse models before clinical use
    """
    )

    # Save all results
    output_data = {
        "proteins": {
            key: {
                "name": SYNOVIAL_PROTEINS[key]["name"],
                "function": SYNOVIAL_PROTEINS[key]["function"],
                "original_sequence": SYNOVIAL_PROTEINS[key]["sequence"],
                "optimized_dna": result["optimized_dna"],
                "metrics": result["metrics"],
            }
            for key, result in all_results.items()
        }
    }

    output_path = results_dir / "codon_optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved results to {output_path}")


if __name__ == "__main__":
    main()
