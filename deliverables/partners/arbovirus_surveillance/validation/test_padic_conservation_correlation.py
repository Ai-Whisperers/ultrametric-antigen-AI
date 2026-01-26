# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic Conservation Correlation Analysis.

This script tests whether p-adic valuation (learned by our trained VAE)
correlates with sequence conservation across Dengue serotypes.

Hypothesis: Conserved positions in primer binding sites should have
lower p-adic valuation (closer to center of Poincaré ball = more fundamental).

Scientific Questions:
1. Do conserved nucleotide positions have different p-adic radii?
2. Does the TrainableCodonEncoder capture conservation patterns?
3. Can p-adic geometry predict primer binding site quality?
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import datetime

import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

# Try to import our trained encoder
try:
    import torch
    from src.encoders import TrainableCodonEncoder
    from src.biology.codons import CODON_TO_INDEX, codon_index_to_triplet
    from src.geometry import poincare_distance
    ENCODER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Cannot import TrainableCodonEncoder: {e}")
    ENCODER_AVAILABLE = False
    CODON_TO_INDEX = {}


# Global cache for encoder embeddings
_CACHED_EMBEDDINGS = None
_CACHED_RADII = None


@dataclass
class PositionAnalysis:
    """Analysis of a single nucleotide position."""
    position: int
    consensus_base: str
    entropy: float
    codon_context: str
    padic_radius: float
    padic_valuation: int


@dataclass
class PrimerAnalysis:
    """Analysis of primer binding site conservation vs p-adic structure."""
    primer_name: str
    serotype: str
    n_positions: int
    positions: list[dict]
    entropy_radius_correlation: float
    entropy_radius_pvalue: float
    entropy_valuation_correlation: float
    entropy_valuation_pvalue: float
    interpretation: str


def get_codon_at_position(sequence: str, position: int) -> Optional[str]:
    """Extract the codon context around a nucleotide position.

    For primer binding sites, we don't have reading frame information,
    so we'll use overlapping triplets (pos-1:pos+2, pos:pos+3).
    """
    if position < 0 or position >= len(sequence):
        return None

    # Get 3 bases starting at position (or as many as available)
    start = position
    end = min(position + 3, len(sequence))
    return sequence[start:end].upper()


def compute_entropy(column: list[str]) -> float:
    """Shannon entropy for a nucleotide column."""
    valid = [b for b in column if b in "ACGT"]
    if len(valid) == 0:
        return 2.0

    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def get_consensus(column: list[str]) -> str:
    """Get consensus base for a column."""
    valid = [b for b in column if b in "ACGT"]
    if not valid:
        return "N"
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def load_trained_encoder(checkpoint_path: Path) -> Optional["TrainableCodonEncoder"]:
    """Load the trained codon encoder."""
    if not ENCODER_AVAILABLE:
        return None

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(checkpoint["model_state_dict"])
        encoder.eval()
        return encoder
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return None


def get_padic_properties(encoder, codon: str) -> tuple[float, int]:
    """Get p-adic radius and valuation for a codon.

    Returns (radius, valuation) tuple.
    """
    global _CACHED_EMBEDDINGS, _CACHED_RADII

    if encoder is None or len(codon) != 3:
        return (0.5, 0)  # Default values

    try:
        # Get embedding for this codon
        codon = codon.upper().replace('U', 'T')
        if 'N' in codon or codon not in CODON_TO_INDEX:
            return (0.5, 0)

        # Cache all embeddings on first call
        if _CACHED_EMBEDDINGS is None:
            with torch.no_grad():
                _CACHED_EMBEDDINGS = encoder.encode_all()  # (64, latent_dim)
                _CACHED_RADII = encoder.get_hyperbolic_radii(_CACHED_EMBEDDINGS)

        # Get index and retrieve cached values
        idx = CODON_TO_INDEX[codon]
        radius = _CACHED_RADII[idx].item()

        # Approximate valuation from radius
        # In p-adic: higher valuation = closer to center = smaller radius
        if radius > 0.85:
            valuation = 0
        elif radius > 0.70:
            valuation = 1
        elif radius > 0.55:
            valuation = 2
        elif radius > 0.40:
            valuation = 3
        elif radius > 0.25:
            valuation = 4
        else:
            valuation = 5 + int((0.25 - radius) / 0.05)

        return (radius, valuation)
    except Exception as e:
        print(f"Error getting p-adic properties for {codon}: {e}")
        return (0.5, 0)


def analyze_primer_conservation(
    primer_name: str,
    serotype: str,
    aligned_sequences: list[str],
    binding_start: int,
    primer_length: int,
    encoder,
) -> Optional[PrimerAnalysis]:
    """Analyze conservation vs p-adic structure at primer binding site."""

    if len(aligned_sequences) < 5:
        return None

    positions = []
    entropies = []
    radii = []
    valuations = []

    for i in range(primer_length):
        abs_pos = binding_start + i

        # Get column from alignment
        column = []
        for seq in aligned_sequences:
            if abs_pos < len(seq):
                column.append(seq[abs_pos])

        if len(column) < 5:
            continue

        # Compute entropy
        entropy = compute_entropy(column)
        consensus = get_consensus(column)

        # Get codon context (take first sequence as reference for codon extraction)
        codon = get_codon_at_position(aligned_sequences[0], abs_pos)
        if codon and len(codon) == 3:
            radius, valuation = get_padic_properties(encoder, codon)
        else:
            radius, valuation = 0.5, 0

        pos_analysis = PositionAnalysis(
            position=i,
            consensus_base=consensus,
            entropy=entropy,
            codon_context=codon or "NNN",
            padic_radius=radius,
            padic_valuation=valuation,
        )
        positions.append(asdict(pos_analysis))
        entropies.append(entropy)
        radii.append(radius)
        valuations.append(valuation)

    if len(positions) < 5:
        return None

    # Compute correlations
    # Hypothesis: Lower entropy (conserved) should correlate with lower radius (central)
    # OR higher valuation (more "fundamental" in p-adic sense)

    ent_rad_corr, ent_rad_p = spearmanr(entropies, radii)
    ent_val_corr, ent_val_p = spearmanr(entropies, valuations)

    # Interpret results
    if ent_rad_corr > 0.3 and ent_rad_p < 0.05:
        interpretation = "SUPPORTED: Conserved positions have lower p-adic radius (more fundamental)"
    elif ent_val_corr < -0.3 and ent_val_p < 0.05:
        interpretation = "SUPPORTED: Conserved positions have higher p-adic valuation"
    elif abs(ent_rad_corr) < 0.1 and abs(ent_val_corr) < 0.1:
        interpretation = "NO SIGNAL: P-adic structure independent of conservation"
    else:
        interpretation = f"WEAK/MIXED: Entropy-radius ρ={ent_rad_corr:.2f}, entropy-valuation ρ={ent_val_corr:.2f}"

    return PrimerAnalysis(
        primer_name=primer_name,
        serotype=serotype,
        n_positions=len(positions),
        positions=positions,
        entropy_radius_correlation=float(ent_rad_corr) if not math.isnan(ent_rad_corr) else 0.0,
        entropy_radius_pvalue=float(ent_rad_p) if not math.isnan(ent_rad_p) else 1.0,
        entropy_valuation_correlation=float(ent_val_corr) if not math.isnan(ent_val_corr) else 0.0,
        entropy_valuation_pvalue=float(ent_val_p) if not math.isnan(ent_val_p) else 1.0,
        interpretation=interpretation,
    )


def run_analysis(cache_dir: Path) -> dict:
    """Run complete p-adic conservation correlation analysis."""

    print("=" * 70)
    print("P-ADIC CONSERVATION CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    print("Testing whether p-adic structure correlates with sequence conservation")
    print()

    # Load strain data
    strain_file = cache_dir / "dengue_strains.json"
    if not strain_file.exists():
        print("ERROR: dengue_strains.json not found. Run test_dengue_strain_variation.py first.")
        return {"error": "Missing strain data"}

    with open(strain_file) as f:
        all_sequences = json.load(f)

    # Load encoder
    encoder_path = project_root / "research" / "codon-encoder" / "training" / "results" / "trained_codon_encoder.pt"
    encoder = load_trained_encoder(encoder_path)

    if encoder:
        print(f"Loaded TrainableCodonEncoder from {encoder_path.name}")
    else:
        print("WARNING: Using approximate p-adic radii (encoder not loaded)")
    print()

    # Primer binding positions
    primer_positions = {
        "CDC_DENV1": {"forward": 8972, "reverse": 9059, "len_f": 20, "len_r": 25},
        "CDC_DENV2": {"forward": 141, "reverse": 833, "len_f": 20, "len_r": 20},
        "CDC_DENV3": {"forward": 9192, "reverse": 1129, "len_f": 22, "len_r": 22},
        "CDC_DENV4": {"forward": 903, "reverse": 972, "len_f": 21, "len_r": 20},
    }

    serotype_map = {
        "CDC_DENV1": "DENV-1",
        "CDC_DENV2": "DENV-2",
        "CDC_DENV3": "DENV-3",
        "CDC_DENV4": "DENV-4",
    }

    results = []

    print("-" * 70)
    print("ANALYZING PRIMER BINDING SITES")
    print("-" * 70)
    print()

    for primer_key, positions in primer_positions.items():
        serotype = serotype_map[primer_key]
        sequences = all_sequences.get(serotype, [])

        # Convert from list of lists if needed
        if sequences and isinstance(sequences[0], list):
            sequences = [s[1] for s in sequences]  # Get sequence strings

        if len(sequences) < 5:
            print(f"{serotype}: Insufficient sequences ({len(sequences)})")
            continue

        print(f"{serotype}:")

        # Analyze forward primer
        result_f = analyze_primer_conservation(
            primer_name=f"{primer_key}_forward",
            serotype=serotype,
            aligned_sequences=sequences,
            binding_start=positions["forward"],
            primer_length=positions["len_f"],
            encoder=encoder,
        )
        if result_f:
            results.append(result_f)
            print(f"  Forward: ρ(entropy,radius)={result_f.entropy_radius_correlation:.3f} "
                  f"(p={result_f.entropy_radius_pvalue:.3f})")

        # Analyze reverse primer
        result_r = analyze_primer_conservation(
            primer_name=f"{primer_key}_reverse",
            serotype=serotype,
            aligned_sequences=sequences,
            binding_start=positions["reverse"],
            primer_length=positions["len_r"],
            encoder=encoder,
        )
        if result_r:
            results.append(result_r)
            print(f"  Reverse: ρ(entropy,radius)={result_r.entropy_radius_correlation:.3f} "
                  f"(p={result_r.entropy_radius_pvalue:.3f})")
        print()

    # Aggregate analysis
    print("-" * 70)
    print("AGGREGATE ANALYSIS")
    print("-" * 70)
    print()

    # Collect all position data
    all_entropies = []
    all_radii = []
    all_valuations = []

    for r in results:
        for pos in r.positions:
            all_entropies.append(pos["entropy"])
            all_radii.append(pos["padic_radius"])
            all_valuations.append(pos["padic_valuation"])

    if len(all_entropies) >= 10:
        overall_ent_rad, overall_ent_rad_p = spearmanr(all_entropies, all_radii)
        overall_ent_val, overall_ent_val_p = spearmanr(all_entropies, all_valuations)

        print(f"Overall entropy-radius correlation:    ρ = {overall_ent_rad:.3f} (p = {overall_ent_rad_p:.3f})")
        print(f"Overall entropy-valuation correlation: ρ = {overall_ent_val:.3f} (p = {overall_ent_val_p:.3f})")
        print()

        # Interpret
        if overall_ent_rad > 0.2:
            print("FINDING: Higher entropy (variable) positions have HIGHER p-adic radius")
            print("         This suggests variable positions are 'peripheral' in p-adic space")
        elif overall_ent_rad < -0.2:
            print("FINDING: Higher entropy (variable) positions have LOWER p-adic radius")
            print("         This is unexpected - variable positions should be peripheral")
        else:
            print("FINDING: No strong correlation between entropy and p-adic radius")
            print("         P-adic structure may encode different information than conservation")
    else:
        overall_ent_rad = 0.0
        overall_ent_rad_p = 1.0
        overall_ent_val = 0.0
        overall_ent_val_p = 1.0
        print("Insufficient data for aggregate analysis")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_primers": len(results),
        "n_positions": len(all_entropies),
        "overall_entropy_radius_correlation": float(overall_ent_rad) if not math.isnan(overall_ent_rad) else 0.0,
        "overall_entropy_radius_pvalue": float(overall_ent_rad_p) if not math.isnan(overall_ent_rad_p) else 1.0,
        "overall_entropy_valuation_correlation": float(overall_ent_val) if not math.isnan(overall_ent_val) else 0.0,
        "overall_entropy_valuation_pvalue": float(overall_ent_val_p) if not math.isnan(overall_ent_val_p) else 1.0,
        "encoder_loaded": encoder is not None,
        "primer_analyses": [asdict(r) for r in results],
    }


def main():
    """Main entry point."""
    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    results = run_analysis(cache_dir)

    # Save results
    output_path = validation_dir / "padic_conservation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
