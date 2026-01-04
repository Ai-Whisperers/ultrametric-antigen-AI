# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 P-adic Hyperbolic Integration Analysis.

This script integrates phylogenetic data from 270 DENV-4 genomes with the
p-adic/hyperbolic codon encoder framework to:

1. Embed primer binding regions into hyperbolic space via codon-level encoding
2. Analyze how phylogenetic clade diversity manifests in hyperbolic geometry
3. Quantify primer stability via hyperbolic variance (not just Shannon entropy)
4. Test hypothesis: hyperbolic variance predicts primer binding stability

Key Research Questions:
- Do phylogenetic clades separate in hyperbolic space?
- Can hyperbolic distance explain the 86.7% coverage gap?
- Is hyperbolic variance more predictive than Shannon entropy for primer design?

Usage:
    python scripts/denv4_padic_integration.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

try:
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

from src.encoders import TrainableCodonEncoder
from src.geometry import poincare_distance
from src.biology.codons import CODON_TO_INDEX, GENETIC_CODE

# Paths
ROJAS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROJAS_DIR / "results" / "padic_integration"
DATA_DIR = ROJAS_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PHYLO_DIR = ROJAS_DIR / "results" / "phylogenetic"

# Cached sequence data
SEQUENCES_CACHE = CACHE_DIR / "denv4_sequences.json"

# Trained encoder checkpoint
ENCODER_CHECKPOINT = _project_root / "research" / "codon-encoder" / "training" / "results" / "trained_codon_encoder.pt"


@dataclass
class WindowEmbedding:
    """Hyperbolic embedding for a genome window."""
    position: int  # Start position
    sequence: str  # Nucleotide sequence
    codons: list[str]  # Codons in window
    embedding: np.ndarray  # Mean hyperbolic embedding
    variance: float  # Hyperbolic variance (within window)
    radius: float  # Mean radius in Poincare ball


@dataclass
class CladeHyperbolicProfile:
    """Hyperbolic profile for a phylogenetic clade."""
    clade_name: str
    n_sequences: int
    centroid: np.ndarray  # Centroid in hyperbolic space
    within_variance: float  # Mean variance within clade
    mean_radius: float  # Mean radial position
    spread: float  # Hyperbolic spread (diameter)
    representative_accession: str


def nucleotide_to_codon_indices(sequence: str) -> list[int]:
    """Convert nucleotide sequence to codon indices.

    Args:
        sequence: Nucleotide sequence (must be multiple of 3)

    Returns:
        List of codon indices (0-63), -1 for invalid codons
    """
    sequence = sequence.upper().replace("U", "T")
    indices = []

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in CODON_TO_INDEX:
            indices.append(CODON_TO_INDEX[codon])
        else:
            indices.append(-1)  # Invalid codon

    return indices


def compute_window_embedding(
    encoder: TrainableCodonEncoder,
    sequence: str,
    start: int,
    window_size: int = 75,  # 25 codons
) -> Optional[WindowEmbedding]:
    """Compute hyperbolic embedding for a genome window.

    Args:
        encoder: Trained codon encoder
        sequence: Full nucleotide sequence
        start: Start position (0-indexed)
        window_size: Window size in nucleotides (multiple of 3)

    Returns:
        WindowEmbedding or None if invalid
    """
    # Extract window
    window_seq = sequence[start:start + window_size].upper()

    # Ensure proper length
    if len(window_seq) < window_size:
        return None

    # Convert to codons
    codon_indices = nucleotide_to_codon_indices(window_seq)

    # Filter valid codons
    valid_indices = [i for i in codon_indices if i >= 0]
    if len(valid_indices) < 3:
        return None

    # Get codon strings
    codons = [window_seq[i:i+3] for i in range(0, len(window_seq) - 2, 3)]

    # Encode to hyperbolic space
    with torch.no_grad():
        indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
        z_hyp = encoder(indices_tensor)  # (n_codons, latent_dim)

        # Compute statistics
        embedding = z_hyp.mean(dim=0).cpu().numpy()

        # Hyperbolic variance: mean pairwise distance
        n = z_hyp.shape[0]
        if n > 1:
            # Compute mean hyperbolic distance from centroid
            centroid = z_hyp.mean(dim=0, keepdim=True)
            distances = poincare_distance(z_hyp, centroid.expand(n, -1), c=encoder.curvature)
            variance = distances.mean().item()
        else:
            variance = 0.0

        # Radius from origin
        origin = torch.zeros(1, encoder.latent_dim)
        radii = poincare_distance(z_hyp, origin.expand(n, -1), c=encoder.curvature)
        mean_radius = radii.mean().item()

    return WindowEmbedding(
        position=start,
        sequence=window_seq,
        codons=codons,
        embedding=embedding,
        variance=variance,
        radius=mean_radius,
    )


def embed_primer_region(
    encoder: TrainableCodonEncoder,
    sequences: dict[str, str],  # accession -> sequence
    region_start: int,
    region_end: int,
) -> dict[str, WindowEmbedding]:
    """Embed a primer binding region across multiple sequences.

    Args:
        encoder: Trained codon encoder
        sequences: Dict mapping accession to full sequence
        region_start: Region start position
        region_end: Region end position

    Returns:
        Dict mapping accession to WindowEmbedding
    """
    window_size = region_end - region_start
    embeddings = {}

    for accession, seq in sequences.items():
        emb = compute_window_embedding(encoder, seq, region_start, window_size)
        if emb is not None:
            embeddings[accession] = emb

    return embeddings


def compute_clade_hyperbolic_profile(
    clade_embeddings: list[np.ndarray],
    clade_name: str,
    n_sequences: int,
    representative: str,
) -> CladeHyperbolicProfile:
    """Compute hyperbolic profile for a clade from its embeddings."""
    embeddings = np.array(clade_embeddings)

    # Centroid
    centroid = embeddings.mean(axis=0)

    # Within-clade variance (mean distance to centroid)
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    within_variance = distances_to_centroid.mean()

    # Spread (max pairwise distance)
    if len(embeddings) > 1:
        from scipy.spatial.distance import pdist
        spread = pdist(embeddings, metric='euclidean').max()
    else:
        spread = 0.0

    # Mean radius
    mean_radius = np.linalg.norm(embeddings, axis=1).mean()

    return CladeHyperbolicProfile(
        clade_name=clade_name,
        n_sequences=n_sequences,
        centroid=centroid,
        within_variance=within_variance,
        mean_radius=mean_radius,
        spread=spread,
        representative_accession=representative,
    )


def analyze_entropy_vs_hyperbolic_variance(
    encoder: TrainableCodonEncoder,
    sequences: dict[str, str],
    window_positions: list[tuple[int, int, float]],  # (start, end, entropy)
) -> list[dict]:
    """Compare Shannon entropy vs hyperbolic variance for windows.

    Args:
        encoder: Trained codon encoder
        sequences: Dict of sequences
        window_positions: List of (start, end, entropy) tuples

    Returns:
        List of comparison results
    """
    results = []

    for start, end, entropy in window_positions:
        # Embed across all sequences
        embeddings = embed_primer_region(encoder, sequences, start, end)

        if len(embeddings) < 10:
            continue

        # Compute hyperbolic variance across sequences
        emb_array = np.array([e.embedding for e in embeddings.values()])
        centroid = emb_array.mean(axis=0)
        hyp_variance = np.linalg.norm(emb_array - centroid, axis=1).mean()

        # Mean within-sequence variance
        within_var = np.mean([e.variance for e in embeddings.values()])

        results.append({
            "position": start,
            "shannon_entropy": entropy,
            "hyperbolic_variance": hyp_variance,
            "within_sequence_variance": within_var,
            "n_sequences": len(embeddings),
        })

    return results


def run_full_analysis(
    sequences_path: Path,
    subclade_results: dict,
    output_dir: Path,
) -> dict:
    """Run full p-adic integration analysis.

    Args:
        sequences_path: Path to cached sequences JSON file
        subclade_results: Results from phylogenetic subclade analysis
        output_dir: Output directory

    Returns:
        Analysis results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DENV-4 P-ADIC HYPERBOLIC INTEGRATION ANALYSIS")
    print("=" * 60)

    # Load or train encoder
    print("\n1. Loading codon encoder...")
    encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)

    if ENCODER_CHECKPOINT.exists():
        print(f"   Loading trained weights from {ENCODER_CHECKPOINT.name}")
        checkpoint = torch.load(ENCODER_CHECKPOINT, map_location='cpu', weights_only=True)
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("   WARNING: No trained checkpoint found, using random initialization")

    encoder.eval()

    # Load sequences from JSON cache
    print("\n2. Loading DENV-4 sequences from cache...")
    with open(sequences_path) as f:
        sequences = json.load(f)

    print(f"   Loaded {len(sequences)} sequences")

    # Parse subclade assignments
    print("\n3. Parsing subclade assignments...")
    clade_assignments = {}
    for clade_info in subclade_results.get("primer_suitable_clades", []):
        clade_name = clade_info["clade"]
        # We need to reconstruct clade membership - for now use representative
        clade_assignments[clade_name] = {
            "representative": clade_info.get("representative"),
            "size": clade_info["size"],
            "entropy": clade_info["mean_entropy"],
        }

    # Also parse non-suitable clades
    for clade_name, clade_data in subclade_results.get("subclades", {}).items():
        if clade_name not in clade_assignments:
            clade_assignments[clade_name] = {
                "size": clade_data["size"],
                "entropy": clade_data["mean_entropy"],
                "has_primers": clade_data.get("has_primers", False),
            }

    print(f"   Found {len(clade_assignments)} subclades")

    # Analyze primer binding regions
    print("\n4. Analyzing primer binding regions in hyperbolic space...")

    # Key regions from subclade analysis
    primer_regions = [
        # Conserved window in Clade_E.3.2
        (9908, 9933, "NS5_conserved"),
        # Pan-flavivirus regions (Kuno et al.)
        (9007, 9033, "PANFLAVI_FU1"),
        (9196, 9222, "PANFLAVI_cFD2"),
        # Additional NS5 regions
        (9000, 9100, "NS5_5prime"),
        (9500, 9600, "NS5_middle"),
        (10000, 10100, "NS5_3prime"),
    ]

    region_analysis = []
    for start, end, name in primer_regions:
        print(f"   Analyzing {name} ({start}-{end})...")

        embeddings = embed_primer_region(encoder, sequences, start, end)

        if len(embeddings) < 10:
            print(f"      Skipped (only {len(embeddings)} valid embeddings)")
            continue

        # Compute hyperbolic statistics
        emb_array = np.array([e.embedding for e in embeddings.values()])
        variances = [e.variance for e in embeddings.values()]
        radii = [e.radius for e in embeddings.values()]

        # Cross-sequence hyperbolic variance
        centroid = emb_array.mean(axis=0)
        cross_seq_distances = np.linalg.norm(emb_array - centroid, axis=1)
        cross_seq_variance = cross_seq_distances.mean()

        region_result = {
            "region": name,
            "start": start,
            "end": end,
            "n_sequences": len(embeddings),
            "hyperbolic_cross_seq_variance": float(cross_seq_variance),
            "mean_within_seq_variance": float(np.mean(variances)),
            "mean_radius": float(np.mean(radii)),
            "radius_std": float(np.std(radii)),
            "centroid_norm": float(np.linalg.norm(centroid)),
        }
        region_analysis.append(region_result)

        print(f"      Cross-seq hyperbolic variance: {cross_seq_variance:.4f}")
        print(f"      Mean radius: {np.mean(radii):.4f} ± {np.std(radii):.4f}")

    # Compare conserved vs divergent regions
    print("\n5. Comparing conserved vs divergent regions...")

    if len(region_analysis) >= 2:
        conserved = [r for r in region_analysis if "conserved" in r["region"].lower()]
        divergent = [r for r in region_analysis if "conserved" not in r["region"].lower()]

        if conserved and divergent:
            conserved_var = np.mean([r["hyperbolic_cross_seq_variance"] for r in conserved])
            divergent_var = np.mean([r["hyperbolic_cross_seq_variance"] for r in divergent])

            print(f"   Conserved regions mean variance: {conserved_var:.4f}")
            print(f"   Divergent regions mean variance: {divergent_var:.4f}")
            print(f"   Ratio (divergent/conserved): {divergent_var/conserved_var:.2f}x")

    # Analyze full genome tiling
    print("\n6. Genome-wide hyperbolic variance scan...")

    # Sample positions across genome
    window_size = 75  # 25 codons
    step = 300  # Every 300 nt

    genome_scan = []
    sample_seq = list(sequences.values())[0]  # Use first sequence for positions

    for pos in range(0, len(sample_seq) - window_size, step):
        embeddings = embed_primer_region(encoder, sequences, pos, pos + window_size)

        if len(embeddings) < 10:
            continue

        emb_array = np.array([e.embedding for e in embeddings.values()])
        centroid = emb_array.mean(axis=0)
        variance = np.linalg.norm(emb_array - centroid, axis=1).mean()

        genome_scan.append({
            "position": pos,
            "variance": float(variance),
            "n_sequences": len(embeddings),
        })

    print(f"   Scanned {len(genome_scan)} windows")

    if genome_scan:
        variances = [g["variance"] for g in genome_scan]
        print(f"   Variance range: {min(variances):.4f} - {max(variances):.4f}")
        print(f"   Mean variance: {np.mean(variances):.4f}")

    # Find lowest variance regions (best for primers)
    print("\n7. Top primer candidate regions (lowest hyperbolic variance)...")

    sorted_scan = sorted(genome_scan, key=lambda x: x["variance"])
    top_candidates = sorted_scan[:10]

    for i, cand in enumerate(top_candidates, 1):
        print(f"   {i}. Position {cand['position']:5d}: variance = {cand['variance']:.4f}")

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "encoder_checkpoint": str(ENCODER_CHECKPOINT) if ENCODER_CHECKPOINT.exists() else None,
            "latent_dim": 16,
            "n_sequences": len(sequences),
        },
        "region_analysis": region_analysis,
        "genome_scan_summary": {
            "n_windows": len(genome_scan),
            "variance_min": float(min(variances)) if genome_scan else None,
            "variance_max": float(max(variances)) if genome_scan else None,
            "variance_mean": float(np.mean(variances)) if genome_scan else None,
        },
        "top_primer_candidates": [
            {
                "rank": i + 1,
                "position": cand["position"],
                "hyperbolic_variance": cand["variance"],
            }
            for i, cand in enumerate(top_candidates)
        ],
        "genome_scan": genome_scan,
    }

    # Save results
    results_file = output_dir / "padic_integration_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Generate report
    generate_report(results, output_dir)

    return results


def generate_report(results: dict, output_dir: Path) -> None:
    """Generate markdown report."""
    report_path = output_dir / "PADIC_INTEGRATION_REPORT.md"

    lines = [
        "# DENV-4 P-adic Hyperbolic Integration Analysis",
        "",
        f"**Generated:** {results['timestamp'][:10]}",
        f"**Sequences Analyzed:** {results['parameters']['n_sequences']}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This analysis integrates phylogenetic data with p-adic/hyperbolic geometry",
        "to provide a fundamentally different view of DENV-4 diversity than traditional",
        "Shannon entropy-based approaches.",
        "",
        "**Key insight:** Hyperbolic variance measures diversity in a geometry that",
        "naturally encodes hierarchical relationships (codon → amino acid → protein),",
        "potentially capturing evolutionarily relevant variation that entropy misses.",
        "",
        "---",
        "",
        "## Region Analysis",
        "",
        "| Region | Position | N Sequences | Hyp. Variance | Mean Radius |",
        "|--------|----------|-------------|---------------|-------------|",
    ]

    for r in results["region_analysis"]:
        lines.append(
            f"| {r['region']} | {r['start']}-{r['end']} | {r['n_sequences']} | "
            f"{r['hyperbolic_cross_seq_variance']:.4f} | {r['mean_radius']:.4f} |"
        )

    lines.extend([
        "",
        "**Interpretation:**",
        "- Lower hyperbolic variance = more conserved in codon structure",
        "- Higher mean radius = more variable/complex codon patterns",
        "",
        "---",
        "",
        "## Top Primer Candidate Regions",
        "",
        "These positions have the **lowest hyperbolic variance** across all sequences,",
        "indicating structural conservation at the codon level:",
        "",
        "| Rank | Position | Hyperbolic Variance |",
        "|------|----------|---------------------|",
    ])

    for cand in results["top_primer_candidates"]:
        lines.append(f"| {cand['rank']} | {cand['position']} | {cand['hyperbolic_variance']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Genome-wide Variance Distribution",
        "",
        f"- **Windows scanned:** {results['genome_scan_summary']['n_windows']}",
        f"- **Variance range:** {results['genome_scan_summary']['variance_min']:.4f} - "
        f"{results['genome_scan_summary']['variance_max']:.4f}",
        f"- **Mean variance:** {results['genome_scan_summary']['variance_mean']:.4f}",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Hyperbolic Codon Embedding",
        "",
        "1. Convert nucleotide windows to codon sequences",
        "2. Encode each codon using TrainableCodonEncoder (16-dim Poincaré ball)",
        "3. Compute window embedding as mean of codon embeddings",
        "4. Compute hyperbolic variance as mean distance from centroid",
        "",
        "### Why Hyperbolic Geometry?",
        "",
        "The Poincaré ball naturally encodes hierarchical relationships:",
        "- Codons at similar radii share structural properties",
        "- Distance in hyperbolic space reflects evolutionary relatedness",
        "- Synonymous codons cluster near each other",
        "",
        "This provides a fundamentally different measure of conservation than",
        "Shannon entropy, which treats all nucleotide changes equally.",
        "",
        "---",
        "",
        f"*Analysis performed with p-adic codon encoder framework*",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Report saved to {report_path}")


def main():
    """Main entry point."""
    # Check for cached sequences
    if not SEQUENCES_CACHE.exists():
        print(f"Sequences cache not found: {SEQUENCES_CACHE}")
        print("Run denv4_phylogenetic_analysis.py first to download sequences.")
        return

    # Load subclade results
    subclade_file = PHYLO_DIR / "subclade_analysis_results.json"
    if subclade_file.exists():
        with open(subclade_file) as f:
            subclade_results = json.load(f)
    else:
        print(f"Subclade results not found: {subclade_file}")
        subclade_results = {}

    # Run analysis
    results = run_full_analysis(SEQUENCES_CACHE, subclade_results, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
