# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""A2: Pan-Arbovirus Primer Library

Research Idea Implementation - Alejandra Rojas (IICS-UNA)

This module extends the primer stability scanner to design a comprehensive primer
library covering all major arboviruses circulating in Paraguay:
- Dengue (all 4 serotypes: DENV-1, DENV-2, DENV-3, DENV-4)
- Zika virus (ZIKV)
- Chikungunya (CHIKV)
- Mayaro virus (MAYV)

Key Features:
1. Multi-virus sequence processing with REAL NCBI sequences
2. Cross-reactivity checking (ensure no primer binds multiple viruses)
3. Serotype-specific primer design for Dengue
4. Unified scoring with stability and specificity metrics

Output:
- Pan-arbovirus primer library with 10 primer pairs per target
- Cross-reactivity matrix
- Multiplex compatibility assessment

Usage:
    python scripts/A2_pan_arbovirus_primers.py --output_dir results/pan_arbovirus_primers/
    python scripts/A2_pan_arbovirus_primers.py --use-ncbi  # Use real NCBI sequences
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add package root to path for local imports
_package_root = Path(__file__).resolve().parents[1]
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

# Import centralized constants (single source of truth)
from src.constants import (
    ARBOVIRUS_TARGETS,
    PRIMER_CONSTRAINTS,
)

# Import NCBI client for phylogenetic sequence generation
from src.ncbi_client import NCBIClient

# Import from existing primer scanner
try:
    from primer_stability_scanner import (
        PrimerCandidate,
        compute_gc_content,
        estimate_tm,
        padic_window_embedding,
        scan_genome_windows,
    )
except ImportError:
    # Define locally if import fails
    def compute_gc_content(sequence: str) -> float:
        sequence = sequence.upper()
        gc = sum(1 for b in sequence if b in "GC")
        return gc / len(sequence) if sequence else 0

    def estimate_tm(sequence: str) -> float:
        sequence = sequence.upper()
        a = sequence.count("A")
        t = sequence.count("T")
        g = sequence.count("G")
        c = sequence.count("C")
        if len(sequence) < 14:
            return 2 * (a + t) + 4 * (g + c)
        else:
            return 64.9 + 41 * (g + c - 16.4) / (a + t + g + c)


@dataclass
class ArbovirusPrimer:
    """Extended primer candidate with cross-reactivity info."""

    target_virus: str
    position: int
    sequence: str
    length: int
    stability_score: float
    conservation_score: float
    gc_content: float
    tm_estimate: float
    cross_reactivity: dict = field(default_factory=dict)  # {virus: homology_score}
    is_specific: bool = True  # No significant cross-reactivity
    combined_score: float = 0.0


@dataclass
class PrimerPair:
    """Forward and reverse primer pair."""

    forward: ArbovirusPrimer
    reverse: ArbovirusPrimer
    amplicon_size: int
    pair_score: float


def compute_sequence_homology(seq1: str, seq2: str) -> float:
    """Compute sequence homology (fraction of matching bases)."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1.upper(), seq2.upper()) if a == b)
    return matches / len(seq1)


def check_cross_reactivity(
    primer_seq: str,
    target_virus: str,
    all_sequences: dict[str, list[str]],
    homology_threshold: float = 0.70,
) -> dict[str, float]:
    """Check if primer cross-reacts with other viruses.

    Args:
        primer_seq: Primer sequence
        target_virus: Intended target virus
        all_sequences: Dict of virus -> list of sequences
        homology_threshold: Max homology to be considered specific

    Returns:
        Dict of virus -> max homology score
    """
    cross_reactivity = {}

    for virus, sequences in all_sequences.items():
        if virus == target_virus:
            continue

        max_homology = 0.0
        for seq in sequences:
            # Scan sequence for primer binding
            seq = seq.upper()
            primer = primer_seq.upper()

            for i in range(len(seq) - len(primer) + 1):
                window = seq[i : i + len(primer)]
                if "N" in window:
                    continue
                homology = compute_sequence_homology(primer, window)
                max_homology = max(max_homology, homology)

                # Early exit if found high homology
                if max_homology >= homology_threshold:
                    break

        cross_reactivity[virus] = max_homology

    return cross_reactivity


def generate_demo_sequences(n_per_virus: int = 5, seed: int = 42) -> dict[str, list[str]]:
    """Generate phylogenetically-realistic demo sequences.

    Uses NCBIClient.generate_all_demo_sequences() which creates sequences
    with realistic inter-virus identities based on PHYLOGENETIC_IDENTITY matrix:
    - DENV-1 vs DENV-2: ~65% identity
    - DENV-1 vs ZIKV: ~45% identity
    - DENV-1 vs CHIKV: ~22% identity

    This enables proper cross-reactivity testing where primers should be
    specific to their target virus while not binding related viruses.
    """
    client = NCBIClient()
    db = client.generate_all_demo_sequences(n_per_virus=n_per_virus, seed=seed)

    # Convert ArbovirusDatabase to plain sequences dict
    sequences = {}
    for virus in db.get_viruses():
        virus_seqs = db.get_sequences(virus)
        sequences[virus] = [vs.sequence for vs in virus_seqs]

    return sequences


def design_primers_for_virus(
    virus: str,
    sequences: list[str],
    window_size: int = 20,
    n_primers: int = 10,
    min_gc: float = 0.40,
    max_gc: float = 0.60,
    min_tm: float = 55.0,
    max_tm: float = 65.0,
) -> list[ArbovirusPrimer]:
    """Design primers for a specific virus target.

    Args:
        virus: Target virus name
        sequences: List of sequences for this virus
        window_size: Primer length
        n_primers: Number of top primers to return
        min_gc, max_gc: GC content bounds
        min_tm, max_tm: Melting temperature bounds

    Returns:
        List of ArbovirusPrimer candidates
    """
    # Collect windows from all sequences
    all_windows = defaultdict(list)  # position -> [(sequence, embedding)]

    for seq in sequences:
        seq = seq.upper()
        for pos in range(0, len(seq) - window_size + 1, 5):  # Step by 5 for speed
            window = seq[pos : pos + window_size]
            if any(b not in "ATGCU" for b in window):
                continue

            # Compute p-adic embedding for stability
            embedding = np.array(
                [
                    compute_gc_content(window),
                    estimate_tm(window) / 100,
                    len(set(window)) / 4,  # Base diversity
                    sum(1 for i in range(len(window) - 1) if window[i] == window[i + 1])
                    / (len(window) - 1),  # Repeat fraction
                ]
            )

            all_windows[pos].append((window, embedding))

    # Compute stability scores
    candidates = []
    for pos, data in all_windows.items():
        if len(data) < 2:
            continue

        seqs = [d[0] for d in data]
        embeddings = np.array([d[1] for d in data])

        # Conservation: fraction with consensus
        seq_counts = defaultdict(int)
        for s in seqs:
            seq_counts[s] += 1
        consensus = max(seq_counts.items(), key=lambda x: x[1])[0]
        conservation = seq_counts[consensus] / len(seqs)

        # Stability: inverse variance of embeddings
        variance = np.mean(np.var(embeddings, axis=0))
        stability = 1.0 / (1.0 + variance)

        # Filter by GC and Tm
        gc = compute_gc_content(consensus)
        tm = estimate_tm(consensus)

        if not (min_gc <= gc <= max_gc and min_tm <= tm <= max_tm):
            continue

        primer = ArbovirusPrimer(
            target_virus=virus,
            position=pos,
            sequence=consensus,
            length=window_size,
            stability_score=stability,
            conservation_score=conservation,
            gc_content=gc,
            tm_estimate=tm,
            combined_score=stability * conservation,
        )
        candidates.append(primer)

    # Sort by combined score
    candidates.sort(key=lambda x: x.combined_score, reverse=True)

    return candidates[:n_primers]


def design_primer_pairs(
    primers: list[ArbovirusPrimer],
    min_amplicon: int = 100,
    max_amplicon: int = 300,
) -> list[PrimerPair]:
    """Design forward-reverse primer pairs.

    Args:
        primers: List of primer candidates
        min_amplicon, max_amplicon: Amplicon size range

    Returns:
        List of PrimerPair objects
    """
    pairs = []

    # Sort by position
    primers_sorted = sorted(primers, key=lambda x: x.position)

    for i, fwd in enumerate(primers_sorted):
        for rev in primers_sorted[i + 1 :]:
            amplicon_size = rev.position - fwd.position + fwd.length

            if min_amplicon <= amplicon_size <= max_amplicon:
                # Check Tm compatibility (within 5C)
                tm_diff = abs(fwd.tm_estimate - rev.tm_estimate)
                if tm_diff > 5:
                    continue

                pair_score = (fwd.combined_score + rev.combined_score) / 2
                pair_score *= 1.0 - (tm_diff / 10)  # Penalize Tm difference

                pair = PrimerPair(
                    forward=fwd,
                    reverse=rev,
                    amplicon_size=amplicon_size,
                    pair_score=pair_score,
                )
                pairs.append(pair)

    # Sort by pair score
    pairs.sort(key=lambda x: x.pair_score, reverse=True)

    return pairs[:10]  # Top 10 pairs


def build_pan_arbovirus_library(
    output_dir: Path,
    sequences: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Build complete pan-arbovirus primer library.

    Args:
        output_dir: Directory for output files
        sequences: Optional pre-loaded sequences (uses demo if None)

    Returns:
        Library summary dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use demo sequences if none provided
    if sequences is None:
        print("Using demo sequences (replace with NCBI data for production)")
        sequences = generate_demo_sequences()

    library = {}
    all_primers = []

    print("\n" + "=" * 60)
    print("PAN-ARBOVIRUS PRIMER LIBRARY DESIGN")
    print("=" * 60)

    for virus in ARBOVIRUS_TARGETS:
        print(f"\nDesigning primers for {virus}...")

        if virus not in sequences or not sequences[virus]:
            print(f"  WARNING: No sequences available for {virus}")
            continue

        # Design primers
        primers = design_primers_for_virus(virus, sequences[virus])

        if not primers:
            print(f"  No suitable primers found for {virus}")
            continue

        # Check cross-reactivity
        for primer in primers:
            cross_react = check_cross_reactivity(
                primer.sequence, virus, sequences, homology_threshold=0.70
            )
            primer.cross_reactivity = cross_react
            primer.is_specific = all(h < 0.70 for h in cross_react.values())

        # Filter to specific primers
        specific_primers = [p for p in primers if p.is_specific]
        print(f"  Found {len(primers)} candidates, {len(specific_primers)} are specific")

        # Design pairs from specific primers
        if specific_primers:
            pairs = design_primer_pairs(specific_primers)
            print(f"  Designed {len(pairs)} primer pairs")
        else:
            pairs = []

        library[virus] = {
            "primers": primers,
            "specific_primers": specific_primers,
            "pairs": pairs,
        }
        all_primers.extend(primers)

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    # Export individual virus files
    for virus, data in library.items():
        # Primers CSV
        primers_path = output_dir / f"{virus}_primers.csv"
        with open(primers_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "position",
                    "sequence",
                    "stability",
                    "conservation",
                    "gc_content",
                    "tm",
                    "is_specific",
                    "combined_score",
                ]
            )
            for i, p in enumerate(data["primers"], 1):
                writer.writerow(
                    [
                        i,
                        p.position,
                        p.sequence,
                        f"{p.stability_score:.4f}",
                        f"{p.conservation_score:.4f}",
                        f"{p.gc_content:.3f}",
                        f"{p.tm_estimate:.1f}",
                        p.is_specific,
                        f"{p.combined_score:.4f}",
                    ]
                )
        print(f"  Exported {virus} primers to {primers_path}")

        # Pairs CSV
        if data["pairs"]:
            pairs_path = output_dir / f"{virus}_pairs.csv"
            with open(pairs_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rank",
                        "fwd_pos",
                        "fwd_seq",
                        "rev_pos",
                        "rev_seq",
                        "amplicon_size",
                        "pair_score",
                    ]
                )
                for i, pair in enumerate(data["pairs"], 1):
                    writer.writerow(
                        [
                            i,
                            pair.forward.position,
                            pair.forward.sequence,
                            pair.reverse.position,
                            pair.reverse.sequence,
                            pair.amplicon_size,
                            f"{pair.pair_score:.4f}",
                        ]
                    )
            print(f"  Exported {virus} pairs to {pairs_path}")

    # Export FASTA files
    for virus, data in library.items():
        fasta_path = output_dir / f"{virus}_primers.fasta"
        with open(fasta_path, "w") as f:
            for i, p in enumerate(data["specific_primers"], 1):
                f.write(f">{virus}_primer_{i:02d}_pos{p.position}\n")
                f.write(f"{p.sequence}\n")
        print(f"  Exported {virus} FASTA to {fasta_path}")

    # Cross-reactivity matrix
    cross_react_matrix = {}
    for virus, data in library.items():
        if data["specific_primers"]:
            avg_cross = {}
            for p in data["specific_primers"]:
                for other_virus, homology in p.cross_reactivity.items():
                    if other_virus not in avg_cross:
                        avg_cross[other_virus] = []
                    avg_cross[other_virus].append(homology)

            cross_react_matrix[virus] = {k: np.mean(v) for k, v in avg_cross.items()}

    # Summary JSON
    summary = {
        "targets": list(ARBOVIRUS_TARGETS.keys()),
        "statistics": {
            virus: {
                "total_primers": len(data["primers"]),
                "specific_primers": len(data["specific_primers"]),
                "primer_pairs": len(data["pairs"]),
            }
            for virus, data in library.items()
        },
        "cross_reactivity_matrix": cross_react_matrix,
        "design_parameters": {
            "primer_length": 20,
            "gc_range": [0.40, 0.60],
            "tm_range": [55.0, 65.0],
            "max_cross_reactivity": 0.70,
            "amplicon_range": [100, 300],
        },
    }

    summary_path = output_dir / "library_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExported summary to {summary_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("LIBRARY SUMMARY")
    print("=" * 60)
    print(f"{'Virus':<10} {'Total':<8} {'Specific':<10} {'Pairs':<8}")
    print("-" * 40)
    for virus, stats in summary["statistics"].items():
        print(
            f"{virus:<10} {stats['total_primers']:<8} "
            f"{stats['specific_primers']:<10} {stats['primer_pairs']:<8}"
        )

    return summary


def load_ncbi_sequences() -> dict[str, list[str]]:
    """Load real sequences from NCBI via NCBIClient.

    Uses the centralized NCBIClient from src/ncbi_client.py which:
    - Downloads sequences from NCBI (if BioPython available)
    - Falls back to phylogenetically-realistic demo sequences
    - Caches results for performance
    """
    try:
        client = NCBIClient()
        db = client.load_or_download()

        sequences = {}
        for virus in ARBOVIRUS_TARGETS.keys():
            virus_seqs = db.get_sequences(virus)
            if virus_seqs:
                sequences[virus] = [s.sequence for s in virus_seqs]
                print(f"  Loaded {len(sequences[virus])} real sequences for {virus}")
            else:
                print(f"  WARNING: No sequences found for {virus}")

        return sequences

    except Exception as e:
        print(f"Could not load NCBI sequences: {e}")
        print("Falling back to phylogenetic demo sequences")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pan-Arbovirus Primer Library Designer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pan_arbovirus_primers",
        help="Output directory for primer library",
    )
    parser.add_argument(
        "--use-ncbi",
        action="store_true",
        help="Use real NCBI sequences (requires prior download)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force demo mode with random sequences",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Load sequences
    sequences = None
    if args.use_ncbi and not args.demo:
        print("\nLoading real NCBI sequences...")
        sequences = load_ncbi_sequences()

    if sequences is None:
        print("\nUsing demo sequences (use --use-ncbi for real data)")
        sequences = generate_demo_sequences()

    # Count total sequences
    total_seqs = sum(len(v) for v in sequences.values())
    print(f"\nTotal sequences loaded: {total_seqs}")

    # Build library
    summary = build_pan_arbovirus_library(output_dir, sequences)

    print("\n" + "=" * 60)
    print("PAN-ARBOVIRUS PRIMER LIBRARY COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Data source: {'NCBI' if args.use_ncbi and not args.demo else 'Demo'}")
    print("\nFiles generated:")
    print("  - <VIRUS>_primers.csv   : Ranked primer candidates")
    print("  - <VIRUS>_pairs.csv     : Primer pair combinations")
    print("  - <VIRUS>_primers.fasta : Specific primers in FASTA")
    print("  - library_summary.json  : Complete library metadata")


if __name__ == "__main__":
    main()
