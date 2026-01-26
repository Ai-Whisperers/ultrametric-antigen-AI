# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Design Degenerate Primers for Functional Convergence Points.

Based on the functional convergence analysis, this script designs actual
degenerate primers for the top candidate positions:
- 5'UTR position 39 (nucleotide ~117)
- E gene position 1254 (nucleotide ~3762)

IUPAC Degenerate Base Codes:
- R = A or G (puRine)
- Y = C or T (pYrimidine)
- S = G or C (Strong)
- W = A or T (Weak)
- K = G or T (Keto)
- M = A or C (aMino)
- B = C or G or T (not A)
- D = A or G or T (not C)
- H = A or C or T (not G)
- V = A or C or G (not T)
- N = A or C or G or T (aNy)

Degeneracy calculation:
- Each IUPAC code contributes a multiplier (R=2, N=4, etc.)
- Total degeneracy = product of all position degeneracies
- Practical limit: 256-4096 for PCR primers
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# IUPAC codes
IUPAC_CODES = {
    frozenset(['A']): 'A',
    frozenset(['C']): 'C',
    frozenset(['G']): 'G',
    frozenset(['T']): 'T',
    frozenset(['A', 'G']): 'R',
    frozenset(['C', 'T']): 'Y',
    frozenset(['G', 'C']): 'S',
    frozenset(['A', 'T']): 'W',
    frozenset(['G', 'T']): 'K',
    frozenset(['A', 'C']): 'M',
    frozenset(['C', 'G', 'T']): 'B',
    frozenset(['A', 'G', 'T']): 'D',
    frozenset(['A', 'C', 'T']): 'H',
    frozenset(['A', 'C', 'G']): 'V',
    frozenset(['A', 'C', 'G', 'T']): 'N',
}

IUPAC_DEGENERACY = {
    'A': 1, 'C': 1, 'G': 1, 'T': 1,
    'R': 2, 'Y': 2, 'S': 2, 'W': 2, 'K': 2, 'M': 2,
    'B': 3, 'D': 3, 'H': 3, 'V': 3,
    'N': 4,
}


class PrimerDesign(NamedTuple):
    """A designed degenerate primer."""
    name: str
    position: int
    length: int
    sequence: str  # IUPAC degenerate sequence
    degeneracy: int
    gc_content: float
    tm_estimate: float  # Basic Tm estimate
    coverage: float  # % of sequences matched
    clade_coverage: dict[str, float]


def load_denv4_data() -> tuple[dict[str, str], dict[str, dict]]:
    """Load DENV-4 sequences and metadata."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        meta_file = json.load(f)
        metadata = meta_file["data"]

    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_file = json.load(f)
        sequences = seq_file["data"]

    return sequences, metadata


def bases_to_iupac(bases: set[str]) -> str:
    """Convert a set of bases to IUPAC code."""
    bases = frozenset(b.upper() for b in bases if b.upper() in 'ACGT')
    return IUPAC_CODES.get(bases, 'N')


def compute_degeneracy(sequence: str) -> int:
    """Compute total degeneracy of an IUPAC sequence."""
    degeneracy = 1
    for base in sequence:
        degeneracy *= IUPAC_DEGENERACY.get(base.upper(), 4)
    return degeneracy


def estimate_tm(sequence: str) -> float:
    """Estimate melting temperature using basic formula.

    For short primers (<14bp): Tm = 2(A+T) + 4(G+C)
    For longer primers: Tm = 64.9 + 41*(G+C-16.4)/(A+T+G+C)

    For degenerate bases, use average contribution.
    """
    # Count bases (treating degenerate as average)
    gc_contrib = {'G': 1, 'C': 1, 'S': 1, 'R': 0.5, 'Y': 0.5, 'K': 0.5, 'M': 0.5,
                  'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33, 'N': 0.5}
    at_contrib = {'A': 1, 'T': 1, 'W': 1, 'R': 0.5, 'Y': 0.5, 'K': 0.5, 'M': 0.5,
                  'B': 0.33, 'V': 0.33, 'D': 0.67, 'H': 0.67, 'N': 0.5}

    gc = sum(gc_contrib.get(b.upper(), 0) for b in sequence)
    at = sum(at_contrib.get(b.upper(), 0) for b in sequence)

    total = gc + at
    if total < 14:
        return 2 * at + 4 * gc
    else:
        return 64.9 + 41 * (gc - 16.4) / total


def compute_gc_content(sequence: str) -> float:
    """Compute GC content (treating degenerate bases as average)."""
    gc_contrib = {'G': 1, 'C': 1, 'S': 1, 'R': 0.5, 'Y': 0.5, 'K': 0.5, 'M': 0.5,
                  'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33, 'N': 0.5, 'A': 0, 'T': 0, 'W': 0}

    gc = sum(gc_contrib.get(b.upper(), 0.5) for b in sequence)
    return gc / len(sequence) if sequence else 0


def expand_iupac(sequence: str) -> list[str]:
    """Expand IUPAC sequence to all possible concrete sequences."""
    IUPAC_EXPAND = {
        'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
        'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
        'K': ['G', 'T'], 'M': ['A', 'C'],
        'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'T'],
    }

    if not sequence:
        return ['']

    first = sequence[0].upper()
    rest = sequence[1:]

    expansions = IUPAC_EXPAND.get(first, [first])
    rest_expanded = expand_iupac(rest)

    return [e + r for e in expansions for r in rest_expanded]


def check_primer_match(primer_seq: str, target_seq: str, max_mismatches: int = 0) -> bool:
    """Check if degenerate primer matches target sequence."""
    if len(primer_seq) > len(target_seq):
        return False

    IUPAC_MATCH = {
        'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T', 'U'},
        'R': {'A', 'G'}, 'Y': {'C', 'T', 'U'}, 'S': {'G', 'C'}, 'W': {'A', 'T', 'U'},
        'K': {'G', 'T', 'U'}, 'M': {'A', 'C'},
        'B': {'C', 'G', 'T', 'U'}, 'D': {'A', 'G', 'T', 'U'},
        'H': {'A', 'C', 'T', 'U'}, 'V': {'A', 'C', 'G'},
        'N': {'A', 'C', 'G', 'T', 'U'},
    }

    mismatches = 0
    for p, t in zip(primer_seq, target_seq):
        if t.upper() not in IUPAC_MATCH.get(p.upper(), set()):
            mismatches += 1
            if mismatches > max_mismatches:
                return False

    return True


def design_primer_at_position(
    sequences: dict[str, str],
    metadata: dict[str, dict],
    start_pos: int,
    length: int = 20,
    name: str = "primer",
    direction: str = "forward",
) -> PrimerDesign:
    """Design a degenerate primer at a specific position.

    Args:
        sequences: Dict of accession -> sequence
        metadata: Dict of accession -> metadata (with clade)
        start_pos: Nucleotide start position (0-indexed)
        length: Primer length
        name: Primer name
        direction: 'forward' or 'reverse' (reverse complement)
    """
    # Extract primer region from all sequences
    primer_regions = []
    clade_regions = {}

    for acc, seq in sequences.items():
        if start_pos + length <= len(seq):
            region = seq[start_pos:start_pos + length].upper().replace('U', 'T')
            if all(b in 'ACGT' for b in region):
                primer_regions.append(region)

                if acc in metadata:
                    clade = metadata[acc]["clade"]
                    if clade not in clade_regions:
                        clade_regions[clade] = []
                    clade_regions[clade].append(region)

    if not primer_regions:
        return None

    # Build degenerate sequence
    degenerate_seq = []
    for pos in range(length):
        bases_at_pos = set(r[pos] for r in primer_regions)
        iupac = bases_to_iupac(bases_at_pos)
        degenerate_seq.append(iupac)

    degenerate_seq = ''.join(degenerate_seq)

    # Reverse complement if needed
    if direction == "reverse":
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
                     'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
                     'K': 'M', 'M': 'K', 'B': 'V', 'V': 'B',
                     'D': 'H', 'H': 'D', 'N': 'N'}
        degenerate_seq = ''.join(complement.get(b, b) for b in reversed(degenerate_seq))

    # Compute metrics
    degeneracy = compute_degeneracy(degenerate_seq)
    gc_content = compute_gc_content(degenerate_seq)
    tm = estimate_tm(degenerate_seq)

    # Compute coverage
    matches = sum(1 for r in primer_regions if check_primer_match(degenerate_seq if direction == "forward" else ''.join(reversed(degenerate_seq)), r))
    coverage = matches / len(primer_regions) if primer_regions else 0

    # Per-clade coverage
    clade_coverage = {}
    for clade, regions in clade_regions.items():
        clade_matches = sum(1 for r in regions if check_primer_match(degenerate_seq if direction == "forward" else ''.join(reversed(degenerate_seq)), r))
        clade_coverage[clade] = clade_matches / len(regions) if regions else 0

    return PrimerDesign(
        name=name,
        position=start_pos,
        length=length,
        sequence=degenerate_seq,
        degeneracy=degeneracy,
        gc_content=gc_content,
        tm_estimate=tm,
        coverage=coverage,
        clade_coverage=clade_coverage,
    )


def optimize_primer_window(
    sequences: dict[str, str],
    metadata: dict[str, dict],
    target_codon_pos: int,
    min_length: int = 18,
    max_length: int = 25,
    max_degeneracy: int = 256,
    name_prefix: str = "primer",
) -> list[PrimerDesign]:
    """Optimize primer around a target codon position.

    Tries different start positions and lengths to minimize degeneracy
    while maintaining coverage.
    """
    nuc_pos = target_codon_pos * 3

    candidates = []

    # Search window around target
    for offset in range(-10, 15):
        start = nuc_pos + offset
        if start < 0:
            continue

        for length in range(min_length, max_length + 1):
            for direction in ["forward", "reverse"]:
                primer = design_primer_at_position(
                    sequences, metadata, start, length,
                    name=f"{name_prefix}_{direction[0].upper()}_{start}",
                    direction=direction,
                )

                if primer and primer.degeneracy <= max_degeneracy and primer.coverage > 0.9:
                    candidates.append(primer)

    # Sort by degeneracy (lower is better)
    candidates.sort(key=lambda p: (p.degeneracy, -p.coverage))

    return candidates[:10]  # Top 10


def main():
    print("=" * 70)
    print("DEGENERATE PRIMER DESIGN FOR FUNCTIONAL CONVERGENCE POINTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Load data
    print("Loading DENV-4 data...")
    sequences, metadata = load_denv4_data()
    print(f"Loaded {len(sequences)} sequences")

    # Target positions from functional convergence analysis
    targets = [
        {"name": "5UTR", "codon_pos": 39, "description": "5'UTR best candidate"},
        {"name": "E_gene", "codon_pos": 1254, "description": "E gene best candidate"},
    ]

    all_designs = {}

    for target in targets:
        print(f"\n{'=' * 70}")
        print(f"DESIGNING PRIMERS FOR {target['name']}")
        print(f"Codon position: {target['codon_pos']} (nucleotide ~{target['codon_pos']*3})")
        print(f"{'=' * 70}")

        # Design forward primers
        print("\nOptimizing forward primers...")
        forward_candidates = optimize_primer_window(
            sequences, metadata,
            target['codon_pos'],
            max_degeneracy=512,
            name_prefix=f"DENV4_{target['name']}_F",
        )

        # Design reverse primers (for amplicon)
        print("Optimizing reverse primers...")
        # Reverse primer ~100bp downstream
        reverse_codon_pos = target['codon_pos'] + 33  # ~100bp amplicon
        reverse_candidates = optimize_primer_window(
            sequences, metadata,
            reverse_codon_pos,
            max_degeneracy=512,
            name_prefix=f"DENV4_{target['name']}_R",
        )

        # Report best primers
        print(f"\n{'TOP FORWARD PRIMERS':=^50}")
        if forward_candidates:
            for i, p in enumerate(forward_candidates[:5], 1):
                print(f"\n{i}. {p.name}")
                print(f"   Sequence: 5'-{p.sequence}-3'")
                print(f"   Position: {p.position}")
                print(f"   Length: {p.length} bp")
                print(f"   Degeneracy: {p.degeneracy}")
                print(f"   GC: {p.gc_content*100:.1f}%")
                print(f"   Tm: {p.tm_estimate:.1f}°C")
                print(f"   Coverage: {p.coverage*100:.1f}%")
                print(f"   Per-clade: {', '.join(f'{c}:{v*100:.0f}%' for c,v in sorted(p.clade_coverage.items()))}")
        else:
            print("   No primers meeting criteria found")

        print(f"\n{'TOP REVERSE PRIMERS':=^50}")
        if reverse_candidates:
            for i, p in enumerate(reverse_candidates[:5], 1):
                print(f"\n{i}. {p.name}")
                print(f"   Sequence: 5'-{p.sequence}-3'")
                print(f"   Position: {p.position}")
                print(f"   Length: {p.length} bp")
                print(f"   Degeneracy: {p.degeneracy}")
                print(f"   GC: {p.gc_content*100:.1f}%")
                print(f"   Tm: {p.tm_estimate:.1f}°C")
                print(f"   Coverage: {p.coverage*100:.1f}%")
        else:
            print("   No primers meeting criteria found")

        # Design primer pair
        if forward_candidates and reverse_candidates:
            best_f = forward_candidates[0]
            best_r = reverse_candidates[0]
            amplicon_size = best_r.position + best_r.length - best_f.position

            print(f"\n{'RECOMMENDED PRIMER PAIR':=^50}")
            print(f"\nForward: {best_f.name}")
            print(f"  5'-{best_f.sequence}-3'")
            print(f"  Pos: {best_f.position}, Deg: {best_f.degeneracy}, Tm: {best_f.tm_estimate:.1f}°C")

            print(f"\nReverse: {best_r.name}")
            print(f"  5'-{best_r.sequence}-3'")
            print(f"  Pos: {best_r.position}, Deg: {best_r.degeneracy}, Tm: {best_r.tm_estimate:.1f}°C")

            print(f"\nAmplicon size: ~{amplicon_size} bp")
            print(f"Total degeneracy: {best_f.degeneracy * best_r.degeneracy}")

        all_designs[target['name']] = {
            "target": target,
            "forward_candidates": [
                {
                    "name": p.name,
                    "sequence": p.sequence,
                    "position": p.position,
                    "length": p.length,
                    "degeneracy": p.degeneracy,
                    "gc_content": p.gc_content,
                    "tm_estimate": p.tm_estimate,
                    "coverage": p.coverage,
                    "clade_coverage": p.clade_coverage,
                }
                for p in forward_candidates[:5]
            ],
            "reverse_candidates": [
                {
                    "name": p.name,
                    "sequence": p.sequence,
                    "position": p.position,
                    "length": p.length,
                    "degeneracy": p.degeneracy,
                    "gc_content": p.gc_content,
                    "tm_estimate": p.tm_estimate,
                    "coverage": p.coverage,
                    "clade_coverage": p.clade_coverage,
                }
                for p in reverse_candidates[:5]
            ],
        }

    # Save results
    results = {
        "_metadata": {
            "schema_version": "1.0",
            "file_type": "primer_design",
            "analysis_type": "degenerate_primer_design",
            "description": "Degenerate primers for DENV-4 functional convergence points",
            "created": datetime.now(timezone.utc).isoformat(),
            "pipeline": "design_degenerate_primers.py",
            "design_criteria": {
                "max_degeneracy": 512,
                "min_length": 18,
                "max_length": 25,
                "min_coverage": 0.9,
            },
            "field_definitions": {
                "sequence": "IUPAC degenerate primer sequence (5' to 3')",
                "degeneracy": "Number of unique sequences in degenerate pool",
                "coverage": "Fraction of sequences with perfect match",
                "clade_coverage": "Per-clade coverage fractions"
            }
        },
        "data": all_designs,
    }

    output_path = RESULTS_DIR / "degenerate_primer_designs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
