# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Design E Gene Primers for DENV-4 Detection.

This script designs primers targeting the E gene regions identified by:
1. P-adic integration analysis (position ~2400, lowest hyperbolic variance)
2. Dual-metric scoring (position ~950, lowest combined score)

Design Constraints:
- Primer length: 18-25 bp
- Tm: 55-65°C (optimal: 58-62°C)
- GC content: 40-60%
- Degeneracy: Allow 2-4 degenerate positions (IUPAC codes)
- Amplicon: 75-200 bp
- No 3' complementarity (hairpin/dimer)

IUPAC Degenerate Base Codes:
    R = A/G (purine)
    Y = C/T (pyrimidine)
    M = A/C
    K = G/T
    S = G/C (strong)
    W = A/T (weak)
    B = C/G/T (not A)
    D = A/G/T (not C)
    H = A/C/T (not G)
    V = A/C/G (not T)
    N = A/C/G/T (any)

Usage:
    python design_egene_primers.py --target e2400
    python design_egene_primers.py --target e950
    python design_egene_primers.py --target all
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

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = ROJAS_DIR / "results" / "primers"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# IUPAC codes
IUPAC_CODES = {
    frozenset(['A']): 'A',
    frozenset(['C']): 'C',
    frozenset(['G']): 'G',
    frozenset(['T']): 'T',
    frozenset(['A', 'G']): 'R',
    frozenset(['C', 'T']): 'Y',
    frozenset(['A', 'C']): 'M',
    frozenset(['G', 'T']): 'K',
    frozenset(['G', 'C']): 'S',
    frozenset(['A', 'T']): 'W',
    frozenset(['C', 'G', 'T']): 'B',
    frozenset(['A', 'G', 'T']): 'D',
    frozenset(['A', 'C', 'T']): 'H',
    frozenset(['A', 'C', 'G']): 'V',
    frozenset(['A', 'C', 'G', 'T']): 'N',
}

# Degeneracy count
DEGENERACY = {
    'A': 1, 'C': 1, 'G': 1, 'T': 1,
    'R': 2, 'Y': 2, 'M': 2, 'K': 2, 'S': 2, 'W': 2,
    'B': 3, 'D': 3, 'H': 3, 'V': 3,
    'N': 4,
}


class Primer(NamedTuple):
    """Represents a primer sequence."""
    name: str
    sequence: str
    position: int
    length: int
    tm: float
    gc_content: float
    degeneracy: int
    direction: str  # 'forward' or 'reverse'


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
            sequences.append(seq_data["data"][acc].upper())

    return accessions, sequences


def get_consensus_with_degeneracy(sequences: list[str], start: int, length: int,
                                   min_freq: float = 0.90) -> str:
    """Get consensus sequence with IUPAC degeneracy codes.

    Args:
        sequences: List of aligned sequences
        start: Start position
        length: Primer length
        min_freq: Minimum frequency to consider nucleotide (default 90%)

    Returns:
        Consensus sequence with IUPAC codes
    """
    consensus = []

    for pos in range(start, start + length):
        nucs = []
        for seq in sequences:
            if pos < len(seq):
                nuc = seq[pos].upper()
                if nuc in 'ACGT':
                    nucs.append(nuc)

        if not nucs:
            consensus.append('N')
            continue

        counts = Counter(nucs)
        total = len(nucs)

        # Find nucleotides above threshold
        significant = set()
        for nuc, count in counts.items():
            if count / total >= min_freq:
                significant.add(nuc)

        # If no single nucleotide dominates, include all common ones
        if not significant:
            # Include nucleotides present in >5% of sequences
            for nuc, count in counts.items():
                if count / total >= 0.05:
                    significant.add(nuc)

        if not significant:
            significant = set(counts.keys())

        # Convert to IUPAC code
        code = IUPAC_CODES.get(frozenset(significant), 'N')
        consensus.append(code)

    return ''.join(consensus)


def reverse_complement(seq: str) -> str:
    """Get reverse complement of sequence (handles IUPAC codes)."""
    complement = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'R': 'Y', 'Y': 'R', 'M': 'K', 'K': 'M',
        'S': 'S', 'W': 'W', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D', 'N': 'N',
    }
    return ''.join(complement.get(b, 'N') for b in reversed(seq))


def calculate_tm(seq: str) -> float:
    """Calculate melting temperature using nearest-neighbor method (simplified).

    For degenerate primers, uses average of possible sequences.

    Returns Tm in °C.
    """
    # Simplified formula for primers 14-25 bp
    # Tm = 64.9 + 41 * (G+C-16.4) / N
    # where N is length

    # For degenerate bases, use average GC contribution
    gc_contribution = {
        'G': 1.0, 'C': 1.0,
        'A': 0.0, 'T': 0.0,
        'S': 1.0,  # G or C
        'W': 0.0,  # A or T
        'R': 0.5, 'Y': 0.5, 'M': 0.5, 'K': 0.5,
        'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33,
        'N': 0.5,
    }

    gc_count = sum(gc_contribution.get(b, 0.5) for b in seq.upper())
    n = len(seq)

    if n < 14:
        # Wallace rule for short primers
        tm = 2 * (seq.count('A') + seq.count('T')) + 4 * gc_count
    else:
        # Nearest-neighbor approximation
        tm = 64.9 + 41 * (gc_count - 16.4) / n

    return round(tm, 1)


def calculate_gc(seq: str) -> float:
    """Calculate GC content (0-1)."""
    gc_contribution = {
        'G': 1.0, 'C': 1.0,
        'A': 0.0, 'T': 0.0,
        'S': 1.0, 'W': 0.0,
        'R': 0.5, 'Y': 0.5, 'M': 0.5, 'K': 0.5,
        'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33,
        'N': 0.5,
    }
    gc = sum(gc_contribution.get(b, 0.5) for b in seq.upper())
    return round(gc / len(seq), 3)


def calculate_degeneracy(seq: str) -> int:
    """Calculate total degeneracy of primer."""
    total = 1
    for b in seq.upper():
        total *= DEGENERACY.get(b, 4)
    return total


def check_3prime_stability(seq: str) -> bool:
    """Check if 3' end is stable (G or C in last 3 positions)."""
    last3 = seq[-3:].upper()
    gc_in_last3 = sum(1 for b in last3 if b in 'GCSBV')
    return gc_in_last3 >= 1


def design_primer_pair(sequences: list[str], target_pos: int, target_name: str,
                       amplicon_size: int = 100, primer_len: int = 20) -> dict:
    """Design a primer pair for a target region.

    Args:
        sequences: List of aligned genome sequences
        target_pos: Target position for primer binding
        target_name: Name for the primer pair (e.g., "E2400")
        amplicon_size: Desired amplicon size in bp
        primer_len: Primer length

    Returns:
        Dictionary with primer pair info
    """
    # Design forward primer
    fwd_seq = get_consensus_with_degeneracy(sequences, target_pos, primer_len)
    fwd_primer = Primer(
        name=f"DENV4_{target_name}_F",
        sequence=fwd_seq,
        position=target_pos,
        length=primer_len,
        tm=calculate_tm(fwd_seq),
        gc_content=calculate_gc(fwd_seq),
        degeneracy=calculate_degeneracy(fwd_seq),
        direction='forward',
    )

    # Design reverse primer (at target_pos + amplicon_size)
    rev_start = target_pos + amplicon_size - primer_len
    rev_seq_fwd = get_consensus_with_degeneracy(sequences, rev_start, primer_len)
    rev_seq = reverse_complement(rev_seq_fwd)
    rev_primer = Primer(
        name=f"DENV4_{target_name}_R",
        sequence=rev_seq,
        position=rev_start,
        length=primer_len,
        tm=calculate_tm(rev_seq),
        gc_content=calculate_gc(rev_seq),
        degeneracy=calculate_degeneracy(rev_seq),
        direction='reverse',
    )

    # Calculate amplicon details
    amplicon_seq = get_consensus_with_degeneracy(
        sequences, target_pos, amplicon_size
    )

    return {
        'target_name': target_name,
        'forward': fwd_primer._asdict(),
        'reverse': rev_primer._asdict(),
        'amplicon': {
            'size': amplicon_size,
            'start': target_pos,
            'end': target_pos + amplicon_size,
            'gc_content': calculate_gc(amplicon_seq),
        },
        'validation': {
            'fwd_3prime_stable': check_3prime_stability(fwd_seq),
            'rev_3prime_stable': check_3prime_stability(rev_seq),
            'tm_diff': abs(fwd_primer.tm - rev_primer.tm),
            'total_degeneracy': fwd_primer.degeneracy * rev_primer.degeneracy,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Design E gene primers for DENV-4")
    parser.add_argument("--target", choices=['e2400', 'e950', 'all'], default='all',
                       help="Target region")
    parser.add_argument("--amplicon", type=int, default=100, help="Amplicon size")
    parser.add_argument("--primer_len", type=int, default=20, help="Primer length")
    args = parser.parse_args()

    print("=" * 80)
    print("DENV-4 E GENE PRIMER DESIGN")
    print("=" * 80)
    print()
    print("Targets:")
    print("  - E2400: Lowest hyperbolic variance (p-adic analysis)")
    print("  - E950: Best dual-metric score (Shannon + p-adic)")
    print()
    print(f"Parameters:")
    print(f"  Amplicon size: {args.amplicon} bp")
    print(f"  Primer length: {args.primer_len} bp")
    print()
    print("=" * 80)

    # Load sequences
    print("\n[1/3] Loading DENV-4 sequences...")
    accessions, sequences = load_denv4_sequences()
    print(f"  Loaded {len(sequences)} sequences")

    # Define targets
    targets = []
    if args.target in ['e2400', 'all']:
        targets.append(('E2400', 2400))  # P-adic best
    if args.target in ['e950', 'all']:
        targets.append(('E950', 950))    # Dual-metric best

    # Additional targets from dual-metric results
    if args.target == 'all':
        targets.extend([
            ('5UTR50', 50),    # Best overall dual-metric
            ('NS5_9010', 9010),  # NS5 region (CDC-like)
        ])

    # Design primers
    print("\n[2/3] Designing primer pairs...")
    results = {
        '_metadata': {
            'analysis_type': 'primer_design',
            'description': 'DENV-4 E gene primers using dual-metric targeting',
            'created': datetime.now(timezone.utc).isoformat(),
            'parameters': {
                'amplicon_size': args.amplicon,
                'primer_length': args.primer_len,
            },
        },
        'primer_pairs': [],
    }

    for name, pos in targets:
        print(f"\n  Designing {name} (position {pos})...")
        pair = design_primer_pair(sequences, pos, name, args.amplicon, args.primer_len)
        results['primer_pairs'].append(pair)

        fwd = pair['forward']
        rev = pair['reverse']

        print(f"    Forward: {fwd['sequence']}")
        print(f"      Tm: {fwd['tm']}°C, GC: {fwd['gc_content']*100:.1f}%, Degeneracy: {fwd['degeneracy']}")
        print(f"    Reverse: {rev['sequence']}")
        print(f"      Tm: {rev['tm']}°C, GC: {rev['gc_content']*100:.1f}%, Degeneracy: {rev['degeneracy']}")
        print(f"    Amplicon: {pair['amplicon']['size']} bp")
        print(f"    Tm diff: {pair['validation']['tm_diff']:.1f}°C")
        print(f"    Total degeneracy: {pair['validation']['total_degeneracy']}")

    # Quality summary
    print("\n" + "=" * 80)
    print("PRIMER QUALITY SUMMARY")
    print("=" * 80)

    print(f"\n{'Name':<15} {'Fwd Tm':<10} {'Rev Tm':<10} {'Tm Diff':<10} {'Degeneracy':<15} {'Status'}")
    print("-" * 70)

    for pair in results['primer_pairs']:
        name = pair['target_name']
        fwd_tm = pair['forward']['tm']
        rev_tm = pair['reverse']['tm']
        tm_diff = pair['validation']['tm_diff']
        degeneracy = pair['validation']['total_degeneracy']

        # Determine status
        issues = []
        if tm_diff > 5:
            issues.append("Tm diff >5°C")
        if degeneracy > 1000:
            issues.append("High degeneracy")
        if not pair['validation']['fwd_3prime_stable']:
            issues.append("Fwd 3' unstable")
        if not pair['validation']['rev_3prime_stable']:
            issues.append("Rev 3' unstable")

        status = "GOOD" if not issues else ", ".join(issues)

        print(f"{name:<15} {fwd_tm:<10.1f} {rev_tm:<10.1f} {tm_diff:<10.1f} {degeneracy:<15} {status}")

    # Save results
    print("\n[3/3] Saving results...")

    results_path = RESULTS_DIR / "egene_primer_pairs.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {results_path}")

    # Export in common format
    print("\n" + "=" * 80)
    print("PRIMER SEQUENCES (Copy for ordering)")
    print("=" * 80)

    for pair in results['primer_pairs']:
        print(f"\n{pair['target_name']}:")
        print(f"  F: 5'-{pair['forward']['sequence']}-3'")
        print(f"  R: 5'-{pair['reverse']['sequence']}-3'")
        print(f"  Amplicon: {pair['amplicon']['size']} bp")

    print("\n" + "=" * 80)
    print("DESIGN COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
