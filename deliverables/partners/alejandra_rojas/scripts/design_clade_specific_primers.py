# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Design Clade-Specific Primer Cocktail for DENV-4.

Since pan-DENV-4 primers are impossible (degeneracy >10^20), this script
designs CLADE-SPECIFIC primers that together provide complete coverage.

Strategy:
    1. Group sequences by clade (5 clades)
    2. For each clade, find the most conserved region
    3. Design primers with minimal degeneracy
    4. Stagger amplicon sizes for multiplex compatibility

Clade Distribution (n=270):
    - Clade_A: 2 sequences (0.7%)
    - Clade_B: 3 sequences (1.1%)
    - Clade_C: 2 sequences (0.7%)
    - Clade_D: 52 sequences (19.3%)
    - Clade_E: 211 sequences (78.1%)

Amplicon Sizes (for multiplex compatibility):
    - Clade_A: 100 bp
    - Clade_B: 120 bp
    - Clade_C: 140 bp
    - Clade_D: 160 bp
    - Clade_E: 180 bp

Usage:
    python design_clade_specific_primers.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter1d

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

DEGENERACY = {
    'A': 1, 'C': 1, 'G': 1, 'T': 1,
    'R': 2, 'Y': 2, 'M': 2, 'K': 2, 'S': 2, 'W': 2,
    'B': 3, 'D': 3, 'H': 3, 'V': 3,
    'N': 4,
}

# Amplicon sizes for multiplex (staggered by 20bp)
AMPLICON_SIZES = {
    'Clade_A': 100,
    'Clade_B': 120,
    'Clade_C': 140,
    'Clade_D': 160,
    'Clade_E': 180,
}


def load_denv4_by_clade() -> dict[str, list[str]]:
    """Load DENV-4 sequences grouped by clade."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_data = json.load(f)

    clade_sequences = {}
    for acc, meta in metadata["data"].items():
        if acc in seq_data["data"]:
            clade = meta.get("clade", "Unknown")
            if clade not in clade_sequences:
                clade_sequences[clade] = []
            clade_sequences[clade].append(seq_data["data"][acc].upper())

    return clade_sequences


def compute_shannon_entropy(nucleotides: list[str]) -> float:
    """Compute Shannon entropy for a column of nucleotides."""
    if not nucleotides:
        return 2.0

    counts = Counter(nucleotides)
    total = len(nucleotides)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def find_best_conserved_region(sequences: list[str], window_size: int = 20,
                                top_n: int = 5) -> list[tuple[int, float]]:
    """Find the most conserved regions in a set of sequences.

    Returns list of (position, mean_entropy) tuples.
    """
    if not sequences:
        return []

    min_len = min(len(s) for s in sequences)
    entropies = []

    for pos in range(min_len):
        nucs = [s[pos] for s in sequences if pos < len(s) and s[pos] in 'ACGT']
        entropies.append(compute_shannon_entropy(nucs))

    entropies = np.array(entropies)

    # Smooth over window
    smoothed = uniform_filter1d(entropies, window_size)

    # Find positions with lowest entropy
    candidates = []
    for pos in range(0, len(smoothed) - window_size, window_size // 2):
        candidates.append((pos, smoothed[pos]))

    # Sort by entropy (lowest first)
    candidates.sort(key=lambda x: x[1])

    return candidates[:top_n]


def get_consensus_with_degeneracy(sequences: list[str], start: int, length: int,
                                   min_freq: float = 0.85) -> str:
    """Get consensus sequence with IUPAC degeneracy codes."""
    consensus = []

    for pos in range(start, start + length):
        nucs = [s[pos].upper() for s in sequences if pos < len(s) and s[pos].upper() in 'ACGT']

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
            for nuc, count in counts.items():
                if count / total >= 0.05:
                    significant.add(nuc)

        if not significant:
            significant = set(counts.keys())

        code = IUPAC_CODES.get(frozenset(significant), 'N')
        consensus.append(code)

    return ''.join(consensus)


def reverse_complement(seq: str) -> str:
    """Get reverse complement of sequence."""
    complement = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'R': 'Y', 'Y': 'R', 'M': 'K', 'K': 'M',
        'S': 'S', 'W': 'W', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D', 'N': 'N',
    }
    return ''.join(complement.get(b, 'N') for b in reversed(seq))


def calculate_tm(seq: str) -> float:
    """Calculate melting temperature."""
    gc_contribution = {
        'G': 1.0, 'C': 1.0, 'A': 0.0, 'T': 0.0,
        'S': 1.0, 'W': 0.0,
        'R': 0.5, 'Y': 0.5, 'M': 0.5, 'K': 0.5,
        'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33,
        'N': 0.5,
    }
    gc_count = sum(gc_contribution.get(b, 0.5) for b in seq.upper())
    n = len(seq)

    if n < 14:
        tm = 2 * (seq.count('A') + seq.count('T')) + 4 * gc_count
    else:
        tm = 64.9 + 41 * (gc_count - 16.4) / n

    return round(tm, 1)


def calculate_gc(seq: str) -> float:
    """Calculate GC content."""
    gc_contribution = {
        'G': 1.0, 'C': 1.0, 'A': 0.0, 'T': 0.0,
        'S': 1.0, 'W': 0.0,
        'R': 0.5, 'Y': 0.5, 'M': 0.5, 'K': 0.5,
        'B': 0.67, 'V': 0.67, 'D': 0.33, 'H': 0.33,
        'N': 0.5,
    }
    gc = sum(gc_contribution.get(b, 0.5) for b in seq.upper())
    return round(gc / len(seq), 3)


def calculate_degeneracy(seq: str) -> int:
    """Calculate total degeneracy."""
    total = 1
    for b in seq.upper():
        total *= DEGENERACY.get(b, 4)
    return total


def design_clade_primer(sequences: list[str], clade: str, target_pos: int,
                        amplicon_size: int, primer_len: int = 20) -> dict:
    """Design a primer pair for a specific clade."""
    # Forward primer
    fwd_seq = get_consensus_with_degeneracy(sequences, target_pos, primer_len)

    # Reverse primer
    rev_start = target_pos + amplicon_size - primer_len
    rev_seq_fwd = get_consensus_with_degeneracy(sequences, rev_start, primer_len)
    rev_seq = reverse_complement(rev_seq_fwd)

    fwd_deg = calculate_degeneracy(fwd_seq)
    rev_deg = calculate_degeneracy(rev_seq)
    total_deg = fwd_deg * rev_deg

    return {
        'clade': clade,
        'n_sequences': len(sequences),
        'target_position': target_pos,
        'forward': {
            'name': f"DENV4_{clade}_F",
            'sequence': fwd_seq,
            'position': target_pos,
            'tm': calculate_tm(fwd_seq),
            'gc': calculate_gc(fwd_seq),
            'degeneracy': fwd_deg,
        },
        'reverse': {
            'name': f"DENV4_{clade}_R",
            'sequence': rev_seq,
            'position': rev_start,
            'tm': calculate_tm(rev_seq),
            'gc': calculate_gc(rev_seq),
            'degeneracy': rev_deg,
        },
        'amplicon_size': amplicon_size,
        'total_degeneracy': total_deg,
        'usable': total_deg < 10000,  # Practical limit for degenerate primers
    }


def main():
    print("=" * 80)
    print("CLADE-SPECIFIC PRIMER COCKTAIL DESIGN")
    print("=" * 80)
    print()
    print("Strategy: Design separate primers for each of 5 DENV-4 clades")
    print("Rationale: Pan-DENV-4 primers have >10^20 degeneracy (impossible)")
    print()
    print("Amplicon sizes (staggered for multiplex):")
    for clade, size in AMPLICON_SIZES.items():
        print(f"  {clade}: {size} bp")
    print()
    print("=" * 80)

    # Load sequences by clade
    print("\n[1/4] Loading sequences by clade...")
    clade_sequences = load_denv4_by_clade()

    for clade, seqs in sorted(clade_sequences.items()):
        print(f"  {clade}: {len(seqs)} sequences")

    # Find best target for each clade
    print("\n[2/4] Finding most conserved regions per clade...")

    best_targets = {}
    for clade, seqs in sorted(clade_sequences.items()):
        if len(seqs) < 2:
            print(f"  {clade}: Only {len(seqs)} sequence, using position 2400 (E gene)")
            best_targets[clade] = 2400
            continue

        candidates = find_best_conserved_region(seqs, window_size=20, top_n=3)
        if candidates:
            best_pos, best_entropy = candidates[0]
            print(f"  {clade}: Best position {best_pos} (entropy={best_entropy:.3f})")
            best_targets[clade] = best_pos
        else:
            print(f"  {clade}: No candidates found, using position 2400")
            best_targets[clade] = 2400

    # Design primers
    print("\n[3/4] Designing clade-specific primers...")

    results = {
        '_metadata': {
            'analysis_type': 'clade_specific_primer_design',
            'description': 'DENV-4 clade-specific primer cocktail',
            'created': datetime.now(timezone.utc).isoformat(),
            'strategy': 'Separate primers per clade due to high within-serotype diversity',
        },
        'primer_pairs': [],
        'summary': {
            'total_clades': len(clade_sequences),
            'total_sequences': sum(len(s) for s in clade_sequences.values()),
        },
    }

    for clade in sorted(clade_sequences.keys()):
        seqs = clade_sequences[clade]
        target_pos = best_targets[clade]
        amplicon_size = AMPLICON_SIZES.get(clade, 150)

        primer = design_clade_primer(seqs, clade, target_pos, amplicon_size)
        results['primer_pairs'].append(primer)

        print(f"\n  {clade} ({len(seqs)} sequences):")
        print(f"    Target: position {target_pos}")
        print(f"    Forward: {primer['forward']['sequence']}")
        print(f"      Tm={primer['forward']['tm']}°C, GC={primer['forward']['gc']*100:.1f}%, "
              f"Deg={primer['forward']['degeneracy']}")
        print(f"    Reverse: {primer['reverse']['sequence']}")
        print(f"      Tm={primer['reverse']['tm']}°C, GC={primer['reverse']['gc']*100:.1f}%, "
              f"Deg={primer['reverse']['degeneracy']}")
        print(f"    Total degeneracy: {primer['total_degeneracy']}")
        print(f"    Status: {'USABLE' if primer['usable'] else 'HIGH DEGENERACY'}")

    # Summary
    print("\n" + "=" * 80)
    print("COCKTAIL SUMMARY")
    print("=" * 80)

    usable = [p for p in results['primer_pairs'] if p['usable']]
    high_deg = [p for p in results['primer_pairs'] if not p['usable']]

    print(f"\nUsable primer pairs: {len(usable)}/{len(results['primer_pairs'])}")

    coverage_seqs = sum(p['n_sequences'] for p in usable)
    total_seqs = results['summary']['total_sequences']
    print(f"Coverage: {coverage_seqs}/{total_seqs} sequences ({100*coverage_seqs/total_seqs:.1f}%)")

    if usable:
        print("\nUsable primers (degeneracy < 10,000):")
        for p in usable:
            print(f"  {p['clade']}: {p['n_sequences']} seqs, degeneracy={p['total_degeneracy']}")

    if high_deg:
        print("\nHigh degeneracy primers (may need optimization):")
        for p in high_deg:
            print(f"  {p['clade']}: {p['n_sequences']} seqs, degeneracy={p['total_degeneracy']}")

    # Save results
    print("\n[4/4] Saving results...")

    results_path = RESULTS_DIR / "clade_specific_primers.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {results_path}")

    # Export for ordering
    print("\n" + "=" * 80)
    print("PRIMER SEQUENCES FOR ORDERING")
    print("=" * 80)

    for p in results['primer_pairs']:
        status = "" if p['usable'] else " [HIGH DEG]"
        print(f"\n{p['clade']} ({p['amplicon_size']}bp){status}:")
        print(f"  F: 5'-{p['forward']['sequence']}-3'")
        print(f"  R: 5'-{p['reverse']['sequence']}-3'")

    print("\n" + "=" * 80)
    print("DESIGN COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
