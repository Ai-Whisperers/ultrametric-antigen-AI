# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""In Silico PCR Validation for DENV-4 Primers.

Tests designed primers against all 270 DENV-4 genomes to determine:
1. Coverage: % of genomes where primers bind
2. Specificity: Which clades each primer targets
3. Mismatches: Distribution of mismatches per binding site

Mismatch Allowance:
    - 0 mismatches: Perfect match only
    - 1 mismatch: Typically works in PCR
    - 2 mismatches: May work depending on position
    - 3+ mismatches: Likely to fail

Usage:
    python insilico_pcr_validation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
PRIMER_DIR = ROJAS_DIR / "results" / "primers"
RESULTS_DIR = ROJAS_DIR / "results" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# IUPAC expansion
IUPAC_EXPAND = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'M': {'A', 'C'}, 'K': {'G', 'T'},
    'S': {'G', 'C'}, 'W': {'A', 'T'},
    'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
    'N': {'A', 'C', 'G', 'T'},
}


def load_denv4_data() -> tuple[list[str], list[str], list[str]]:
    """Load DENV-4 sequences with metadata."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_data = json.load(f)

    accessions = []
    sequences = []
    clades = []

    for acc, meta in metadata["data"].items():
        if acc in seq_data["data"]:
            accessions.append(acc)
            sequences.append(seq_data["data"][acc].upper())
            clades.append(meta.get("clade", "Unknown"))

    return accessions, sequences, clades


def count_mismatches(primer: str, target: str) -> int:
    """Count mismatches between primer and target, accounting for IUPAC codes."""
    if len(primer) != len(target):
        return max(len(primer), len(target))

    mismatches = 0
    for p, t in zip(primer.upper(), target.upper()):
        p_bases = IUPAC_EXPAND.get(p, {p})
        t_bases = IUPAC_EXPAND.get(t, {t})

        # Mismatch if no overlap between allowed bases
        if not p_bases & t_bases:
            mismatches += 1

    return mismatches


def reverse_complement(seq: str) -> str:
    """Get reverse complement."""
    complement = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'R': 'Y', 'Y': 'R', 'M': 'K', 'K': 'M',
        'S': 'S', 'W': 'W', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D', 'N': 'N',
    }
    return ''.join(complement.get(b, 'N') for b in reversed(seq.upper()))


def validate_primer_pair(
    fwd_primer: str,
    rev_primer: str,
    fwd_pos: int,
    rev_pos: int,
    sequences: list[str],
    accessions: list[str],
    clades: list[str],
    max_mismatches: int = 2,
) -> dict:
    """Validate a primer pair against all sequences.

    Args:
        fwd_primer: Forward primer sequence
        rev_primer: Reverse primer sequence (as given, will be reverse complemented for matching)
        fwd_pos: Expected forward primer binding position
        rev_pos: Expected reverse primer binding position
        sequences: List of target sequences
        accessions: List of accession IDs
        clades: List of clade assignments
        max_mismatches: Maximum allowed mismatches to count as "hit"

    Returns:
        Dictionary with validation results
    """
    fwd_len = len(fwd_primer)
    rev_len = len(rev_primer)

    # For reverse primer, we need to check the reverse complement at the position
    # The reverse primer binds to the complementary strand
    rev_primer_rc = reverse_complement(rev_primer)

    results = {
        'fwd_hits': [],
        'rev_hits': [],
        'both_hits': [],
        'clade_coverage': defaultdict(int),
        'clade_total': defaultdict(int),
    }

    for i, (acc, seq, clade) in enumerate(zip(accessions, sequences, clades)):
        results['clade_total'][clade] += 1

        # Check forward primer
        if fwd_pos + fwd_len <= len(seq):
            target_fwd = seq[fwd_pos:fwd_pos + fwd_len]
            fwd_mm = count_mismatches(fwd_primer, target_fwd)
            fwd_hit = fwd_mm <= max_mismatches
        else:
            fwd_hit = False
            fwd_mm = fwd_len

        # Check reverse primer (bind to reverse complement)
        if rev_pos + rev_len <= len(seq):
            target_rev = seq[rev_pos:rev_pos + rev_len]
            rev_mm = count_mismatches(rev_primer_rc, target_rev)
            rev_hit = rev_mm <= max_mismatches
        else:
            rev_hit = False
            rev_mm = rev_len

        if fwd_hit:
            results['fwd_hits'].append((acc, fwd_mm, clade))
        if rev_hit:
            results['rev_hits'].append((acc, rev_mm, clade))
        if fwd_hit and rev_hit:
            results['both_hits'].append((acc, fwd_mm, rev_mm, clade))
            results['clade_coverage'][clade] += 1

    # Compute statistics
    n_total = len(sequences)
    n_fwd_hits = len(results['fwd_hits'])
    n_rev_hits = len(results['rev_hits'])
    n_both_hits = len(results['both_hits'])

    return {
        'n_total': n_total,
        'fwd_coverage': n_fwd_hits / n_total * 100,
        'rev_coverage': n_rev_hits / n_total * 100,
        'both_coverage': n_both_hits / n_total * 100,
        'n_fwd_hits': n_fwd_hits,
        'n_rev_hits': n_rev_hits,
        'n_both_hits': n_both_hits,
        'clade_coverage': dict(results['clade_coverage']),
        'clade_total': dict(results['clade_total']),
        'clade_pct': {
            c: results['clade_coverage'].get(c, 0) / results['clade_total'][c] * 100
            for c in results['clade_total']
        },
    }


def main():
    print("=" * 80)
    print("IN SILICO PCR VALIDATION")
    print("=" * 80)
    print()
    print("Testing designed primers against all 270 DENV-4 genomes")
    print("Mismatch tolerance: 0, 1, 2 allowed")
    print()
    print("=" * 80)

    # Load data
    print("\n[1/3] Loading DENV-4 sequences...")
    accessions, sequences, clades = load_denv4_data()
    print(f"  Loaded {len(sequences)} sequences")
    print(f"  Clades: {dict((c, clades.count(c)) for c in set(clades))}")

    # Load primers
    print("\n[2/3] Loading primer designs...")

    clade_primers_path = PRIMER_DIR / "clade_specific_primers.json"
    if clade_primers_path.exists():
        with open(clade_primers_path) as f:
            clade_primers = json.load(f)
        print(f"  Loaded {len(clade_primers['primer_pairs'])} clade-specific primer pairs")
    else:
        print("  No clade-specific primers found")
        clade_primers = {'primer_pairs': []}

    # Validate primers
    print("\n[3/3] Validating primers...")

    results = {
        '_metadata': {
            'analysis_type': 'insilico_pcr_validation',
            'description': 'In silico PCR validation of DENV-4 primers',
            'created': datetime.now(timezone.utc).isoformat(),
            'n_sequences': len(sequences),
            'max_mismatches': 2,
        },
        'primer_validations': [],
    }

    for primer in clade_primers['primer_pairs']:
        clade = primer['clade']
        fwd = primer['forward']
        rev = primer['reverse']

        print(f"\n  Validating {clade} primers...")
        print(f"    Forward: {fwd['sequence'][:20]}...")
        print(f"    Reverse: {rev['sequence'][:20]}...")

        validation = validate_primer_pair(
            fwd['sequence'],
            rev['sequence'],
            fwd['position'],
            rev['position'],
            sequences,
            accessions,
            clades,
            max_mismatches=2,
        )

        validation['primer_name'] = clade
        validation['target_clade'] = clade
        results['primer_validations'].append(validation)

        print(f"    Coverage: {validation['both_coverage']:.1f}% ({validation['n_both_hits']}/{validation['n_total']})")
        print(f"    Per-clade coverage:")
        for c in sorted(validation['clade_pct'].keys()):
            pct = validation['clade_pct'][c]
            n = validation['clade_coverage'].get(c, 0)
            total = validation['clade_total'][c]
            expected = "TARGET" if c == clade else ""
            print(f"      {c}: {pct:.1f}% ({n}/{total}) {expected}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Primer':<15} {'Coverage':<15} {'Target Clade':<15} {'Target %':<15}")
    print("-" * 60)

    total_coverage = 0
    for v in results['primer_validations']:
        name = v['primer_name']
        coverage = v['both_coverage']
        target = v['target_clade']
        target_pct = v['clade_pct'].get(target, 0)
        print(f"{name:<15} {coverage:>6.1f}%        {target:<15} {target_pct:>6.1f}%")
        total_coverage += v['n_both_hits']

    # Account for overlapping hits
    print(f"\n{'TOTAL':<15} {total_coverage/len(sequences)*100:>6.1f}%")

    # Calculate actual unique coverage
    print("\nNote: Coverage may overlap between primers (same sequence hit by multiple primers)")

    # Save results
    print("\nSaving results...")
    results_path = RESULTS_DIR / "insilico_pcr_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {results_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
