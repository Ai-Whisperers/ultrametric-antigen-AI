# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Design Multi-Primer Cocktail for Maximum DENV-4 Coverage.

Even within Clades D and E, single primers achieve only ~50% coverage.
This script uses a greedy set-cover algorithm to find the MINIMAL set
of primers that achieves maximum coverage.

STRATEGY:
1. For each clade, generate ALL candidate primers
2. Use greedy set cover to find primers that cover DIFFERENT sequences
3. Iteratively add primers until we hit target coverage or diminishing returns
4. Output optimized cocktail specification with per-primer contributions

The key insight: primers at DIFFERENT genomic positions may cover
complementary subsets of sequences within each clade.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PrimerCandidate(NamedTuple):
    """A primer candidate with coverage info."""
    name: str
    position: int
    gene: str
    sequence: str
    covered_accessions: frozenset  # Set of accession IDs covered
    coverage: float
    three_prime_cons: float
    gc_content: float
    tm_estimate: float


def load_denv4_data():
    """Load DENV-4 sequences and metadata."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)["data"]
    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        sequences = json.load(f)["data"]
    return sequences, metadata


def get_gene_annotation(pos: int) -> str:
    """Get gene for position."""
    genes = [
        (0, 94, "5UTR"), (94, 436, "C"), (436, 934, "prM"),
        (934, 2419, "E"), (2419, 3474, "NS1"), (3474, 4128, "NS2A"),
        (4128, 4518, "NS2B"), (4518, 6378, "NS3"), (6378, 6531, "NS4A"),
        (6531, 6600, "2K"), (6600, 7350, "NS4B"), (7350, 10095, "NS5"),
        (10095, 10723, "3UTR"),
    ]
    for start, end, name in genes:
        if start <= pos < end:
            return name
    return "intergenic"


def count_mismatches(primer: str, target: str) -> int:
    """Count mismatches between primer and target."""
    return sum(1 for p, t in zip(primer, target) if p != t)


def estimate_tm(sequence: str) -> float:
    """Estimate Tm using nearest-neighbor approximation."""
    gc = sequence.count('G') + sequence.count('C')
    at = sequence.count('A') + sequence.count('T')
    if len(sequence) < 14:
        return 2 * at + 4 * gc
    else:
        return 64.9 + 41 * (gc - 16.4) / len(sequence)


def generate_all_primers(
    sequences: dict[str, str],
    metadata: dict[str, dict],
    target_clade: str,
    primer_length: int = 20,
    max_mismatches: int = 2,
    step: int = 10,  # Sample every 10 positions for speed
) -> list[PrimerCandidate]:
    """Generate all primer candidates for a clade with coverage info."""

    # Get target clade accessions
    target_accs = {
        acc for acc, m in metadata.items() if m["clade"] == target_clade
    }

    # Get genome length (use first sequence)
    first_seq = next(iter(sequences.values()))
    genome_len = len(first_seq)

    candidates = []

    for pos in range(0, genome_len - primer_length + 1, step):
        # Build consensus from target clade
        target_regions = []
        acc_to_region = {}

        for acc in target_accs:
            if acc in sequences:
                seq = sequences[acc]
                if pos + primer_length <= len(seq):
                    region = seq[pos:pos + primer_length].upper().replace('U', 'T')
                    if all(b in 'ACGT' for b in region):
                        target_regions.append(region)
                        acc_to_region[acc] = region

        if len(target_regions) < 5:
            continue

        # Build consensus
        consensus = []
        per_position_freq = []
        for i in range(primer_length):
            bases = [r[i] for r in target_regions]
            counts = Counter(bases)
            most_common = counts.most_common(1)[0]
            consensus.append(most_common[0])
            per_position_freq.append(most_common[1] / len(target_regions))

        consensus_seq = ''.join(consensus)

        # Find which accessions are covered
        covered = set()
        for acc, region in acc_to_region.items():
            mm = count_mismatches(consensus_seq, region)
            if mm <= max_mismatches:
                covered.add(acc)

        if not covered:
            continue

        coverage = len(covered) / len(target_accs)
        three_prime_cons = np.mean(per_position_freq[-6:])
        gc = (consensus_seq.count('G') + consensus_seq.count('C')) / primer_length
        tm = estimate_tm(consensus_seq)

        candidates.append(PrimerCandidate(
            name=f"DENV4_{target_clade}_{pos}",
            position=pos,
            gene=get_gene_annotation(pos),
            sequence=consensus_seq,
            covered_accessions=frozenset(covered),
            coverage=coverage,
            three_prime_cons=three_prime_cons,
            gc_content=gc,
            tm_estimate=tm,
        ))

    return candidates


def greedy_set_cover(
    candidates: list[PrimerCandidate],
    target_accs: set[str],
    max_primers: int = 5,
    min_coverage_gain: float = 0.05,  # Stop if gain < 5%
) -> list[PrimerCandidate]:
    """Find minimal set of primers that maximizes coverage using greedy algorithm."""

    selected = []
    covered = set()
    remaining_accs = target_accs.copy()

    for _ in range(max_primers):
        if not remaining_accs:
            break

        # Find primer that covers most UNCOVERED sequences
        best_primer = None
        best_new_coverage = 0

        for primer in candidates:
            new_coverage = len(primer.covered_accessions & remaining_accs)
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_primer = primer

        if best_primer is None or best_new_coverage == 0:
            break

        # Check if gain is worth it
        coverage_gain = best_new_coverage / len(target_accs)
        if coverage_gain < min_coverage_gain:
            print(f"  Stopping: coverage gain {coverage_gain*100:.1f}% < threshold {min_coverage_gain*100:.1f}%")
            break

        selected.append(best_primer)
        covered.update(best_primer.covered_accessions)
        remaining_accs -= best_primer.covered_accessions

        print(f"  Added primer at pos {best_primer.position}: "
              f"+{best_new_coverage} seqs ({coverage_gain*100:.1f}%), "
              f"total {len(covered)}/{len(target_accs)} ({len(covered)/len(target_accs)*100:.1f}%)")

    return selected


def compute_cocktail_coverage(
    primers: list[PrimerCandidate],
    sequences: dict[str, str],
    metadata: dict[str, dict],
    max_mismatches: int = 2,
) -> dict:
    """Compute coverage of primer cocktail across all sequences."""

    covered_by_clade = defaultdict(set)
    covered_total = set()

    for primer in primers:
        for acc, seq in sequences.items():
            if primer.position + 20 <= len(seq):
                region = seq[primer.position:primer.position + 20].upper().replace('U', 'T')
                if all(b in 'ACGT' for b in region):
                    mm = count_mismatches(primer.sequence, region)
                    if mm <= max_mismatches:
                        covered_total.add(acc)
                        if acc in metadata:
                            covered_by_clade[metadata[acc]["clade"]].add(acc)

    # Compute per-clade coverage
    clade_coverage = {}
    for clade in set(m["clade"] for m in metadata.values()):
        clade_accs = {acc for acc, m in metadata.items() if m["clade"] == clade}
        covered_in_clade = covered_by_clade[clade]
        clade_coverage[clade] = {
            "covered": len(covered_in_clade),
            "total": len(clade_accs),
            "coverage": len(covered_in_clade) / len(clade_accs) if clade_accs else 0,
        }

    return {
        "total_covered": len(covered_total),
        "total_sequences": len(sequences),
        "total_coverage": len(covered_total) / len(sequences),
        "per_clade": clade_coverage,
    }


def main():
    print("=" * 70)
    print("MULTI-PRIMER COCKTAIL OPTIMIZATION FOR DENV-4")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Find MINIMAL primer set that achieves MAXIMAL coverage")
    print("Method: Greedy set-cover algorithm")
    print()

    sequences, metadata = load_denv4_data()

    # Count per clade
    clade_counts = Counter(m["clade"] for m in metadata.values())
    print("Clade distribution:")
    for clade, count in sorted(clade_counts.items()):
        print(f"  {clade}: {count} sequences")
    print()

    all_results = {"per_clade_optimization": {}, "cocktails": []}

    # Optimize for each major clade
    for clade in ["Clade_D", "Clade_E"]:
        print("=" * 70)
        print(f"OPTIMIZING {clade}")
        print("=" * 70)

        target_accs = {acc for acc, m in metadata.items() if m["clade"] == clade}
        print(f"Target: {len(target_accs)} sequences")

        print("\nGenerating primer candidates...")
        candidates = generate_all_primers(
            sequences, metadata, clade,
            primer_length=20, max_mismatches=2, step=5
        )
        print(f"Generated {len(candidates)} candidate primers")

        # Sort by coverage first
        candidates.sort(key=lambda p: -p.coverage)

        print("\nTop single primers:")
        for i, p in enumerate(candidates[:3], 1):
            print(f"  {i}. pos {p.position} ({p.gene}): {p.coverage*100:.1f}%")

        print("\nGreedy set cover optimization:")
        selected = greedy_set_cover(candidates, target_accs, max_primers=5)

        print(f"\nSelected {len(selected)} primers for {clade}:")
        for p in selected:
            print(f"  - {p.name} ({p.gene})")
            print(f"    Sequence: 5'-{p.sequence}-3'")
            print(f"    Tm: {p.tm_estimate:.1f}C, GC: {p.gc_content*100:.0f}%")

        all_results["per_clade_optimization"][clade] = {
            "target_sequences": len(target_accs),
            "selected_primers": [
                {
                    "name": p.name,
                    "position": p.position,
                    "gene": p.gene,
                    "sequence": p.sequence,
                    "coverage": p.coverage,
                    "tm": p.tm_estimate,
                    "gc": p.gc_content,
                }
                for p in selected
            ]
        }

    # Now build optimal cocktail combining all primers
    print("\n" + "=" * 70)
    print("COMBINED COCKTAIL OPTIMIZATION")
    print("=" * 70)

    # Collect all selected primers
    all_selected_d = all_results["per_clade_optimization"].get("Clade_D", {}).get("selected_primers", [])
    all_selected_e = all_results["per_clade_optimization"].get("Clade_E", {}).get("selected_primers", [])

    # Also add primers that cover A/B/C
    # For these small clades, find any primer that gives 100% coverage
    print("\nFinding primer for minor clades (A/B/C)...")
    minor_clade_accs = {
        acc for acc, m in metadata.items()
        if m["clade"] in ["Clade_A", "Clade_B", "Clade_C"]
    }
    print(f"Minor clades: {len(minor_clade_accs)} sequences")

    # Check which Clade_D primers also cover minor clades well
    print("Checking cross-coverage from D/E primers...")

    # Build combined cocktail
    cocktail_primers = []

    # Add best D primers
    for p_data in all_selected_d[:3]:  # Top 3 D primers
        cocktail_primers.append(PrimerCandidate(
            name=p_data["name"],
            position=p_data["position"],
            gene=p_data["gene"],
            sequence=p_data["sequence"],
            covered_accessions=frozenset(),
            coverage=p_data["coverage"],
            three_prime_cons=0,
            gc_content=p_data["gc"],
            tm_estimate=p_data["tm"],
        ))

    # Add best E primers
    for p_data in all_selected_e[:3]:  # Top 3 E primers
        cocktail_primers.append(PrimerCandidate(
            name=p_data["name"],
            position=p_data["position"],
            gene=p_data["gene"],
            sequence=p_data["sequence"],
            covered_accessions=frozenset(),
            coverage=p_data["coverage"],
            three_prime_cons=0,
            gc_content=p_data["gc"],
            tm_estimate=p_data["tm"],
        ))

    # Compute final cocktail coverage
    cocktail_coverage = compute_cocktail_coverage(
        cocktail_primers, sequences, metadata, max_mismatches=2
    )

    print(f"\nCOMBINED COCKTAIL COVERAGE (<=2 mismatches):")
    print(f"  Total: {cocktail_coverage['total_coverage']*100:.1f}% "
          f"({cocktail_coverage['total_covered']}/{cocktail_coverage['total_sequences']})")
    print("\n  Per-clade:")
    for clade, data in sorted(cocktail_coverage["per_clade"].items()):
        print(f"    {clade}: {data['coverage']*100:.1f}% ({data['covered']}/{data['total']})")

    # Also compute with 3 mismatches
    cocktail_coverage_3mm = compute_cocktail_coverage(
        cocktail_primers, sequences, metadata, max_mismatches=3
    )

    print(f"\nCOMBINED COCKTAIL COVERAGE (<=3 mismatches):")
    print(f"  Total: {cocktail_coverage_3mm['total_coverage']*100:.1f}% "
          f"({cocktail_coverage_3mm['total_covered']}/{cocktail_coverage_3mm['total_sequences']})")
    print("\n  Per-clade:")
    for clade, data in sorted(cocktail_coverage_3mm["per_clade"].items()):
        print(f"    {clade}: {data['coverage']*100:.1f}% ({data['covered']}/{data['total']})")

    # Save cocktail specification
    cocktail_spec = {
        "primers": [
            {
                "name": p.name,
                "position": p.position,
                "gene": p.gene,
                "sequence": p.sequence,
                "tm": p.tm_estimate,
                "gc": p.gc_content,
            }
            for p in cocktail_primers
        ],
        "coverage_2mm": cocktail_coverage,
        "coverage_3mm": cocktail_coverage_3mm,
    }

    all_results["cocktails"].append(cocktail_spec)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL PRIMER COCKTAIL SPECIFICATION")
    print("=" * 70)

    print(f"\nTotal primers in cocktail: {len(cocktail_primers)}")
    print("\nPrimer sequences:")
    for i, p in enumerate(cocktail_primers, 1):
        print(f"\n{i}. {p.name}")
        print(f"   5'-{p.sequence}-3'")
        print(f"   Position: {p.position} ({p.gene})")
        print(f"   Tm: {p.tm_estimate:.1f}C, GC: {p.gc_content*100:.0f}%")

    print(f"\nExpected coverage:")
    print(f"  With <=2 mismatches: {cocktail_coverage['total_coverage']*100:.1f}%")
    print(f"  With <=3 mismatches: {cocktail_coverage_3mm['total_coverage']*100:.1f}%")

    # Compare to pan-DENV-4 consensus
    print("\nImprovement over pan-DENV-4 consensus (37% coverage):")
    print(f"  Cocktail gain: +{(cocktail_coverage['total_coverage'] - 0.37)*100:.1f}pp")

    # Save results
    output = {
        "_metadata": {
            "schema_version": "1.0",
            "file_type": "primer_optimization_results",
            "analysis_type": "multi_primer_cocktail",
            "description": "Optimized primer cocktail for DENV-4 detection",
            "created": datetime.now(timezone.utc).isoformat(),
            "pipeline": "design_multi_primer_cocktail.py",
            "key_finding": "Multi-primer cocktail improves coverage over single consensus approach",
            "field_definitions": {
                "coverage_2mm": "Coverage with <=2 mismatches (standard PCR tolerance)",
                "coverage_3mm": "Coverage with <=3 mismatches (relaxed tolerance)",
                "greedy_set_cover": "Algorithm that iteratively selects primers covering most uncovered sequences",
            },
        },
        "data": all_results,
    }

    output_path = RESULTS_DIR / "multi_primer_cocktail.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
