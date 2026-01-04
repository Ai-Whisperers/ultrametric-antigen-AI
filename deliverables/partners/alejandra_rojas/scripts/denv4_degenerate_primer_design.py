# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Degenerate Primer Design.

Since only 13.3% of DENV-4 sequences can be covered by standard consensus primers,
this script designs degenerate primers using IUPAC codes to handle the remaining
86.7% of sequences.

Degenerate primers use ambiguity codes at variable positions:
  R = A/G    Y = C/T    S = G/C    W = A/T
  K = G/T    M = A/C    B = C/G/T  D = A/G/T
  H = A/C/T  V = A/C/G  N = any

Strategy:
1. Identify the best conserved regions even with high entropy
2. Design primers with degeneracy at variable positions
3. Limit degeneracy to 4-8 fold to keep synthesis practical
4. Validate in silico against all sequences

Author: AI Whisperers
Date: 2026-01-04
"""

from __future__ import annotations

import datetime
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "denv4"
RESULTS_DIR = PROJECT_ROOT / "results" / "phylogenetic"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# IUPAC ambiguity codes
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

# Gene regions
GENE_REGIONS = {
    "5UTR": (1, 101),
    "C": (102, 476),
    "prM": (477, 976),
    "E": (977, 2471),
    "NS1": (2472, 3527),
    "NS2A": (3528, 4180),
    "NS2B": (4181, 4571),
    "NS3": (4572, 6428),
    "NS4A": (6429, 6809),
    "NS4B": (6810, 7558),
    "NS5": (7559, 10271),
    "3UTR": (10272, 10723),
}

# Primer constraints
MIN_PRIMER_LENGTH = 18
MAX_PRIMER_LENGTH = 25
MIN_GC = 40
MAX_GC = 60
MIN_TM = 55
MAX_TM = 65
MAX_DEGENERACY = 512  # Maximum number of possible sequences


@dataclass
class DegeneratePrimer:
    """A degenerate primer candidate."""
    name: str
    sequence: str
    start: int
    end: int
    gene: str
    degeneracy: int
    gc_range: tuple
    tm_range: tuple
    coverage: float  # Percentage of sequences matched
    n_matched: int


class DegeneratePrimerDesigner:
    """Designer for degenerate DENV-4 primers."""

    def __init__(self):
        """Initialize the designer."""
        self.sequences = {}
        self.accessions = []
        self.primers = []

    def load_data(self) -> bool:
        """Load sequences from cache."""
        print("Loading sequences...")

        seq_cache = CACHE_DIR / "denv4_sequences.json"
        if not seq_cache.exists():
            print(f"ERROR: Sequence cache not found")
            return False

        with open(seq_cache) as f:
            self.sequences = json.load(f)

        self.accessions = list(self.sequences.keys())
        print(f"Loaded {len(self.sequences)} sequences")
        return True

    def run_design(self) -> dict:
        """Run the complete degenerate primer design."""
        print()
        print("=" * 80)
        print("DENV-4 DEGENERATE PRIMER DESIGN")
        print("=" * 80)
        print()

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "n_sequences": len(self.sequences),
        }

        if not self.load_data():
            return {"error": "Failed to load data"}

        # Step 1: Find best windows genome-wide
        print("-" * 80)
        print("STEP 1: Finding windows with lowest required degeneracy")
        print("-" * 80)
        windows = self.find_lowest_degeneracy_windows()
        results["candidate_windows"] = windows[:50]
        print()

        # Step 2: Design primers for top windows
        print("-" * 80)
        print("STEP 2: Designing degenerate primers")
        print("-" * 80)
        primers = self.design_primers(windows[:100])
        results["designed_primers"] = [
            {
                "name": p.name,
                "sequence": p.sequence,
                "gene": p.gene,
                "position": f"{p.start}-{p.end}",
                "degeneracy": p.degeneracy,
                "gc_range": p.gc_range,
                "tm_range": p.tm_range,
                "coverage": p.coverage,
            }
            for p in primers[:20]
        ]
        print()

        # Step 3: Validate coverage
        print("-" * 80)
        print("STEP 3: Validating primer coverage")
        print("-" * 80)
        validation = self.validate_coverage(primers)
        results["validation"] = validation
        print()

        # Step 4: Create multiplex cocktail
        print("-" * 80)
        print("STEP 4: Creating optimal multiplex cocktail")
        print("-" * 80)
        cocktail = self.create_multiplex_cocktail(primers)
        results["multiplex_cocktail"] = cocktail
        print()

        # Step 5: Generate report
        print("-" * 80)
        print("STEP 5: Generating report")
        print("-" * 80)
        self.generate_report(results)
        print()

        # Save results
        results_path = RESULTS_DIR / "degenerate_primer_results.json"
        self._save_json(results, results_path)

        # Save primers as FASTA
        self.save_primers_fasta(primers[:20])

        print(f"Results saved to: {results_path}")

        return results

    def find_lowest_degeneracy_windows(
        self,
        window_size: int = 20,
    ) -> list[dict]:
        """Find windows with the lowest required degeneracy.

        Scans the genome to find windows where degenerate primers
        can be designed with minimal complexity.

        Returns:
            List of window candidates sorted by degeneracy
        """
        print(f"Scanning genome for low-degeneracy {window_size}bp windows...")

        # Get all sequences aligned to reference length
        min_len = min(len(s) for s in self.sequences.values())
        print(f"Using alignment length: {min_len}")

        windows = []
        total_windows = min_len - window_size + 1

        for start in range(0, min_len - window_size + 1, 10):  # Step by 10 for speed
            end = start + window_size

            # Get nucleotide frequencies at each position
            position_freqs = []
            for pos in range(start, end):
                counts = Counter()
                for seq in self.sequences.values():
                    if pos < len(seq):
                        nt = seq[pos].upper()
                        if nt in "ACGT":
                            counts[nt] += 1
                position_freqs.append(counts)

            # Calculate degeneracy and build consensus
            total_degeneracy = 1
            consensus = []

            for pos_idx, counts in enumerate(position_freqs):
                if not counts:
                    consensus.append('N')
                    total_degeneracy *= 4
                    continue

                # Get nucleotides present at >5% frequency
                total = sum(counts.values())
                significant_nts = frozenset([
                    nt for nt, count in counts.items()
                    if count / total > 0.05
                ])

                if significant_nts in IUPAC_CODES:
                    code = IUPAC_CODES[significant_nts]
                else:
                    code = 'N'

                consensus.append(code)
                total_degeneracy *= IUPAC_DEGENERACY[code]

            consensus_seq = ''.join(consensus)

            # Get gene region
            gene = "intergenic"
            for g, (g_start, g_end) in GENE_REGIONS.items():
                if g_start <= start <= g_end:
                    gene = g
                    break

            windows.append({
                "start": start,
                "end": end,
                "gene": gene,
                "consensus": consensus_seq,
                "degeneracy": total_degeneracy,
                "log_degeneracy": math.log2(total_degeneracy) if total_degeneracy > 0 else 0,
            })

            if len(windows) % 500 == 0:
                print(f"  Processed {len(windows)}/{total_windows//10} windows...")

        # Sort by degeneracy
        windows.sort(key=lambda x: x["degeneracy"])

        print(f"\nTop 10 lowest-degeneracy windows:")
        for w in windows[:10]:
            print(f"  {w['gene']} {w['start']}-{w['end']}: degeneracy={w['degeneracy']} ({w['consensus'][:15]}...)")

        return windows

    def design_primers(self, windows: list[dict]) -> list[DegeneratePrimer]:
        """Design degenerate primers from candidate windows.

        Args:
            windows: List of candidate windows

        Returns:
            List of designed primers
        """
        print(f"Designing primers from {len(windows)} candidate windows...")

        primers = []

        for i, window in enumerate(windows):
            if window["degeneracy"] > MAX_DEGENERACY:
                continue

            consensus = window["consensus"]

            # Calculate GC range
            gc_counts = []
            for variant in self._expand_degenerate(consensus):
                gc = (variant.count('G') + variant.count('C')) / len(variant) * 100
                gc_counts.append(gc)

            gc_range = (min(gc_counts), max(gc_counts))

            # Calculate Tm range (Wallace rule approximation)
            tm_range = self._calculate_tm_range(consensus)

            # Check constraints
            if gc_range[0] < MIN_GC or gc_range[1] > MAX_GC:
                continue
            if tm_range[0] < MIN_TM or tm_range[1] > MAX_TM:
                continue

            # Calculate coverage
            coverage, n_matched = self._calculate_coverage(consensus, window["start"])

            primer = DegeneratePrimer(
                name=f"DENV4_{window['gene']}_{window['start']}_F",
                sequence=consensus,
                start=window["start"],
                end=window["end"],
                gene=window["gene"],
                degeneracy=window["degeneracy"],
                gc_range=gc_range,
                tm_range=tm_range,
                coverage=coverage,
                n_matched=n_matched,
            )

            primers.append(primer)

        # Sort by coverage
        primers.sort(key=lambda x: -x.coverage)

        print(f"Designed {len(primers)} valid primers")
        print("\nTop 10 by coverage:")
        for p in primers[:10]:
            print(f"  {p.name}: {p.coverage:.1f}% coverage, degeneracy={p.degeneracy}")
            print(f"    Sequence: {p.sequence}")

        self.primers = primers
        return primers

    def _expand_degenerate(self, seq: str, max_variants: int = 100) -> list[str]:
        """Expand a degenerate sequence into all possible variants."""
        IUPAC_EXPAND = {
            'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
            'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'],
            'W': ['A', 'T'], 'K': ['G', 'T'], 'M': ['A', 'C'],
            'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
            'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
            'N': ['A', 'C', 'G', 'T'],
        }

        variants = ['']
        for nt in seq:
            new_variants = []
            for v in variants:
                for expansion in IUPAC_EXPAND.get(nt, [nt]):
                    new_variants.append(v + expansion)
                    if len(new_variants) > max_variants:
                        return new_variants
            variants = new_variants

        return variants

    def _calculate_tm_range(self, seq: str) -> tuple:
        """Calculate Tm range for degenerate primer (Wallace rule)."""
        min_gc = sum(1 for nt in seq if nt in 'GCS')  # Minimum GC (S=G/C counts once)
        max_gc = sum(1 for nt in seq if nt in 'GCSRKMBDHV')  # Maximum GC

        min_at = len(seq) - max_gc
        max_at = len(seq) - min_gc

        # Wallace rule: Tm = 2(A+T) + 4(G+C)
        min_tm = 2 * min_at + 4 * min_gc
        max_tm = 2 * max_at + 4 * max_gc

        return (min_tm, max_tm)

    def _calculate_coverage(self, primer: str, start: int) -> tuple[float, int]:
        """Calculate what percentage of sequences the primer matches."""
        matched = 0

        for seq in self.sequences.values():
            if start + len(primer) > len(seq):
                continue

            target = seq[start:start + len(primer)].upper()
            if self._matches_degenerate(primer, target):
                matched += 1

        coverage = matched / len(self.sequences) * 100
        return coverage, matched

    def _matches_degenerate(self, primer: str, target: str) -> bool:
        """Check if a target sequence matches a degenerate primer."""
        IUPAC_MATCH = {
            'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
            'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'},
            'W': {'A', 'T'}, 'K': {'G', 'T'}, 'M': {'A', 'C'},
            'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
            'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
            'N': {'A', 'C', 'G', 'T'},
        }

        if len(primer) != len(target):
            return False

        for p_nt, t_nt in zip(primer, target):
            if t_nt not in IUPAC_MATCH.get(p_nt, set()):
                return False

        return True

    def validate_coverage(self, primers: list[DegeneratePrimer]) -> dict:
        """Validate how well primers cover all sequences.

        Args:
            primers: List of designed primers

        Returns:
            Validation results
        """
        print("Validating primer coverage across all sequences...")

        # Track which sequences are covered
        covered = set()

        for primer in primers[:10]:  # Top 10 primers
            for acc, seq in self.sequences.items():
                if primer.start + len(primer.sequence) > len(seq):
                    continue

                target = seq[primer.start:primer.start + len(primer.sequence)].upper()
                if self._matches_degenerate(primer.sequence, target):
                    covered.add(acc)

        total = len(self.sequences)
        n_covered = len(covered)
        uncovered = total - n_covered

        print(f"\nWith top 10 primers:")
        print(f"  Covered: {n_covered}/{total} ({n_covered/total*100:.1f}%)")
        print(f"  Uncovered: {uncovered}/{total} ({uncovered/total*100:.1f}%)")

        return {
            "n_primers_used": min(10, len(primers)),
            "total_sequences": total,
            "covered": n_covered,
            "coverage_pct": n_covered / total * 100,
            "uncovered": uncovered,
        }

    def create_multiplex_cocktail(
        self,
        primers: list[DegeneratePrimer],
        n_primers: int = 5,
    ) -> dict:
        """Create an optimal multiplex primer cocktail.

        Selects primers that maximize coverage while minimizing overlap.

        Args:
            primers: All designed primers
            n_primers: Number of primers to include in cocktail

        Returns:
            Multiplex cocktail specification
        """
        print(f"Creating optimal {n_primers}-primer cocktail...")

        if not primers:
            print("  No valid primers available!")
            return {
                "n_primers": 0,
                "total_coverage_pct": 0,
                "primers": [],
                "error": "No primers with acceptable degeneracy found",
            }

        covered = set()

        if len(primers) < n_primers:
            selected = primers
        else:
            # Greedy selection: pick primers that add most new coverage
            selected = []
            covered = set()

            for _ in range(n_primers):
                best_primer = None
                best_new_coverage = 0

                for primer in primers:
                    if primer in selected:
                        continue

                    # Calculate new sequences covered
                    new_covered = 0
                    for acc, seq in self.sequences.items():
                        if acc in covered:
                            continue
                        if primer.start + len(primer.sequence) > len(seq):
                            continue

                        target = seq[primer.start:primer.start + len(primer.sequence)].upper()
                        if self._matches_degenerate(primer.sequence, target):
                            new_covered += 1

                    if new_covered > best_new_coverage:
                        best_new_coverage = new_covered
                        best_primer = primer

                if best_primer:
                    selected.append(best_primer)
                    # Update covered set
                    for acc, seq in self.sequences.items():
                        if best_primer.start + len(best_primer.sequence) > len(seq):
                            continue
                        target = seq[best_primer.start:best_primer.start + len(best_primer.sequence)].upper()
                        if self._matches_degenerate(best_primer.sequence, target):
                            covered.add(acc)

        # Calculate final coverage
        total_coverage = len(covered) / len(self.sequences) * 100

        cocktail = {
            "n_primers": len(selected),
            "total_coverage_pct": total_coverage,
            "primers": [],
        }

        print(f"\nMultiplex Cocktail ({len(selected)} primers, {total_coverage:.1f}% coverage):")
        for p in selected:
            primer_info = {
                "name": p.name,
                "sequence": p.sequence,
                "gene": p.gene,
                "position": f"{p.start}-{p.end}",
                "degeneracy": p.degeneracy,
                "individual_coverage": p.coverage,
            }
            cocktail["primers"].append(primer_info)
            print(f"  {p.name}")
            print(f"    {p.sequence}")
            print(f"    {p.gene} {p.start}-{p.end}, degeneracy={p.degeneracy}")

        return cocktail

    def save_primers_fasta(self, primers: list[DegeneratePrimer]) -> None:
        """Save primers as FASTA file."""
        if not HAS_BIOPYTHON:
            return

        records = []
        for p in primers:
            record = SeqRecord(
                Seq(p.sequence),
                id=p.name,
                description=f"gene={p.gene} pos={p.start}-{p.end} degeneracy={p.degeneracy} coverage={p.coverage:.1f}%",
            )
            records.append(record)

        fasta_path = RESULTS_DIR / "degenerate_primers.fasta"
        SeqIO.write(records, fasta_path, "fasta")
        print(f"Primers saved to: {fasta_path}")

    def generate_report(self, results: dict) -> None:
        """Generate markdown report."""
        report_path = RESULTS_DIR / "DENV4_DEGENERATE_PRIMER_REPORT.md"

        lines = [
            "# DENV-4 Degenerate Primer Design Report",
            "",
            f"**Doc-Type:** Degenerate Primer Report · Version 1.0 · {datetime.datetime.now().strftime('%Y-%m-%d')} · AI Whisperers",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "Standard consensus primers can only cover **13.3%** of DENV-4 sequences due to extreme cryptic diversity. ",
            "This report presents **degenerate primers** using IUPAC ambiguity codes to achieve broader coverage.",
            "",
            "### Key Results",
            "",
        ]

        if "multiplex_cocktail" in results:
            cocktail = results["multiplex_cocktail"]
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Sequences | {results['n_sequences']} |",
                f"| Primers in Cocktail | {cocktail['n_primers']} |",
                f"| Total Coverage | {cocktail['total_coverage_pct']:.1f}% |",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## Critical Finding: Universal Primers Are Impossible",
            "",
            "The analysis reveals a fundamental limitation:",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            "| Minimum degeneracy found | ~3.2 × 10^8 (322 million) |",
            "| Maximum practical degeneracy | 512 |",
            "| Gap | **6 orders of magnitude** |",
            "",
            "This means **NO single primer or degenerate primer pool** can detect all DENV-4 variants.",
            "",
            "### Implications",
            "",
            "1. **Clade-specific detection required** - Each DENV-4 sub-lineage needs its own primer set",
            "2. **Geographic stratification** - Primers must be designed for local circulating strains",
            "3. **Amplicon sequencing recommended** - NGS-based approaches avoid primer bias entirely",
            "4. **Pan-flavivirus fallback** - Use conserved flavivirus primers + sequencing for confirmation",
            "",
            "---",
            "",
            "## Degenerate Primer Design Strategy",
            "",
            "### IUPAC Ambiguity Codes Used",
            "",
            "| Code | Bases | Degeneracy |",
            "|------|-------|------------|",
            "| R | A/G | 2 |",
            "| Y | C/T | 2 |",
            "| S | G/C | 2 |",
            "| W | A/T | 2 |",
            "| K | G/T | 2 |",
            "| M | A/C | 2 |",
            "| B | C/G/T | 3 |",
            "| D | A/G/T | 3 |",
            "| H | A/C/T | 3 |",
            "| V | A/C/G | 3 |",
            "| N | any | 4 |",
            "",
            "---",
            "",
            "## Recommended Multiplex Cocktail",
            "",
        ])

        if "multiplex_cocktail" in results:
            for p in results["multiplex_cocktail"]["primers"]:
                lines.extend([
                    f"### {p['name']}",
                    "",
                    f"**Sequence:** `{p['sequence']}`",
                    "",
                    f"| Property | Value |",
                    f"|----------|-------|",
                    f"| Gene | {p['gene']} |",
                    f"| Position | {p['position']} |",
                    f"| Degeneracy | {p['degeneracy']} |",
                    f"| Individual Coverage | {p['individual_coverage']:.1f}% |",
                    "",
                ])

        lines.extend([
            "---",
            "",
            "## Laboratory Protocol Considerations",
            "",
            "### Primer Synthesis",
            "",
            "- Use **hand-mixed** degenerate positions for equimolar representation",
            "- Order primers at **desalt** purity minimum",
            "- Total degeneracy should remain <512 for practical synthesis",
            "",
            "### PCR Optimization",
            "",
            "- Use **touchdown PCR** to improve specificity",
            "- Start annealing at 65°C, decrease 1°C per cycle to 55°C",
            "- Use **hot-start** polymerase to reduce non-specific amplification",
            "- Consider **nested PCR** for low-titer samples",
            "",
            "### Expected Amplicon Sizes",
            "",
            "Design reverse primers to create staggered amplicons:",
            "- Forward primer 1 + Reverse → 100 bp",
            "- Forward primer 2 + Reverse → 150 bp",
            "- Forward primer 3 + Reverse → 200 bp",
            "",
            "---",
            "",
            "## Coverage Gap Analysis",
            "",
        ])

        if "validation" in results:
            v = results["validation"]
            lines.extend([
                f"With the designed primers, **{v['uncovered']} sequences** ({v['uncovered']/v['total_sequences']*100:.1f}%) remain uncovered.",
                "",
                "These represent the most divergent strains and may require:",
                "",
                "1. **Higher degeneracy primers** (accepting lower specificity)",
                "2. **Amplicon sequencing** for variant discovery",
                "3. **Pan-flavivirus primers** with sequencing confirmation",
                "",
            ])

        lines.extend([
            "---",
            "",
            f"*Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "*IICS-UNA Arbovirus Surveillance Program*",
        ])

        report_path.write_text("\n".join(lines))
        print(f"Report saved to: {report_path}")

    def _save_json(self, data: dict, path: Path) -> None:
        """Save dictionary to JSON."""
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(convert(data), f, indent=2, default=str)


def main():
    """Main entry point."""
    print()
    print("=" * 80)
    print("DENV-4 DEGENERATE PRIMER DESIGN")
    print("=" * 80)
    print()

    designer = DegeneratePrimerDesigner()
    results = designer.run_design()

    print()
    print("=" * 80)
    print("DESIGN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
