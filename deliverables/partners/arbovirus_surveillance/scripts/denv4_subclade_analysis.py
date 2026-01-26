# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Sub-clade Analysis for Primer Design.

The initial phylogenetic analysis revealed that the two largest clades (D and E)
containing 263/270 sequences (97%) have NO conserved windows suitable for primer
design (mean entropy >1.4).

This script performs adaptive sub-clustering to identify smaller, more homogeneous
groups where conserved regions exist.

Strategy:
1. Focus on the large clades (D, E)
2. Sub-cluster until within-clade entropy drops below threshold
3. Identify conserved windows in each sub-clade
4. Map sub-clades to known genotypes where possible

Author: AI Whisperers
Date: 2026-01-04
"""

from __future__ import annotations

import datetime
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import cluster
from scipy.spatial.distance import squareform

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "denv4"
RESULTS_DIR = PROJECT_ROOT / "results" / "phylogenetic"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Thresholds
MAX_ENTROPY_FOR_PRIMERS = 0.5  # Maximum acceptable entropy for primer binding
MIN_CLADE_SIZE = 5  # Minimum sequences per sub-clade
MAX_SUBCLADE_DEPTH = 4  # Maximum recursion depth


@dataclass
class SubCladeInfo:
    """Information about a sub-clade."""
    name: str
    parent: str
    depth: int
    size: int
    members: list = field(default_factory=list)
    mean_entropy: float = 0.0
    conserved_windows: list = field(default_factory=list)
    has_primers: bool = False
    representative: Optional[str] = None


class DENV4SubcladeAnalyzer:
    """Adaptive sub-clustering for primer-suitable clades."""

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

    def __init__(self):
        """Initialize the analyzer."""
        self.sequences = {}
        self.distance_matrix = None
        self.accessions = []
        self.subclades = {}
        self.primer_candidates = []

    def load_data(self) -> bool:
        """Load sequences and distance matrix from previous analysis."""
        print("Loading data from previous analysis...")

        # Load sequences
        seq_cache = CACHE_DIR / "denv4_sequences.json"
        if not seq_cache.exists():
            print(f"ERROR: Sequence cache not found at {seq_cache}")
            return False

        with open(seq_cache) as f:
            self.sequences = json.load(f)

        # Load distance matrix
        dist_path = DATA_DIR / "distance_matrix.npy"
        if not dist_path.exists():
            print(f"ERROR: Distance matrix not found at {dist_path}")
            return False

        self.distance_matrix = np.load(dist_path)

        # Load accession order
        acc_path = DATA_DIR / "accession_order.json"
        if acc_path.exists():
            with open(acc_path) as f:
                self.accessions = json.load(f)
        else:
            self.accessions = list(self.sequences.keys())

        print(f"Loaded {len(self.sequences)} sequences")
        return True

    def run_subclade_analysis(self) -> dict:
        """Run the complete sub-clade analysis."""
        print()
        print("=" * 80)
        print("DENV-4 SUB-CLADE ANALYSIS FOR PRIMER DESIGN")
        print("=" * 80)
        print()

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "max_entropy": MAX_ENTROPY_FOR_PRIMERS,
                "min_clade_size": MIN_CLADE_SIZE,
                "max_depth": MAX_SUBCLADE_DEPTH,
            },
        }

        if not self.load_data():
            return {"error": "Failed to load data"}

        # Step 1: Identify optimal number of clades
        print("-" * 80)
        print("STEP 1: Finding optimal clade count for primer design")
        print("-" * 80)
        optimal_k, k_analysis = self.find_optimal_clade_count()
        results["optimal_k"] = optimal_k
        results["k_analysis"] = k_analysis
        print()

        # Step 2: Perform adaptive sub-clustering
        print("-" * 80)
        print("STEP 2: Adaptive sub-clustering")
        print("-" * 80)
        subclades = self.adaptive_subclustering()
        results["subclades"] = {
            name: {
                "size": info.size,
                "mean_entropy": info.mean_entropy,
                "has_primers": info.has_primers,
                "n_conserved_windows": len(info.conserved_windows),
            }
            for name, info in subclades.items()
        }
        print()

        # Step 3: Identify primer-suitable clades
        print("-" * 80)
        print("STEP 3: Identifying primer-suitable sub-clades")
        print("-" * 80)
        suitable = self.identify_primer_suitable_clades()
        results["primer_suitable_clades"] = suitable
        print()

        # Step 4: Design primer candidates for each suitable clade
        print("-" * 80)
        print("STEP 4: Designing primer candidates")
        print("-" * 80)
        primers = self.design_primer_candidates()
        results["primer_candidates"] = primers
        print()

        # Step 5: Generate coverage analysis
        print("-" * 80)
        print("STEP 5: Coverage analysis")
        print("-" * 80)
        coverage = self.analyze_primer_coverage()
        results["coverage"] = coverage
        print()

        # Step 6: Generate report
        print("-" * 80)
        print("STEP 6: Generating report")
        print("-" * 80)
        self.generate_subclade_report(results)
        print()

        # Save results
        results_path = RESULTS_DIR / "subclade_analysis_results.json"
        self._save_json(results, results_path)
        print(f"Results saved to: {results_path}")

        return results

    def find_optimal_clade_count(self) -> tuple[int, dict]:
        """Find the optimal number of clades for primer design.

        Tests different k values and checks what percentage of sequences
        end up in clades with conserved regions.

        Returns:
            Tuple of (optimal_k, analysis_dict)
        """
        print("Testing clade counts from 5 to 30...")

        condensed = squareform(self.distance_matrix)
        linkage = cluster.hierarchy.linkage(condensed, method='average')

        k_results = {}

        for k in range(5, 31, 5):
            labels = cluster.hierarchy.fcluster(linkage, t=k, criterion='maxclust')

            # Analyze each clade
            clades_with_primers = 0
            sequences_covered = 0

            for clade_id in range(1, k + 1):
                member_indices = np.where(labels == clade_id)[0]
                members = [self.accessions[i] for i in member_indices]

                if len(members) < MIN_CLADE_SIZE:
                    continue

                # Compute entropy for this clade
                seqs = [self.sequences[acc] for acc in members]
                entropy = self._compute_mean_entropy(seqs)

                if entropy < MAX_ENTROPY_FOR_PRIMERS:
                    clades_with_primers += 1
                    sequences_covered += len(members)

            coverage_pct = sequences_covered / len(self.accessions) * 100

            k_results[k] = {
                "n_clades": k,
                "clades_with_primers": clades_with_primers,
                "sequences_covered": sequences_covered,
                "coverage_pct": coverage_pct,
            }

            print(f"  k={k}: {clades_with_primers} clades with primers, {coverage_pct:.1f}% coverage")

        # Find optimal k (maximize coverage)
        optimal_k = max(k_results.keys(), key=lambda k: k_results[k]["coverage_pct"])

        print(f"\nOptimal k={optimal_k} with {k_results[optimal_k]['coverage_pct']:.1f}% coverage")

        return optimal_k, k_results

    def adaptive_subclustering(self) -> dict[str, SubCladeInfo]:
        """Perform adaptive sub-clustering.

        Recursively splits clades until entropy is below threshold.

        Returns:
            Dictionary of sub-clade info
        """
        print("Performing adaptive sub-clustering...")

        condensed = squareform(self.distance_matrix)
        linkage = cluster.hierarchy.linkage(condensed, method='average')

        # Start with initial clustering
        initial_k = 5
        labels = cluster.hierarchy.fcluster(linkage, t=initial_k, criterion='maxclust')

        # Create initial clades
        initial_clades = {}
        for clade_id in range(1, initial_k + 1):
            member_indices = np.where(labels == clade_id)[0].tolist()
            members = [self.accessions[i] for i in member_indices]

            if len(members) >= MIN_CLADE_SIZE:
                clade_name = f"Clade_{chr(ord('A') + clade_id - 1)}"
                initial_clades[clade_name] = SubCladeInfo(
                    name=clade_name,
                    parent="root",
                    depth=0,
                    size=len(members),
                    members=members,
                )

        # Recursively sub-cluster clades with high entropy
        self.subclades = {}
        for clade_name, clade_info in initial_clades.items():
            self._recursive_subcluster(clade_info, 0)

        # Summary
        print(f"\nIdentified {len(self.subclades)} sub-clades:")
        for name, info in sorted(self.subclades.items(), key=lambda x: -x[1].size):
            status = "HAS PRIMERS" if info.has_primers else "no primers"
            print(f"  {name}: {info.size} seqs, entropy={info.mean_entropy:.3f} [{status}]")

        return self.subclades

    def _recursive_subcluster(
        self,
        clade: SubCladeInfo,
        depth: int,
    ) -> None:
        """Recursively sub-cluster a clade until entropy is acceptable."""
        # Compute entropy for this clade
        seqs = [self.sequences[acc] for acc in clade.members]
        entropy = self._compute_mean_entropy(seqs)
        clade.mean_entropy = entropy

        # Find conserved windows
        conserved = self._find_conserved_windows(seqs)
        clade.conserved_windows = conserved
        clade.has_primers = len(conserved) > 0

        # Check if we should stop
        if entropy < MAX_ENTROPY_FOR_PRIMERS or depth >= MAX_SUBCLADE_DEPTH or len(clade.members) < MIN_CLADE_SIZE * 2:
            # Terminal clade - add to results
            self.subclades[clade.name] = clade
            return

        # Need to sub-cluster
        print(f"  Sub-clustering {clade.name} (n={len(clade.members)}, entropy={entropy:.3f})...")

        # Get sub-distance matrix for this clade
        member_indices = [self.accessions.index(acc) for acc in clade.members]
        sub_dist = self.distance_matrix[np.ix_(member_indices, member_indices)]

        # Cluster into 2-3 sub-clades
        n_subclusters = min(3, len(clade.members) // MIN_CLADE_SIZE)
        if n_subclusters < 2:
            self.subclades[clade.name] = clade
            return

        condensed = squareform(sub_dist)
        linkage = cluster.hierarchy.linkage(condensed, method='average')
        labels = cluster.hierarchy.fcluster(linkage, t=n_subclusters, criterion='maxclust')

        # Create sub-clades
        for sub_id in range(1, n_subclusters + 1):
            sub_indices = np.where(labels == sub_id)[0]
            sub_members = [clade.members[i] for i in sub_indices]

            if len(sub_members) < MIN_CLADE_SIZE:
                continue

            sub_name = f"{clade.name}.{sub_id}"
            sub_clade = SubCladeInfo(
                name=sub_name,
                parent=clade.name,
                depth=depth + 1,
                size=len(sub_members),
                members=sub_members,
            )

            self._recursive_subcluster(sub_clade, depth + 1)

    def _compute_mean_entropy(self, sequences: list[str]) -> float:
        """Compute mean per-position entropy for a set of sequences."""
        if len(sequences) < 2:
            return 0.0

        min_len = min(len(s) for s in sequences)
        sample_positions = list(range(0, min_len, 100))  # Sample every 100bp for speed

        entropies = []
        for pos in sample_positions:
            counts = Counter()
            for seq in sequences:
                if pos < len(seq):
                    nt = seq[pos].upper()
                    if nt in "ACGT":
                        counts[nt] += 1

            total = sum(counts.values())
            if total > 0:
                entropy = 0
                for count in counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                entropies.append(entropy)

        return np.mean(entropies) if entropies else 0.0

    def _find_conserved_windows(
        self,
        sequences: list[str],
        window_size: int = 25,
        max_entropy: float = 0.3,
    ) -> list[dict]:
        """Find conserved windows suitable for primers."""
        if len(sequences) < 2:
            return []

        min_len = min(len(s) for s in sequences)

        # Compute full entropy profile
        entropy_profile = np.zeros(min_len)
        for pos in range(min_len):
            counts = Counter()
            for seq in sequences:
                nt = seq[pos].upper()
                if nt in "ACGT":
                    counts[nt] += 1

            total = sum(counts.values())
            if total > 0:
                for count in counts.values():
                    p = count / total
                    if p > 0:
                        entropy_profile[pos] -= p * math.log2(p)

        # Find windows with low average entropy
        conserved = []
        for start in range(min_len - window_size + 1):
            end = start + window_size
            window_entropy = np.mean(entropy_profile[start:end])

            if window_entropy < max_entropy:
                gene = self._get_gene_at_position(start)
                conserved.append({
                    "start": start,
                    "end": end,
                    "mean_entropy": float(window_entropy),
                    "gene": gene,
                })

        # Sort by entropy
        conserved.sort(key=lambda x: x["mean_entropy"])

        # Return top 50, avoiding overlaps
        filtered = []
        used_positions = set()
        for window in conserved:
            positions = set(range(window["start"], window["end"]))
            if not positions & used_positions:
                filtered.append(window)
                used_positions |= positions
                if len(filtered) >= 50:
                    break

        return filtered

    def _get_gene_at_position(self, position: int) -> str:
        """Get gene name at position."""
        for gene, (start, end) in self.GENE_REGIONS.items():
            if start <= position <= end:
                return gene
        return "intergenic"

    def identify_primer_suitable_clades(self) -> list[dict]:
        """Identify clades suitable for primer design.

        Returns:
            List of suitable clade information
        """
        suitable = []

        for name, info in self.subclades.items():
            if info.has_primers:
                # Select representative
                rep = self._select_representative(info.members)
                info.representative = rep

                suitable.append({
                    "clade": name,
                    "size": info.size,
                    "mean_entropy": info.mean_entropy,
                    "n_conserved_windows": len(info.conserved_windows),
                    "top_windows": info.conserved_windows[:5],
                    "representative": rep,
                })

        suitable.sort(key=lambda x: -x["size"])

        print(f"\nFound {len(suitable)} primer-suitable sub-clades:")
        total_covered = sum(x["size"] for x in suitable)
        coverage = total_covered / len(self.accessions) * 100

        for s in suitable[:10]:
            print(f"  {s['clade']}: {s['size']} seqs, {s['n_conserved_windows']} windows")

        print(f"\nTotal coverage: {total_covered}/{len(self.accessions)} ({coverage:.1f}%)")

        return suitable

    def _select_representative(self, members: list[str]) -> str:
        """Select the most central sequence as representative."""
        if len(members) == 1:
            return members[0]

        member_indices = [self.accessions.index(acc) for acc in members]

        min_avg_dist = float('inf')
        rep = members[0]

        for acc in members:
            idx = self.accessions.index(acc)
            avg_dist = np.mean([
                self.distance_matrix[idx, other_idx]
                for other_idx in member_indices
                if other_idx != idx
            ])
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                rep = acc

        return rep

    def design_primer_candidates(self) -> list[dict]:
        """Design primer candidates for each suitable clade.

        Returns:
            List of primer candidate information
        """
        print("Designing primer candidates for suitable clades...")

        primers = []

        for name, info in self.subclades.items():
            if not info.has_primers:
                continue

            # Get top conserved windows
            windows = info.conserved_windows[:10]

            # Get representative sequence for consensus
            rep_seq = self.sequences[info.representative]

            for i, window in enumerate(windows[:3]):  # Top 3 per clade
                start, end = window["start"], window["end"]

                # Extract consensus sequence
                consensus = rep_seq[start:end]

                # Check GC content
                gc_count = consensus.upper().count("G") + consensus.upper().count("C")
                gc_content = gc_count / len(consensus) * 100

                # Estimate Tm (simple approximation)
                at = len(consensus) - gc_count
                tm = 2 * at + 4 * gc_count

                primer = {
                    "clade": name,
                    "window_rank": i + 1,
                    "gene": window["gene"],
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "sequence": consensus,
                    "gc_content": gc_content,
                    "estimated_tm": tm,
                    "entropy": window["mean_entropy"],
                    "n_sequences_covered": info.size,
                }

                primers.append(primer)

        # Sort by coverage
        primers.sort(key=lambda x: -x["n_sequences_covered"])

        print(f"Designed {len(primers)} primer candidates")

        # Show top candidates
        for p in primers[:10]:
            print(f"  {p['clade']}: {p['gene']} {p['start']}-{p['end']} "
                  f"(GC={p['gc_content']:.0f}%, Tm~{p['estimated_tm']}°C)")

        self.primer_candidates = primers
        return primers

    def analyze_primer_coverage(self) -> dict:
        """Analyze how many sequences are covered by designed primers.

        Returns:
            Coverage analysis dictionary
        """
        print("Analyzing primer coverage...")

        # Get all covered sequences
        covered_seqs = set()
        for name, info in self.subclades.items():
            if info.has_primers:
                covered_seqs.update(info.members)

        total = len(self.accessions)
        covered = len(covered_seqs)
        uncovered = total - covered

        print(f"\nCoverage Summary:")
        print(f"  Total sequences: {total}")
        print(f"  Covered by primers: {covered} ({covered/total*100:.1f}%)")
        print(f"  Not covered: {uncovered} ({uncovered/total*100:.1f}%)")

        # Analyze uncovered sequences
        uncovered_seqs = [acc for acc in self.accessions if acc not in covered_seqs]

        coverage = {
            "total_sequences": total,
            "covered_sequences": covered,
            "coverage_pct": covered / total * 100,
            "uncovered_sequences": uncovered,
            "uncovered_pct": uncovered / total * 100,
            "uncovered_accessions": uncovered_seqs[:20],  # First 20
        }

        if uncovered > 0:
            print(f"\nUncovered sequences may require:")
            print("  1. More granular sub-clustering")
            print("  2. Degenerate primers with IUPAC codes")
            print("  3. Separate multiplex tier for rare variants")

        return coverage

    def generate_subclade_report(self, results: dict) -> None:
        """Generate markdown report for sub-clade analysis."""
        report_path = RESULTS_DIR / "DENV4_SUBCLADE_PRIMER_REPORT.md"

        lines = [
            "# DENV-4 Sub-clade Analysis for Primer Design",
            "",
            f"**Doc-Type:** Sub-clade Primer Report · Version 1.0 · {datetime.datetime.now().strftime('%Y-%m-%d')} · AI Whisperers",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"Initial phylogenetic analysis showed that {len(self.sequences)} DENV-4 sequences cluster into 5 major clades, "
            f"but the two largest (97% of sequences) have **NO conserved regions** suitable for primer design.",
            "",
            "This sub-clade analysis performs **adaptive sub-clustering** to identify smaller, more homogeneous "
            "groups where conserved primer binding sites exist.",
            "",
            "### Key Results",
            "",
        ]

        if "coverage" in results:
            cov = results["coverage"]
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Sequences | {cov['total_sequences']} |",
                f"| Sequences with Primers | {cov['covered_sequences']} ({cov['coverage_pct']:.1f}%) |",
                f"| Sequences without Primers | {cov['uncovered_sequences']} ({cov['uncovered_pct']:.1f}%) |",
                f"| Primer-Suitable Sub-clades | {len([s for s in self.subclades.values() if s.has_primers])} |",
                f"| Total Sub-clades | {len(self.subclades)} |",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## Sub-clade Summary",
            "",
            "| Sub-clade | Size | Entropy | Has Primers | Top Gene Region |",
            "|-----------|------|---------|-------------|-----------------|",
        ])

        for name, info in sorted(self.subclades.items(), key=lambda x: -x[1].size):
            has_primers = "Yes" if info.has_primers else "No"
            top_gene = info.conserved_windows[0]["gene"] if info.conserved_windows else "-"
            lines.append(
                f"| {name} | {info.size} | {info.mean_entropy:.3f} | {has_primers} | {top_gene} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Primer Candidates",
            "",
            "Top primer candidates by sequence coverage:",
            "",
            "| Clade | Gene | Position | Length | GC% | Tm | Entropy | Coverage |",
            "|-------|------|----------|--------|-----|-----|---------|----------|",
        ])

        for p in self.primer_candidates[:20]:
            lines.append(
                f"| {p['clade']} | {p['gene']} | {p['start']}-{p['end']} | "
                f"{p['length']}bp | {p['gc_content']:.0f}% | {p['estimated_tm']}°C | "
                f"{p['entropy']:.3f} | {p['n_sequences_covered']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Recommended Multiplex Strategy",
            "",
            "Based on this analysis, the recommended approach is:",
            "",
            "### Tier 1: High-Coverage Primers",
            "",
            "Use primers from the largest primer-suitable sub-clades to cover the majority of sequences.",
            "",
        ])

        # Top 5 clades
        suitable_clades = [
            (name, info) for name, info in self.subclades.items() if info.has_primers
        ]
        suitable_clades.sort(key=lambda x: -x[1].size)

        for name, info in suitable_clades[:5]:
            if info.conserved_windows:
                w = info.conserved_windows[0]
                lines.append(f"- **{name}** ({info.size} seqs): {w['gene']} {w['start']}-{w['end']}")

        lines.extend([
            "",
            "### Tier 2: Rare Variant Coverage",
            "",
            "For sequences not covered by Tier 1, consider:",
            "",
            "1. **Degenerate primers** with IUPAC codes for variable positions",
            "2. **Nested PCR** with outer pan-DENV-4 primers",
            "3. **Amplicon sequencing** for novel variants",
            "",
            "---",
            "",
            "## Gap Analysis: Uncovered Sequences",
            "",
        ])

        if "coverage" in results and results["coverage"]["uncovered_sequences"] > 0:
            lines.append(
                f"**{results['coverage']['uncovered_sequences']} sequences** remain uncovered by current primer candidates."
            )
            lines.append("")
            lines.append("These likely represent:")
            lines.append("- Highly divergent or recombinant strains")
            lines.append("- Novel genotypes not in clustering reference set")
            lines.append("- Sylvatic strains with different evolutionary history")
            lines.append("")
            lines.append("**Recommended action:** Analyze uncovered sequences separately for potential new primer targets.")
        else:
            lines.append("All sequences are covered by designed primer candidates.")

        lines.extend([
            "",
            "---",
            "",
            f"*Analysis completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "*IICS-UNA Arbovirus Surveillance Program*",
        ])

        report_path.write_text("\n".join(lines))
        print(f"Report saved to: {report_path}")

    def _save_json(self, data: dict, path: Path) -> None:
        """Save dictionary to JSON with type conversion."""
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        with open(path, "w") as f:
            json.dump(convert(data), f, indent=2, default=str)


def main():
    """Main entry point."""
    print()
    print("=" * 80)
    print("DENV-4 SUB-CLADE ANALYSIS")
    print("=" * 80)
    print()

    analyzer = DENV4SubcladeAnalyzer()
    results = analyzer.run_subclade_analysis()

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
