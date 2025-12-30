# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Primer Designer for Arbovirus RT-PCR.

This module provides tools for designing RT-PCR primers targeting
arbovirus genomes with:
- GC content and Tm optimization
- Cross-reactivity checking
- Conservation scoring
- Multiplex compatibility

Example:
    >>> designer = PrimerDesigner(database)
    >>> primers = designer.design_primers("DENV-1", n_pairs=10)
    >>> for p in primers:
    ...     print(f"{p.forward_seq} / {p.reverse_seq}: {p.score:.2f}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from .constants import PRIMER_CONSTRAINTS, NUCLEOTIDE_COMPLEMENT

if TYPE_CHECKING:
    from .ncbi_client import ArbovirusDatabase


@dataclass
class PrimerCandidate:
    """A single primer candidate.

    Attributes:
        sequence: Primer sequence (5' to 3')
        position: Start position in genome
        length: Primer length
        gc_content: GC content (0-1)
        tm: Melting temperature
        direction: "forward" or "reverse"
        score: Overall quality score
    """

    sequence: str
    position: int
    length: int
    gc_content: float
    tm: float
    direction: str
    score: float = 0.0
    hairpin_tm: Optional[float] = None
    self_dimer_tm: Optional[float] = None

    def reverse_complement(self) -> str:
        """Get reverse complement of primer."""
        complement = {
            "A": "T", "T": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "g": "c", "c": "g",
        }
        return "".join(complement.get(b, b) for b in reversed(self.sequence))


@dataclass
class PrimerPair:
    """A pair of forward and reverse primers.

    Attributes:
        forward: Forward primer
        reverse: Reverse primer
        amplicon_size: Expected amplicon size
        amplicon_start: Start position of amplicon
        tm_diff: Difference in Tm between primers
        target_virus: Target virus
        score: Combined quality score
    """

    forward: PrimerCandidate
    reverse: PrimerCandidate
    amplicon_size: int
    amplicon_start: int
    tm_diff: float
    target_virus: str
    score: float = 0.0
    cross_reactive_with: list[str] = field(default_factory=list)
    conservation_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "forward_seq": self.forward.sequence,
            "reverse_seq": self.reverse.sequence,
            "forward_tm": self.forward.tm,
            "reverse_tm": self.reverse.tm,
            "forward_gc": self.forward.gc_content,
            "reverse_gc": self.reverse.gc_content,
            "amplicon_size": self.amplicon_size,
            "amplicon_start": self.amplicon_start,
            "tm_diff": self.tm_diff,
            "target_virus": self.target_virus,
            "score": self.score,
            "conservation_score": self.conservation_score,
            "cross_reactive_with": self.cross_reactive_with,
        }


@dataclass
class CrossReactivityResult:
    """Result of cross-reactivity analysis.

    Attributes:
        primer_seq: Primer sequence tested
        matches: Dict of virus -> similarity score
        is_specific: Whether primer is specific (no cross-reactivity)
    """

    primer_seq: str
    matches: dict[str, float]
    is_specific: bool
    threshold: float = 0.8

    def get_cross_reactive_viruses(self) -> list[str]:
        """Get list of viruses with potential cross-reactivity."""
        return [
            virus for virus, sim in self.matches.items()
            if sim >= self.threshold
        ]


class PrimerDesigner:
    """Designer for RT-PCR primers targeting arboviruses.

    Features:
    - Constraint-based design (GC, Tm, length)
    - Cross-reactivity checking
    - Conservation scoring across sequences
    - Multiplex compatibility analysis

    Example:
        >>> designer = PrimerDesigner(database)
        >>> primers = designer.design_primers("DENV-1")
        >>> cross_check = designer.check_cross_reactivity(primers[0])
    """

    def __init__(
        self,
        database: "ArbovirusDatabase" = None,
        constraints: dict = None,
    ):
        """Initialize primer designer.

        Args:
            database: ArbovirusDatabase with sequences
            constraints: Custom primer constraints
        """
        self.database = database
        self.constraints = constraints or PRIMER_CONSTRAINTS

    def compute_gc_content(self, sequence: str) -> float:
        """Compute GC content of sequence."""
        sequence = sequence.upper()
        gc = sum(1 for b in sequence if b in "GC")
        return gc / len(sequence) if sequence else 0.0

    def estimate_tm(self, sequence: str) -> float:
        """Estimate melting temperature using nearest-neighbor method.

        Args:
            sequence: Primer sequence

        Returns:
            Estimated Tm in Celsius
        """
        sequence = sequence.upper()

        # Simple estimation (Wallace rule for short, improved for longer)
        a = sequence.count("A")
        t = sequence.count("T")
        g = sequence.count("G")
        c = sequence.count("C")
        length = len(sequence)

        if length < 14:
            # Wallace rule
            return 2 * (a + t) + 4 * (g + c)
        else:
            # Improved formula
            return 64.9 + 41 * (g + c - 16.4) / length

    def check_gc_clamp(self, sequence: str) -> bool:
        """Check if primer has appropriate GC clamp at 3' end."""
        last_5 = sequence[-5:].upper()
        gc_count = sum(1 for b in last_5 if b in "GC")

        constraints = self.constraints.get("gc_clamp", {})
        min_gc = constraints.get("min_gc_3prime", 1)
        max_gc = constraints.get("max_gc_3prime", 3)

        return min_gc <= gc_count <= max_gc

    def score_primer(self, candidate: PrimerCandidate) -> float:
        """Score a primer candidate based on constraints.

        Higher score = better primer.
        """
        score = 100.0

        # Length penalty
        length_c = self.constraints["length"]
        if candidate.length < length_c["min"]:
            score -= 20 * (length_c["min"] - candidate.length)
        elif candidate.length > length_c["max"]:
            score -= 20 * (candidate.length - length_c["max"])

        # GC content penalty
        gc_c = self.constraints["gc_content"]
        if candidate.gc_content < gc_c["min"]:
            score -= 50 * (gc_c["min"] - candidate.gc_content)
        elif candidate.gc_content > gc_c["max"]:
            score -= 50 * (candidate.gc_content - gc_c["max"])

        # Tm penalty
        tm_c = self.constraints["tm"]
        if candidate.tm < tm_c["min"]:
            score -= 5 * (tm_c["min"] - candidate.tm)
        elif candidate.tm > tm_c["max"]:
            score -= 5 * (candidate.tm - tm_c["max"])

        # GC clamp bonus
        if self.check_gc_clamp(candidate.sequence):
            score += 5

        return max(0, score)

    def find_primer_candidates(
        self,
        sequence: str,
        direction: str = "forward",
        window_start: int = 0,
        window_end: int = None,
        max_candidates: int = 100,
    ) -> list[PrimerCandidate]:
        """Find primer candidates in a sequence region.

        Args:
            sequence: Target sequence
            direction: "forward" or "reverse"
            window_start: Start position for search
            window_end: End position for search
            max_candidates: Maximum candidates to return

        Returns:
            List of PrimerCandidate objects
        """
        if window_end is None:
            window_end = len(sequence)

        length_c = self.constraints["length"]
        candidates = []

        for pos in range(window_start, window_end - length_c["max"]):
            for length in range(length_c["min"], length_c["max"] + 1):
                primer_seq = sequence[pos:pos + length].upper()

                # Skip if contains ambiguous bases
                if any(b not in "ACGT" for b in primer_seq):
                    continue

                gc = self.compute_gc_content(primer_seq)
                tm = self.estimate_tm(primer_seq)

                # Quick filter
                gc_c = self.constraints["gc_content"]
                tm_c = self.constraints["tm"]

                if not (gc_c["min"] - 0.1 <= gc <= gc_c["max"] + 0.1):
                    continue
                if not (tm_c["min"] - 5 <= tm <= tm_c["max"] + 5):
                    continue

                candidate = PrimerCandidate(
                    sequence=primer_seq,
                    position=pos,
                    length=length,
                    gc_content=gc,
                    tm=tm,
                    direction=direction,
                )
                candidate.score = self.score_primer(candidate)
                candidates.append(candidate)

        # Sort by score and return top candidates
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_candidates]

    def design_primer_pairs(
        self,
        sequence: str,
        target_virus: str,
        n_pairs: int = 10,
        conserved_regions: list[tuple[int, int]] = None,
    ) -> list[PrimerPair]:
        """Design primer pairs for a target sequence.

        Args:
            sequence: Target genome sequence
            target_virus: Virus identifier
            n_pairs: Number of primer pairs to design
            conserved_regions: List of (start, end) conserved regions

        Returns:
            List of PrimerPair objects
        """
        amplicon_c = self.constraints["amplicon"]
        tm_c = self.constraints["tm"]

        pairs = []

        # If conserved regions provided, focus on those
        if conserved_regions:
            regions = conserved_regions
        else:
            # Use whole genome
            regions = [(0, len(sequence))]

        for region_start, region_end in regions:
            # Find forward primers in first part of region
            forward_end = region_start + (region_end - region_start) // 2
            forwards = self.find_primer_candidates(
                sequence, "forward",
                window_start=region_start,
                window_end=forward_end,
            )

            # Find reverse primers in second part
            reverse_start = forward_end
            reverses = self.find_primer_candidates(
                sequence, "reverse",
                window_start=reverse_start,
                window_end=region_end,
            )

            # Match pairs
            for fwd in forwards[:20]:
                for rev in reverses[:20]:
                    amplicon_size = rev.position + rev.length - fwd.position

                    if not (amplicon_c["min"] <= amplicon_size <= amplicon_c["max"]):
                        continue

                    tm_diff = abs(fwd.tm - rev.tm)
                    if tm_diff > tm_c["max_diff"]:
                        continue

                    pair_score = (fwd.score + rev.score) / 2 - 5 * tm_diff

                    pair = PrimerPair(
                        forward=fwd,
                        reverse=rev,
                        amplicon_size=amplicon_size,
                        amplicon_start=fwd.position,
                        tm_diff=tm_diff,
                        target_virus=target_virus,
                        score=pair_score,
                    )
                    pairs.append(pair)

        # Sort by score and return top pairs
        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs[:n_pairs]

    def design_primers(
        self,
        virus: str,
        n_pairs: int = 10,
    ) -> list[PrimerPair]:
        """Design primers for a virus using database sequences.

        Args:
            virus: Virus identifier
            n_pairs: Number of pairs to design

        Returns:
            List of PrimerPair objects
        """
        if self.database is None:
            raise ValueError("No database provided")

        sequences = self.database.get_sequences(virus)
        if not sequences:
            raise ValueError(f"No sequences for {virus}")

        # Use consensus or first sequence
        consensus = self.database.get_consensus(virus)
        target_seq = consensus or sequences[0].sequence

        # Get conserved regions from constants
        from .constants import ARBOVIRUS_TARGETS
        target_info = ARBOVIRUS_TARGETS.get(virus, {})
        conserved = target_info.get("conserved_regions", None)

        return self.design_primer_pairs(
            target_seq, virus, n_pairs, conserved
        )

    def check_cross_reactivity(
        self,
        primer: PrimerCandidate,
        threshold: float = 0.8,
    ) -> CrossReactivityResult:
        """Check primer for cross-reactivity with other viruses.

        Args:
            primer: Primer to check
            threshold: Similarity threshold for cross-reactivity

        Returns:
            CrossReactivityResult
        """
        if self.database is None:
            return CrossReactivityResult(
                primer_seq=primer.sequence,
                matches={},
                is_specific=True,
                threshold=threshold,
            )

        matches = {}

        for virus in self.database.get_viruses():
            sequences = self.database.get_sequences(virus)

            max_similarity = 0.0
            for seq in sequences[:10]:  # Check first 10 sequences
                similarity = self._compute_similarity(
                    primer.sequence, seq.sequence
                )
                max_similarity = max(max_similarity, similarity)

            matches[virus] = max_similarity

        is_specific = all(sim < threshold for sim in matches.values())

        return CrossReactivityResult(
            primer_seq=primer.sequence,
            matches=matches,
            is_specific=is_specific,
            threshold=threshold,
        )

    def _compute_similarity(
        self,
        primer: str,
        genome: str,
    ) -> float:
        """Compute maximum similarity of primer to genome.

        Uses simple substring matching.
        """
        primer = primer.upper()
        genome = genome.upper()
        primer_len = len(primer)

        max_similarity = 0.0

        for i in range(len(genome) - primer_len + 1):
            window = genome[i:i + primer_len]
            matches = sum(1 for a, b in zip(primer, window) if a == b)
            similarity = matches / primer_len
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def export_primers(
        self,
        pairs: list[PrimerPair],
        output_path: Path,
        format: str = "json",
    ) -> Path:
        """Export designed primers.

        Args:
            pairs: List of primer pairs
            output_path: Output file path
            format: "json" or "csv"

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "primers": [p.to_dict() for p in pairs],
                "total_pairs": len(pairs),
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "target_virus", "forward_seq", "reverse_seq",
                        "forward_tm", "reverse_tm", "amplicon_size", "score",
                    ],
                )
                writer.writeheader()
                for p in pairs:
                    writer.writerow({
                        "target_virus": p.target_virus,
                        "forward_seq": p.forward.sequence,
                        "reverse_seq": p.reverse.sequence,
                        "forward_tm": f"{p.forward.tm:.1f}",
                        "reverse_tm": f"{p.reverse.tm:.1f}",
                        "amplicon_size": p.amplicon_size,
                        "score": f"{p.score:.2f}",
                    })

        return output_path
