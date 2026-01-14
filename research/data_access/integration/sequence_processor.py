"""
Sequence processing utilities for codon extraction and analysis.

Bridges raw sequence data to the p-adic hyperbolic encoding framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import re

import pandas as pd


# Standard genetic code (imported from centralized biology module)
from src.biology.codons import GENETIC_CODE, AMINO_ACID_TO_CODONS as AA_TO_CODONS


@dataclass
class CodonInfo:
    """Information about a single codon."""
    codon: str
    position: int  # 1-based position in sequence
    amino_acid: str
    is_valid: bool = True

    @property
    def nucleotides(self) -> tuple[str, str, str]:
        """Get individual nucleotides."""
        return (self.codon[0], self.codon[1], self.codon[2])


@dataclass
class SequenceInfo:
    """Processed sequence information."""
    sequence_id: str
    raw_sequence: str
    codons: list[CodonInfo]
    gene: Optional[str] = None
    organism: Optional[str] = None

    @property
    def length(self) -> int:
        """Sequence length in nucleotides."""
        return len(self.raw_sequence)

    @property
    def codon_count(self) -> int:
        """Number of complete codons."""
        return len(self.codons)

    @property
    def translated_sequence(self) -> str:
        """Get amino acid sequence."""
        return "".join(c.amino_acid for c in self.codons)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with codon details."""
        records = []
        for codon in self.codons:
            records.append({
                "sequence_id": self.sequence_id,
                "position": codon.position,
                "codon": codon.codon,
                "amino_acid": codon.amino_acid,
                "n1": codon.nucleotides[0],
                "n2": codon.nucleotides[1],
                "n3": codon.nucleotides[2],
            })
        return pd.DataFrame(records)


class SequenceProcessor:
    """
    Process DNA/RNA sequences for codon analysis.

    Provides utilities for:
    - Sequence validation and cleaning
    - Codon extraction
    - Frame detection
    - Mutation position mapping
    """

    VALID_DNA = set("ATGC")
    VALID_RNA = set("AUGC")
    AMBIGUOUS = set("NRYWSMKHBVD")

    def __init__(self, allow_ambiguous: bool = False):
        """
        Initialize processor.

        Args:
            allow_ambiguous: Whether to allow ambiguous nucleotides
        """
        self.allow_ambiguous = allow_ambiguous

    def clean_sequence(self, sequence: str) -> str:
        """
        Clean and normalize a sequence.

        Args:
            sequence: Raw sequence string

        Returns:
            Cleaned sequence (uppercase, no whitespace)
        """
        # Remove whitespace and newlines
        cleaned = re.sub(r"\s+", "", sequence)
        # Convert to uppercase
        cleaned = cleaned.upper()
        # Convert RNA to DNA
        cleaned = cleaned.replace("U", "T")
        return cleaned

    def validate_sequence(self, sequence: str) -> tuple[bool, list[str]]:
        """
        Validate a sequence.

        Args:
            sequence: Sequence to validate

        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        sequence = self.clean_sequence(sequence)

        if not sequence:
            return False, ["Empty sequence"]

        # Check for valid characters
        valid_chars = self.VALID_DNA
        if self.allow_ambiguous:
            valid_chars = valid_chars | self.AMBIGUOUS

        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            messages.append(f"Invalid characters: {invalid_chars}")

        # Check length is divisible by 3 for complete codons
        if len(sequence) % 3 != 0:
            messages.append(f"Sequence length ({len(sequence)}) not divisible by 3")

        # Check for start codon
        if len(sequence) >= 3 and sequence[:3] != "ATG":
            messages.append("Sequence does not start with ATG start codon")

        is_valid = len(messages) == 0 or (
            len(messages) == 1 and "start codon" in messages[0]
        )

        return is_valid, messages

    def extract_codons(
        self,
        sequence: str,
        frame: int = 0,
    ) -> Iterator[CodonInfo]:
        """
        Extract codons from a sequence.

        Args:
            sequence: DNA sequence
            frame: Reading frame (0, 1, or 2)

        Yields:
            CodonInfo for each codon
        """
        sequence = self.clean_sequence(sequence)

        position = 1
        for i in range(frame, len(sequence) - 2, 3):
            codon = sequence[i:i + 3]
            amino_acid = GENETIC_CODE.get(codon, "X")
            is_valid = all(n in self.VALID_DNA for n in codon)

            yield CodonInfo(
                codon=codon,
                position=position,
                amino_acid=amino_acid,
                is_valid=is_valid,
            )
            position += 1

    def process_sequence(
        self,
        sequence: str,
        sequence_id: str = "query",
        gene: Optional[str] = None,
    ) -> SequenceInfo:
        """
        Process a sequence into structured format.

        Args:
            sequence: DNA sequence
            sequence_id: Identifier
            gene: Gene name

        Returns:
            Processed sequence information
        """
        cleaned = self.clean_sequence(sequence)
        codons = list(self.extract_codons(cleaned))

        return SequenceInfo(
            sequence_id=sequence_id,
            raw_sequence=cleaned,
            codons=codons,
            gene=gene,
        )

    def find_best_frame(self, sequence: str) -> int:
        """
        Find the best reading frame based on start codon and stop codon positions.

        Args:
            sequence: DNA sequence

        Returns:
            Best frame (0, 1, or 2)
        """
        sequence = self.clean_sequence(sequence)
        best_frame = 0
        best_score = -1

        for frame in range(3):
            score = 0
            stop_found = False

            for i in range(frame, len(sequence) - 2, 3):
                codon = sequence[i:i + 3]

                # Check for start codon
                if codon == "ATG" and i == frame:
                    score += 10

                # Check for stop codon
                if codon in ("TAA", "TAG", "TGA"):
                    if i + 3 >= len(sequence) - 2:
                        score += 5  # Stop at end is good
                    else:
                        score -= 5  # Internal stop is bad
                    stop_found = True

                # Valid codon
                if codon in GENETIC_CODE:
                    score += 1

            if score > best_score:
                best_score = score
                best_frame = frame

        return best_frame

    def get_codon_at_position(
        self,
        sequence: str,
        aa_position: int,
        frame: int = 0,
    ) -> Optional[str]:
        """
        Get the codon at a specific amino acid position.

        Args:
            sequence: DNA sequence
            aa_position: 1-based amino acid position
            frame: Reading frame

        Returns:
            Codon at position or None if out of range
        """
        sequence = self.clean_sequence(sequence)
        nucleotide_position = frame + (aa_position - 1) * 3

        if nucleotide_position + 3 > len(sequence):
            return None

        return sequence[nucleotide_position:nucleotide_position + 3]

    def get_synonymous_codons(self, codon: str) -> list[str]:
        """
        Get all synonymous codons (encoding same amino acid).

        Args:
            codon: Reference codon

        Returns:
            List of synonymous codons including input
        """
        aa = GENETIC_CODE.get(codon, "X")
        return AA_TO_CODONS.get(aa, [codon])

    def calculate_codon_statistics(
        self,
        sequence: str,
    ) -> pd.DataFrame:
        """
        Calculate codon usage statistics for a sequence.

        Args:
            sequence: DNA sequence

        Returns:
            DataFrame with codon counts and frequencies
        """
        sequence = self.clean_sequence(sequence)
        codon_counts = {}

        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i + 3]
            if codon in GENETIC_CODE:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1

        total = sum(codon_counts.values())
        records = []

        for codon, count in sorted(codon_counts.items()):
            aa = GENETIC_CODE[codon]
            synonymous = AA_TO_CODONS[aa]
            synonymous_total = sum(codon_counts.get(c, 0) for c in synonymous)

            records.append({
                "codon": codon,
                "amino_acid": aa,
                "count": count,
                "frequency": count / total if total > 0 else 0,
                "rscu": (count * len(synonymous) / synonymous_total)
                if synonymous_total > 0 else 0,
            })

        return pd.DataFrame(records)
