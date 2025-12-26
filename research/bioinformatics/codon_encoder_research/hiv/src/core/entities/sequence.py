"""
Sequence entity - represents a biological sequence.

Supports DNA, RNA, and protein sequences with codon iteration
and translation capabilities.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

from .codon import Codon


class SequenceType(Enum):
    """Type of biological sequence."""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


@dataclass(slots=True)
class Sequence:
    """
    Represents a biological sequence (DNA, RNA, or protein).

    Attributes:
        raw: The raw sequence string
        sequence_type: Type of sequence (DNA, RNA, protein)
        id: Optional identifier
        metadata: Additional metadata dictionary
    """

    raw: str
    sequence_type: SequenceType
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize sequence."""
        self.raw = self.raw.upper().replace(" ", "").replace("\n", "")
        if self.sequence_type == SequenceType.RNA:
            self.raw = self.raw.replace("U", "T")

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int | slice) -> str:
        return self.raw[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def codons(self) -> Iterator[Codon]:
        """
        Iterate over codons (for DNA/RNA sequences only).

        Yields:
            Codon objects for each triplet

        Raises:
            ValueError: If called on protein sequence
        """
        if self.sequence_type == SequenceType.PROTEIN:
            raise ValueError("Cannot get codons from protein sequence")

        for i in range(0, len(self.raw) - 2, 3):
            yield Codon(self.raw[i : i + 3])

    def codon_list(self) -> list[Codon]:
        """Get list of all codons."""
        return list(self.codons())

    def translate(self) -> "Sequence":
        """
        Translate nucleotide sequence to protein.

        Returns:
            New Sequence with protein sequence

        Raises:
            ValueError: If already a protein sequence
        """
        if self.sequence_type == SequenceType.PROTEIN:
            return self

        aa_seq = "".join(codon.amino_acid for codon in self.codons())
        return Sequence(
            raw=aa_seq,
            sequence_type=SequenceType.PROTEIN,
            id=f"{self.id}_translated" if self.id else None,
            metadata={**self.metadata, "source_type": self.sequence_type.value},
        )

    def subsequence(self, start: int, end: int) -> "Sequence":
        """Extract subsequence."""
        return Sequence(
            raw=self.raw[start:end],
            sequence_type=self.sequence_type,
            id=f"{self.id}_{start}_{end}" if self.id else None,
            metadata={**self.metadata, "parent_start": start, "parent_end": end},
        )

    @property
    def gc_content(self) -> float:
        """Calculate GC content (for DNA/RNA)."""
        if self.sequence_type == SequenceType.PROTEIN:
            raise ValueError("GC content not applicable to protein")
        gc = sum(1 for n in self.raw if n in "GC")
        return gc / len(self.raw) if self.raw else 0.0

    def __str__(self) -> str:
        if len(self.raw) > 50:
            return f"{self.raw[:50]}... ({len(self.raw)} bases)"
        return self.raw

    def __repr__(self) -> str:
        return f"Sequence({self.raw[:20]}..., type={self.sequence_type.value})"
