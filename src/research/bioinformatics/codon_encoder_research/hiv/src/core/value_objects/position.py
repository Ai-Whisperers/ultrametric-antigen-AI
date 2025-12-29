"""
Position value objects - represent genomic/protein positions.

Uses HXB2 reference coordinates as the standard.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Position:
    """
    Generic sequence position.

    Attributes:
        value: 1-indexed position number
        reference: Reference system name
    """

    value: int
    reference: str = "unknown"

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError(f"Position must be >= 1: {self.value}")

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class HXB2Position:
    """
    Position in HXB2 reference coordinates.

    HXB2 (K03455) is the standard HIV-1 reference sequence.

    Attributes:
        protein: Protein name (e.g., 'PR', 'RT', 'gp120')
        position: Position within protein (1-indexed)
        nucleotide_start: Optional nucleotide position in genome
    """

    protein: str
    position: int
    nucleotide_start: Optional[int] = None

    # HXB2 protein boundaries (nucleotide coordinates)
    PROTEIN_COORDS: dict = {
        "gag": (790, 2292),
        "pol": (2085, 5096),
        "pr": (2253, 2549),  # Protease
        "rt": (2550, 3869),  # Reverse transcriptase
        "in": (4230, 5096),  # Integrase
        "vif": (5041, 5619),
        "vpr": (5559, 5850),
        "tat": (5831, 8469),  # split
        "rev": (5970, 8653),  # split
        "vpu": (6062, 6310),
        "env": (6225, 8795),
        "gp120": (6225, 7758),
        "gp41": (7759, 8795),
        "nef": (8797, 9417),
    }

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError(f"Position must be >= 1: {self.position}")
        protein_lower = self.protein.lower()
        if protein_lower not in self.PROTEIN_COORDS:
            pass  # Allow unknown proteins

    @classmethod
    def from_nucleotide(cls, nucleotide_pos: int) -> Optional["HXB2Position"]:
        """
        Convert nucleotide position to protein position.

        Args:
            nucleotide_pos: Position in HXB2 nucleotide coordinates

        Returns:
            HXB2Position if within known protein, None otherwise
        """
        for protein, (start, end) in cls.PROTEIN_COORDS.items():
            if start <= nucleotide_pos <= end:
                aa_pos = (nucleotide_pos - start) // 3 + 1
                return cls(
                    protein=protein,
                    position=aa_pos,
                    nucleotide_start=nucleotide_pos,
                )
        return None

    def to_nucleotide_range(self) -> tuple[int, int]:
        """
        Get nucleotide range for this codon position.

        Returns:
            Tuple of (start, end) nucleotide positions
        """
        protein_lower = self.protein.lower()
        if protein_lower not in self.PROTEIN_COORDS:
            raise ValueError(f"Unknown protein: {self.protein}")

        start, _ = self.PROTEIN_COORDS[protein_lower]
        codon_start = start + (self.position - 1) * 3
        return (codon_start, codon_start + 2)

    def __str__(self) -> str:
        return f"{self.protein.upper()}:{self.position}"

    def __repr__(self) -> str:
        return f"HXB2Position({self.protein}, {self.position})"
