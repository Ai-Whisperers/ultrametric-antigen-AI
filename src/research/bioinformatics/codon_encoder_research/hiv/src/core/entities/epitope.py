"""
Epitope entity - represents an immune epitope.

Epitopes are short peptide sequences recognized by the immune system,
either by T-cells (CTL epitopes) or B-cells (antibody epitopes).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EpitopeType(Enum):
    """Type of epitope."""
    CTL = "ctl"           # Cytotoxic T-lymphocyte (CD8+)
    HELPER = "helper"     # Helper T-cell (CD4+)
    ANTIBODY = "antibody" # B-cell/antibody


@dataclass(slots=True)
class Epitope:
    """
    Represents an immune epitope.

    Attributes:
        sequence: Peptide sequence (amino acids)
        protein: Source protein (e.g., 'Gag', 'Pol', 'Env')
        start_position: Start position in protein (1-indexed, HXB2)
        end_position: End position in protein (1-indexed, HXB2)
        epitope_type: Type of epitope (CTL, helper, antibody)
        hla_restrictions: List of restricting HLA alleles
        id: Optional identifier
        metadata: Additional metadata
    """

    sequence: str
    protein: str
    start_position: int
    end_position: int
    epitope_type: EpitopeType = EpitopeType.CTL
    hla_restrictions: list[str] = field(default_factory=list)
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate epitope."""
        self.sequence = self.sequence.upper()
        if len(self.sequence) < 8 or len(self.sequence) > 15:
            pass  # Allow unusual lengths but log warning
        if self.start_position < 1:
            raise ValueError(f"Start position must be >= 1: {self.start_position}")
        if self.end_position < self.start_position:
            raise ValueError(
                f"End position must be >= start: {self.end_position} < {self.start_position}"
            )

    @property
    def length(self) -> int:
        """Get epitope length."""
        return len(self.sequence)

    @property
    def positions(self) -> list[int]:
        """Get list of all positions covered."""
        return list(range(self.start_position, self.end_position + 1))

    @property
    def anchor_positions(self) -> tuple[int, int]:
        """
        Get anchor positions (P2 and PΩ for CTL epitopes).

        Returns:
            Tuple of (P2 position, PΩ position)
        """
        if self.epitope_type != EpitopeType.CTL:
            raise ValueError("Anchor positions only defined for CTL epitopes")
        p2 = self.start_position + 1  # Second position
        p_omega = self.end_position    # Last position
        return (p2, p_omega)

    def overlaps(self, other: "Epitope") -> bool:
        """Check if this epitope overlaps with another."""
        if self.protein != other.protein:
            return False
        return not (
            self.end_position < other.start_position
            or other.end_position < self.start_position
        )

    def contains_position(self, position: int) -> bool:
        """Check if epitope contains a specific position."""
        return self.start_position <= position <= self.end_position

    def __str__(self) -> str:
        hla_str = f" ({','.join(self.hla_restrictions)})" if self.hla_restrictions else ""
        return f"{self.protein}:{self.sequence}{hla_str}"

    def __repr__(self) -> str:
        return f"Epitope({self.protein}, {self.sequence}, {self.start_position}-{self.end_position})"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "protein": self.protein,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "epitope_type": self.epitope_type.value,
            "hla_restrictions": self.hla_restrictions,
            "id": self.id,
        }
