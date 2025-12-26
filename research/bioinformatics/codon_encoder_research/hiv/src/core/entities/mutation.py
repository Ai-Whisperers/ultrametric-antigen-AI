"""
Mutation entity - represents a sequence change.

Mutations are changes from a wild-type residue to a mutant residue
at a specific position. Core unit for resistance and escape analysis.
"""
from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True, slots=True)
class Mutation:
    """
    Represents a mutation (amino acid substitution).

    Attributes:
        wild_type: Original residue (single letter)
        position: 1-indexed position in protein
        mutant: New residue (single letter, '*' for stop)
        protein: Optional protein name (e.g., 'PR', 'RT', 'IN')
    """

    wild_type: str
    position: int
    mutant: str
    protein: Optional[str] = None

    _PATTERN: re.Pattern = re.compile(r"^([A-Z])(\d+)([A-Z*])$")

    def __post_init__(self) -> None:
        """Validate mutation."""
        if len(self.wild_type) != 1 or not self.wild_type.isalpha():
            raise ValueError(f"Invalid wild type: {self.wild_type}")
        if self.position < 1:
            raise ValueError(f"Position must be >= 1: {self.position}")
        if len(self.mutant) != 1 or (not self.mutant.isalpha() and self.mutant != "*"):
            raise ValueError(f"Invalid mutant: {self.mutant}")

    @classmethod
    def from_string(cls, mutation_str: str, protein: Optional[str] = None) -> "Mutation":
        """
        Parse mutation from string format.

        Supports formats:
        - 'D30N' - standard format
        - 'PR:D30N' - with protein prefix

        Args:
            mutation_str: Mutation in string format
            protein: Optional protein override

        Returns:
            Parsed Mutation object

        Raises:
            ValueError: If format is invalid
        """
        mutation_str = mutation_str.strip().upper()

        # Handle protein prefix
        if ":" in mutation_str:
            prefix, mutation_str = mutation_str.split(":", 1)
            protein = protein or prefix

        match = cls._PATTERN.match(mutation_str)
        if not match:
            raise ValueError(f"Invalid mutation format: {mutation_str}")

        return cls(
            wild_type=match.group(1),
            position=int(match.group(2)),
            mutant=match.group(3),
            protein=protein,
        )

    @property
    def is_synonymous(self) -> bool:
        """Check if synonymous (same residue)."""
        return self.wild_type == self.mutant

    @property
    def is_nonsense(self) -> bool:
        """Check if nonsense (introduces stop codon)."""
        return self.mutant == "*"

    def __str__(self) -> str:
        base = f"{self.wild_type}{self.position}{self.mutant}"
        if self.protein:
            return f"{self.protein}:{base}"
        return base

    def __repr__(self) -> str:
        return f"Mutation({self!s})"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "wild_type": self.wild_type,
            "position": self.position,
            "mutant": self.mutant,
            "protein": self.protein,
        }
