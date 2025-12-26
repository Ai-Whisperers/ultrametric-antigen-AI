"""
HLA allele value object - represents Human Leukocyte Antigen alleles.

HLA alleles determine which epitopes an individual can present
and are crucial for understanding CTL escape patterns.
"""
from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True, slots=True)
class HLAAllele:
    """
    Represents an HLA allele.

    HLA naming convention: Gene*AlleleGroup:SpecificAllele
    Example: A*02:01, B*57:01, C*07:02

    Attributes:
        gene: HLA gene (A, B, C, DRB1, etc.)
        allele_group: Two-digit allele group
        specific_allele: Optional specific allele (2-4 digits)
        full_name: Original full allele name
    """

    gene: str
    allele_group: str
    specific_allele: Optional[str] = None
    full_name: Optional[str] = None

    # Pattern for parsing HLA names
    _PATTERN: re.Pattern = re.compile(
        r"^([A-Z]+)\*?(\d{2}):?(\d{2,4})?$", re.IGNORECASE
    )

    @classmethod
    def from_string(cls, hla_str: str) -> "HLAAllele":
        """
        Parse HLA allele from string.

        Supports formats:
        - 'A*02:01'
        - 'A*0201'
        - 'A02:01'
        - 'A0201'

        Args:
            hla_str: HLA allele string

        Returns:
            Parsed HLAAllele

        Raises:
            ValueError: If format is invalid
        """
        hla_str = hla_str.strip().upper().replace("-", "")

        match = cls._PATTERN.match(hla_str)
        if not match:
            raise ValueError(f"Invalid HLA format: {hla_str}")

        return cls(
            gene=match.group(1),
            allele_group=match.group(2),
            specific_allele=match.group(3),
            full_name=hla_str,
        )

    @property
    def supertype(self) -> str:
        """Get HLA supertype (gene + allele group)."""
        return f"{self.gene}*{self.allele_group}"

    @property
    def is_class_i(self) -> bool:
        """Check if Class I HLA (A, B, C)."""
        return self.gene in ("A", "B", "C")

    @property
    def is_class_ii(self) -> bool:
        """Check if Class II HLA (DR, DQ, DP)."""
        return self.gene.startswith(("DR", "DQ", "DP"))

    def matches_supertype(self, other: "HLAAllele") -> bool:
        """Check if alleles share supertype."""
        return self.supertype == other.supertype

    def __str__(self) -> str:
        if self.specific_allele:
            return f"{self.gene}*{self.allele_group}:{self.specific_allele}"
        return f"{self.gene}*{self.allele_group}"

    def __repr__(self) -> str:
        return f"HLAAllele({self!s})"
