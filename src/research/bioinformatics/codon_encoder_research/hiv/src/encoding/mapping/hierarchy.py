"""
Hierarchical codon encoding.

Captures the multi-level structure:
nucleotide -> codon -> amino acid -> property class
"""
from dataclasses import dataclass

from .genetic_code import GENETIC_CODE
from .amino_acid_properties import (
    HYDROPHOBIC, POLAR, CHARGED_POSITIVE, CHARGED_NEGATIVE, AROMATIC
)
from ..padic.number import PadicNumber


@dataclass
class CodonHierarchy:
    """
    Hierarchical representation of a codon.

    Levels:
    - Level 0: Nucleotide sequence (finest)
    - Level 1: Amino acid
    - Level 2: Property class (coarsest)
    """

    codon: str
    amino_acid: str
    property_class: str
    degeneracy: int  # Number of synonymous codons

    @classmethod
    def from_codon(cls, codon: str) -> "CodonHierarchy":
        """Create hierarchy from codon."""
        codon = codon.upper().replace("U", "T")
        aa = GENETIC_CODE.get(codon, "X")

        # Determine property class
        if aa in HYDROPHOBIC:
            prop_class = "hydrophobic"
        elif aa in POLAR:
            prop_class = "polar"
        elif aa in CHARGED_POSITIVE:
            prop_class = "positive"
        elif aa in CHARGED_NEGATIVE:
            prop_class = "negative"
        elif aa in AROMATIC:
            prop_class = "aromatic"
        elif aa == "*":
            prop_class = "stop"
        else:
            prop_class = "other"

        # Count synonymous codons
        degeneracy = sum(1 for c, a in GENETIC_CODE.items() if a == aa)

        return cls(
            codon=codon,
            amino_acid=aa,
            property_class=prop_class,
            degeneracy=degeneracy,
        )

    def same_amino_acid(self, other: "CodonHierarchy") -> bool:
        """Check if same amino acid (synonymous)."""
        return self.amino_acid == other.amino_acid

    def same_property_class(self, other: "CodonHierarchy") -> bool:
        """Check if same property class (conservative change)."""
        return self.property_class == other.property_class


def hierarchical_encoding(
    codon: str,
    prime: int = 3,
    precision: int = 10
) -> tuple[PadicNumber, PadicNumber, PadicNumber]:
    """
    Encode codon at multiple hierarchical levels.

    Returns p-adic numbers for:
    1. Full codon (64 values)
    2. Amino acid (21 values including stop)
    3. Property class (6 values)

    Args:
        codon: 3-letter codon
        prime: Prime for p-adic representation
        precision: Precision

    Returns:
        Tuple of (codon_padic, aa_padic, class_padic)
    """
    from .codon_to_padic import codon_to_padic_number

    hierarchy = CodonHierarchy.from_codon(codon)

    # Full codon encoding
    codon_padic = codon_to_padic_number(codon, prime, precision)

    # Amino acid encoding (0-20)
    aa_order = "ACDEFGHIKLMNPQRSTVWY*"
    aa_index = aa_order.index(hierarchy.amino_acid) if hierarchy.amino_acid in aa_order else 20
    aa_padic = PadicNumber.from_integer(aa_index, prime, precision)

    # Property class encoding (0-5)
    class_order = ["hydrophobic", "polar", "positive", "negative", "aromatic", "stop", "other"]
    class_index = class_order.index(hierarchy.property_class)
    class_padic = PadicNumber.from_integer(class_index, prime, precision)

    return (codon_padic, aa_padic, class_padic)


def hierarchical_distance(
    codon1: str,
    codon2: str,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> float:
    """
    Calculate weighted hierarchical distance.

    Combines distances at codon, amino acid, and property levels.

    Args:
        codon1: First codon
        codon2: Second codon
        weights: Weights for (codon, aa, class) levels

    Returns:
        Weighted distance
    """
    from ..padic.distance import normalized_padic_distance

    enc1 = hierarchical_encoding(codon1)
    enc2 = hierarchical_encoding(codon2)

    distances = [
        normalized_padic_distance(e1, e2)
        for e1, e2 in zip(enc1, enc2)
    ]

    return sum(w * d for w, d in zip(weights, distances))


def is_synonymous_mutation(codon1: str, codon2: str) -> bool:
    """Check if codons are synonymous (same amino acid)."""
    h1 = CodonHierarchy.from_codon(codon1)
    h2 = CodonHierarchy.from_codon(codon2)
    return h1.same_amino_acid(h2)


def is_conservative_mutation(codon1: str, codon2: str) -> bool:
    """Check if mutation is conservative (same property class)."""
    h1 = CodonHierarchy.from_codon(codon1)
    h2 = CodonHierarchy.from_codon(codon2)
    return h1.same_property_class(h2)
