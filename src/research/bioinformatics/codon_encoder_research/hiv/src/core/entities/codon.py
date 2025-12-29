"""
Codon entity - fundamental unit of genetic encoding.

A codon is a triplet of nucleotides that encodes an amino acid.
This is the atomic unit for p-adic hyperbolic encoding.
"""
from dataclasses import dataclass

# Standard genetic code (DNA codons -> amino acids)
GENETIC_CODE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS: frozenset[str] = frozenset({"TAA", "TAG", "TGA"})
VALID_NUCLEOTIDES: frozenset[str] = frozenset("ACGTU")


@dataclass(frozen=True, slots=True)
class Codon:
    """
    Represents a single codon (triplet of nucleotides).

    This is an immutable value object - once created, it cannot be changed.
    """

    nucleotides: str

    def __post_init__(self) -> None:
        """Validate codon."""
        if len(self.nucleotides) != 3:
            raise ValueError(f"Codon must be 3 nucleotides: {self.nucleotides}")
        upper = self.nucleotides.upper()
        if not all(n in VALID_NUCLEOTIDES for n in upper):
            raise ValueError(f"Invalid nucleotides: {self.nucleotides}")
        # Normalize to uppercase (bypass frozen)
        object.__setattr__(self, "nucleotides", upper.replace("U", "T"))

    @property
    def amino_acid(self) -> str:
        """Get the translated amino acid (single letter code)."""
        return GENETIC_CODE.get(self.nucleotides, "X")

    @property
    def is_stop(self) -> bool:
        """Check if this is a stop codon."""
        return self.nucleotides in STOP_CODONS

    @property
    def is_start(self) -> bool:
        """Check if this is a start codon (ATG/Methionine)."""
        return self.nucleotides == "ATG"

    def __str__(self) -> str:
        return self.nucleotides

    def __repr__(self) -> str:
        return f"Codon({self.nucleotides!r})"
