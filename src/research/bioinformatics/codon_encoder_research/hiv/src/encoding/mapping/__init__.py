"""
Codon mapping module.

Maps biological sequences to mathematical representations
using the genetic code and amino acid properties.
"""
from .genetic_code import GENETIC_CODE, codon_to_amino_acid, amino_acid_to_codons
from .amino_acid_properties import AMINO_ACID_PROPERTIES, get_aa_vector
from .codon_to_padic import codon_to_padic_number, NUCLEOTIDE_VALUES
from .hierarchy import CodonHierarchy, hierarchical_encoding

__all__ = [
    # Genetic code
    "GENETIC_CODE",
    "codon_to_amino_acid",
    "amino_acid_to_codons",
    # Amino acid properties
    "AMINO_ACID_PROPERTIES",
    "get_aa_vector",
    # P-adic mapping
    "codon_to_padic_number",
    "NUCLEOTIDE_VALUES",
    # Hierarchy
    "CodonHierarchy",
    "hierarchical_encoding",
]
