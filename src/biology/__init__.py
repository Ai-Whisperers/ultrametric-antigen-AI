# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized biology constants - Single Source of Truth.

This module contains ALL biology-related constants used throughout the codebase:
- Genetic code (codon to amino acid mapping)
- Nucleotide/base mappings
- Amino acid properties (hydrophobicity, charge, volume, polarity)
- Codon usage tables

Import from here instead of defining in individual modules.

Usage:
    from src.biology import (
        GENETIC_CODE,
        AMINO_ACID_PROPERTIES,
        codon_to_amino_acid,
    )
"""

from .amino_acids import (
    AMINO_ACID_3LETTER,
    AMINO_ACID_PROPERTIES,
    STANDARD_AMINO_ACIDS,
    get_amino_acid_charge,
    get_amino_acid_property,
)
from .codons import (
    BASE_TO_IDX,
    CODON_TO_INDEX,
    GENETIC_CODE,
    IDX_TO_BASE,
    INDEX_TO_CODON,
    codon_index_to_triplet,
    codon_to_amino_acid,
    triplet_to_codon_index,
)

__all__ = [
    # Amino acids
    "STANDARD_AMINO_ACIDS",
    "AMINO_ACID_3LETTER",
    "AMINO_ACID_PROPERTIES",
    "get_amino_acid_property",
    "get_amino_acid_charge",
    # Codons and genetic code
    "GENETIC_CODE",
    "BASE_TO_IDX",
    "IDX_TO_BASE",
    "CODON_TO_INDEX",
    "INDEX_TO_CODON",
    "codon_to_amino_acid",
    "codon_index_to_triplet",
    "triplet_to_codon_index",
]
