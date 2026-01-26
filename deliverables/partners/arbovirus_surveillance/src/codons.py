# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Codon tables and genetic code for arbovirus analysis.

Self-contained module providing standard genetic code mappings
for the Alejandra Rojas arbovirus surveillance package.
"""

from __future__ import annotations

from typing import Dict, List

# Standard Genetic Code (NCBI table 1)
GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Reverse mapping: amino acid to list of codons
AMINO_ACID_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in AMINO_ACID_TO_CODONS:
        AMINO_ACID_TO_CODONS[aa] = []
    AMINO_ACID_TO_CODONS[aa].append(codon)

# Codon to index (0-63 for standard codons)
NUCLEOTIDES = "TCAG"  # Order matches standard codon tables
CODON_TO_INDEX: Dict[str, int] = {}
INDEX_TO_CODON: Dict[int, str] = {}

for i, base1 in enumerate(NUCLEOTIDES):
    for j, base2 in enumerate(NUCLEOTIDES):
        for k, base3 in enumerate(NUCLEOTIDES):
            codon = base1 + base2 + base3
            idx = i * 16 + j * 4 + k
            CODON_TO_INDEX[codon] = idx
            INDEX_TO_CODON[idx] = codon


def codon_index_to_triplet(idx: int) -> str:
    """Convert codon index (0-63) to triplet string.

    Args:
        idx: Codon index (0-63)

    Returns:
        Three-letter codon string
    """
    return INDEX_TO_CODON.get(idx, "NNN")


def translate_sequence(seq: str) -> str:
    """Translate nucleotide sequence to amino acid sequence.

    Args:
        seq: Nucleotide sequence (multiple of 3)

    Returns:
        Amino acid sequence
    """
    seq = seq.upper().replace("U", "T")
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = GENETIC_CODE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def get_codon_degeneracy(codon: str) -> int:
    """Get degeneracy of a codon (how many codons encode same AA).

    Args:
        codon: Three-letter codon

    Returns:
        Number of synonymous codons
    """
    aa = GENETIC_CODE.get(codon.upper(), None)
    if aa is None or aa == "*":
        return 0
    return len(AMINO_ACID_TO_CODONS[aa])
