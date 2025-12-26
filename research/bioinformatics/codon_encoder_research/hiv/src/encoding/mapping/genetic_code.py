"""
Genetic code data and utilities.

The standard genetic code maps 64 codons to 20 amino acids + stop.
"""

# Standard genetic code (DNA)
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

# Reverse mapping: amino acid -> list of codons
REVERSE_GENETIC_CODE: dict[str, list[str]] = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in REVERSE_GENETIC_CODE:
        REVERSE_GENETIC_CODE[aa] = []
    REVERSE_GENETIC_CODE[aa].append(codon)

# Degeneracy of each amino acid
CODON_DEGENERACY: dict[str, int] = {
    aa: len(codons) for aa, codons in REVERSE_GENETIC_CODE.items()
}


def codon_to_amino_acid(codon: str) -> str:
    """
    Translate codon to amino acid.

    Args:
        codon: 3-letter codon (DNA)

    Returns:
        Single-letter amino acid code ('X' if unknown)
    """
    return GENETIC_CODE.get(codon.upper().replace("U", "T"), "X")


def amino_acid_to_codons(aa: str) -> list[str]:
    """
    Get all codons encoding an amino acid.

    Args:
        aa: Single-letter amino acid code

    Returns:
        List of codons encoding this amino acid
    """
    return REVERSE_GENETIC_CODE.get(aa.upper(), [])


def is_synonymous(codon1: str, codon2: str) -> bool:
    """
    Check if two codons are synonymous (encode same AA).

    Args:
        codon1: First codon
        codon2: Second codon

    Returns:
        True if synonymous
    """
    return codon_to_amino_acid(codon1) == codon_to_amino_acid(codon2)


def codon_to_position_changes(codon1: str, codon2: str) -> list[int]:
    """
    Get positions that differ between codons.

    Args:
        codon1: First codon
        codon2: Second codon

    Returns:
        List of positions (0, 1, 2) that differ
    """
    c1 = codon1.upper()
    c2 = codon2.upper()
    return [i for i in range(3) if c1[i] != c2[i]]


def transition_transversion(codon1: str, codon2: str) -> dict:
    """
    Classify nucleotide changes as transitions or transversions.

    Transitions: A<->G, C<->T (purines<->purines, pyrimidines<->pyrimidines)
    Transversions: purine<->pyrimidine

    Args:
        codon1: First codon
        codon2: Second codon

    Returns:
        Dict with 'transitions' and 'transversions' counts
    """
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}

    transitions = 0
    transversions = 0

    for n1, n2 in zip(codon1.upper(), codon2.upper()):
        if n1 != n2:
            same_class = (
                (n1 in purines and n2 in purines) or
                (n1 in pyrimidines and n2 in pyrimidines)
            )
            if same_class:
                transitions += 1
            else:
                transversions += 1

    return {"transitions": transitions, "transversions": transversions}
