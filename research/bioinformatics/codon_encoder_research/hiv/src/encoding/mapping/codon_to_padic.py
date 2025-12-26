"""
Codon to p-adic number mapping.

Maps the 64 codons to p-adic integers using a base-3 representation
(3 nucleotides per codon, each with 4 possible values).

We use p=3 to reflect the 3-position structure of codons.
"""
from ..padic.number import PadicNumber


# Nucleotide to integer mapping
# Ordered by chemical similarity (purines: A,G; pyrimidines: C,T)
NUCLEOTIDE_VALUES: dict[str, int] = {
    "A": 0,  # Adenine (purine)
    "G": 1,  # Guanine (purine)
    "C": 2,  # Cytosine (pyrimidine)
    "T": 3,  # Thymine (pyrimidine)
    "U": 3,  # Uracil (RNA) = Thymine equivalent
}

# Alternative mapping preserving transition relationships
NUCLEOTIDE_VALUES_TRANSITION: dict[str, int] = {
    "A": 0,
    "G": 2,  # A<->G transition
    "C": 1,
    "T": 3,  # C<->T transition
    "U": 3,
}


def codon_to_padic_number(
    codon: str,
    prime: int = 3,
    precision: int = 10,
    use_transition_order: bool = False
) -> PadicNumber:
    """
    Convert codon to p-adic number.

    The codon is encoded as a base-4 number, then represented
    in p-adic form. Position 1 (first nucleotide) gets highest weight.

    Args:
        codon: 3-letter codon string
        prime: Prime for p-adic representation
        precision: Number of p-adic digits
        use_transition_order: Use alternative ordering

    Returns:
        P-adic representation of codon
    """
    if len(codon) != 3:
        raise ValueError(f"Codon must be 3 nucleotides: {codon}")

    codon = codon.upper().replace("U", "T")

    mapping = (
        NUCLEOTIDE_VALUES_TRANSITION if use_transition_order
        else NUCLEOTIDE_VALUES
    )

    # Convert to integer: pos1 * 16 + pos2 * 4 + pos3
    value = 0
    for i, nuc in enumerate(codon):
        if nuc not in mapping:
            raise ValueError(f"Invalid nucleotide: {nuc}")
        value = value * 4 + mapping[nuc]

    return PadicNumber.from_integer(value, prime=prime, precision=precision)


def padic_to_codon(p: PadicNumber) -> str:
    """
    Convert p-adic number back to codon.

    Args:
        p: P-adic number

    Returns:
        3-letter codon string
    """
    value = p.to_integer()

    # Inverse mapping
    inv_map = {v: k for k, v in NUCLEOTIDE_VALUES.items() if k != "U"}

    nucleotides = []
    for _ in range(3):
        nucleotides.append(inv_map[value % 4])
        value //= 4

    return "".join(reversed(nucleotides))


def codon_to_digits(codon: str) -> tuple[int, int, int]:
    """
    Get individual position values for codon.

    Args:
        codon: 3-letter codon

    Returns:
        Tuple of (pos1_value, pos2_value, pos3_value)
    """
    codon = codon.upper().replace("U", "T")
    return tuple(NUCLEOTIDE_VALUES[n] for n in codon)


def codon_hamming_distance(codon1: str, codon2: str) -> int:
    """
    Calculate Hamming distance between codons.

    Args:
        codon1: First codon
        codon2: Second codon

    Returns:
        Number of positions that differ
    """
    c1 = codon1.upper().replace("U", "T")
    c2 = codon2.upper().replace("U", "T")
    return sum(1 for a, b in zip(c1, c2) if a != b)


def single_step_neighbors(codon: str) -> list[str]:
    """
    Get all codons reachable by single nucleotide change.

    Args:
        codon: Starting codon

    Returns:
        List of neighboring codons (9 total)
    """
    neighbors = []
    codon = codon.upper().replace("U", "T")
    nucleotides = ["A", "G", "C", "T"]

    for pos in range(3):
        for nuc in nucleotides:
            if nuc != codon[pos]:
                new_codon = codon[:pos] + nuc + codon[pos + 1:]
                neighbors.append(new_codon)

    return neighbors
