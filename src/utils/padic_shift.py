# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic shift wrapper functions for biological sequence analysis.

This module implements the core p-adic shift operations for encoding
biological sequences (DNA, codons, amino acids) into p-adic space.

Key functions:
- padic_shift: Core shift operation for sequence positions
- codon_padic_distance: Distance between codons in 3-adic space
- sequence_padic_encoding: Full sequence encoding

The 3-adic framework is particularly suited for codon analysis because:
- Codons have 3 positions
- Each position has 4 nucleotides (mapped to 0-3)
- The p-adic distance captures wobble position effects

References:
- 2006_Kozyrev_Padic_Analysis_Methods.md
- 1975_Wong_CoEvolution_Theory.md
"""

from dataclasses import dataclass
from typing import Sequence

import torch

from src.biology.codons import codon_to_index, index_to_codon


def index_to_rna_codon(index: int) -> str:
    """Convert a codon index (0-63) to RNA codon (using U instead of T).

    Args:
        index: Codon index from 0 to 63

    Returns:
        3-letter RNA codon string (using U for uracil)
    """
    return index_to_codon(index).replace("T", "U")


@dataclass
class PAdicShiftResult:
    """Result of p-adic shift operation."""

    shift_value: float  # The actual shift value
    valuation: int  # p-adic valuation
    digits: list[int]  # p-adic digit expansion
    canonical_form: str  # String representation


def padic_valuation(n: int, p: int = 3) -> int:
    """Compute p-adic valuation v_p(n).

    The p-adic valuation is the largest power of p that divides n.
    v_p(0) is defined as infinity (we return a large number).

    Args:
        n: Integer to compute valuation for
        p: Prime base (default 3)

    Returns:
        Valuation v_p(n)
    """
    if n == 0:
        return 100  # Represent infinity

    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def padic_norm(n: int, p: int = 3) -> float:
    """Compute p-adic norm |n|_p = p^(-v_p(n)).

    Args:
        n: Integer to compute norm for
        p: Prime base (default 3)

    Returns:
        p-adic norm
    """
    if n == 0:
        return 0.0
    v = padic_valuation(n, p)
    return float(p) ** (-v)


def padic_distance(a: int, b: int, p: int = 3) -> float:
    """Compute p-adic distance d_p(a, b) = |a - b|_p.

    This is an ultrametric: d(a,c) <= max(d(a,b), d(b,c))

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default 3)

    Returns:
        p-adic distance
    """
    return padic_norm(a - b, p)


def padic_digits(n: int, p: int = 3, n_digits: int = 4) -> list[int]:
    """Compute p-adic digit expansion of n.

    Returns the first n_digits of the p-adic expansion:
    n = a_0 + a_1*p + a_2*p^2 + ...

    Args:
        n: Integer to expand
        p: Prime base (default 3)
        n_digits: Number of digits to compute

    Returns:
        List of p-adic digits [a_0, a_1, a_2, ...]
    """
    n = abs(n)
    digits = []
    for _ in range(n_digits):
        digits.append(n % p)
        n //= p
    return digits


def padic_shift(
    value: int,
    shift_amount: int = 1,
    p: int = 3,
) -> PAdicShiftResult:
    """Perform p-adic shift operation.

    Shifts the p-adic representation by the specified amount.
    This is equivalent to multiplication/division by powers of p
    in the p-adic domain.

    Args:
        value: Input integer value
        shift_amount: Number of positions to shift (positive = right, negative = left)
        p: Prime base (default 3)

    Returns:
        PAdicShiftResult with shifted value and metadata
    """
    if shift_amount >= 0:
        # Right shift: divide by p^shift_amount
        shifted = value // (p ** shift_amount)
    else:
        # Left shift: multiply by p^|shift_amount|
        shifted = value * (p ** abs(shift_amount))

    digits = padic_digits(shifted, p)
    valuation = padic_valuation(shifted, p)

    # Create canonical form string
    canonical = f"{shifted}_({p})"

    return PAdicShiftResult(
        shift_value=float(shifted),
        valuation=valuation,
        digits=digits,
        canonical_form=canonical,
    )


def codon_padic_distance(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute p-adic distance between two codons.

    Args:
        codon1: First codon (3 letters)
        codon2: Second codon (3 letters)
        p: Prime base (default 3)

    Returns:
        p-adic distance
    """
    idx1 = codon_to_index(codon1)
    idx2 = codon_to_index(codon2)
    return padic_distance(idx1, idx2, p)


def sequence_padic_encoding(
    sequence: str,
    p: int = 3,
    n_digits: int = 4,
) -> torch.Tensor:
    """Encode a nucleotide sequence into p-adic representation.

    Converts sequence to codon indices and then to p-adic digits.

    Args:
        sequence: Nucleotide sequence (length must be multiple of 3)
        p: Prime base (default 3)
        n_digits: Number of p-adic digits per codon

    Returns:
        Tensor of shape (n_codons, n_digits) containing p-adic digits
    """
    sequence = sequence.upper().replace("T", "U")

    if len(sequence) % 3 != 0:
        # Pad to multiple of 3
        padding = 3 - (len(sequence) % 3)
        sequence += "N" * padding

    n_codons = len(sequence) // 3
    encodings = []

    for i in range(n_codons):
        codon = sequence[i * 3 : (i + 1) * 3]
        if "N" in codon:
            # Unknown nucleotide - use zero encoding
            encodings.append([0] * n_digits)
        else:
            idx = codon_to_index(codon)
            digits = padic_digits(idx, p, n_digits)
            encodings.append(digits)

    return torch.tensor(encodings, dtype=torch.float32)


def batch_padic_distance(
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    p: int = 3,
) -> torch.Tensor:
    """Compute p-adic distances for batches of index pairs.

    Args:
        indices1: First indices (batch,)
        indices2: Second indices (batch,)
        p: Prime base (default 3)

    Returns:
        p-adic distances (batch,)
    """
    diff = torch.abs(indices1.long() - indices2.long())

    # Compute valuations
    valuations = torch.zeros_like(diff, dtype=torch.float)
    for k in range(1, 10):  # Up to p^9
        divisible = (diff % (p ** k) == 0) & (diff > 0)
        valuations[divisible] = k

    # Compute norms
    distances = torch.where(
        diff == 0,
        torch.zeros_like(valuations),
        torch.pow(float(p), -valuations),
    )

    return distances


def padic_distance_matrix(
    indices: torch.Tensor,
    p: int = 3,
) -> torch.Tensor:
    """Compute pairwise p-adic distance matrix.

    Args:
        indices: Tensor of indices (n,)
        p: Prime base (default 3)

    Returns:
        Distance matrix (n, n)
    """
    n = len(indices)
    indices_expanded = indices.unsqueeze(0).expand(n, -1)

    # Compute all pairwise differences
    diff = torch.abs(indices_expanded - indices.unsqueeze(1))

    # Compute valuations
    valuations = torch.zeros_like(diff, dtype=torch.float)
    for k in range(1, 10):
        divisible = (diff % (p ** k) == 0) & (diff > 0)
        valuations[divisible] = k

    # Compute distances
    distances = torch.where(
        diff == 0,
        torch.zeros_like(valuations),
        torch.pow(float(p), -valuations),
    )

    return distances


class PAdicSequenceEncoder:
    """Encoder for biological sequences using p-adic representation.

    Provides methods for encoding DNA, RNA, and protein sequences
    into p-adic space for analysis.
    """

    def __init__(self, p: int = 3, n_digits: int = 4):
        """Initialize encoder.

        Args:
            p: Prime base (default 3)
            n_digits: Number of p-adic digits to compute
        """
        self.p = p
        self.n_digits = n_digits

        # Precompute p-adic digits for all 64 codons
        self.codon_digits = torch.zeros(64, n_digits)
        for i in range(64):
            digits = padic_digits(i, p, n_digits)
            self.codon_digits[i] = torch.tensor(digits, dtype=torch.float)

        # Precompute 64x64 distance matrix
        indices = torch.arange(64)
        self.distance_matrix = padic_distance_matrix(indices, p)

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode a nucleotide sequence.

        Args:
            sequence: DNA/RNA sequence

        Returns:
            p-adic encoding tensor
        """
        return sequence_padic_encoding(sequence, self.p, self.n_digits)

    def encode_codons(self, codons: Sequence[str]) -> torch.Tensor:
        """Encode a list of codons.

        Args:
            codons: List of 3-letter codon strings

        Returns:
            Tensor of codon indices (n_codons,)
        """
        indices = [codon_to_index(c) for c in codons]
        return torch.tensor(indices)

    def get_padic_digits(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Get p-adic digit representations for codon indices.

        Args:
            codon_indices: Tensor of codon indices (batch, seq_len)

        Returns:
            p-adic digits (batch, seq_len, n_digits)
        """
        return self.codon_digits[codon_indices.long()]

    def compute_distances(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute pairwise p-adic distances.

        Vectorized implementation using advanced indexing (5-10x faster than loops).

        Args:
            indices: Codon indices (batch, seq_len)

        Returns:
            Distance matrix (batch, seq_len, seq_len)
        """
        batch_size, seq_len = indices.shape
        indices = indices.long()

        # Use advanced indexing instead of triple loop
        # Expand indices for broadcasting: (batch, seq, 1) and (batch, 1, seq)
        i_idx = indices.unsqueeze(2).expand(-1, -1, seq_len)
        j_idx = indices.unsqueeze(1).expand(-1, seq_len, -1)

        # Flatten, index into precomputed matrix, reshape
        flat_i = i_idx.reshape(-1)
        flat_j = j_idx.reshape(-1)

        # Move distance_matrix to same device as indices if needed
        dist_matrix = self.distance_matrix
        if dist_matrix.device != indices.device:
            dist_matrix = dist_matrix.to(indices.device)

        distances = dist_matrix[flat_i, flat_j].reshape(batch_size, seq_len, seq_len)

        return distances


class PAdicCodonAnalyzer:
    """Analyzer for codon usage patterns using p-adic metrics.

    Provides tools for analyzing synonymous codon usage,
    codon bias, and evolutionary distances.
    """

    # Genetic code table (standard)
    CODON_TABLE = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
        "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    def __init__(self, p: int = 3):
        """Initialize analyzer.

        Args:
            p: Prime base (default 3)
        """
        self.p = p
        self.encoder = PAdicSequenceEncoder(p=p)

        # Build synonymous codon groups
        self.synonymous_groups = self._build_synonymous_groups()

    def _build_synonymous_groups(self) -> dict[str, list[str]]:
        """Build mapping from amino acid to synonymous codons."""
        groups: dict[str, list[str]] = {}
        for codon, aa in self.CODON_TABLE.items():
            if aa not in groups:
                groups[aa] = []
            groups[aa].append(codon)
        return groups

    def synonymous_padic_spread(self, amino_acid: str) -> float:
        """Compute p-adic spread of synonymous codons for an amino acid.

        A larger spread indicates synonymous codons are more "distant"
        in p-adic space, suggesting wobble position effects.

        Args:
            amino_acid: Single-letter amino acid code

        Returns:
            Average pairwise p-adic distance among synonymous codons
        """
        codons = self.synonymous_groups.get(amino_acid.upper(), [])
        if len(codons) <= 1:
            return 0.0

        distances = []
        for i, c1 in enumerate(codons):
            for c2 in codons[i + 1 :]:
                d = codon_padic_distance(c1, c2, self.p)
                distances.append(d)

        return sum(distances) / len(distances) if distances else 0.0

    def codon_bias_score(self, sequence: str) -> float:
        """Compute codon bias score based on p-adic distances.

        Higher scores indicate preference for p-adically "close"
        synonymous codons.

        Args:
            sequence: Nucleotide sequence

        Returns:
            Codon bias score
        """
        sequence = sequence.upper().replace("T", "U")
        if len(sequence) < 3:
            return 0.0

        n_codons = len(sequence) // 3
        total_distance = 0.0
        count = 0

        for i in range(n_codons):
            codon = sequence[i * 3 : (i + 1) * 3]
            if "N" in codon:
                continue

            aa = self.CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue

            # Compare to other synonymous codons
            for syn_codon in self.synonymous_groups.get(aa, []):
                if syn_codon != codon:
                    d = codon_padic_distance(codon, syn_codon, self.p)
                    total_distance += d
                    count += 1

        return total_distance / count if count > 0 else 0.0
