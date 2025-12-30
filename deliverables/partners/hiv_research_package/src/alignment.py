# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV Sequence Alignment Module.

This module provides sequence alignment functionality for HIV resistance
analysis, including alignment to reference sequences (HXB2) and mutation
detection.

Example:
    >>> aligner = HIVSequenceAligner()
    >>> result = aligner.align(sequence)
    >>> mutations = aligner.detect_mutations(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reference sequences (HXB2 K03455)
HXB2_RT_REFERENCE = """PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV
FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL
DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI
YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWT
VQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAE
LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGA
HTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTP
PLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNKGRQKVVTLTDTTNQKTELQ
AIYLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHK
GIGGNEQVDKLVSAGIRKVL"""

HXB2_IN_REFERENCE = """FLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGI
WQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSN
FTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAV
FIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAK
LLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"""


@dataclass
class AlignmentResult:
    """Result of sequence alignment.

    Attributes:
        query_sequence: Original query sequence
        aligned_query: Query aligned to reference
        aligned_reference: Reference sequence
        score: Alignment score
        identity: Sequence identity (0-1)
        coverage: Alignment coverage (0-1)
        gaps: Number of gaps
        mutations: List of detected mutations
        gene: Gene aligned to (RT, IN, PR)
    """

    query_sequence: str
    aligned_query: str
    aligned_reference: str
    score: float
    identity: float
    coverage: float
    gaps: int
    mutations: list[dict]
    gene: str


@dataclass
class MutationResult:
    """A detected mutation.

    Attributes:
        position: Position in reference (1-indexed)
        reference: Reference amino acid
        mutation: Mutant amino acid
        notation: Standard notation (e.g., K103N)
        gene: Gene (RT, IN, PR)
        is_resistance: Whether this is a known resistance mutation
    """

    position: int
    reference: str
    mutation: str
    notation: str
    gene: str
    is_resistance: bool = False


class HIVSequenceAligner:
    """Aligner for HIV sequences.

    Aligns query sequences to HXB2 reference to enable
    position-based mutation detection.

    Example:
        >>> aligner = HIVSequenceAligner()
        >>> result = aligner.align(rt_sequence, gene="RT")
        >>> print(f"Identity: {result.identity:.1%}")
    """

    # Scoring matrix for alignment
    MATCH_SCORE = 2
    MISMATCH_SCORE = -1
    GAP_OPEN = -5
    GAP_EXTEND = -1

    def __init__(self):
        """Initialize aligner with reference sequences."""
        self.references = {
            "RT": self._clean_sequence(HXB2_RT_REFERENCE),
            "IN": self._clean_sequence(HXB2_IN_REFERENCE),
        }

    def _clean_sequence(self, seq: str) -> str:
        """Clean and validate sequence."""
        return "".join(seq.upper().split())

    def align(
        self,
        query: str,
        gene: str = "RT",
        method: str = "simple",
    ) -> AlignmentResult:
        """Align query sequence to reference.

        Args:
            query: Query amino acid sequence
            gene: Gene to align to ("RT" or "IN")
            method: Alignment method ("simple" or "sw" for Smith-Waterman)

        Returns:
            AlignmentResult with alignment details
        """
        query = self._clean_sequence(query)
        reference = self.references.get(gene)

        if reference is None:
            raise ValueError(f"Unknown gene: {gene}")

        if method == "simple":
            return self._simple_align(query, reference, gene)
        else:
            return self._smith_waterman(query, reference, gene)

    def _simple_align(
        self,
        query: str,
        reference: str,
        gene: str,
    ) -> AlignmentResult:
        """Simple pairwise alignment (ungapped).

        For sequences of similar length, performs position-by-position
        comparison.
        """
        # Pad shorter sequence
        max_len = max(len(query), len(reference))
        query_padded = query.ljust(max_len, "-")
        ref_padded = reference.ljust(max_len, "-")

        # Count matches and mismatches
        matches = 0
        mismatches = 0
        gaps = 0
        mutations = []

        for i, (q, r) in enumerate(zip(query_padded, ref_padded)):
            if q == "-" or r == "-":
                gaps += 1
            elif q == r:
                matches += 1
            else:
                mismatches += 1
                mutations.append({
                    "position": i + 1,
                    "reference": r,
                    "mutation": q,
                    "notation": f"{r}{i + 1}{q}",
                })

        total_aligned = matches + mismatches
        identity = matches / total_aligned if total_aligned > 0 else 0
        coverage = min(len(query), len(reference)) / len(reference)
        score = matches * self.MATCH_SCORE + mismatches * self.MISMATCH_SCORE

        return AlignmentResult(
            query_sequence=query,
            aligned_query=query_padded[:len(reference)],
            aligned_reference=ref_padded[:len(reference)],
            score=score,
            identity=identity,
            coverage=coverage,
            gaps=gaps,
            mutations=mutations,
            gene=gene,
        )

    def _smith_waterman(
        self,
        query: str,
        reference: str,
        gene: str,
    ) -> AlignmentResult:
        """Smith-Waterman local alignment.

        More accurate for sequences that may not be full-length.
        """
        m, n = len(query), len(reference)

        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1))
        traceback = np.zeros((m + 1, n + 1), dtype=int)

        # Fill scoring matrix
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1, j - 1] + (
                    self.MATCH_SCORE if query[i - 1] == reference[j - 1]
                    else self.MISMATCH_SCORE
                )
                delete = score_matrix[i - 1, j] + self.GAP_EXTEND
                insert = score_matrix[i, j - 1] + self.GAP_EXTEND

                score_matrix[i, j] = max(0, match, delete, insert)

                if score_matrix[i, j] == match:
                    traceback[i, j] = 1  # Diagonal
                elif score_matrix[i, j] == delete:
                    traceback[i, j] = 2  # Up
                elif score_matrix[i, j] == insert:
                    traceback[i, j] = 3  # Left

                if score_matrix[i, j] > max_score:
                    max_score = score_matrix[i, j]
                    max_pos = (i, j)

        # Traceback
        aligned_query = []
        aligned_ref = []
        i, j = max_pos

        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            if traceback[i, j] == 1:  # Diagonal
                aligned_query.append(query[i - 1])
                aligned_ref.append(reference[j - 1])
                i -= 1
                j -= 1
            elif traceback[i, j] == 2:  # Up
                aligned_query.append(query[i - 1])
                aligned_ref.append("-")
                i -= 1
            else:  # Left
                aligned_query.append("-")
                aligned_ref.append(reference[j - 1])
                j -= 1

        aligned_query = "".join(reversed(aligned_query))
        aligned_ref = "".join(reversed(aligned_ref))

        # Calculate statistics
        matches = sum(1 for a, b in zip(aligned_query, aligned_ref) if a == b and a != "-")
        total = len(aligned_query)
        gaps = sum(1 for c in aligned_query + aligned_ref if c == "-")

        # Detect mutations
        mutations = []
        ref_pos = j  # Starting position in reference
        for i, (q, r) in enumerate(zip(aligned_query, aligned_ref)):
            if q != "-" and r != "-" and q != r:
                mutations.append({
                    "position": ref_pos + i + 1,
                    "reference": r,
                    "mutation": q,
                    "notation": f"{r}{ref_pos + i + 1}{q}",
                })

        identity = matches / (total - gaps) if (total - gaps) > 0 else 0
        coverage = (total - gaps) / len(reference)

        return AlignmentResult(
            query_sequence=query,
            aligned_query=aligned_query,
            aligned_reference=aligned_ref,
            score=max_score,
            identity=identity,
            coverage=coverage,
            gaps=gaps,
            mutations=mutations,
            gene=gene,
        )

    def detect_mutations(
        self,
        alignment_result: AlignmentResult,
    ) -> list[MutationResult]:
        """Detect mutations from alignment result.

        Args:
            alignment_result: AlignmentResult from align()

        Returns:
            List of MutationResult objects
        """
        from .constants import (
            TDR_MUTATIONS,
            WHO_SDRM_NRTI,
            WHO_SDRM_NNRTI,
            WHO_SDRM_INSTI,
        )

        results = []

        # Combine all known resistance mutations
        known_mutations = set()
        if alignment_result.gene == "RT":
            known_mutations.update(WHO_SDRM_NRTI)
            known_mutations.update(WHO_SDRM_NNRTI)
        elif alignment_result.gene == "IN":
            known_mutations.update(WHO_SDRM_INSTI)

        for mut in alignment_result.mutations:
            notation = mut["notation"]
            is_resistance = notation in known_mutations

            results.append(MutationResult(
                position=mut["position"],
                reference=mut["reference"],
                mutation=mut["mutation"],
                notation=notation,
                gene=alignment_result.gene,
                is_resistance=is_resistance,
            ))

        return results

    def align_multiple(
        self,
        sequences: list[str],
        gene: str = "RT",
    ) -> list[AlignmentResult]:
        """Align multiple sequences.

        Args:
            sequences: List of sequences to align
            gene: Gene to align to

        Returns:
            List of AlignmentResult objects
        """
        return [self.align(seq, gene) for seq in sequences]


def generate_demo_alignment():
    """Generate demo alignment data for testing."""
    # Simulate a sequence with K103N mutation
    reference = HIVSequenceAligner()._clean_sequence(HXB2_RT_REFERENCE)

    # Create mutant sequence
    mutant = list(reference)
    mutant[102] = "N"  # K103N (0-indexed = 102)

    return "".join(mutant)
