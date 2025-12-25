# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CRISPR off-target analysis type definitions.

This module contains data structures for CRISPR analysis:
- MismatchType: Enumeration of mismatch categories
- OffTargetSite: Single off-target site data
- GuideSafetyProfile: Safety analysis for a guide RNA

Single responsibility: Type definitions only.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MismatchType(Enum):
    """Types of mismatches in CRISPR targeting."""

    MATCH = "match"
    TRANSITION = "transition"  # Purine-purine or pyrimidine-pyrimidine
    TRANSVERSION = "transversion"  # Purine-pyrimidine
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass
class OffTargetSite:
    """Represents a potential off-target site.

    Attributes:
        sequence: 20nt target sequence
        chromosome: Chromosome name
        position: Genomic position
        strand: Strand (+ or -)
        pam: PAM sequence (typically NGG)
        mismatches: List of (position, ref, alt) tuples
        mismatch_count: Total number of mismatches
        seed_mismatches: Mismatches in seed region (positions 1-12)
        padic_distance: p-Adic distance from target
        hyperbolic_distance: Hyperbolic embedding distance
        predicted_activity: Cleavage probability (0-1)
    """

    sequence: str
    chromosome: str
    position: int
    strand: str
    pam: str
    mismatches: list[tuple[int, str, str]]
    mismatch_count: int
    seed_mismatches: int
    padic_distance: float
    hyperbolic_distance: float
    predicted_activity: float


@dataclass
class GuideSafetyProfile:
    """Safety profile for a guide RNA.

    Attributes:
        guide_sequence: The guide RNA sequence
        total_offtargets: Total number of off-targets analyzed
        high_risk_offtargets: Off-targets with activity > threshold
        seed_region_offtargets: Off-targets with seed region matches
        min_padic_distance: Minimum p-adic distance to any off-target
        safety_radius: Hyperbolic safety radius
        specificity_score: Overall specificity (0-1, higher is safer)
        recommended: Whether this guide is recommended
    """

    guide_sequence: str
    total_offtargets: int
    high_risk_offtargets: int
    seed_region_offtargets: int
    min_padic_distance: float
    safety_radius: float
    specificity_score: float
    recommended: bool


# Nucleotide encoding constants
NUCLEOTIDE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
IDX_TO_NUCLEOTIDE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

# PAM sequences for different Cas variants
PAM_SEQUENCES = {
    "SpCas9": "NGG",
    "SaCas9": "NNGRRT",
    "Cas12a": "TTTV",
    "xCas9": "NG",
}


__all__ = [
    "MismatchType",
    "OffTargetSite",
    "GuideSafetyProfile",
    "NUCLEOTIDE_TO_IDX",
    "IDX_TO_NUCLEOTIDE",
    "PAM_SEQUENCES",
]
