# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared immunology utilities for disease modules.

This module consolidates duplicated immunology patterns from disease modules:
- Epitope encoding and sequence analysis
- HLA genetic risk computation
- P-adic valuation and Goldilocks zone detection
- Shared dataclasses for epitope analysis

Usage:
    from src.analysis.immunology import (
        encode_amino_acid_sequence,
        compute_hla_genetic_risk,
        compute_padic_valuation,
        compute_goldilocks_score,
        EpitopeAnalysisResult,
    )
"""

from .epitope_encoding import (
    AMINO_ACID_INDEX,
    AMINO_ACID_PROPERTIES,
    encode_amino_acid_sequence,
    sequence_to_indices,
)
from .genetic_risk import (
    HLARiskProfile,
    compute_hla_genetic_risk,
)
from .padic_utils import (
    compute_goldilocks_score,
    compute_padic_distance,
    compute_padic_valuation,
)
from .types import (
    EpitopeAnalysisResult,
    HLAAlleleRisk,
)

__all__ = [
    # Epitope encoding
    "AMINO_ACID_INDEX",
    "AMINO_ACID_PROPERTIES",
    "encode_amino_acid_sequence",
    "sequence_to_indices",
    # Genetic risk
    "HLARiskProfile",
    "compute_hla_genetic_risk",
    # P-adic utilities
    "compute_padic_valuation",
    "compute_padic_distance",
    "compute_goldilocks_score",
    # Types
    "EpitopeAnalysisResult",
    "HLAAlleleRisk",
]
