# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Core domain layer - Single Source of Truth.

This module contains the fundamental domain concepts that all other
modules depend on. Changes here affect the entire codebase.

Modules:
    ternary: Ternary algebra (valuation, distance, operations)
    padic_math: P-adic mathematics (distance, norm, goldilocks)
    metrics: Hyperbolic geometry metrics (consolidated from metrics/)

Usage:
    from src.core import TERNARY

    # All ternary operations go through the singleton
    v = TERNARY.valuation(indices)
    d = TERNARY.distance(i, j)
    t = TERNARY.to_ternary(indices)
"""

from .ternary import (TERNARY, TernarySpace, distance, from_ternary,
                      to_ternary, valuation)
from .metrics import (
    compute_3adic_valuation,
    compute_ranking_correlation_hyperbolic,
    poincare_distance,
    project_to_poincare,
)
from .padic_math import (
    DEFAULT_P,
    PADIC_INFINITY,
    padic_valuation,
    padic_norm,
    padic_distance,
    padic_digits,
    padic_distance_vectorized,
    padic_distance_matrix,
    padic_distance_batch,
    compute_goldilocks_score,
    is_in_goldilocks_zone,
    compute_hierarchical_embedding,
)

# Constants exposed at module level for convenience
N_OPERATIONS = TERNARY.N_OPERATIONS
N_DIGITS = TERNARY.N_DIGITS
MAX_VALUATION = TERNARY.MAX_VALUATION

__all__ = [
    # Ternary
    "TernarySpace",
    "TERNARY",
    "valuation",
    "distance",
    "to_ternary",
    "from_ternary",
    "N_OPERATIONS",
    "N_DIGITS",
    "MAX_VALUATION",
    # Hyperbolic metrics (consolidated from metrics/)
    "project_to_poincare",
    "poincare_distance",
    "compute_3adic_valuation",
    "compute_ranking_correlation_hyperbolic",
    # P-adic math
    "DEFAULT_P",
    "PADIC_INFINITY",
    "padic_valuation",
    "padic_norm",
    "padic_distance",
    "padic_digits",
    "padic_distance_vectorized",
    "padic_distance_matrix",
    "padic_distance_batch",
    "compute_goldilocks_score",
    "is_in_goldilocks_zone",
    "compute_hierarchical_embedding",
]
