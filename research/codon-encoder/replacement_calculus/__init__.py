"""Replacement Calculus: Structured Substitution Framework.

Core principle: Global optima must be implemented THROUGH local minima,
not ON TOP OF them.

This module provides:
- LocalMinimum: Groups representing stable configurations
- Morphism: Structure-preserving transformations
- Groupoid: Global optima as collections of local coordinates
- Invariants: (valuation, redundancy, symmetry_rank)

Usage:
    from research.codon_encoder.replacement_calculus import (
        LocalMinimum,
        Morphism,
        Groupoid,
        compute_invariants,
    )

Key insight: Local minima are not traps—they are coordinates.
"""

from .invariants import (
    valuation,
    redundancy,
    symmetry_rank,
    invariant_tuple,
    InvariantTuple,
)
from .groups import LocalMinimum, Constraint
from .morphisms import Morphism, is_valid_morphism
from .groupoids import Groupoid, find_escape_path

__all__ = [
    # Invariants
    "valuation",
    "redundancy",
    "symmetry_rank",
    "invariant_tuple",
    "InvariantTuple",
    # Groups
    "LocalMinimum",
    "Constraint",
    # Morphisms
    "Morphism",
    "is_valid_morphism",
    # Groupoids
    "Groupoid",
    "find_escape_path",
]
