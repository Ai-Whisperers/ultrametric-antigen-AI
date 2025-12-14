"""Core domain layer - Single Source of Truth.

This module contains the fundamental domain concepts that all other
modules depend on. Changes here affect the entire codebase.

Modules:
    ternary: Ternary algebra (valuation, distance, operations)

Usage:
    from src.core import TERNARY

    # All ternary operations go through the singleton
    v = TERNARY.valuation(indices)
    d = TERNARY.distance(i, j)
    t = TERNARY.to_ternary(indices)
"""

from .ternary import (
    TernarySpace,
    TERNARY,
    valuation,
    distance,
    to_ternary,
    from_ternary,
)

__all__ = [
    'TernarySpace',
    'TERNARY',
    'valuation',
    'distance',
    'to_ternary',
    'from_ternary',
]
