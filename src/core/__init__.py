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

# Constants exposed at module level for convenience
N_OPERATIONS = TERNARY.N_OPERATIONS
N_DIGITS = TERNARY.N_DIGITS
MAX_VALUATION = TERNARY.MAX_VALUATION

__all__ = [
    'TernarySpace',
    'TERNARY',
    'valuation',
    'distance',
    'to_ternary',
    'from_ternary',
    'N_OPERATIONS',
    'N_DIGITS',
    'MAX_VALUATION',
]
