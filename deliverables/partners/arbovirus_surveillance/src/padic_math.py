# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic mathematics for arbovirus sequence analysis.

Self-contained module providing p-adic valuation and related functions
for the Alejandra Rojas arbovirus surveillance package.

The 3-adic framework is used because:
- Codons have 3 positions (natural ternary structure)
- Hierarchical viral evolution follows p-adic topology
"""

from __future__ import annotations


# ============================================================================
# Constants
# ============================================================================

DEFAULT_P = 3  # Default prime base (3-adic for ternary/codon structure)
PADIC_INFINITY_INT = 100  # Integer representation of infinity


# ============================================================================
# Core P-adic Operations
# ============================================================================


def padic_valuation(n: int, p: int = DEFAULT_P) -> int:
    """Compute p-adic valuation v_p(n).

    The p-adic valuation is the largest power of p that divides n.
    v_p(0) is defined as infinity (returns 100 for compatibility).

    Args:
        n: Integer to compute valuation for
        p: Prime base (default 3)

    Returns:
        Valuation v_p(n), or 100 for n=0 (representing infinity)

    Examples:
        >>> padic_valuation(9, 3)   # 9 = 3^2
        2
        >>> padic_valuation(6, 3)   # 6 = 2 * 3^1
        1
        >>> padic_valuation(5, 3)   # 5 is not divisible by 3
        0
    """
    if n == 0:
        return PADIC_INFINITY_INT

    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def padic_norm(n: int, p: int = DEFAULT_P) -> float:
    """Compute p-adic norm |n|_p = p^(-v_p(n)).

    The p-adic norm is the multiplicative inverse of the valuation.
    Smaller norm = more divisible by p = "closer to zero" in p-adic topology.

    Args:
        n: Integer to compute norm for
        p: Prime base (default 3)

    Returns:
        P-adic norm, or 0.0 for n=0
    """
    if n == 0:
        return 0.0
    v = padic_valuation(n, p)
    return float(p) ** (-v)


def padic_distance(a: int, b: int, p: int = DEFAULT_P) -> float:
    """Compute p-adic distance d_p(a, b) = |a - b|_p.

    The p-adic distance satisfies the ultrametric inequality:
    d(a, c) <= max(d(a, b), d(b, c))

    This means "triangles are isoceles" - a key property for
    hierarchical viral evolution analysis.

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default 3)

    Returns:
        P-adic distance between a and b
    """
    return padic_norm(a - b, p)
