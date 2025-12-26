"""
P-adic distance and metric functions.

The p-adic metric is an ultrametric, meaning it satisfies the
strong triangle inequality: d(x,z) <= max(d(x,y), d(y,z)).

This property makes it ideal for representing hierarchical
genetic relationships.
"""
from .number import PadicNumber
from .arithmetic import padic_subtract


def padic_norm(a: PadicNumber) -> float:
    """
    Calculate p-adic norm (absolute value).

    |a|_p = p^(-v_p(a)) where v_p is the p-adic valuation.

    Args:
        a: P-adic number

    Returns:
        P-adic absolute value
    """
    return a.norm()


def padic_distance(a: PadicNumber, b: PadicNumber) -> float:
    """
    Calculate p-adic distance between two numbers.

    d_p(a, b) = |a - b|_p = p^(-v_p(a-b))

    This is an ultrametric distance.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        P-adic distance

    Raises:
        ValueError: If primes don't match
    """
    if a.prime != b.prime:
        raise ValueError(f"Prime mismatch: {a.prime} vs {b.prime}")

    if a == b:
        return 0.0

    diff = padic_subtract(a, b)
    return diff.norm()


def padic_valuation_distance(a: PadicNumber, b: PadicNumber) -> int:
    """
    Calculate distance in terms of p-adic valuation.

    Returns v_p(a - b), the power of p dividing a - b.
    Larger values mean "closer" in p-adic sense.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        P-adic valuation of difference
    """
    if a == b:
        return a.precision  # Infinity

    diff = padic_subtract(a, b)
    return diff.valuation()


def is_ultrametric(a: PadicNumber, b: PadicNumber, c: PadicNumber) -> bool:
    """
    Verify ultrametric inequality for three points.

    d(a,c) <= max(d(a,b), d(b,c))

    Args:
        a, b, c: Three p-adic numbers

    Returns:
        True if ultrametric inequality holds (should always be True)
    """
    d_ac = padic_distance(a, c)
    d_ab = padic_distance(a, b)
    d_bc = padic_distance(b, c)

    return d_ac <= max(d_ab, d_bc) + 1e-10  # Allow small float error


def normalized_padic_distance(a: PadicNumber, b: PadicNumber) -> float:
    """
    Calculate normalized p-adic distance in [0, 1].

    Useful for combining with other distance metrics.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        Distance normalized to [0, 1]
    """
    d = padic_distance(a, b)
    # Maximum distance is 1 (when valuation is 0)
    return min(d, 1.0)


def hierarchical_level(a: PadicNumber, b: PadicNumber) -> int:
    """
    Determine the hierarchical level at which a and b diverge.

    Higher levels mean more similar (share more hierarchy).

    For codons:
    - Level 0: Different at first nucleotide
    - Level 1: Same first nucleotide, differ at second
    - Level 2: Same first two, differ at third

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        Hierarchical level of divergence
    """
    if a == b:
        return len(a.digits)

    diff = padic_subtract(a, b)
    return diff.valuation()
