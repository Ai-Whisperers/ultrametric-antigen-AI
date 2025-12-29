"""
Poincaré disk distance functions.

The hyperbolic distance in the Poincaré disk is:
d(x, y) = 2 * arctanh(||(-x) ⊕ y||)

where ⊕ is Möbius addition.
"""
from .point import PoincarePoint, EPS, MAX_NORM
from .operations import mobius_add
import math


def poincare_distance(x: PoincarePoint, y: PoincarePoint) -> float:
    """
    Calculate hyperbolic distance in Poincaré disk.

    d(x, y) = 2 * arctanh(||-x ⊕ y||)

    This is the geodesic distance in hyperbolic space.

    Args:
        x: First point
        y: Second point

    Returns:
        Hyperbolic distance
    """
    if x.dimension != y.dimension:
        raise ValueError(f"Dimension mismatch: {x.dimension} vs {y.dimension}")

    # Compute -x ⊕ y
    neg_x = PoincarePoint(
        coords=tuple(-c for c in x.coords),
        curvature=x.curvature
    )
    diff = mobius_add(neg_x, y)

    # Distance = 2 * arctanh(||diff||)
    diff_norm = min(diff.norm, MAX_NORM)
    return 2 * math.atanh(diff_norm)


def geodesic_distance(x: PoincarePoint, y: PoincarePoint) -> float:
    """
    Alias for poincare_distance (geodesic = shortest path).
    """
    return poincare_distance(x, y)


def distance_from_origin(x: PoincarePoint) -> float:
    """
    Distance from origin (simpler formula).

    d(0, x) = 2 * arctanh(||x||)
    """
    x_norm = min(x.norm, MAX_NORM)
    return 2 * math.atanh(x_norm)


def poincare_squared_distance(x: PoincarePoint, y: PoincarePoint) -> float:
    """
    Squared hyperbolic distance (for optimization).

    Avoids some computation when only relative distances matter.
    """
    # Direct formula for squared distance
    x_norm_sq = x.norm_squared
    y_norm_sq = y.norm_squared
    diff_sq = sum((a - b) ** 2 for a, b in zip(x.coords, y.coords))

    num = diff_sq
    denom = (1 - x_norm_sq) * (1 - y_norm_sq) + EPS

    # This is ||−x ⊕ y||²
    gamma = num / denom

    # d² = 4 * arctanh²(sqrt(gamma))
    sqrt_gamma = min(math.sqrt(gamma), MAX_NORM)
    return 4 * math.atanh(sqrt_gamma) ** 2


def conformal_factor(x: PoincarePoint) -> float:
    """
    Conformal factor at point x.

    λ_x = 2 / (1 - ||x||²)

    This relates the hyperbolic metric to the Euclidean metric.
    """
    return 2 / (1 - x.norm_squared + EPS)


def klein_norm_from_poincare(x: PoincarePoint) -> float:
    """
    Convert Poincaré norm to Klein model norm.

    Used for some computations that are simpler in Klein model.
    """
    r = x.norm
    return 2 * r / (1 + r * r)
