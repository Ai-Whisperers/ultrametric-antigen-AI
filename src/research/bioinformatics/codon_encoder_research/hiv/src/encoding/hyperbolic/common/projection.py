"""
Projections between hyperbolic models.

Supports Poincaré disk, hyperboloid (Lorentz), and Klein models.
"""
from ..poincare.point import PoincarePoint, EPS
import math


def project_to_hyperboloid(p: PoincarePoint) -> tuple[float, ...]:
    """
    Project Poincaré disk point to hyperboloid model.

    The hyperboloid model uses coordinates (x_0, x_1, ..., x_n) where
    -x_0² + x_1² + ... + x_n² = -1 (for curvature -1).

    Args:
        p: Point in Poincaré disk

    Returns:
        Coordinates in hyperboloid model (x_0, x_1, ..., x_n)
    """
    r_sq = p.norm_squared

    # Time component
    x_0 = (1 + r_sq) / (1 - r_sq + EPS)

    # Space components
    scale = 2 / (1 - r_sq + EPS)
    space = tuple(c * scale for c in p.coords)

    return (x_0,) + space


def project_to_poincare(hyperboloid_coords: tuple[float, ...]) -> PoincarePoint:
    """
    Project hyperboloid point to Poincaré disk.

    Args:
        hyperboloid_coords: (x_0, x_1, ..., x_n) in hyperboloid model

    Returns:
        Point in Poincaré disk
    """
    x_0 = hyperboloid_coords[0]
    space = hyperboloid_coords[1:]

    # Projection formula
    scale = 1 / (1 + x_0 + EPS)
    poincare_coords = tuple(c * scale for c in space)

    return PoincarePoint(coords=poincare_coords)


def project_to_klein(p: PoincarePoint) -> tuple[float, ...]:
    """
    Project Poincaré disk point to Klein model.

    The Klein model uses the same unit disk but with different metric.
    Geodesics are straight lines in Klein model.

    Args:
        p: Point in Poincaré disk

    Returns:
        Coordinates in Klein model
    """
    r_sq = p.norm_squared
    scale = 2 / (1 + r_sq + EPS)

    return tuple(c * scale for c in p.coords)


def klein_to_poincare(klein_coords: tuple[float, ...]) -> PoincarePoint:
    """
    Project Klein model point to Poincaré disk.

    Args:
        klein_coords: Coordinates in Klein model

    Returns:
        Point in Poincaré disk
    """
    k_norm_sq = sum(c ** 2 for c in klein_coords)

    # Projection formula
    scale = 1 / (1 + math.sqrt(1 - k_norm_sq + EPS))
    poincare_coords = tuple(c * scale for c in klein_coords)

    return PoincarePoint(coords=poincare_coords)


def project_euclidean_to_poincare(
    euclidean_coords: tuple[float, ...],
    scale: float = 1.0
) -> PoincarePoint:
    """
    Project Euclidean coordinates into Poincaré disk.

    Uses tanh scaling to map R^n -> B^n.

    Args:
        euclidean_coords: Coordinates in Euclidean space
        scale: Scaling factor before tanh

    Returns:
        Point in Poincaré disk
    """
    # Calculate Euclidean norm
    norm = math.sqrt(sum(c ** 2 for c in euclidean_coords))
    if norm < EPS:
        return PoincarePoint.origin(len(euclidean_coords))

    # Apply tanh scaling
    target_norm = math.tanh(scale * norm)
    poincare_coords = tuple(c * target_norm / norm for c in euclidean_coords)

    return PoincarePoint(coords=poincare_coords)
