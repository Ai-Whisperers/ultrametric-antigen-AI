"""
Curvature computations for hyperbolic space.

Curvature relates to evolutionary constraints:
high curvature = high constraint = slow evolution.
"""
from ..poincare.point import PoincarePoint, EPS
import math


def gaussian_curvature(point: PoincarePoint) -> float:
    """
    Gaussian curvature at a point in Poincaré disk.

    For the Poincaré disk, Gaussian curvature is constant K = -1
    (or K = c for curvature parameter c).

    Args:
        point: Point in disk

    Returns:
        Gaussian curvature (constant for hyperbolic space)
    """
    return point.curvature


def compute_curvature(
    points: list[PoincarePoint],
    center: PoincarePoint = None
) -> float:
    """
    Estimate local curvature from neighborhood of points.

    Uses variance of distances as a proxy for curvature.
    Higher variance = less uniform = higher effective curvature.

    Args:
        points: Neighborhood points
        center: Center point (default: centroid)

    Returns:
        Estimated local curvature
    """
    from ..poincare.distance import poincare_distance
    from ..poincare.geodesic import frechet_mean

    if len(points) < 3:
        return abs(points[0].curvature) if points else 1.0

    # Use Fréchet mean as center if not provided
    if center is None:
        center = frechet_mean(points)

    # Calculate distances to center
    distances = [poincare_distance(center, p) for p in points]

    # Mean and variance
    mean_dist = sum(distances) / len(distances)
    variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)

    # Curvature proxy: normalized variance
    # Higher variance = points spread non-uniformly = higher curvature effect
    curvature_estimate = 1.0 + variance / (mean_dist + EPS)

    return curvature_estimate


def sectional_curvature(
    p1: PoincarePoint,
    p2: PoincarePoint,
    p3: PoincarePoint
) -> float:
    """
    Estimate sectional curvature from three points.

    Uses deviation from flat-space triangle inequality.

    Args:
        p1, p2, p3: Three points

    Returns:
        Sectional curvature estimate
    """
    from ..poincare.distance import poincare_distance

    d12 = poincare_distance(p1, p2)
    d23 = poincare_distance(p2, p3)
    d13 = poincare_distance(p1, p3)

    # In flat space: d13 = d12 + d23 (if on a line)
    # In hyperbolic space: d13 < d12 + d23 (always)
    # Curvature relates to how much "less"

    max_d = max(d12, d23)
    if max_d < EPS:
        return 0.0

    # Ratio indicates curvature
    ratio = d13 / (d12 + d23 + EPS)

    # 1.0 = flat, < 1.0 = hyperbolic
    return -math.log(ratio + EPS)


def radial_curvature_gradient(point: PoincarePoint) -> float:
    """
    Curvature gradient in radial direction.

    Points near boundary experience "higher" effective curvature
    due to metric distortion.

    Args:
        point: Point in disk

    Returns:
        Radial curvature gradient
    """
    # Conformal factor λ = 2 / (1 - ||x||²)
    # Increases toward boundary
    r_sq = point.norm_squared
    lambda_sq = 4 / ((1 - r_sq) ** 2 + EPS)

    # Gradient of curvature is related to gradient of λ
    return lambda_sq / 4  # Normalized
