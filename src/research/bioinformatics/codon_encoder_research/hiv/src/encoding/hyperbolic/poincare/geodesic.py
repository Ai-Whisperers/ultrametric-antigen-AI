"""
Geodesic computation in Poincaré disk.

Geodesics in the Poincaré disk are arcs of circles
orthogonal to the boundary (or diameters through origin).
"""
from .point import PoincarePoint, EPS
from .operations import exp_map, log_map
import math


def geodesic_path(
    x: PoincarePoint,
    y: PoincarePoint,
    steps: int = 10
) -> list[PoincarePoint]:
    """
    Compute points along geodesic from x to y.

    Uses linear interpolation in the tangent space at x,
    then maps back to the disk.

    Args:
        x: Start point
        y: End point
        steps: Number of points to generate

    Returns:
        List of points along geodesic
    """
    if steps < 2:
        return [x, y]

    # Get tangent vector at x pointing to y
    v = log_map(y, x)

    path = []
    for i in range(steps):
        t = i / (steps - 1)
        # Scale tangent vector
        v_t = tuple(vi * t for vi in v)
        # Map back to disk
        point = exp_map(v_t, x)
        path.append(point)

    return path


def geodesic_midpoint(x: PoincarePoint, y: PoincarePoint) -> PoincarePoint:
    """
    Compute midpoint of geodesic from x to y.

    Args:
        x: First point
        y: Second point

    Returns:
        Geodesic midpoint
    """
    # Get tangent vector
    v = log_map(y, x)

    # Half the tangent vector
    v_half = tuple(vi / 2 for vi in v)

    # Map back
    return exp_map(v_half, x)


def geodesic_interpolate(
    x: PoincarePoint,
    y: PoincarePoint,
    t: float
) -> PoincarePoint:
    """
    Interpolate along geodesic.

    Args:
        x: Start point
        y: End point
        t: Interpolation parameter (0 = x, 1 = y)

    Returns:
        Point at parameter t along geodesic
    """
    v = log_map(y, x)
    v_t = tuple(vi * t for vi in v)
    return exp_map(v_t, x)


def geodesic_direction(x: PoincarePoint, y: PoincarePoint) -> tuple[float, ...]:
    """
    Get unit direction vector from x toward y (in tangent space at x).

    Args:
        x: Source point
        y: Target point

    Returns:
        Unit tangent vector at x pointing toward y
    """
    v = log_map(y, x)
    norm = math.sqrt(sum(vi ** 2 for vi in v))
    if norm < EPS:
        return tuple(0.0 for _ in v)
    return tuple(vi / norm for vi in v)


def extend_geodesic(
    x: PoincarePoint,
    direction: tuple[float, ...],
    distance: float
) -> PoincarePoint:
    """
    Extend from x in given direction by hyperbolic distance.

    Args:
        x: Starting point
        direction: Unit tangent vector at x
        distance: Hyperbolic distance to travel

    Returns:
        Endpoint
    """
    # Scale direction by distance
    v = tuple(d * distance for d in direction)
    return exp_map(v, x)


def is_on_geodesic(
    p: PoincarePoint,
    x: PoincarePoint,
    y: PoincarePoint,
    tolerance: float = 1e-5
) -> bool:
    """
    Check if point p lies on geodesic from x to y.

    Args:
        p: Point to check
        x: Geodesic start
        y: Geodesic end
        tolerance: Distance tolerance

    Returns:
        True if p is on the geodesic
    """
    from .distance import poincare_distance

    d_xy = poincare_distance(x, y)
    d_xp = poincare_distance(x, p)
    d_py = poincare_distance(p, y)

    # On geodesic if d(x,p) + d(p,y) = d(x,y)
    return abs(d_xp + d_py - d_xy) < tolerance


def frechet_mean(
    points: list[PoincarePoint],
    weights: list[float] = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> PoincarePoint:
    """
    Compute Fréchet mean (barycenter) of points.

    The Fréchet mean minimizes sum of squared distances.

    Args:
        points: List of points
        weights: Optional weights (default: uniform)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Fréchet mean point
    """
    if not points:
        raise ValueError("Cannot compute mean of empty list")

    if weights is None:
        weights = [1.0 / len(points)] * len(points)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # Initialize at first point
    mean = points[0]

    for _ in range(max_iter):
        # Compute weighted sum of tangent vectors
        tangent_sum = [0.0] * mean.dimension

        for p, w in zip(points, weights):
            v = log_map(p, mean)
            for i in range(len(tangent_sum)):
                tangent_sum[i] += w * v[i]

        # Check convergence
        norm = math.sqrt(sum(t ** 2 for t in tangent_sum))
        if norm < tol:
            break

        # Update mean
        mean = exp_map(tuple(tangent_sum), mean)

    return mean
