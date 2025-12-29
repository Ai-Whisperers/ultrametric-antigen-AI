"""
Poincaré disk operations.

Implements core hyperbolic operations:
- Möbius addition (group operation)
- Exponential map (tangent space -> manifold)
- Logarithmic map (manifold -> tangent space)
- Parallel transport
"""
from .point import PoincarePoint, EPS, MAX_NORM
import math


def mobius_add(x: PoincarePoint, y: PoincarePoint) -> PoincarePoint:
    """
    Möbius addition in the Poincaré disk.

    x ⊕ y = ((1 + 2<x,y> + ||y||²)x + (1 - ||x||²)y) /
            (1 + 2<x,y> + ||x||²||y||²)

    This is the group operation for hyperbolic space.

    Args:
        x: First point
        y: Second point

    Returns:
        Sum x ⊕ y
    """
    if x.dimension != y.dimension:
        raise ValueError(f"Dimension mismatch: {x.dimension} vs {y.dimension}")

    x_norm_sq = x.norm_squared
    y_norm_sq = y.norm_squared
    xy_dot = sum(a * b for a, b in zip(x.coords, y.coords))

    # Numerator coefficients
    num_x = 1 + 2 * xy_dot + y_norm_sq
    num_y = 1 - x_norm_sq

    # Denominator
    denom = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq
    denom = max(denom, EPS)  # Numerical stability

    # Compute result
    result = tuple(
        (num_x * a + num_y * b) / denom
        for a, b in zip(x.coords, y.coords)
    )

    return PoincarePoint(coords=result, curvature=x.curvature)


def mobius_scalar_mult(r: float, x: PoincarePoint) -> PoincarePoint:
    """
    Möbius scalar multiplication.

    r ⊗ x = tanh(r * arctanh(||x||)) * x / ||x||

    Args:
        r: Scalar
        x: Point

    Returns:
        r ⊗ x
    """
    x_norm = x.norm
    if x_norm < EPS:
        return x

    # Compute new norm
    new_norm = math.tanh(r * math.atanh(min(x_norm, MAX_NORM)))

    # Scale coordinates
    scale = new_norm / x_norm
    return PoincarePoint(
        coords=tuple(c * scale for c in x.coords),
        curvature=x.curvature
    )


def exp_map(v: tuple[float, ...], base: PoincarePoint = None) -> PoincarePoint:
    """
    Exponential map from tangent space to Poincaré disk.

    Maps a tangent vector at base point to a point on the manifold.

    exp_x(v) = x ⊕ (tanh(||v|| / 2) * v / ||v||)

    Args:
        v: Tangent vector (at base point)
        base: Base point (default: origin)

    Returns:
        Point on manifold
    """
    if base is None:
        base = PoincarePoint.origin(len(v))

    v_norm = math.sqrt(sum(x * x for x in v))
    if v_norm < EPS:
        return base

    # Direction in tangent space
    direction = tuple(x / v_norm for x in v)

    # Scale by tanh(||v|| / 2)
    scale = math.tanh(v_norm / 2)
    scaled_v = PoincarePoint(
        coords=tuple(d * scale for d in direction),
        curvature=base.curvature
    )

    # Möbius add to base
    return mobius_add(base, scaled_v)


def log_map(y: PoincarePoint, base: PoincarePoint = None) -> tuple[float, ...]:
    """
    Logarithmic map from Poincaré disk to tangent space.

    Inverse of exponential map.

    log_x(y) = 2 * arctanh(||-x ⊕ y||) * (-x ⊕ y) / ||-x ⊕ y||

    Args:
        y: Point on manifold
        base: Base point (default: origin)

    Returns:
        Tangent vector at base
    """
    if base is None:
        base = PoincarePoint.origin(y.dimension)

    # Compute -base ⊕ y
    neg_base = PoincarePoint(
        coords=tuple(-c for c in base.coords),
        curvature=base.curvature
    )
    diff = mobius_add(neg_base, y)

    diff_norm = diff.norm
    if diff_norm < EPS:
        return tuple(0.0 for _ in range(y.dimension))

    # Scale by 2 * arctanh(||diff||)
    scale = 2 * math.atanh(min(diff_norm, MAX_NORM)) / diff_norm

    return tuple(c * scale for c in diff.coords)


def parallel_transport(
    v: tuple[float, ...],
    x: PoincarePoint,
    y: PoincarePoint
) -> tuple[float, ...]:
    """
    Parallel transport a tangent vector from x to y.

    Transports vector v from tangent space at x to tangent space at y
    along the geodesic connecting them.

    Args:
        v: Tangent vector at x
        x: Source point
        y: Target point

    Returns:
        Transported vector at y
    """
    # Use Möbius gyration for parallel transport
    x_norm_sq = x.norm_squared
    y_norm_sq = y.norm_squared
    xy_dot = sum(a * b for a, b in zip(x.coords, y.coords))

    # Conformal factor
    lambda_x = 2 / (1 - x_norm_sq + EPS)
    lambda_y = 2 / (1 - y_norm_sq + EPS)

    # Transport factor
    factor = lambda_x / lambda_y

    # Simplified transport (works for small distances)
    return tuple(factor * vi for vi in v)


def project_to_disk(x: PoincarePoint) -> PoincarePoint:
    """
    Project point back into disk if outside.

    Args:
        x: Point (possibly outside disk)

    Returns:
        Point inside disk
    """
    norm = x.norm
    if norm < MAX_NORM:
        return x

    scale = MAX_NORM / (norm + EPS)
    return PoincarePoint(
        coords=tuple(c * scale for c in x.coords),
        curvature=x.curvature
    )
