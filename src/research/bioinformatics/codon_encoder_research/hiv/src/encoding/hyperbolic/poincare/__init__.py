"""
Poincaré disk model of hyperbolic geometry.

The Poincaré disk is the unit disk {z : |z| < 1} with the metric
ds² = 4(dx² + dy²)/(1 - x² - y²)².

This model is conformal (preserves angles) and bounded,
making it ideal for visualization and computation.
"""
from .point import PoincarePoint
from .operations import mobius_add, exp_map, log_map, parallel_transport
from .distance import poincare_distance, geodesic_distance
from .geodesic import geodesic_path, geodesic_midpoint

__all__ = [
    "PoincarePoint",
    "mobius_add",
    "exp_map",
    "log_map",
    "parallel_transport",
    "poincare_distance",
    "geodesic_distance",
    "geodesic_path",
    "geodesic_midpoint",
]
