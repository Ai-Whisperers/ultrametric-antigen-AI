"""
Hyperbolic geometry module.

Implements operations in hyperbolic space, primarily using
the Poincaré disk model. Hyperbolic geometry naturally
represents tree-like evolutionary structures.
"""
from .poincare.point import PoincarePoint
from .poincare.operations import mobius_add, exp_map, log_map
from .poincare.distance import poincare_distance
from .poincare.geodesic import geodesic_path

__all__ = [
    "PoincarePoint",
    "mobius_add",
    "exp_map",
    "log_map",
    "poincare_distance",
    "geodesic_path",
]
