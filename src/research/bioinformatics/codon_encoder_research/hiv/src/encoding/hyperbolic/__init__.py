"""
Hyperbolic geometry module.

Implements operations in hyperbolic space, primarily using
the Poincaré disk model. Hyperbolic geometry naturally
represents tree-like evolutionary structures.

V5.12.2 Geometry Compliance:
- poincare_distance(): Proper geodesic distance formula
- exp_map/log_map: Correct tangent space transformations
- All radii use hyperbolic distance from origin
- Euclidean .norm() only inside transformation formulas (correct)
- Verified: 2024-12-30
"""

__version__ = "2.1.0"
__geometry_version__ = "V5.12.2"

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
