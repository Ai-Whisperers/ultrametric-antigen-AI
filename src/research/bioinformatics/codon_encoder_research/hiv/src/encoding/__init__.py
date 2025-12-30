"""
Encoding layer - p-adic hyperbolic codon encoding.

This layer implements the mathematical framework for encoding
biological sequences into p-adic hyperbolic space.

V5.12.2 Geometry Compliance:
- All distance computations use proper Poincaré ball formulas
- Radii computed via hyperbolic distance from origin, not Euclidean norm
- Verified: 2024-12-30
"""

__version__ = "2.1.0"
__geometry_version__ = "V5.12.2"

from .encoder import PadicHyperbolicEncoder

__all__ = [
    "PadicHyperbolicEncoder",
]
