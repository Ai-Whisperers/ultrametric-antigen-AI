"""
P-adic arithmetic module.

Implements p-adic numbers and arithmetic for the encoding framework.
P-adic numbers naturally capture hierarchical genetic relationships.

V5.12.2 Geometry Compliance:
- padic_distance(): Uses proper p-adic ultrametric
- padic_norm(): Correct p-adic valuation-based norm
- Integration with Poincaré ball uses hyperbolic formulas
- Verified: 2024-12-30
"""

__version__ = "2.1.0"
__geometry_version__ = "V5.12.2"

from .number import PadicNumber
from .arithmetic import padic_add, padic_multiply, padic_subtract
from .distance import padic_distance, padic_norm

__all__ = [
    "PadicNumber",
    "padic_add",
    "padic_multiply",
    "padic_subtract",
    "padic_distance",
    "padic_norm",
]
