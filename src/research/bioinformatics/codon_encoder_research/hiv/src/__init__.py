"""
HIV Codon Encoder Research - Source Package

P-adic hyperbolic codon encoding for HIV bioinformatics analysis.
Uses Poincaré ball geometry to capture hierarchical genetic relationships.

V5.12.2 Geometry Compliance:
- All distance/radius computations use proper hyperbolic formulas
- Euclidean .norm() only used where mathematically correct
- Full audit completed: 2024-12-30

Submodules:
- encoding: P-adic hyperbolic encoder implementation
- encoding.hyperbolic: Poincaré ball geometry operations
- encoding.padic: P-adic arithmetic and distance
- core: Domain entities and interfaces
- infrastructure: Configuration and data access
"""

__version__ = "2.1.0"
__geometry_version__ = "V5.12.2"
