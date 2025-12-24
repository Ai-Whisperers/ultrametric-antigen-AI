"""Geometry module for hyperbolic and p-adic operations.

This module provides geoopt-backed implementations of hyperbolic geometry
operations, providing numerical stability and optimized performance.

Key components:
- PoincareBall: Manifold operations via geoopt
- Stable exp/log maps with automatic edge-case handling
- RiemannianAdam optimizer support
"""

from .poincare import (
    get_manifold,
    poincare_distance,
    project_to_poincare,
    exp_map_zero,
    log_map_zero,
    mobius_add,
    lambda_x,
    parallel_transport,
    PoincareModule,
    get_riemannian_optimizer,
    GEOOPT_AVAILABLE
)

__all__ = [
    'get_manifold',
    'poincare_distance',
    'project_to_poincare',
    'exp_map_zero',
    'log_map_zero',
    'mobius_add',
    'lambda_x',
    'parallel_transport',
    'PoincareModule',
    'get_riemannian_optimizer',
    'GEOOPT_AVAILABLE'
]
