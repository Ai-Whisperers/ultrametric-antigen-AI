"""Utility functions for Ternary VAE.

This module contains:
- Coverage tracking utilities
- Reproducibility utilities (seed management)
- Precomputed ternary LUTs (P1 optimization)

For data generation, use src.data instead.
For hyperbolic metrics, use src.metrics instead.
"""

from .metrics import (
    evaluate_coverage,
    compute_latent_entropy,
    compute_diversity_score,
    CoverageTracker
)
from .reproducibility import set_seed, get_generator
from .ternary_lut import (
    VALUATION_LUT,
    TERNARY_LUT,
    get_valuation_batch,
    get_ternary_batch,
    get_3adic_distance,
    get_3adic_distance_batch,
)

__all__ = [
    # Coverage metrics
    'evaluate_coverage',
    'compute_latent_entropy',
    'compute_diversity_score',
    'CoverageTracker',
    # Reproducibility
    'set_seed',
    'get_generator',
    # Ternary LUTs (P1 optimization)
    'VALUATION_LUT',
    'TERNARY_LUT',
    'get_valuation_batch',
    'get_ternary_batch',
    'get_3adic_distance',
    'get_3adic_distance_batch',
]
