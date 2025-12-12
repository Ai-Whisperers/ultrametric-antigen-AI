"""Utility functions for Ternary VAE.

This module contains:
- Coverage tracking utilities
- Reproducibility utilities (seed management)

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

__all__ = [
    # Coverage metrics
    'evaluate_coverage',
    'compute_latent_entropy',
    'compute_diversity_score',
    'CoverageTracker',
    # Reproducibility
    'set_seed',
    'get_generator',
]
