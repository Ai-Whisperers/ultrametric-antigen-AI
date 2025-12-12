"""Utility functions for Ternary VAE.

This module contains legacy coverage tracking utilities.
For data generation, use src.data instead.
For hyperbolic metrics, use src.metrics instead.
"""

from .metrics import (
    evaluate_coverage,
    compute_latent_entropy,
    compute_diversity_score,
    CoverageTracker
)

__all__ = [
    'evaluate_coverage',
    'compute_latent_entropy',
    'compute_diversity_score',
    'CoverageTracker',
]
