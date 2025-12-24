# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Metrics computation components.

This module contains evaluation metrics:
- hyperbolic: 3-adic ranking correlation using Poincare geometry (v5.10)
- coverage: Coverage evaluation (unique operations learned)
- entropy: Latent space entropy computation
- reconstruction: Reconstruction accuracy metrics
- tracking: Coverage and metrics tracking
"""

from .hyperbolic import (
    project_to_poincare,
    poincare_distance,
    compute_3adic_valuation,
    compute_ranking_correlation_hyperbolic
)

__all__ = [
    'project_to_poincare',
    'poincare_distance',
    'compute_3adic_valuation',
    'compute_ranking_correlation_hyperbolic'
]
