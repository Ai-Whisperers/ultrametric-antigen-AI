# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""p-Adic loss functions for 3-adic manifold coherence.

This package implements losses from implement.md Phase 1A/1B:
- PAdicMetricLoss: Force latent distances to match 3-adic distances
- PAdicRankingLoss: Order-preserving ranking with triplet loss
- PAdicRankingLossV2: Hard negative mining + hierarchical margin
- PAdicRankingLossHyperbolic: Poincaré distance + radial hierarchy
- PAdicNormLoss: Enforce MSB/LSB hierarchy via p-adic norm

Also exports triplet mining utilities for custom loss implementations.

Usage:
    from src.losses.padic import PAdicRankingLossV2, PAdicRankingLossHyperbolic

    # Euclidean ranking loss
    loss_v2 = PAdicRankingLossV2(base_margin=0.05, hard_negative_ratio=0.5)
    ranking_loss, metrics = loss_v2(z, batch_indices)

    # Hyperbolic ranking loss
    loss_hyp = PAdicRankingLossHyperbolic(curvature=2.0, radial_weight=0.1)
    total_loss, metrics = loss_hyp(z, batch_indices)
"""

from .metric_loss import PAdicMetricLoss
from .norm_loss import PAdicNormLoss
from .ranking_hyperbolic import PAdicRankingLossHyperbolic
from .ranking_loss import PAdicRankingLoss
from .ranking_v2 import PAdicRankingLossV2
from .triplet_mining import (
    EuclideanTripletMiner,
    HyperbolicTripletMiner,
    TripletBatch,
    TripletMiner,
    compute_3adic_valuation_batch,
)

__all__ = [
    # Core losses
    "PAdicMetricLoss",
    "PAdicRankingLoss",
    "PAdicRankingLossV2",
    "PAdicRankingLossHyperbolic",
    "PAdicNormLoss",
    # Triplet mining utilities
    "TripletBatch",
    "TripletMiner",
    "EuclideanTripletMiner",
    "HyperbolicTripletMiner",
    "compute_3adic_valuation_batch",
]
