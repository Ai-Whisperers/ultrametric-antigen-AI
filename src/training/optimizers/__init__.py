# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training optimizers module.

Provides specialized optimizers for mixed Euclidean/Hyperbolic training:
- MixedRiemannianOptimizer: Handles both standard and manifold parameters
- MultiObjectiveOptimizer: Multi-objective optimization support

Consolidated from src/optimizers/ for better organization.
"""

from src.training.optimizers.multi_objective import (
    NSGAII,
    NSGAConfig,
    ParetoFrontOptimizer,
    compute_crowding_distance,
    fast_non_dominated_sort,
)
from src.training.optimizers.riemannian import (
    HyperbolicScheduler,
    MixedRiemannianOptimizer,
    OptimizerConfig,
    create_optimizer,
)

__all__ = [
    # Multi-objective optimization
    "ParetoFrontOptimizer",
    "NSGAII",
    "NSGAConfig",
    "fast_non_dominated_sort",
    "compute_crowding_distance",
    # Riemannian optimization
    "MixedRiemannianOptimizer",
    "HyperbolicScheduler",
    "OptimizerConfig",
    "create_optimizer",
]
