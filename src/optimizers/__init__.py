# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Low-level optimizer algorithms for hyperbolic and multi-objective training.

This module provides specialized optimization ALGORITHMS designed for:
1. Training on hyperbolic (Riemannian) manifolds
2. Multi-objective optimization with Pareto fronts

Note:
    This module contains optimizer IMPLEMENTATIONS (algorithms).
    For high-level optimization workflows (sequence design, etc.),
    see src/optimization/ instead.

    - src/optimizers/ (this) = How to optimize (optimizer algorithms)
    - src/optimization/ = What to optimize (sequence design workflows)

Optimizer Categories:
    **Multi-Objective Optimization**:
        ParetoFrontOptimizer: General Pareto front optimization
        NSGAII: Non-dominated Sorting Genetic Algorithm II
        fast_non_dominated_sort: Efficient Pareto ranking
        compute_crowding_distance: Diversity preservation metric

    **Riemannian Optimization**:
        MixedRiemannianOptimizer: Combined Euclidean/Riemannian optimizer
        HyperbolicScheduler: Learning rate scheduling for hyperbolic space
        create_optimizer: Factory function for optimizer creation

Example:
    >>> from src.optimizers import create_optimizer, OptimizerConfig
    >>> config = OptimizerConfig(lr=1e-3, riemannian_lr=1e-4)
    >>> optimizer = create_optimizer(model.parameters(), config)

    >>> # Multi-objective optimization
    >>> from src.optimizers import NSGAII, NSGAConfig
    >>> nsga = NSGAII(NSGAConfig(pop_size=100))
    >>> pareto_front = nsga.optimize(objectives)

Note:
    Riemannian optimizers use Riemannian gradient descent to respect
    the geometry of hyperbolic space (Poincaré ball model).
"""

from .multi_objective import (NSGAII, NSGAConfig, ParetoFrontOptimizer,
                              compute_crowding_distance,
                              fast_non_dominated_sort)
from .riemannian import (HyperbolicScheduler, MixedRiemannianOptimizer,
                         OptimizerConfig, create_optimizer)

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
