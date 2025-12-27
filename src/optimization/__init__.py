# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Optimization Module.

Specialized optimization algorithms for vaccine design and
bioinformatics applications, including:
- Spin glass-based vaccine optimization
- Natural gradient methods
- Fisher-Rao metric optimization
"""

from src.optimization.vaccine_optimizer import (
    VaccineOptimizer,
    VaccineOptimizerConfig,
    VaccineCandidate,
    OptimizationResult,
    ImmunogenicityLandscape,
)

from src.optimization.natural_gradient import (
    VAENaturalGradient,
    AdaptiveNaturalGradient,
    FisherRaoSGD,
    VAEFisherEstimator,
)

__all__ = [
    # Vaccine optimization
    "VaccineOptimizer",
    "VaccineOptimizerConfig",
    "VaccineCandidate",
    "OptimizationResult",
    "ImmunogenicityLandscape",
    # Natural gradient optimization
    "VAENaturalGradient",
    "AdaptiveNaturalGradient",
    "FisherRaoSGD",
    "VAEFisherEstimator",
]
