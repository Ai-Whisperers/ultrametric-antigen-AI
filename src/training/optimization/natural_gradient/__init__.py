# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Natural Gradient Optimization Module.

Specialized optimizers for VAE training using information geometry:
- VAENaturalGradient: Natural gradient specifically for VAE encoders
- AdaptiveNaturalGradient: Adapts damping based on training dynamics
- FisherRaoSGD: SGD with Fisher-Rao metric preconditioning

These optimizers account for the information geometry of the latent
space, leading to more stable and efficient training.

References:
- Amari (1998): Natural Gradient Works Efficiently in Learning
- Hoffman (2013): Stochastic Variational Inference
- Salimbeni (2018): Natural Gradients in Practice
"""

from src.training.optimization.natural_gradient.fisher_optimizer import (
    AdaptiveNaturalGradient,
    FisherRaoSGD,
    VAEFisherEstimator,
    VAENaturalGradient,
)

__all__ = [
    "VAENaturalGradient",
    "AdaptiveNaturalGradient",
    "FisherRaoSGD",
    "VAEFisherEstimator",
]
