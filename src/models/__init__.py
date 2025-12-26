# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Model definitions for Ternary VAE (canonical V5.11 architecture)."""

from .curriculum import ContinuousCurriculumModule, CurriculumScheduler
from .differentiable_controller import DifferentiableController
from .homeostasis import HomeostasisController, compute_Q
from .hyperbolic_projection import (DualHyperbolicProjection,
                                    HyperbolicProjection)
from .swarm_vae import AgentConfig, AgentRole, PheromoneField, SwarmAgent, SwarmVAE
from .ternary_vae import (FrozenDecoder, FrozenEncoder, TernaryVAEV5_11,
                          TernaryVAEV5_11_OptionC, TernaryVAEV5_11_PartialFreeze)

# Canonical exports (V5.11 architecture)
TernaryVAE = TernaryVAEV5_11
TernaryVAE_PartialFreeze = TernaryVAEV5_11_PartialFreeze
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC  # Deprecated alias

__all__ = [
    # Canonical (V5.11)
    "TernaryVAE",
    "TernaryVAE_PartialFreeze",
    "TernaryVAE_OptionC",  # Deprecated alias
    "TernaryVAEV5_11",
    "TernaryVAEV5_11_PartialFreeze",
    "TernaryVAEV5_11_OptionC",  # Deprecated alias
    "FrozenEncoder",
    "FrozenDecoder",
    # Projections
    "HyperbolicProjection",
    "DualHyperbolicProjection",
    # Controllers
    "DifferentiableController",
    # Curriculum
    "ContinuousCurriculumModule",
    "CurriculumScheduler",
    # Homeostasis (V5.11.7 + V5.11.8)
    "HomeostasisController",
    "compute_Q",
    # Swarm VAE (multi-agent architecture)
    "SwarmVAE",
    "SwarmAgent",
    "AgentConfig",
    "AgentRole",
    "PheromoneField",
]
