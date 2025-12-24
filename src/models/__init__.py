"""Model definitions for Ternary VAE (canonical V5.11 architecture)."""

from .ternary_vae import TernaryVAEV5_11, TernaryVAEV5_11_OptionC, FrozenEncoder, FrozenDecoder
from .hyperbolic_projection import HyperbolicProjection, DualHyperbolicProjection
from .differentiable_controller import DifferentiableController, ThreeBodyController
from .curriculum import ContinuousCurriculumModule, CurriculumScheduler
from .homeostasis import HomeostasisController

# Canonical exports (V5.11 architecture)
TernaryVAE = TernaryVAEV5_11
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC

__all__ = [
    # Canonical (V5.11)
    'TernaryVAE',
    'TernaryVAE_OptionC',
    'TernaryVAEV5_11',
    'TernaryVAEV5_11_OptionC',
    'FrozenEncoder',
    'FrozenDecoder',
    # Projections
    'HyperbolicProjection',
    'DualHyperbolicProjection',
    # Controllers
    'DifferentiableController',
    'ThreeBodyController',
    # Curriculum
    'ContinuousCurriculumModule',
    'CurriculumScheduler',
    # Homeostasis (V5.11.7)
    'HomeostasisController',
]
