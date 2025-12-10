"""Loss function components.

This module contains loss computation separated from model architecture:
- ReconstructionLoss: Cross-entropy reconstruction loss
- KLDivergenceLoss: KL divergence with free bits support
- EntropyRegularization: Entropy-based regularization
- RepulsionLoss: Latent space repulsion
- DualVAELoss: Aggregated loss for dual VAE system
- PAdicMetricLoss: 3-adic metric alignment (Phase 1A)
- PAdicNormLoss: MSB/LSB hierarchy regularizer (Phase 1B)
- AdaptiveRankingLoss: Multi-scale ranking loss for ultrametric approximation
- HierarchicalNormLoss: MSB/LSB variance hierarchy
- CuriosityModule: Density-based exploration drive
- SymbioticBridge: MI-based coupling between VAEs
- AlgebraicClosureLoss: Homomorphism constraint
"""

from .dual_vae_loss import (
    ReconstructionLoss,
    KLDivergenceLoss,
    EntropyRegularization,
    RepulsionLoss,
    DualVAELoss
)

from .padic_losses import (
    PAdicMetricLoss,
    PAdicRankingLoss,
    PAdicNormLoss
)

from .appetitive_losses import (
    AdaptiveRankingLoss,
    HierarchicalNormLoss,
    CuriosityModule,
    SymbioticBridge,
    AlgebraicClosureLoss,
    ViolationBuffer
)

__all__ = [
    'ReconstructionLoss',
    'KLDivergenceLoss',
    'EntropyRegularization',
    'RepulsionLoss',
    'DualVAELoss',
    'PAdicMetricLoss',
    'PAdicRankingLoss',
    'PAdicNormLoss',
    'AdaptiveRankingLoss',
    'HierarchicalNormLoss',
    'CuriosityModule',
    'SymbioticBridge',
    'AlgebraicClosureLoss',
    'ViolationBuffer'
]
