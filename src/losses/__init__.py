"""Loss function components.

This module contains loss computation separated from model architecture:
- ReconstructionLoss: Cross-entropy reconstruction loss
- KLDivergenceLoss: KL divergence with free bits support
- EntropyRegularization: Entropy-based regularization
- RepulsionLoss: Latent space repulsion
- DualVAELoss: Aggregated loss for dual VAE system
"""

from .dual_vae_loss import (
    ReconstructionLoss,
    KLDivergenceLoss,
    EntropyRegularization,
    RepulsionLoss,
    DualVAELoss
)

__all__ = [
    'ReconstructionLoss',
    'KLDivergenceLoss',
    'EntropyRegularization',
    'RepulsionLoss',
    'DualVAELoss'
]
