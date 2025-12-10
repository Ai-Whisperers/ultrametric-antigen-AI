"""Model definitions for Ternary VAE v5.6, v5.7, and Appetitive VAE."""

from .ternary_vae_v5_6 import DualNeuralVAEV5
from .ternary_vae_v5_7 import DualNeuralVAEV5_7, StateNetV3
from .appetitive_vae import AppetitiveDualVAE, create_appetitive_vae

__all__ = [
    'DualNeuralVAEV5',
    'DualNeuralVAEV5_7',
    'StateNetV3',
    'AppetitiveDualVAE',
    'create_appetitive_vae'
]
