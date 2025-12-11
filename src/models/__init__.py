"""Model definitions for Ternary VAE v5.6, v5.7, v5.10, and Appetitive VAE."""

from .ternary_vae_v5_6 import DualNeuralVAEV5
from .ternary_vae_v5_7 import DualNeuralVAEV5_7, StateNetV3
from .ternary_vae_v5_10 import DualNeuralVAEV5_10, StateNetV4
from .appetitive_vae import AppetitiveDualVAE, create_appetitive_vae

__all__ = [
    'DualNeuralVAEV5',
    'DualNeuralVAEV5_7',
    'DualNeuralVAEV5_10',
    'StateNetV3',
    'StateNetV4',
    'AppetitiveDualVAE',
    'create_appetitive_vae'
]
