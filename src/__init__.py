"""Ternary VAE v5.10 - Pure Hyperbolic Geometry.

This package implements Dual Neural VAEs for learning 3-adic algebraic
structure in ternary operation space using hyperbolic geometry.

Key modules:
- models: VAE architectures (v5.6 legacy, v5.7 metric-aware, v5.10 hyperbolic)
- losses: Loss functions including hyperbolic prior/recon (v5.10)
- training: Trainers, schedulers, monitoring
- data: Ternary operation generation and datasets
- metrics: Hyperbolic correlation metrics
"""

__version__ = "5.10.0"
__author__ = "AI Whisperers"
__license__ = "MIT"

# v5.10 canonical model
from .models.ternary_vae_v5_10 import DualNeuralVAEV5_10

# Data (canonical location)
from .data import generate_all_ternary_operations, TernaryOperationDataset

# Training
from .training import (
    TernaryVAETrainer,
    HyperbolicVAETrainer,
    TrainingMonitor
)

# Metrics
from .metrics import compute_ranking_correlation_hyperbolic

# Legacy model (for backwards compatibility)
from .models.ternary_vae_v5_6 import DualNeuralVAEV5

__all__ = [
    # v5.10 canonical
    'DualNeuralVAEV5_10',
    'HyperbolicVAETrainer',
    'compute_ranking_correlation_hyperbolic',
    # Data
    'generate_all_ternary_operations',
    'TernaryOperationDataset',
    # Training
    'TernaryVAETrainer',
    'TrainingMonitor',
    # Legacy
    'DualNeuralVAEV5',
]
