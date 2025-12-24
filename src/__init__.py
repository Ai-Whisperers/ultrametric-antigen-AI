# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE - Canonical V5.11 Architecture.

This package implements Dual Neural VAEs for learning 3-adic algebraic
structure in ternary operation space using hyperbolic geometry.

Key modules:
- models: VAE architectures (canonical v5.11 with frozen encoder)
- losses: Loss functions including p-adic geodesic loss
- training: Trainers, schedulers, monitoring
- data: Ternary operation generation and datasets
- metrics: Hyperbolic correlation metrics
"""

__version__ = "5.11.0"
__author__ = "AI Whisperers"
__license__ = "PolyForm-Noncommercial-1.0.0"

# Canonical model (V5.11)
from .models.ternary_vae import TernaryVAEV5_11, TernaryVAEV5_11_OptionC

# Canonical aliases
TernaryVAE = TernaryVAEV5_11
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC

# Data
from .data import generate_all_ternary_operations, TernaryOperationDataset

# Training
from .training import (
    TernaryVAETrainer,
    HyperbolicVAETrainer,
    TrainingMonitor
)

# Metrics
from .metrics import compute_ranking_correlation_hyperbolic

__all__ = [
    # Canonical (V5.11)
    'TernaryVAE',
    'TernaryVAE_OptionC',
    'TernaryVAEV5_11',
    'TernaryVAEV5_11_OptionC',
    # Data
    'generate_all_ternary_operations',
    'TernaryOperationDataset',
    # Training
    'TernaryVAETrainer',
    'HyperbolicVAETrainer',
    'TrainingMonitor',
    # Metrics
    'compute_ranking_correlation_hyperbolic',
]
