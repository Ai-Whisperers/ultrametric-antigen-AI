# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized configuration module.

This module provides:
- Constants: All magic numbers and defaults in one place
- Schema: Typed, validated configuration classes
- Loader: YAML + environment variable configuration loading

Usage:
    from src.config import load_config, TrainingConfig, EPSILON

    # Load configuration
    config = load_config("config.yaml")

    # Access constants
    from src.config.constants import EPSILON, DEFAULT_CURVATURE

    # Create config programmatically
    config = TrainingConfig(epochs=500, batch_size=128)
"""

# Constants
from .constants import (
    # Numerical stability
    EPSILON,
    EPSILON_LOG,
    EPSILON_NORM,
    EPSILON_TEMP,
    # Geometry
    CURVATURE_MAX,
    CURVATURE_MIN,
    DEFAULT_CURVATURE,
    DEFAULT_LATENT_DIM,
    DEFAULT_MAX_RADIUS,
    RADIUS_MAX,
    RADIUS_MIN,
    # Training
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FREE_BITS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    # Gradient balance
    GRAD_EMA_MOMENTUM,
    GRAD_SCALE_MAX,
    GRAD_SCALE_MIN,
    # Ternary space
    MAX_VALUATION,
    N_TERNARY_DIGITS,
    N_TERNARY_OPERATIONS,
    TERNARY_BASE,
    # Loss functions
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_N_TRIPLETS,
    DEFAULT_RANKING_MARGIN,
    DEFAULT_REPULSION_SIGMA,
    # Observability
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_EMBEDDING_INTERVAL,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_HISTOGRAM_INTERVAL,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_TENSORBOARD_DIR,
)

# Schema
from .schema import (
    ConfigValidationError,
    GeometryConfig,
    LossWeights,
    OptimizerConfig,
    RankingConfig,
    TrainingConfig,
    VAEConfig,
)

# Loader
from .loader import load_config, save_config

__all__ = [
    # Loader functions
    "load_config",
    "save_config",
    # Schema classes
    "TrainingConfig",
    "GeometryConfig",
    "LossWeights",
    "OptimizerConfig",
    "RankingConfig",
    "VAEConfig",
    "ConfigValidationError",
    # Key constants (most commonly used)
    "EPSILON",
    "DEFAULT_CURVATURE",
    "DEFAULT_MAX_RADIUS",
    "DEFAULT_LATENT_DIM",
    "N_TERNARY_OPERATIONS",
]
