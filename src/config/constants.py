# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized constants - Single Source of Truth for magic numbers.

This module contains ALL numerical constants, thresholds, and default values
used throughout the codebase. Import from here instead of hardcoding values.

Usage:
    from src.config.constants import EPSILON, DEFAULT_CURVATURE

Categories:
    - Numerical stability (epsilons)
    - Geometry defaults
    - Training defaults
    - Gradient balance bounds
    - Ternary space constants
"""

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Standard epsilon for division and normalization
EPSILON = 1e-8

# Epsilon for logarithm operations (prevents log(0))
EPSILON_LOG = 1e-10

# Epsilon for norm computations
EPSILON_NORM = 1e-6

# Epsilon for softmax temperature scaling
EPSILON_TEMP = 1e-4


# =============================================================================
# GEOMETRY DEFAULTS (Poincaré Ball / Hyperbolic)
# =============================================================================

# Default hyperbolic curvature (c > 0 for hyperbolic space)
DEFAULT_CURVATURE = 1.0

# Maximum radius in Poincaré ball (must be < 1/sqrt(c))
DEFAULT_MAX_RADIUS = 0.95

# Default latent space dimension
DEFAULT_LATENT_DIM = 16

# Curvature bounds for learnable curvature
CURVATURE_MIN = 0.1
CURVATURE_MAX = 10.0

# Radius bounds for projections
RADIUS_MIN = 0.0
RADIUS_MAX = 0.99


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

# Batch size
DEFAULT_BATCH_SIZE = 256

# Learning rate
DEFAULT_LEARNING_RATE = 1e-3

# Gradient clipping threshold
DEFAULT_GRAD_CLIP = 1.0

# Weight decay for AdamW
DEFAULT_WEIGHT_DECAY = 1e-4

# Default number of epochs
DEFAULT_EPOCHS = 300

# Early stopping patience
DEFAULT_PATIENCE = 150

# Free bits for KL divergence
DEFAULT_FREE_BITS = 0.3


# =============================================================================
# GRADIENT BALANCE BOUNDS
# =============================================================================

# Minimum gradient scale factor
GRAD_SCALE_MIN = 0.5

# Maximum gradient scale factor
GRAD_SCALE_MAX = 2.0

# EMA momentum for gradient tracking
GRAD_EMA_MOMENTUM = 0.99


# =============================================================================
# TERNARY SPACE CONSTANTS
# =============================================================================

# Number of ternary digits (3^9 = 19,683 operations)
N_TERNARY_DIGITS = 9

# Base for ternary operations
TERNARY_BASE = 3

# Total number of unique ternary operations
N_TERNARY_OPERATIONS = TERNARY_BASE ** N_TERNARY_DIGITS  # 19,683

# Maximum 3-adic valuation possible
MAX_VALUATION = N_TERNARY_DIGITS  # 9


# =============================================================================
# LOSS FUNCTION DEFAULTS
# =============================================================================

# Repulsion loss bandwidth
DEFAULT_REPULSION_SIGMA = 0.5

# Ranking loss margin
DEFAULT_RANKING_MARGIN = 0.1

# Number of triplets for ranking loss
DEFAULT_N_TRIPLETS = 500

# Hard negative mining ratio
DEFAULT_HARD_NEGATIVE_RATIO = 0.5


# =============================================================================
# OBSERVABILITY DEFAULTS
# =============================================================================

# TensorBoard log interval (batches)
DEFAULT_LOG_INTERVAL = 10

# Checkpoint save frequency (epochs)
DEFAULT_CHECKPOINT_FREQ = 10

# Evaluation sample count
DEFAULT_EVAL_SAMPLES = 1000

# Histogram logging interval (epochs)
DEFAULT_HISTOGRAM_INTERVAL = 10

# Embedding visualization interval (epochs)
DEFAULT_EMBEDDING_INTERVAL = 50


# =============================================================================
# PATH DEFAULTS (can be overridden by environment variables)
# =============================================================================

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Default TensorBoard runs directory
DEFAULT_TENSORBOARD_DIR = "runs"


__all__ = [
    # Numerical stability
    "EPSILON",
    "EPSILON_LOG",
    "EPSILON_NORM",
    "EPSILON_TEMP",
    # Geometry
    "DEFAULT_CURVATURE",
    "DEFAULT_MAX_RADIUS",
    "DEFAULT_LATENT_DIM",
    "CURVATURE_MIN",
    "CURVATURE_MAX",
    "RADIUS_MIN",
    "RADIUS_MAX",
    # Training
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_GRAD_CLIP",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_EPOCHS",
    "DEFAULT_PATIENCE",
    "DEFAULT_FREE_BITS",
    # Gradient balance
    "GRAD_SCALE_MIN",
    "GRAD_SCALE_MAX",
    "GRAD_EMA_MOMENTUM",
    # Ternary space
    "N_TERNARY_DIGITS",
    "TERNARY_BASE",
    "N_TERNARY_OPERATIONS",
    "MAX_VALUATION",
    # Loss functions
    "DEFAULT_REPULSION_SIGMA",
    "DEFAULT_RANKING_MARGIN",
    "DEFAULT_N_TRIPLETS",
    "DEFAULT_HARD_NEGATIVE_RATIO",
    # Observability
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_CHECKPOINT_FREQ",
    "DEFAULT_EVAL_SAMPLES",
    "DEFAULT_HISTOGRAM_INTERVAL",
    "DEFAULT_EMBEDDING_INTERVAL",
    # Paths
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_TENSORBOARD_DIR",
]
