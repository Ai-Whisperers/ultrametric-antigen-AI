# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Training orchestration components.

This module contains components for managing the training process:
- BaseTrainer: Abstract base with defensive patterns (safe division, val_loader guards)
- TernaryVAETrainer: Main training loop (single responsibility)
- HyperbolicVAETrainer: Hyperbolic geometry trainer (canonical)
- Schedulers: Parameter scheduling (temperature, beta, learning rate)
- Monitor: Logging and metrics tracking (TensorBoard + file)
- ConfigSchema: Typed configuration validation
- Environment: Pre-training environment checks
"""

from .base import BaseTrainer, STATENET_KEYS
from .schedulers import (
    TemperatureScheduler,
    BetaScheduler,
    LearningRateScheduler,
    linear_schedule,
    cyclic_schedule
)
from .monitor import TrainingMonitor
from .trainer import TernaryVAETrainer
from .hyperbolic_trainer import HyperbolicVAETrainer
from .config_schema import (
    TrainingConfig,
    ModelConfig,
    ConfigValidationError,
    validate_config,
    config_to_dict
)
from .environment import (
    EnvironmentStatus,
    validate_environment,
    require_valid_environment
)

__all__ = [
    # Base trainer
    'BaseTrainer',
    'STATENET_KEYS',
    # Trainers
    'TernaryVAETrainer',
    'HyperbolicVAETrainer',
    # Schedulers
    'TemperatureScheduler',
    'BetaScheduler',
    'LearningRateScheduler',
    'linear_schedule',
    'cyclic_schedule',
    # Monitoring
    'TrainingMonitor',
    # Config validation
    'TrainingConfig',
    'ModelConfig',
    'ConfigValidationError',
    'validate_config',
    'config_to_dict',
    # Environment validation
    'EnvironmentStatus',
    'validate_environment',
    'require_valid_environment'
]
