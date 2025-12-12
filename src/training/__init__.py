"""Training orchestration components.

This module contains components for managing the training process:
- Trainer: Main training loop (single responsibility)
- HyperbolicVAETrainer: Pure hyperbolic geometry trainer (v5.10)
- AppetitiveVAETrainer: Trainer with bio-inspired appetite losses
- Schedulers: Parameter scheduling (temperature, beta, learning rate)
- Monitor: Logging and metrics tracking (TensorBoard + file)
- ConfigSchema: Typed configuration validation
- Environment: Pre-training environment checks
"""

from .schedulers import (
    TemperatureScheduler,
    BetaScheduler,
    LearningRateScheduler,
    linear_schedule,
    cyclic_schedule
)
from .monitor import TrainingMonitor
from .trainer import TernaryVAETrainer
from .appetitive_trainer import AppetitiveVAETrainer
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
    # Trainers
    'TernaryVAETrainer',
    'HyperbolicVAETrainer',
    'AppetitiveVAETrainer',
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
