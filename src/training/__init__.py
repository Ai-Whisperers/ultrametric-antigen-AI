"""Training orchestration components.

This module contains components for managing the training process:
- TernaryVAETrainer: Main training loop (single responsibility)
- HyperbolicVAETrainer: Hyperbolic geometry trainer (canonical)
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
