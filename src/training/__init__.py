"""Training orchestration components.

This module contains components for managing the training process:
- Trainer: Main training loop (single responsibility)
- AppetitiveVAETrainer: Trainer with bio-inspired appetite losses
- Schedulers: Parameter scheduling (temperature, beta, learning rate)
- Monitor: Logging and metrics tracking
- Validators: Validation logic
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

__all__ = [
    'TernaryVAETrainer',
    'AppetitiveVAETrainer',
    'TemperatureScheduler',
    'BetaScheduler',
    'LearningRateScheduler',
    'linear_schedule',
    'cyclic_schedule',
    'TrainingMonitor'
]
