# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training callbacks module.

This module provides a callback system for customizing training behavior
without modifying the core trainer code.

Categories:
    - Base: TrainingCallback protocol and CallbackList
    - Early Stopping: Various stopping strategies
    - Logging: Structured logging and TensorBoard
    - Checkpointing: Model saving and rotation

Usage:
    from src.training.callbacks import (
        CallbackList,
        EarlyStoppingCallback,
        LoggingCallback,
        CheckpointCallback,
    )

    callbacks = CallbackList([
        LoggingCallback(log_interval=10),
        EarlyStoppingCallback(patience=50),
        CheckpointCallback(checkpoint_dir="checkpoints"),
    ])

    trainer = Trainer(callbacks=callbacks)
"""

from .base import CallbackList, TrainingCallback
from .checkpointing import BestModelCallback, CheckpointCallback
from .early_stopping import (
    CorrelationDropCallback,
    CoveragePlateauCallback,
    EarlyStoppingCallback,
)
from .logging import LoggingCallback, ProgressCallback, TensorBoardCallback

__all__ = [
    # Base
    "TrainingCallback",
    "CallbackList",
    # Early Stopping
    "EarlyStoppingCallback",
    "CoveragePlateauCallback",
    "CorrelationDropCallback",
    # Logging
    "LoggingCallback",
    "TensorBoardCallback",
    "ProgressCallback",
    # Checkpointing
    "CheckpointCallback",
    "BestModelCallback",
]
