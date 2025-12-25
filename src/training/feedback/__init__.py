# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training feedback controllers for adaptive training.

This module provides feedback systems that modulate training based on
metrics like coverage, correlation, and learning dynamics.

Components:
    - ContinuousFeedbackController: Coverage-based ranking weight adaptation
    - CorrelationEarlyStop: Correlation-based early stopping
    - ExplorationBoostController: Coverage stall detection and exploration boost
"""

from .continuous_feedback import ContinuousFeedbackController
from .correlation_feedback import CorrelationEarlyStop
from .exploration_boost import ExplorationBoostController

__all__ = [
    "ContinuousFeedbackController",
    "CorrelationEarlyStop",
    "ExplorationBoostController",
]
