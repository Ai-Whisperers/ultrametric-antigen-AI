# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Observability layer - Decoupled from training.

This module provides observability components that are decoupled
from the training loop:

- MetricsBuffer: In-memory buffer (zero I/O during training)
- AsyncTensorBoardWriter: Async I/O in background thread
- CoverageEvaluator: Vectorized coverage evaluation

Architecture:
    Training Loop                    Observability Layer
    ─────────────                    ───────────────────
    train_epoch()
         │
         ├──> buffer.record()  ──────> MetricsBuffer (in-memory)
         │                                   │
         └──> evaluator.evaluate()           │ (drain periodically)
                   │                         v
                   │              AsyncTensorBoardWriter
                   │                         │
                   v                         v (background thread)
              CoverageStats            TensorBoard files

Benefits:
    - Training not blocked by I/O
    - Single flush per epoch (not 3-5)
    - Coverage evaluation uses vectorized ops
    - Easy to disable observability for benchmarks
"""

from .async_writer import AsyncTensorBoardWriter, NullWriter, create_writer
from .coverage import CoverageEvaluator, CoverageStats, evaluate_model_coverage
from .logging import (
    ColoredFormatter,
    LogContext,
    StructuredFormatter,
    get_logger,
    setup_logging,
)
from .metrics_buffer import MetricRecord, MetricsBuffer, ScopedMetrics
from .training_history import TrainingHistory, TrainingState

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "StructuredFormatter",
    "ColoredFormatter",
    # Metrics buffer
    "MetricsBuffer",
    "MetricRecord",
    "ScopedMetrics",
    # Async writer
    "AsyncTensorBoardWriter",
    "NullWriter",
    "create_writer",
    # Coverage
    "CoverageEvaluator",
    "CoverageStats",
    "evaluate_model_coverage",
    # Training history
    "TrainingHistory",
    "TrainingState",
]
