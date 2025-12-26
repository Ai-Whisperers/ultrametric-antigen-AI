"""Tests for training monitoring module.

Tests cover:
- TrainingMonitor initialization
- MetricsTracker tracking and best value updates
- FileLogger logging capabilities
"""

from __future__ import annotations

import pytest

from src.training.monitoring import MetricsTracker


class TestMetricsTracker:
    """Tests for MetricsTracker component."""

    def test_initialization(self):
        """Tracker should initialize with default values."""
        tracker = MetricsTracker()
        assert tracker.best_val_loss == float("inf")
        assert tracker.patience_counter == 0
        assert tracker.global_step == 0

    def test_empty_histories(self):
        """Tracker should start with empty histories."""
        tracker = MetricsTracker()
        assert len(tracker.coverage_A_history) == 0
        assert len(tracker.coverage_B_history) == 0
        assert len(tracker.H_A_history) == 0
        assert len(tracker.H_B_history) == 0

    def test_update_histories(self):
        """update_histories should append values."""
        tracker = MetricsTracker()

        tracker.update_histories(
            H_A=1.5,
            H_B=1.6,
            coverage_A=100,
            coverage_B=200,
        )

        assert len(tracker.H_A_history) == 1
        assert len(tracker.H_B_history) == 1
        assert len(tracker.coverage_A_history) == 1
        assert len(tracker.coverage_B_history) == 1

        assert tracker.H_A_history[0] == 1.5
        assert tracker.H_B_history[0] == 1.6
        assert tracker.coverage_A_history[0] == 100
        assert tracker.coverage_B_history[0] == 200

    def test_multiple_history_updates(self):
        """Multiple updates should accumulate in histories."""
        tracker = MetricsTracker()

        tracker.update_histories(H_A=1.0, H_B=1.0, coverage_A=100, coverage_B=100)
        tracker.update_histories(H_A=1.5, H_B=1.5, coverage_A=150, coverage_B=150)
        tracker.update_histories(H_A=2.0, H_B=2.0, coverage_A=200, coverage_B=200)

        assert len(tracker.H_A_history) == 3
        assert tracker.H_A_history[-1] == 2.0
        assert tracker.coverage_A_history[-1] == 200

    def test_best_val_loss_update(self):
        """best_val_loss should be updatable."""
        tracker = MetricsTracker()

        # Initial value
        assert tracker.best_val_loss == float("inf")

        # Update
        tracker.best_val_loss = 0.5
        assert tracker.best_val_loss == 0.5

        # Should not auto-update - manual tracking
        tracker.best_val_loss = 0.3
        assert tracker.best_val_loss == 0.3

    def test_update_correlation(self):
        """update_correlation should update history."""
        tracker = MetricsTracker()

        tracker.update_correlation(
            corr_A_hyp=0.9,
            corr_B_hyp=0.85,
            corr_A_euc=0.8,
            corr_B_euc=0.75,
        )

        assert len(tracker.correlation_hyp_history) == 1
        assert len(tracker.correlation_euc_history) == 1

    def test_global_step_tracking(self):
        """global_step should be trackable."""
        tracker = MetricsTracker()

        assert tracker.global_step == 0
        tracker.global_step = 100
        assert tracker.global_step == 100

    def test_patience_counter(self):
        """patience_counter should be modifiable."""
        tracker = MetricsTracker()

        assert tracker.patience_counter == 0
        tracker.patience_counter = 5
        assert tracker.patience_counter == 5


class TestTrainingMonitor:
    """Tests for TrainingMonitor facade."""

    def test_initialization_with_defaults(self):
        """Monitor should initialize with default parameters."""
        from src.training.monitor import TrainingMonitor

        # Disable file logging for test
        monitor = TrainingMonitor(log_to_file=False)

        assert monitor.eval_num_samples == 100000
        assert monitor.experiment_name is not None
        assert monitor.metrics is not None

    def test_custom_experiment_name(self):
        """Monitor should accept custom experiment name."""
        from src.training.monitor import TrainingMonitor

        monitor = TrainingMonitor(
            experiment_name="test_experiment", log_to_file=False
        )

        assert monitor.experiment_name == "test_experiment"

    def test_best_val_loss_property(self):
        """best_val_loss should delegate to metrics tracker."""
        from src.training.monitor import TrainingMonitor

        monitor = TrainingMonitor(log_to_file=False)
        monitor.best_val_loss = 0.123

        assert monitor.best_val_loss == 0.123

    def test_metrics_tracker_exists(self):
        """Monitor should have metrics tracker."""
        from src.training.monitor import TrainingMonitor

        monitor = TrainingMonitor(log_to_file=False)
        assert hasattr(monitor, "metrics")
        assert isinstance(monitor.metrics, MetricsTracker)


class TestFileLogger:
    """Tests for FileLogger component."""

    def test_initialization_without_file(self):
        """Logger should initialize without file logging."""
        from src.training.monitoring import FileLogger

        logger = FileLogger(log_to_file=False)
        assert logger.logger is None

    def test_log_message(self):
        """Logger should accept log messages."""
        from src.training.monitoring import FileLogger

        logger = FileLogger(log_to_file=False)
        # Should not raise even without file logging
        logger.log("Test message")

    def test_experiment_name(self):
        """Logger should store experiment name."""
        from src.training.monitoring import FileLogger

        logger = FileLogger(log_to_file=False, experiment_name="my_experiment")
        assert logger.experiment_name == "my_experiment"


class TestCoverageEvaluator:
    """Tests for CoverageEvaluator component."""

    def test_initialization(self):
        """Evaluator should initialize with num_samples."""
        from src.training.monitoring import CoverageEvaluator

        evaluator = CoverageEvaluator(num_samples=1000)
        assert evaluator.num_samples == 1000

    def test_default_num_samples(self):
        """Evaluator should have reasonable default."""
        from src.training.monitoring import CoverageEvaluator

        evaluator = CoverageEvaluator()
        assert evaluator.num_samples > 0


class TestTensorBoardLogger:
    """Tests for TensorBoardLogger component."""

    def test_initialization_without_dir(self):
        """Logger should handle no tensorboard_dir gracefully."""
        from src.training.monitoring import TensorBoardLogger

        logger = TensorBoardLogger(tensorboard_dir=None, experiment_name="test")
        assert logger.writer is None

    def test_initialization_with_disabled(self):
        """Logger should handle disabled state."""
        from src.training.monitoring import TensorBoardLogger

        logger = TensorBoardLogger(tensorboard_dir=None, experiment_name="test")
        # Should not raise
        logger.close()
