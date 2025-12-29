# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for conformal prediction methods.

Tests cover:
- Split Conformal Classifier
- Adaptive Conformal Classifier (APS)
- RAPS Classifier
- Conformal Regressor
- Conformalized Quantile Regression
- Coverage evaluation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.uncertainty.conformal import (
    AdaptiveConformalClassifier,
    ConformalPredictionWrapper,
    ConformalRegressor,
    ConformizedQuantileRegressor,
    PredictionSet,
    RAPSConformalClassifier,
    RegressionInterval,
    SplitConformalClassifier,
    evaluate_conformal_coverage,
)


class TestSplitConformalClassifier:
    """Tests for Split Conformal Prediction."""

    def test_initialization(self):
        """Test split conformal initialization."""
        scc = SplitConformalClassifier(alpha=0.1)

        assert scc.alpha == 0.1
        assert not scc.is_calibrated

    def test_calibrate(self):
        """Test calibration on validation data."""
        scc = SplitConformalClassifier(alpha=0.1)

        # Create probabilities using softmax manually
        logits = np.random.randn(100, 5) + np.eye(5)[np.random.randint(0, 5, 100)] * 2
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        scc.calibrate(probs, labels)

        assert scc.is_calibrated
        assert scc._threshold is not None

    def test_predict(self):
        """Test prediction set generation."""
        scc = SplitConformalClassifier(alpha=0.1)

        # Create calibration data
        np.random.seed(42)
        cal_probs = np.random.dirichlet(np.ones(5), size=200)
        cal_labels = np.random.randint(0, 5, 200)

        scc.calibrate(cal_probs, cal_labels)

        # Test prediction
        test_probs = np.random.dirichlet(np.ones(5), size=50)
        pred_sets = scc.predict(test_probs)

        assert isinstance(pred_sets, PredictionSet)
        assert len(pred_sets.sets) == 50
        assert len(pred_sets.set_sizes) == 50

    def test_coverage_guarantee(self):
        """Test that coverage is approximately 1-alpha."""
        scc = SplitConformalClassifier(alpha=0.1)

        np.random.seed(42)
        # Create data where true class has high probability
        n_cal = 500
        n_test = 500

        cal_probs = np.random.dirichlet(np.ones(5) * 0.5, size=n_cal)
        cal_labels = cal_probs.argmax(axis=1)  # True label is argmax

        scc.calibrate(cal_probs, cal_labels)

        test_probs = np.random.dirichlet(np.ones(5) * 0.5, size=n_test)
        test_labels = test_probs.argmax(axis=1)

        pred_sets = scc.predict(test_probs)

        # Check coverage
        covered = sum(
            test_labels[i] in pred_sets.sets[i]
            for i in range(n_test)
        )
        coverage = covered / n_test

        # Coverage should be approximately 1 - alpha = 0.9
        assert coverage >= 0.85  # Allow some slack

    def test_uncalibrated_raises(self):
        """Test that prediction without calibration raises error."""
        scc = SplitConformalClassifier(alpha=0.1)

        probs = np.random.dirichlet(np.ones(5), size=10)

        with pytest.raises(RuntimeError, match="calibrate"):
            scc.predict(probs)


class TestAdaptiveConformalClassifier:
    """Tests for Adaptive Prediction Sets (APS)."""

    def test_initialization(self):
        """Test APS initialization."""
        aps = AdaptiveConformalClassifier(alpha=0.1)

        assert aps.alpha == 0.1
        assert aps.randomize is True

    def test_calibrate(self):
        """Test APS calibration."""
        aps = AdaptiveConformalClassifier(alpha=0.1)

        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(5), size=200)
        labels = np.random.randint(0, 5, 200)

        aps.calibrate(probs, labels)

        assert aps.is_calibrated

    def test_predict(self):
        """Test APS prediction."""
        aps = AdaptiveConformalClassifier(alpha=0.1)

        np.random.seed(42)
        cal_probs = np.random.dirichlet(np.ones(5), size=200)
        cal_labels = np.random.randint(0, 5, 200)

        aps.calibrate(cal_probs, cal_labels)

        test_probs = np.random.dirichlet(np.ones(5), size=50)
        pred_sets = aps.predict(test_probs)

        assert isinstance(pred_sets, PredictionSet)
        assert len(pred_sets.sets) == 50

    def test_adaptive_set_sizes(self):
        """Test that APS produces adaptive set sizes."""
        aps = AdaptiveConformalClassifier(alpha=0.1)

        np.random.seed(42)
        cal_probs = np.random.dirichlet(np.ones(5), size=200)
        cal_labels = np.random.randint(0, 5, 200)

        aps.calibrate(cal_probs, cal_labels)

        # Create test data with varying confidence
        high_conf = np.array([[0.9, 0.025, 0.025, 0.025, 0.025]])
        low_conf = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])

        sets_high = aps.predict(high_conf)
        sets_low = aps.predict(low_conf)

        # High confidence should have smaller sets
        assert sets_high.set_sizes[0] <= sets_low.set_sizes[0]


class TestRAPSConformalClassifier:
    """Tests for Regularized Adaptive Prediction Sets."""

    def test_initialization(self):
        """Test RAPS initialization."""
        raps = RAPSConformalClassifier(alpha=0.1, k_reg=2, lambda_reg=0.01)

        assert raps.alpha == 0.1
        assert raps.k_reg == 2
        assert raps.lambda_reg == 0.01

    def test_calibrate(self):
        """Test RAPS calibration."""
        raps = RAPSConformalClassifier(alpha=0.1)

        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(5), size=200)
        labels = np.random.randint(0, 5, 200)

        raps.calibrate(probs, labels)

        assert raps.is_calibrated

    def test_regularization_reduces_set_size(self):
        """Test that regularization tends to reduce set sizes."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(10), size=500)
        labels = np.random.randint(0, 10, 500)

        # Without regularization
        raps_no_reg = RAPSConformalClassifier(alpha=0.1, lambda_reg=0.0)
        raps_no_reg.calibrate(probs[:250], labels[:250])
        sets_no_reg = raps_no_reg.predict(probs[250:])

        # With regularization
        raps_reg = RAPSConformalClassifier(alpha=0.1, lambda_reg=0.1)
        raps_reg.calibrate(probs[:250], labels[:250])
        sets_reg = raps_reg.predict(probs[250:])

        # Regularized should have smaller average set size
        assert sets_reg.average_set_size() <= sets_no_reg.average_set_size() + 1


class TestConformalRegressor:
    """Tests for Conformal Regression."""

    def test_initialization(self):
        """Test conformal regressor initialization."""
        cr = ConformalRegressor(alpha=0.1)

        assert cr.alpha == 0.1
        assert not cr.is_calibrated

    def test_calibrate(self):
        """Test calibration on regression data."""
        cr = ConformalRegressor(alpha=0.1)

        np.random.seed(42)
        preds = np.random.randn(100)
        labels = preds + np.random.randn(100) * 0.5  # Add noise

        cr.calibrate(preds, labels)

        assert cr.is_calibrated
        assert cr._threshold is not None

    def test_predict(self):
        """Test prediction interval generation."""
        cr = ConformalRegressor(alpha=0.1)

        np.random.seed(42)
        cal_preds = np.random.randn(200)
        cal_labels = cal_preds + np.random.randn(200) * 0.5

        cr.calibrate(cal_preds, cal_labels)

        test_preds = np.random.randn(50)
        intervals = cr.predict(test_preds)

        assert isinstance(intervals, RegressionInterval)
        assert len(intervals.lower) == 50
        assert len(intervals.upper) == 50
        assert all(intervals.upper >= intervals.lower)

    def test_coverage(self):
        """Test that coverage matches target."""
        cr = ConformalRegressor(alpha=0.1)

        np.random.seed(42)
        n_cal = 500
        n_test = 500

        cal_preds = np.random.randn(n_cal)
        cal_labels = cal_preds + np.random.randn(n_cal) * 0.5

        cr.calibrate(cal_preds, cal_labels)

        test_preds = np.random.randn(n_test)
        test_labels = test_preds + np.random.randn(n_test) * 0.5

        intervals = cr.predict(test_preds, test_labels)

        assert intervals.coverage >= 0.85


class TestConformizedQuantileRegressor:
    """Tests for Conformalized Quantile Regression."""

    def test_initialization(self):
        """Test CQR initialization."""
        cqr = ConformizedQuantileRegressor(alpha=0.1)

        assert cqr.alpha == 0.1

    def test_calibrate(self):
        """Test CQR calibration."""
        cqr = ConformizedQuantileRegressor(alpha=0.1)

        np.random.seed(42)
        cal_lower = np.random.randn(100) - 0.5
        cal_upper = cal_lower + 1.0
        cal_labels = (cal_lower + cal_upper) / 2 + np.random.randn(100) * 0.3

        cqr.calibrate(cal_lower, cal_upper, cal_labels)

        assert cqr.is_calibrated

    def test_predict(self):
        """Test CQR prediction."""
        cqr = ConformizedQuantileRegressor(alpha=0.1)

        np.random.seed(42)
        cal_lower = np.random.randn(200) - 0.5
        cal_upper = cal_lower + 1.0
        cal_labels = (cal_lower + cal_upper) / 2 + np.random.randn(200) * 0.3

        cqr.calibrate(cal_lower, cal_upper, cal_labels)

        test_lower = np.random.randn(50) - 0.5
        test_upper = test_lower + 1.0

        intervals = cqr.predict(test_lower, test_upper)

        assert isinstance(intervals, RegressionInterval)
        # Intervals should be properly ordered (lower < upper)
        assert all(intervals.lower < intervals.upper)
        # Width should be consistent (CQR adjusts by constant threshold)
        widths = intervals.upper - intervals.lower
        assert np.allclose(widths, widths[0], atol=1e-5)


class TestConformalPredictionWrapper:
    """Tests for ConformalPredictionWrapper."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def test_initialization(self):
        """Test wrapper initialization."""
        model = nn.Linear(10, 5)
        wrapper = ConformalPredictionWrapper(model, alpha=0.1, method="aps")

        assert wrapper.alpha == 0.1
        assert isinstance(wrapper.conformal, AdaptiveConformalClassifier)

    def test_different_methods(self):
        """Test different conformal methods."""
        model = nn.Linear(10, 5)

        wrapper_split = ConformalPredictionWrapper(model, method="split")
        wrapper_aps = ConformalPredictionWrapper(model, method="aps")
        wrapper_raps = ConformalPredictionWrapper(model, method="raps")

        assert isinstance(wrapper_split.conformal, SplitConformalClassifier)
        assert isinstance(wrapper_aps.conformal, AdaptiveConformalClassifier)
        assert isinstance(wrapper_raps.conformal, RAPSConformalClassifier)

    def test_predict_sets(self, simple_model):
        """Test prediction set generation."""
        wrapper = ConformalPredictionWrapper(simple_model, alpha=0.1)

        # Create calibration data
        torch.manual_seed(42)
        cal_x = torch.randn(100, 10)
        cal_y = torch.randint(0, 5, (100,))

        # Create simple data loader
        from torch.utils.data import DataLoader, TensorDataset
        cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=32)

        wrapper.calibrate(cal_loader, device="cpu")

        # Test prediction
        test_x = torch.randn(20, 10)
        pred_sets = wrapper.predict_sets(test_x)

        assert isinstance(pred_sets, PredictionSet)
        assert len(pred_sets.sets) == 20

    def test_predict_with_confidence(self, simple_model):
        """Test prediction with confidence output."""
        wrapper = ConformalPredictionWrapper(simple_model, alpha=0.1)

        torch.manual_seed(42)
        cal_x = torch.randn(100, 10)
        cal_y = torch.randint(0, 5, (100,))

        from torch.utils.data import DataLoader, TensorDataset
        cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=32)

        wrapper.calibrate(cal_loader, device="cpu")

        test_x = torch.randn(20, 10)
        result = wrapper.predict_with_confidence(test_x)

        assert "predictions" in result
        assert "prediction_sets" in result
        assert "set_sizes" in result
        assert "coverage_guarantee" in result


class TestEvaluateConformalCoverage:
    """Tests for coverage evaluation function."""

    def test_evaluate_coverage(self):
        """Test coverage evaluation."""
        np.random.seed(42)

        # Create and calibrate predictor
        predictor = SplitConformalClassifier(alpha=0.1)
        probs = np.random.dirichlet(np.ones(5), size=300)
        labels = np.random.randint(0, 5, 300)

        predictor.calibrate(probs[:200], labels[:200])

        # Evaluate
        metrics = evaluate_conformal_coverage(
            predictor,
            probs[200:],
            labels[200:],
        )

        assert "empirical_coverage" in metrics
        assert "target_coverage" in metrics
        assert "average_set_size" in metrics
        assert metrics["target_coverage"] == 0.9


class TestPredictionSet:
    """Tests for PredictionSet dataclass."""

    def test_average_set_size(self):
        """Test average set size computation."""
        pred_set = PredictionSet(
            sets=[[0], [0, 1], [0, 1, 2]],
            set_sizes=np.array([1, 2, 3]),
            coverage=0.9,
            alpha=0.1,
            threshold=0.5,
        )

        assert pred_set.average_set_size() == pytest.approx(2.0)

    def test_coverage_rate(self):
        """Test coverage rate method."""
        pred_set = PredictionSet(
            sets=[[0]],
            set_sizes=np.array([1]),
            coverage=0.9,
            alpha=0.1,
            threshold=0.5,
        )

        assert pred_set.coverage_rate() == 0.9


class TestRegressionInterval:
    """Tests for RegressionInterval dataclass."""

    def test_average_width(self):
        """Test average width computation."""
        interval = RegressionInterval(
            lower=np.array([0.0, 1.0, 2.0]),
            upper=np.array([1.0, 3.0, 4.0]),
            coverage=0.9,
            alpha=0.1,
            width=np.array([1.0, 2.0, 2.0]),
        )

        assert interval.average_width() == pytest.approx(5.0 / 3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
