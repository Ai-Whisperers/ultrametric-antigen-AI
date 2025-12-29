# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for calibration methods.

Tests cover:
- Temperature Scaling
- Vector Scaling
- Platt Scaling
- Isotonic Calibration
- Label Smoothing Loss
- Focal Loss
- Calibration Metrics
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.uncertainty.calibration import (
    CalibrationMetrics,
    CalibratedModel,
    FocalLossCalibration,
    IsotonicCalibration,
    LabelSmoothingLoss,
    PlattScaling,
    TemperatureScaling,
    VectorScaling,
    compute_calibration_metrics,
)


class TestTemperatureScaling:
    """Tests for Temperature Scaling calibration."""

    def test_initialization(self):
        """Test temperature scaling initialization."""
        ts = TemperatureScaling()
        assert ts.temperature.item() == pytest.approx(1.0)
        assert not ts.is_fitted

    def test_initialization_custom_temp(self):
        """Test custom initial temperature."""
        ts = TemperatureScaling(init_temperature=2.0)
        assert ts.temperature.item() == pytest.approx(2.0)

    def test_fit(self):
        """Test fitting temperature scaling."""
        ts = TemperatureScaling()

        # Create overconfident predictions
        logits = torch.randn(100, 5) * 3  # High magnitude = overconfident
        labels = torch.randint(0, 5, (100,))

        ts.fit(logits, labels)

        assert ts.is_fitted
        # Temperature should increase to reduce confidence
        assert ts.temperature.item() > 1.0

    def test_calibrate(self):
        """Test applying calibration."""
        ts = TemperatureScaling()

        logits = torch.randn(100, 5) * 3
        labels = torch.randint(0, 5, (100,))

        ts.fit(logits, labels)
        probs = ts.calibrate(logits)

        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-5)

    def test_get_temperature(self):
        """Test getting temperature value."""
        ts = TemperatureScaling(init_temperature=1.5)
        assert ts.get_temperature() == pytest.approx(1.5)

    def test_forward_pass(self):
        """Test forward pass (same as calibrate)."""
        ts = TemperatureScaling()

        logits = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))
        ts.fit(logits, labels)

        probs_forward = ts(logits)
        probs_calibrate = ts.calibrate(logits)

        assert torch.allclose(probs_forward, probs_calibrate)


class TestVectorScaling:
    """Tests for Vector Scaling calibration."""

    def test_initialization(self):
        """Test vector scaling initialization."""
        vs = VectorScaling(num_classes=5)

        assert vs.W.shape == (5,)
        assert vs.b.shape == (5,)
        assert torch.allclose(vs.W, torch.ones(5))
        assert torch.allclose(vs.b, torch.zeros(5))

    def test_fit(self):
        """Test fitting vector scaling."""
        vs = VectorScaling(num_classes=3)

        logits = torch.randn(50, 3) * 2
        labels = torch.randint(0, 3, (50,))

        vs.fit(logits, labels, max_epochs=50)

        assert vs.is_fitted

    def test_calibrate(self):
        """Test applying vector scaling."""
        vs = VectorScaling(num_classes=3)

        logits = torch.randn(50, 3)
        labels = torch.randint(0, 3, (50,))

        vs.fit(logits, labels, max_epochs=50)
        probs = vs.calibrate(logits)

        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=1), torch.ones(50), atol=1e-5)


class TestPlattScaling:
    """Tests for Platt Scaling calibration."""

    def test_initialization(self):
        """Test Platt scaling initialization."""
        ps = PlattScaling()

        assert ps.a.item() == pytest.approx(1.0)
        assert ps.b.item() == pytest.approx(0.0)

    def test_fit_binary(self):
        """Test fitting on binary classification."""
        ps = PlattScaling()

        # Binary classification scores
        logits = torch.randn(100)
        labels = (logits > 0).long()  # Simple threshold labels

        ps.fit(logits, labels)

        assert ps.is_fitted

    def test_calibrate_binary(self):
        """Test calibrating binary classification."""
        ps = PlattScaling()

        logits = torch.randn(100)
        labels = (logits > 0).long()

        ps.fit(logits, labels)
        probs = ps.calibrate(logits)

        assert probs.shape == (100, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-5)


class TestLabelSmoothingLoss:
    """Tests for Label Smoothing Loss."""

    def test_initialization(self):
        """Test label smoothing initialization."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        assert loss_fn.smoothing == 0.1

    def test_forward(self):
        """Test forward pass."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        loss = loss_fn(logits, labels)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_no_smoothing_equals_ce(self):
        """Test that zero smoothing equals cross entropy."""
        loss_smooth = LabelSmoothingLoss(smoothing=0.0)

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        smooth_loss = loss_smooth(logits, labels)
        ce_loss = F.cross_entropy(logits, labels)

        assert torch.isclose(smooth_loss, ce_loss, atol=1e-5)

    def test_smoothing_reduces_confidence(self):
        """Test that smoothing reduces model confidence."""
        loss_no_smooth = LabelSmoothingLoss(smoothing=0.0)
        loss_smooth = LabelSmoothingLoss(smoothing=0.2)

        # Perfect predictions (high confidence)
        logits = torch.zeros(10, 5)
        labels = torch.zeros(10, dtype=torch.long)
        logits[:, 0] = 10.0  # Very confident in class 0

        # Smoothed loss should be higher (penalizes overconfidence)
        loss_no = loss_no_smooth(logits, labels)
        loss_yes = loss_smooth(logits, labels)

        assert loss_yes > loss_no


class TestFocalLoss:
    """Tests for Focal Loss calibration."""

    def test_initialization(self):
        """Test focal loss initialization."""
        fl = FocalLossCalibration(gamma=2.0)
        assert fl.gamma == 2.0

    def test_forward(self):
        """Test forward pass."""
        fl = FocalLossCalibration(gamma=2.0)

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        loss = fl(logits, labels)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gamma_zero_equals_ce(self):
        """Test that gamma=0 gives cross entropy."""
        fl = FocalLossCalibration(gamma=0.0)

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        focal_loss = fl(logits, labels)
        ce_loss = F.cross_entropy(logits, labels)

        assert torch.isclose(focal_loss, ce_loss, atol=1e-5)

    def test_higher_gamma_lower_easy_loss(self):
        """Test that higher gamma reduces easy example contribution."""
        fl_low = FocalLossCalibration(gamma=0.5)
        fl_high = FocalLossCalibration(gamma=3.0)

        # Easy example (high confidence, correct)
        logits = torch.zeros(1, 3)
        logits[0, 0] = 10.0  # Very confident
        labels = torch.tensor([0])

        loss_low = fl_low(logits, labels)
        loss_high = fl_high(logits, labels)

        # Higher gamma should give lower loss for easy examples
        assert loss_high < loss_low


class TestCalibrationMetrics:
    """Tests for calibration metrics computation."""

    def test_compute_metrics(self):
        """Test computing calibration metrics."""
        # Well-calibrated predictions
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))

        metrics = compute_calibration_metrics(probs, labels)

        assert isinstance(metrics, CalibrationMetrics)
        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.mce <= 1
        assert metrics.brier >= 0
        assert metrics.nll >= 0

    def test_reliability_diagram(self):
        """Test reliability diagram output."""
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        labels = torch.randint(0, 5, (100,))

        metrics = compute_calibration_metrics(probs, labels, n_bins=10)

        assert "accuracies" in metrics.reliability_diagram
        assert "confidences" in metrics.reliability_diagram
        assert "counts" in metrics.reliability_diagram
        assert len(metrics.reliability_diagram["accuracies"]) == 10

    def test_perfect_calibration_low_ece(self):
        """Test that perfect calibration gives low ECE."""
        # Create perfectly calibrated predictions
        n_samples = 1000
        probs = torch.zeros(n_samples, 2)

        # 50% confidence predictions that are right 50% of the time
        probs[:, 0] = 0.5
        probs[:, 1] = 0.5
        labels = torch.randint(0, 2, (n_samples,))

        metrics = compute_calibration_metrics(probs, labels)

        # ECE should be relatively low for well-calibrated
        assert metrics.ece < 0.2


class TestCalibratedModel:
    """Tests for CalibratedModel wrapper."""

    def test_initialization(self):
        """Test calibrated model initialization."""
        model = nn.Linear(10, 5)
        calibrator = TemperatureScaling()

        cm = CalibratedModel(model, calibrator)

        assert cm.model is model
        assert cm.calibrator is calibrator

    def test_forward(self):
        """Test forward pass returns calibrated probabilities."""
        model = nn.Linear(10, 5)
        calibrator = TemperatureScaling()

        # Fit calibrator
        x = torch.randn(50, 10)
        logits = model(x)
        labels = torch.randint(0, 5, (50,))
        calibrator.fit(logits.detach(), labels)

        cm = CalibratedModel(model, calibrator)
        probs = cm(x)

        assert probs.shape == (50, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(50), atol=1e-5)

    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty output."""
        model = nn.Linear(10, 5)
        calibrator = TemperatureScaling()

        x = torch.randn(50, 10)
        logits = model(x)
        labels = torch.randint(0, 5, (50,))
        calibrator.fit(logits.detach(), labels)

        cm = CalibratedModel(model, calibrator)
        result = cm.predict_with_uncertainty(x)

        assert "predictions" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert result["predictions"].shape == (50,)
        assert result["confidence"].shape == (50,)


class TestIsotonicCalibration:
    """Tests for Isotonic Regression calibration."""

    def test_initialization(self):
        """Test isotonic calibration initialization."""
        ic = IsotonicCalibration()
        assert not ic.is_fitted

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available",
    )
    def test_fit_and_calibrate(self):
        """Test fitting and calibrating with isotonic regression."""
        ic = IsotonicCalibration()

        logits = torch.randn(100, 3) * 2
        labels = torch.randint(0, 3, (100,))

        ic.fit(logits, labels)

        assert ic.is_fitted

        probs = ic.calibrate(logits)
        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
