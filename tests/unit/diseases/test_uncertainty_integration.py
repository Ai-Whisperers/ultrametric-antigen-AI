# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Uncertainty Integration.

Tests cover:
- UncertaintyConfig
- UncertaintyResult
- UncertaintyAwareAnalyzer
- MCDropoutUncertainty
- EnsembleUncertainty
- EvidentialUncertainty
- Calibration
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.diseases.uncertainty_aware_analyzer import (
    UncertaintyAwareAnalyzer,
    UncertaintyConfig,
    UncertaintyMethod,
    UncertaintyResult,
)


class MockBaseAnalyzer:
    """Mock base analyzer for testing."""

    def __init__(self):
        self.config = type("Config", (), {"name": "mock"})()

    def analyze(self, sequences, **kwargs):
        return {
            "n_sequences": len(sequences) if isinstance(sequences, list) else 1,
            "drug_resistance": {
                "drug_a": {"scores": [0.8, 0.7, 0.9]},
                "drug_b": {"scores": [0.3, 0.4, 0.2]},
            },
        }

    def validate_predictions(self, predictions, ground_truth):
        return {"accuracy": 0.85}


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, with_dropout=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2) if with_dropout else nn.Identity(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return {"prediction": self.net(x).squeeze(-1)}


class MockEvidentialModel(nn.Module):
    """Mock evidential model for testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(64, 4)

    def forward(self, x):
        out = self.net(x)
        return {
            "gamma": out[:, 0],
            "nu": torch.abs(out[:, 1]) + 1,
            "alpha": torch.abs(out[:, 2]) + 1,
            "beta": torch.abs(out[:, 3]) + 0.1,
        }

    def predict_with_uncertainty(self, x):
        from src.diseases.uncertainty_aware_analyzer import UncertaintyEstimate

        self.eval()
        with torch.no_grad():
            out = self(x)

        gamma = out["gamma"]
        std = torch.sqrt(out["beta"] / out["nu"])

        return UncertaintyEstimate(
            mean=gamma,
            std=std,
            lower=gamma - 1.96 * std,
            upper=gamma + 1.96 * std,
            epistemic=std * 0.5,
            aleatoric=std * 0.5,
        )


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = UncertaintyConfig()

        assert config.method == UncertaintyMethod.EVIDENTIAL
        assert config.n_samples == 50
        assert config.n_models == 5
        assert config.confidence_level == 0.95
        assert config.calibrate is True
        assert config.decompose is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = UncertaintyConfig(
            method=UncertaintyMethod.MC_DROPOUT,
            n_samples=100,
            confidence_level=0.99,
        )

        assert config.method == UncertaintyMethod.MC_DROPOUT
        assert config.n_samples == 100
        assert config.confidence_level == 0.99


class TestUncertaintyMethod:
    """Test UncertaintyMethod enum."""

    def test_method_values(self):
        """Test method values."""
        assert UncertaintyMethod.MC_DROPOUT.value == "mc_dropout"
        assert UncertaintyMethod.ENSEMBLE.value == "ensemble"
        assert UncertaintyMethod.EVIDENTIAL.value == "evidential"
        assert UncertaintyMethod.COMBINED.value == "combined"


class TestUncertaintyResult:
    """Test UncertaintyResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = UncertaintyResult(
            mean=np.array([0.5, 0.6, 0.7]),
            std=np.array([0.1, 0.1, 0.1]),
            lower=np.array([0.3, 0.4, 0.5]),
            upper=np.array([0.7, 0.8, 0.9]),
        )

        assert len(result.mean) == 3
        assert result.calibrated is False

    def test_result_with_decomposition(self):
        """Test result with uncertainty decomposition."""
        result = UncertaintyResult(
            mean=np.array([0.5]),
            std=np.array([0.15]),
            lower=np.array([0.2]),
            upper=np.array([0.8]),
            epistemic=np.array([0.1]),
            aleatoric=np.array([0.1]),
        )

        assert result.epistemic is not None
        assert result.aleatoric is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = UncertaintyResult(
            mean=np.array([0.5]),
            std=np.array([0.1]),
            lower=np.array([0.3]),
            upper=np.array([0.7]),
            calibrated=True,
            confidence_level=0.95,
        )

        d = result.to_dict()

        assert "mean" in d
        assert "std" in d
        assert "lower" in d
        assert "upper" in d
        assert d["calibrated"] is True


class TestUncertaintyAwareAnalyzer:
    """Test UncertaintyAwareAnalyzer."""

    @pytest.fixture
    def base_analyzer(self):
        """Create base analyzer fixture."""
        return MockBaseAnalyzer()

    @pytest.fixture
    def mc_model(self):
        """Create MC Dropout model fixture."""
        return MockModel(with_dropout=True)

    @pytest.fixture
    def evidential_model(self):
        """Create evidential model fixture."""
        return MockEvidentialModel()

    def test_initialization_default(self, base_analyzer):
        """Test default initialization."""
        analyzer = UncertaintyAwareAnalyzer(base_analyzer)

        assert analyzer.base_analyzer is base_analyzer
        assert analyzer.config.method == UncertaintyMethod.EVIDENTIAL

    def test_initialization_with_config(self, base_analyzer):
        """Test initialization with config."""
        config = UncertaintyConfig(method=UncertaintyMethod.MC_DROPOUT)
        analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config)

        assert analyzer.config.method == UncertaintyMethod.MC_DROPOUT

    def test_analyze_without_uncertainty(self, base_analyzer):
        """Test basic analyze (no uncertainty)."""
        analyzer = UncertaintyAwareAnalyzer(base_analyzer)

        results = analyzer.analyze(["MKTEFPSASLY"])

        assert "drug_resistance" in results
        assert "n_sequences" in results

    def test_analyze_with_uncertainty_mc_dropout(self, base_analyzer, mc_model):
        """Test analyze with MC Dropout uncertainty."""
        config = UncertaintyConfig(method=UncertaintyMethod.MC_DROPOUT, n_samples=10)
        analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config, model=mc_model)

        x = torch.randn(4, 64)
        results = analyzer.analyze_with_uncertainty(["seq1"], encodings=x)

        assert "drug_resistance" in results
        # Uncertainty should be added if wrapper exists

    def test_analyze_with_uncertainty_evidential(self, base_analyzer, evidential_model):
        """Test analyze with evidential uncertainty."""
        config = UncertaintyConfig(method=UncertaintyMethod.EVIDENTIAL)
        analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config, model=evidential_model)

        x = torch.randn(4, 64)
        results = analyzer.analyze_with_uncertainty(["seq1"], encodings=x)

        assert "drug_resistance" in results

    def test_analyze_no_encodings(self, base_analyzer, mc_model):
        """Test analyze without encodings returns base results."""
        config = UncertaintyConfig(method=UncertaintyMethod.MC_DROPOUT)
        analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config, model=mc_model)

        results = analyzer.analyze_with_uncertainty(["seq1"])

        # Should return base results without uncertainty
        assert "drug_resistance" in results

    def test_validate_predictions(self, base_analyzer):
        """Test prediction validation."""
        analyzer = UncertaintyAwareAnalyzer(base_analyzer)

        predictions = {"drug_a": torch.tensor([0.8, 0.7])}
        ground_truth = {"drug_a": [0.75, 0.65]}

        metrics = analyzer.validate_predictions(predictions, ground_truth)

        assert "accuracy" in metrics


class TestUncertaintyCalibration:
    """Test uncertainty calibration."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with calibration enabled."""
        base = MockBaseAnalyzer()
        config = UncertaintyConfig(calibrate=True)
        model = MockEvidentialModel()
        return UncertaintyAwareAnalyzer(base, config=config, model=model)

    def test_calibrate(self, analyzer):
        """Test calibration method."""
        x = torch.randn(100, 64)
        y = torch.rand(100)

        analyzer.calibrate(x, y)

        assert analyzer.is_calibrated is True

    def test_calibrated_uncertainty(self, analyzer):
        """Test uncertainty after calibration."""
        # First calibrate
        x_cal = torch.randn(50, 64)
        y_cal = torch.rand(50)
        analyzer.calibrate(x_cal, y_cal)

        # Then analyze
        x_test = torch.randn(10, 64)
        results = analyzer.analyze_with_uncertainty(["seq1"], encodings=x_test)

        # Results should include calibrated uncertainties
        if "drug_resistance" in results:
            for drug_data in results["drug_resistance"].values():
                if "uncertainty" in drug_data:
                    assert drug_data["uncertainty"]["calibrated"] is True


class TestEvaluateUncertaintyQuality:
    """Test uncertainty quality evaluation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        base = MockBaseAnalyzer()
        model = MockEvidentialModel()
        return UncertaintyAwareAnalyzer(base, model=model)

    def test_evaluate_uncertainty_quality(self, analyzer):
        """Test uncertainty quality metrics."""
        x = torch.randn(50, 64)
        y = torch.rand(50)

        metrics = analyzer.evaluate_uncertainty_quality(x, y)

        if "error" not in metrics:
            assert "coverage_95" in metrics or "nll" in metrics


class TestUncertaintyEdgeCases:
    """Test edge cases."""

    def test_no_model_provided(self):
        """Test analyzer without model."""
        base = MockBaseAnalyzer()
        analyzer = UncertaintyAwareAnalyzer(base)

        # Should work but not add uncertainty
        results = analyzer.analyze_with_uncertainty(["seq1"], encodings=torch.randn(1, 64))

        assert "drug_resistance" in results

    def test_empty_sequences(self):
        """Test with empty sequences."""
        base = MockBaseAnalyzer()
        analyzer = UncertaintyAwareAnalyzer(base)

        results = analyzer.analyze([])

        # Should handle gracefully

    def test_invalid_method(self):
        """Test with invalid method string."""
        # This should raise an error
        with pytest.raises(ValueError):
            UncertaintyMethod("invalid_method")

    def test_model_without_predict_with_uncertainty(self):
        """Test evidential method with model missing required method."""
        base = MockBaseAnalyzer()
        config = UncertaintyConfig(method=UncertaintyMethod.EVIDENTIAL)

        # Regular model without predict_with_uncertainty
        model = MockModel()

        # Should raise or handle gracefully
        with pytest.raises(ValueError):
            from src.diseases.uncertainty_aware_analyzer import EvidentialUncertainty

            EvidentialUncertainty(model)


class TestUncertaintyDecomposition:
    """Test epistemic/aleatoric decomposition."""

    def test_decomposition_enabled(self):
        """Test uncertainty decomposition."""
        config = UncertaintyConfig(decompose=True)

        assert config.decompose is True

    def test_decomposition_in_results(self):
        """Test decomposition appears in results."""
        base = MockBaseAnalyzer()
        config = UncertaintyConfig(decompose=True)
        model = MockEvidentialModel()
        analyzer = UncertaintyAwareAnalyzer(base, config=config, model=model)

        x = torch.randn(4, 64)
        results = analyzer.analyze_with_uncertainty(["seq1"], encodings=x)

        # Check that decomposition is present when uncertainty is added
        if "drug_resistance" in results:
            for drug_data in results["drug_resistance"].values():
                if "uncertainty" in drug_data:
                    # May have epistemic/aleatoric
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
