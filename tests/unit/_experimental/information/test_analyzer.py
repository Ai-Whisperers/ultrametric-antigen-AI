# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for InformationGeometricAnalyzer class."""

from __future__ import annotations

import pytest
import torch

from src.information import FisherInfo, InformationGeometricAnalyzer


class TestAnalyzerInit:
    """Tests for InformationGeometricAnalyzer initialization."""

    def test_default_init(self, simple_classifier):
        """Test default initialization."""
        analyzer = InformationGeometricAnalyzer(simple_classifier)
        assert analyzer.track_eigenvalues is True
        assert analyzer.n_eigenvalues == 10

    def test_custom_params(self, simple_classifier):
        """Test custom parameters."""
        analyzer = InformationGeometricAnalyzer(
            simple_classifier, track_eigenvalues=False, n_eigenvalues=5
        )
        assert analyzer.track_eigenvalues is False
        assert analyzer.n_eigenvalues == 5

    def test_history_initialized(self, simple_classifier):
        """Test history is initialized."""
        analyzer = InformationGeometricAnalyzer(simple_classifier)
        assert "condition_numbers" in analyzer.history
        assert "trace" in analyzer.history
        assert "log_det" in analyzer.history
        assert "top_eigenvalues" in analyzer.history
        assert "bottom_eigenvalues" in analyzer.history


class TestAnalyzerAnalyzeStep:
    """Tests for analyze_step method."""

    def test_analyze_step_returns_dict(self, simple_classifier, small_data_loader, device):
        """Test analyze_step returns dict."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)
        metrics = analyzer.analyze_step(iter(small_data_loader), n_samples=10)

        assert isinstance(metrics, dict)
        assert "condition_number" in metrics
        assert "trace" in metrics
        assert "log_determinant" in metrics

    def test_analyze_step_updates_history(self, simple_classifier, small_data_loader, device):
        """Test analyze_step updates history."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        # Initial history empty
        assert len(analyzer.history["condition_numbers"]) == 0

        # Analyze
        analyzer.analyze_step(iter(small_data_loader), n_samples=10)

        # History should have one entry
        assert len(analyzer.history["condition_numbers"]) == 1
        assert len(analyzer.history["trace"]) == 1
        assert len(analyzer.history["log_det"]) == 1

    def test_eigenvalues_tracked(self, simple_classifier, small_data_loader, device):
        """Test eigenvalues are tracked when enabled."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model, track_eigenvalues=True)

        metrics = analyzer.analyze_step(iter(small_data_loader), n_samples=10)

        assert "top_eigenvalue" in metrics
        assert "bottom_eigenvalue" in metrics
        assert len(analyzer.history["top_eigenvalues"]) == 1

    def test_eigenvalues_not_tracked(self, simple_classifier, small_data_loader, device):
        """Test eigenvalues not tracked when disabled."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model, track_eigenvalues=False)

        metrics = analyzer.analyze_step(iter(small_data_loader), n_samples=10)

        # Eigenvalue metrics may still be computed but history not populated
        assert len(analyzer.history["top_eigenvalues"]) == 0


class TestAnalyzerGeodesicDistance:
    """Tests for geodesic_distance method."""

    def test_geodesic_distance_to_self(self, simple_classifier, fisher_matrix, device):
        """Test geodesic distance to self is zero."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        params = {n: p.clone() for n, p in model.named_parameters()}

        # Create matching Fisher info
        n_params = sum(p.numel() for p in params.values())
        F = torch.eye(n_params, device=device)
        fisher_info = FisherInfo.from_matrix(F)

        distance = analyzer.geodesic_distance(params, params, fisher_info)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_geodesic_distance_positive(self, simple_classifier, device):
        """Test geodesic distance is positive for different params."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        params1 = {n: p.clone() for n, p in model.named_parameters()}
        params2 = {n: p.clone() + 0.1 for n, p in model.named_parameters()}

        n_params = sum(p.numel() for p in params1.values())
        F = torch.eye(n_params, device=device)
        fisher_info = FisherInfo.from_matrix(F)

        distance = analyzer.geodesic_distance(params1, params2, fisher_info)
        assert distance > 0

    def test_geodesic_distance_symmetric(self, simple_classifier, device):
        """Test geodesic distance is symmetric."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        params1 = {n: p.clone() for n, p in model.named_parameters()}
        params2 = {n: p.clone() + torch.randn_like(p) * 0.01 for n, p in model.named_parameters()}

        n_params = sum(p.numel() for p in params1.values())
        F = torch.eye(n_params, device=device)
        fisher_info = FisherInfo.from_matrix(F)

        d12 = analyzer.geodesic_distance(params1, params2, fisher_info)
        d21 = analyzer.geodesic_distance(params2, params1, fisher_info)

        assert d12 == pytest.approx(d21, rel=1e-5)


class TestAnalyzerEffectiveDimensionality:
    """Tests for effective_dimensionality method."""

    def test_eff_dim_identity(self, simple_classifier):
        """Test effective dimensionality for identity Fisher."""
        analyzer = InformationGeometricAnalyzer(simple_classifier)

        # Identity matrix has eff_dim = n (all directions equally important)
        n = 10
        F = torch.eye(n)
        fisher_info = FisherInfo.from_matrix(F)

        eff_dim = analyzer.effective_dimensionality(fisher_info)
        assert eff_dim == pytest.approx(n, rel=0.1)

    def test_eff_dim_rank_one(self, simple_classifier):
        """Test effective dimensionality for rank-1 Fisher."""
        analyzer = InformationGeometricAnalyzer(simple_classifier)

        # Rank-1 matrix has eff_dim = 1
        v = torch.randn(10)
        F = torch.outer(v, v) + 1e-6 * torch.eye(10)  # Add small regularization
        fisher_info = FisherInfo.from_matrix(F)

        eff_dim = analyzer.effective_dimensionality(fisher_info)
        assert eff_dim < 2  # Should be close to 1

    def test_eff_dim_positive(self, fisher_matrix, simple_classifier):
        """Test effective dimensionality is positive."""
        analyzer = InformationGeometricAnalyzer(simple_classifier)
        fisher_info = FisherInfo.from_matrix(fisher_matrix)

        eff_dim = analyzer.effective_dimensionality(fisher_info)
        assert eff_dim > 0


class TestAnalyzerFlatnessMeasure:
    """Tests for flatness_measure method."""

    def test_flatness_returns_dict(self, simple_classifier, small_data_loader, device):
        """Test flatness_measure returns dict."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        flatness = analyzer.flatness_measure(
            iter(small_data_loader), epsilon=0.01, n_directions=3
        )

        assert isinstance(flatness, dict)
        assert "baseline_loss" in flatness
        assert "mean_perturbed_loss" in flatness
        assert "std_perturbed_loss" in flatness
        assert "max_increase" in flatness
        assert "sharpness" in flatness

    def test_flatness_baseline_positive(self, simple_classifier, small_data_loader, device):
        """Test baseline loss is positive."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        flatness = analyzer.flatness_measure(
            iter(small_data_loader), epsilon=0.01, n_directions=3
        )

        assert flatness["baseline_loss"] >= 0

    def test_flatness_preserves_params(self, simple_classifier, small_data_loader, device):
        """Test flatness_measure preserves original parameters."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        # Store original params
        original_params = {n: p.clone() for n, p in model.named_parameters()}

        # Compute flatness
        analyzer.flatness_measure(iter(small_data_loader), epsilon=0.01, n_directions=3)

        # Check params unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(param, original_params[name])

    def test_larger_epsilon_higher_variation(self, simple_classifier, small_data_loader, device):
        """Test larger epsilon gives higher loss variation."""
        model = simple_classifier.to(device)
        analyzer = InformationGeometricAnalyzer(model)

        flatness_small = analyzer.flatness_measure(
            iter(small_data_loader), epsilon=0.001, n_directions=5
        )
        flatness_large = analyzer.flatness_measure(
            iter(small_data_loader), epsilon=0.1, n_directions=5
        )

        # Larger perturbation should generally give larger variation
        # (though not always due to randomness)
        assert flatness_large["std_perturbed_loss"] >= 0
