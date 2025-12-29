# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for FisherInformationEstimator class."""

from __future__ import annotations

import pytest
import torch

from src.information import FisherInfo, FisherInformationEstimator


class TestFisherEstimatorInit:
    """Tests for FisherInformationEstimator initialization."""

    def test_default_init(self, simple_classifier):
        """Test default initialization."""
        estimator = FisherInformationEstimator(simple_classifier)
        assert estimator.method == "empirical"
        assert estimator.damping == 1e-4
        assert estimator.block_diagonal is True

    def test_custom_method(self, simple_classifier):
        """Test custom method."""
        estimator = FisherInformationEstimator(simple_classifier, method="true")
        assert estimator.method == "true"

    def test_custom_damping(self, simple_classifier):
        """Test custom damping."""
        estimator = FisherInformationEstimator(simple_classifier, damping=1e-3)
        assert estimator.damping == 1e-3

    def test_n_params_computed(self, simple_classifier):
        """Test n_params is computed correctly."""
        estimator = FisherInformationEstimator(simple_classifier)
        expected = sum(p.numel() for p in simple_classifier.parameters())
        assert estimator.n_params == expected


class TestFisherEstimatorBlockDiagonal:
    """Tests for block-diagonal estimation."""

    def test_block_diagonal_returns_dict(self, simple_classifier, small_data_loader):
        """Test block-diagonal returns dict of FisherInfo."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=True
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        assert isinstance(result, dict)
        for name, info in result.items():
            assert isinstance(info, FisherInfo)

    def test_block_per_parameter(self, simple_classifier, small_data_loader):
        """Test one block per parameter."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=True
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        param_names = [n for n, p in simple_classifier.named_parameters() if p.requires_grad]
        assert set(result.keys()) == set(param_names)

    def test_block_size_matches_param(self, simple_classifier, small_data_loader):
        """Test block size matches parameter size."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=True
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        for name, param in simple_classifier.named_parameters():
            if param.requires_grad:
                size = param.numel()
                assert result[name].matrix.shape == (size, size)


class TestFisherEstimatorFull:
    """Tests for full Fisher matrix estimation."""

    def test_full_returns_fisher_info(self, simple_classifier, small_data_loader):
        """Test full estimation returns FisherInfo."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        assert isinstance(result, FisherInfo)

    def test_full_matrix_size(self, simple_classifier, small_data_loader):
        """Test full matrix has correct size."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        n_params = estimator.n_params
        assert result.matrix.shape == (n_params, n_params)

    def test_full_matrix_symmetric(self, simple_classifier, small_data_loader):
        """Test full matrix is symmetric."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        diff = (result.matrix - result.matrix.T).abs().max()
        assert diff < 1e-5

    def test_full_matrix_psd(self, simple_classifier, small_data_loader):
        """Test full matrix is positive semi-definite."""
        estimator = FisherInformationEstimator(
            simple_classifier, block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        # All eigenvalues should be non-negative (after damping)
        assert (result.eigenvalues >= 0).all()


class TestFisherEstimatorMethods:
    """Tests for empirical vs true Fisher."""

    def test_empirical_method(self, simple_classifier, small_data_loader):
        """Test empirical Fisher method."""
        estimator = FisherInformationEstimator(
            simple_classifier, method="empirical", block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)
        assert result is not None

    def test_true_method(self, simple_classifier, small_data_loader):
        """Test true Fisher method."""
        estimator = FisherInformationEstimator(
            simple_classifier, method="true", block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)
        assert result is not None


class TestFisherEstimatorDamping:
    """Tests for damping behavior."""

    def test_damping_applied(self, simple_classifier, small_data_loader):
        """Test damping is applied to result."""
        damping = 0.1
        estimator = FisherInformationEstimator(
            simple_classifier, damping=damping, block_diagonal=False
        )
        result = estimator.estimate(iter(small_data_loader), n_samples=10)

        # Minimum eigenvalue should be at least damping
        assert result.eigenvalues.min() >= damping * 0.9  # Allow small numerical error

    def test_higher_damping_better_condition(self, simple_classifier, small_data_loader):
        """Test higher damping improves condition number."""
        estimator_low = FisherInformationEstimator(
            simple_classifier, damping=1e-6, block_diagonal=False
        )
        estimator_high = FisherInformationEstimator(
            simple_classifier, damping=1e-2, block_diagonal=False
        )

        result_low = estimator_low.estimate(iter(small_data_loader), n_samples=10)
        result_high = estimator_high.estimate(iter(small_data_loader), n_samples=10)

        assert result_high.condition_number <= result_low.condition_number
