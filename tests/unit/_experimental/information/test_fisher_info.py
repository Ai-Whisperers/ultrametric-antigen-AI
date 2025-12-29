# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for FisherInfo dataclass."""

from __future__ import annotations

import pytest
import torch

from src.information import FisherInfo


class TestFisherInfoCreation:
    """Tests for FisherInfo creation."""

    def test_basic_creation(self, fisher_matrix):
        """Test basic FisherInfo creation."""
        info = FisherInfo(matrix=fisher_matrix)
        assert info.matrix is not None
        assert info.matrix.shape == fisher_matrix.shape

    def test_from_matrix(self, fisher_matrix):
        """Test from_matrix classmethod."""
        info = FisherInfo.from_matrix(fisher_matrix)

        assert info.matrix is not None
        assert info.eigenvalues is not None
        assert info.eigenvectors is not None
        assert info.condition_number is not None
        assert info.trace is not None
        assert info.log_determinant is not None

    def test_eigenvalues_positive(self, fisher_matrix):
        """Test eigenvalues are positive after clamping."""
        info = FisherInfo.from_matrix(fisher_matrix)
        assert (info.eigenvalues > 0).all()

    def test_condition_number_positive(self, fisher_matrix):
        """Test condition number is positive."""
        info = FisherInfo.from_matrix(fisher_matrix)
        assert info.condition_number > 0

    def test_trace_positive(self, fisher_matrix):
        """Test trace is positive for PSD matrix."""
        info = FisherInfo.from_matrix(fisher_matrix)
        assert info.trace > 0


class TestFisherInfoProperties:
    """Tests for FisherInfo computed properties."""

    def test_eigenvalue_count(self, fisher_matrix):
        """Test eigenvalue count matches matrix size."""
        info = FisherInfo.from_matrix(fisher_matrix)
        assert len(info.eigenvalues) == fisher_matrix.shape[0]

    def test_eigenvector_shape(self, fisher_matrix):
        """Test eigenvector shape."""
        info = FisherInfo.from_matrix(fisher_matrix)
        n = fisher_matrix.shape[0]
        assert info.eigenvectors.shape == (n, n)

    def test_trace_matches_eigenvalue_sum(self, fisher_matrix):
        """Test trace equals sum of eigenvalues."""
        info = FisherInfo.from_matrix(fisher_matrix)
        assert abs(info.trace - info.eigenvalues.sum().item()) < 1e-4

    def test_log_det_matches_eigenvalue_log_sum(self, fisher_matrix):
        """Test log determinant equals sum of log eigenvalues."""
        info = FisherInfo.from_matrix(fisher_matrix)
        expected = info.eigenvalues.log().sum().item()
        assert abs(info.log_determinant - expected) < 1e-4


class TestFisherInfoEdgeCases:
    """Tests for edge cases."""

    def test_identity_matrix(self):
        """Test with identity matrix."""
        I = torch.eye(5)
        info = FisherInfo.from_matrix(I)

        assert info.condition_number == pytest.approx(1.0, rel=1e-4)
        assert info.trace == pytest.approx(5.0, rel=1e-4)
        assert info.log_determinant == pytest.approx(0.0, abs=1e-4)

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        diag = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        info = FisherInfo.from_matrix(diag)

        assert info.condition_number == pytest.approx(4.0, rel=1e-4)
        assert info.trace == pytest.approx(10.0, rel=1e-4)

    def test_small_eigenvalue_clamping(self):
        """Test that small eigenvalues are clamped."""
        # Create matrix with very small eigenvalue
        D = torch.diag(torch.tensor([1e-15, 1.0, 2.0]))
        info = FisherInfo.from_matrix(D)

        # All eigenvalues should be >= 1e-10
        assert (info.eigenvalues >= 1e-10).all()
