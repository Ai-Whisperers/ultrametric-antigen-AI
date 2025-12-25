# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Poincare distance computation in src/geometry/poincare.py.

Tests pairwise and matrix distance computations.
"""

import pytest
import torch
from src.geometry.poincare import (
    poincare_distance,
    poincare_distance_matrix,
    project_to_poincare,
)


class TestPoincareDistanceBasic:
    """Basic distance computation tests."""

    def test_distance_to_self_is_zero(self, device):
        """d(x, x) = 0."""
        x = torch.tensor([[0.5, 0.3]], dtype=torch.float32, device=device)
        x = project_to_poincare(x)
        d = poincare_distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)

    def test_distance_symmetry(self, device):
        """d(x, y) = d(y, x)."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        d_xy = poincare_distance(x, y)
        d_yx = poincare_distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_non_negative(self, device):
        """Distance should always be non-negative."""
        x = project_to_poincare(torch.randn(10, 8, device=device) * 0.5)
        y = project_to_poincare(torch.randn(10, 8, device=device) * 0.5)
        d = poincare_distance(x, y)
        assert (d >= 0).all()

    def test_distance_is_finite(self, device):
        """Distance should always be finite."""
        x = project_to_poincare(torch.randn(10, 8, device=device) * 0.5)
        y = project_to_poincare(torch.randn(10, 8, device=device) * 0.5)
        d = poincare_distance(x, y)
        assert torch.isfinite(d).all()


class TestPoincareDistanceShape:
    """Tests for output shape handling."""

    def test_distance_keepdim_false(self, device):
        """Test keepdim=False parameter."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        d = poincare_distance(x, y, keepdim=False)
        assert d.shape == (5,)

    def test_distance_keepdim_true(self, device):
        """Test keepdim=True parameter."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        d = poincare_distance(x, y, keepdim=True)
        assert d.shape == (5, 1)

    def test_single_point(self, device):
        """Test distance for single point."""
        x = project_to_poincare(torch.randn(1, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(1, 4, device=device) * 0.5)
        d = poincare_distance(x, y)
        assert d.shape == (1,)


class TestPoincareDistanceCurvature:
    """Tests for curvature effects on distance."""

    def test_distance_different_curvature(self, device):
        """Distance with different curvature."""
        x = project_to_poincare(torch.randn(3, 2, device=device) * 0.3)
        y = project_to_poincare(torch.randn(3, 2, device=device) * 0.3)

        d1 = poincare_distance(x, y, c=1.0)
        d2 = poincare_distance(x, y, c=2.0)

        # Different curvatures should give different distances
        assert not torch.allclose(d1, d2)

    def test_higher_curvature_larger_distance(self, device):
        """Higher curvature should generally give larger distances."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)

        d_low = poincare_distance(x, y, c=0.5)
        d_high = poincare_distance(x, y, c=2.0)

        # Higher curvature typically means larger distances
        assert d_high.mean() > d_low.mean()


class TestPoincareDistanceTriangleInequality:
    """Tests for triangle inequality property."""

    def test_triangle_inequality(self, device):
        """Verify dist(a,c) <= dist(a,b) + dist(b,c)."""
        a = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        b = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        c = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)

        d_ab = poincare_distance(a, b, keepdim=False)
        d_bc = poincare_distance(b, c, keepdim=False)
        d_ac = poincare_distance(a, c, keepdim=False)

        # Allow slight numerical error margin
        assert (d_ac <= d_ab + d_bc + 1e-5).all()

    def test_triangle_inequality_multiple_dims(self, device):
        """Test triangle inequality with various dimensions."""
        for dim in [2, 4, 8, 16]:
            a = project_to_poincare(torch.randn(5, dim, device=device) * 0.5)
            b = project_to_poincare(torch.randn(5, dim, device=device) * 0.5)
            c = project_to_poincare(torch.randn(5, dim, device=device) * 0.5)

            d_ab = poincare_distance(a, b)
            d_bc = poincare_distance(b, c)
            d_ac = poincare_distance(a, c)

            assert (d_ac <= d_ab + d_bc + 1e-5).all()


class TestPoincareDistanceMatrix:
    """Test pairwise distance matrix computation."""

    def test_distance_matrix_shape(self, device):
        """Distance matrix should be n x n."""
        z = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        D = poincare_distance_matrix(z)
        assert D.shape == (10, 10)

    def test_distance_matrix_diagonal_zero(self, device):
        """Diagonal should be zero (d(x, x) = 0)."""
        z = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        D = poincare_distance_matrix(z)
        diag = D.diag()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_distance_matrix_symmetric(self, device):
        """Distance matrix should be symmetric."""
        torch.manual_seed(42)  # Reproducible
        z = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        D = poincare_distance_matrix(z)
        assert torch.allclose(D, D.T, atol=1e-4)

    def test_distance_matrix_non_negative(self, device):
        """All distances should be non-negative."""
        z = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        D = poincare_distance_matrix(z)
        assert (D >= -1e-6).all()

    def test_distance_matrix_curvature(self, device):
        """Different curvatures should give different matrices."""
        z = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        D1 = poincare_distance_matrix(z, c=1.0)
        D2 = poincare_distance_matrix(z, c=2.0)
        assert not torch.allclose(D1, D2)

    def test_distance_matrix_single_point(self, device):
        """Test with single point."""
        z = project_to_poincare(torch.randn(1, 4, device=device) * 0.5)
        D = poincare_distance_matrix(z)
        assert D.shape == (1, 1)
        assert D.item() == pytest.approx(0.0, abs=1e-6)
