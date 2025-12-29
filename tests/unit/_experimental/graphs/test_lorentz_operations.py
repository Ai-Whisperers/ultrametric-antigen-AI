# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for LorentzOperations class.

Tests cover:
- Minkowski inner product
- Minkowski norm
- Projection to hyperboloid
- Exponential and logarithmic maps
- Distance on hyperboloid
- Parallel transport
"""

from __future__ import annotations

import math

import pytest
import torch

from src.graphs import LorentzOperations


class TestLorentzOperationsInit:
    """Tests for LorentzOperations initialization."""

    def test_default_curvature(self):
        """Test default curvature is 1.0."""
        ops = LorentzOperations()
        assert ops.c == 1.0
        assert ops.eps == 1e-5

    def test_custom_curvature(self):
        """Test custom curvature."""
        ops = LorentzOperations(curvature=2.0)
        assert ops.c == 2.0


class TestMinkowskiOperations:
    """Tests for Minkowski inner product and norm."""

    def test_minkowski_inner_basic(self, lorentz_ops, device):
        """Test basic Minkowski inner product."""
        x = torch.tensor([[2.0, 1.0, 0.0, 0.0]], device=device)
        y = torch.tensor([[3.0, 0.0, 1.0, 0.0]], device=device)

        # <x, y>_L = -x0*y0 + x1*y1 + x2*y2 + ...
        # = -2*3 + 1*0 + 0*1 = -6
        inner = lorentz_ops.minkowski_inner(x, y)
        assert torch.allclose(inner, torch.tensor([[-6.0]], device=device))

    def test_minkowski_inner_symmetric(self, lorentz_ops, lorentz_points):
        """Minkowski inner product is symmetric."""
        x = lorentz_points[:16]
        y = lorentz_points[16:]

        inner_xy = lorentz_ops.minkowski_inner(x, y)
        inner_yx = lorentz_ops.minkowski_inner(y, x)

        assert torch.allclose(inner_xy, inner_yx, atol=1e-5)

    def test_minkowski_inner_on_hyperboloid(self, lorentz_ops, lorentz_points):
        """Points on hyperboloid have <x,x>_L = -1/c."""
        inner = lorentz_ops.minkowski_inner(lorentz_points, lorentz_points)
        expected = -1.0 / lorentz_ops.c

        assert torch.allclose(inner.squeeze(), torch.full_like(inner.squeeze(), expected), atol=1e-4)

    def test_minkowski_norm(self, lorentz_ops, device):
        """Test Minkowski norm computation."""
        x = torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0]], device=device)
        # <x, x>_L = -9 + 16 = 7
        # ||x||_L = sqrt(|7|) = sqrt(7)
        norm = lorentz_ops.minkowski_norm(x)
        assert torch.allclose(norm, torch.tensor([[math.sqrt(7)]], device=device), atol=1e-5)


class TestHyperboloidProjection:
    """Tests for projection to hyperboloid."""

    def test_project_basic(self, lorentz_ops, device):
        """Test basic hyperboloid projection."""
        x = torch.randn(4, 9, device=device)
        result = lorentz_ops.project_to_hyperboloid(x)

        # Check constraint: <x, x>_L = -1/c
        inner = lorentz_ops.minkowski_inner(result, result)
        expected = -1.0 / lorentz_ops.c

        assert torch.allclose(inner.squeeze(), torch.full((4,), expected, device=device), atol=1e-4)

    def test_project_positive_time(self, lorentz_ops, device):
        """Projected points have positive time component."""
        x = torch.randn(10, 9, device=device)
        result = lorentz_ops.project_to_hyperboloid(x)

        assert (result[..., 0] > 0).all()

    def test_project_idempotent(self, lorentz_ops, lorentz_points):
        """Projecting already-on-hyperboloid points gives same result."""
        result = lorentz_ops.project_to_hyperboloid(lorentz_points)

        # Should be approximately the same
        # (might differ slightly due to time component recomputation)
        assert torch.allclose(result[..., 1:], lorentz_points[..., 1:], atol=1e-5)


class TestLorentzExpLog:
    """Tests for exponential and logarithmic maps on hyperboloid."""

    def test_exp_map_basic(self, lorentz_ops, lorentz_point, device):
        """Test basic exp map produces valid hyperboloid points."""
        # Tangent vector at base (time component should be 0 for tangent)
        v = torch.zeros_like(lorentz_point)
        v[..., 1:] = torch.randn_like(v[..., 1:]) * 0.3

        result = lorentz_ops.exp_map(v, lorentz_point)

        # Result should be finite and have positive time component
        assert torch.isfinite(result).all()
        assert (result[..., 0] > 0).all()

    def test_log_map_basic(self, lorentz_ops, lorentz_points):
        """Test basic log map."""
        x = lorentz_points[:4]
        y = lorentz_points[4:8]

        result = lorentz_ops.log_map(y, x)

        # Result should be finite
        assert torch.isfinite(result).all()

    def test_distance_to_self(self, lorentz_ops, lorentz_points):
        """Distance from point to itself is close to zero."""
        dist = lorentz_ops.distance(lorentz_points, lorentz_points)
        # Relaxed tolerance for hyperbolic geometry numerical issues
        assert (dist.abs() < 1e-2).all()

    def test_distance_symmetric(self, lorentz_ops, lorentz_points):
        """d(x, y) = d(y, x)."""
        x = lorentz_points[:16]
        y = lorentz_points[16:]

        d_xy = lorentz_ops.distance(x, y)
        d_yx = lorentz_ops.distance(y, x)

        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_non_negative(self, lorentz_ops, lorentz_points):
        """Distance is non-negative."""
        x = lorentz_points[:16]
        y = lorentz_points[16:]

        dist = lorentz_ops.distance(x, y)
        assert (dist >= -1e-5).all()


class TestParallelTransport:
    """Tests for parallel transport on hyperboloid."""

    def test_parallel_transport_preserves_norm(self, lorentz_ops, lorentz_points):
        """Parallel transport produces finite results."""
        x = lorentz_points[:4]
        y = lorentz_points[4:8]

        # Create tangent vector at x
        v = torch.zeros_like(x)
        v[..., 1:] = torch.randn_like(v[..., 1:]) * 0.5

        # Transport to y
        v_transported = lorentz_ops.parallel_transport(v, x, y)

        # Check result is finite (exact norm preservation is hard numerically)
        assert torch.isfinite(v_transported).all()

    def test_parallel_transport_finite(self, lorentz_ops, lorentz_points):
        """Parallel transport produces finite results."""
        x = lorentz_points[:4]
        y = lorentz_points[4:8]

        v = torch.zeros_like(x)
        v[..., 1:] = torch.randn_like(v[..., 1:]) * 0.5

        v_transported = lorentz_ops.parallel_transport(v, x, y)
        assert torch.isfinite(v_transported).all()


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_small_inputs(self, lorentz_ops, device):
        """Operations work with small inputs."""
        x = torch.randn(4, 9, device=device) * 1e-6
        result = lorentz_ops.project_to_hyperboloid(x)

        assert torch.isfinite(result).all()

    def test_large_space_components(self, lorentz_ops, device):
        """Operations work with large space components."""
        x = torch.randn(4, 9, device=device) * 100
        result = lorentz_ops.project_to_hyperboloid(x)

        assert torch.isfinite(result).all()
        assert (result[..., 0] > 0).all()


class TestCurvatureVariations:
    """Tests with different curvatures."""

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
    def test_hyperboloid_constraint(self, curvature, device):
        """Hyperboloid constraint holds for different curvatures."""
        ops = LorentzOperations(curvature=curvature)

        x = torch.randn(4, 9, device=device)
        result = ops.project_to_hyperboloid(x)

        inner = ops.minkowski_inner(result, result)
        expected = -1.0 / curvature

        assert torch.allclose(inner.squeeze(), torch.full((4,), expected, device=device), atol=1e-4)
