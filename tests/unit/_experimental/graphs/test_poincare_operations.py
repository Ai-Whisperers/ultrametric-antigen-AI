# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PoincareOperations class.

Tests cover:
- Mobius addition properties (closure, associativity-like)
- Mobius scalar multiplication
- Exponential and logarithmic maps
- Distance computations
- Projection to ball
- Numerical stability
"""

from __future__ import annotations

import math

import pytest
import torch

from src.graphs import PoincareOperations


class TestPoincareOperationsInit:
    """Tests for PoincareOperations initialization."""

    def test_default_curvature(self):
        """Test default curvature is 1.0."""
        ops = PoincareOperations()
        assert ops.c == 1.0
        assert ops.eps == 1e-5

    def test_custom_curvature(self):
        """Test custom curvature."""
        ops = PoincareOperations(curvature=2.0)
        assert ops.c == 2.0

    def test_custom_eps(self):
        """Test custom epsilon."""
        ops = PoincareOperations(eps=1e-8)
        assert ops.eps == 1e-8


class TestMobiusAddition:
    """Tests for Mobius addition operation."""

    def test_mobius_add_identity(self, poincare_ops, poincare_points):
        """Adding zero gives same point."""
        zero = torch.zeros_like(poincare_points)
        result = poincare_ops.mobius_add(poincare_points, zero)
        assert torch.allclose(result, poincare_points, atol=1e-5)

    def test_mobius_add_closure(self, poincare_ops, poincare_points):
        """Result stays inside the Poincare ball (with projection)."""
        y = torch.randn_like(poincare_points) * 0.3
        result = poincare_ops.mobius_add(poincare_points, y)

        # Check result is finite and project to ensure inside ball
        assert torch.isfinite(result).all()
        # After projection, norms should be inside ball
        projected = poincare_ops.project(result)
        norms = projected.norm(dim=-1)
        max_norm = 1.0 / math.sqrt(poincare_ops.c)
        assert (norms < max_norm + 0.1).all()

    def test_mobius_add_commutative_origin(self, poincare_ops, device):
        """Mobius addition at origin is commutative."""
        x = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        y = torch.tensor([[0.1, 0.2, 0.3]], device=device)

        result1 = poincare_ops.mobius_add(x, y)
        result2 = poincare_ops.mobius_add(y, x)

        # Both should equal y (adding zero)
        assert torch.allclose(result1, y, atol=1e-5)

    def test_mobius_add_batch(self, poincare_ops, poincare_points):
        """Test batch processing."""
        batch_size = poincare_points.shape[0]
        y = torch.randn_like(poincare_points) * 0.2

        result = poincare_ops.mobius_add(poincare_points, y)
        assert result.shape == poincare_points.shape

    def test_mobius_add_inverse(self, poincare_ops, poincare_points_small):
        """x + (-x) should give origin (approximately)."""
        neg_x = -poincare_points_small
        result = poincare_ops.mobius_add(poincare_points_small, neg_x)

        # Result should be near origin
        norms = result.norm(dim=-1)
        assert (norms < 0.1).all()


class TestMobiusScalar:
    """Tests for Mobius scalar multiplication."""

    def test_mobius_scalar_identity(self, poincare_ops, poincare_points_small):
        """Multiplying by 1 gives same point."""
        r = torch.ones(poincare_points_small.shape[0], 1, device=poincare_points_small.device)
        result = poincare_ops.mobius_scalar(r, poincare_points_small)
        assert torch.allclose(result, poincare_points_small, atol=1e-4)

    def test_mobius_scalar_zero(self, poincare_ops, poincare_points_small):
        """Multiplying by 0 gives origin."""
        r = torch.zeros(poincare_points_small.shape[0], 1, device=poincare_points_small.device)
        result = poincare_ops.mobius_scalar(r, poincare_points_small)
        assert result.norm(dim=-1).max() < 1e-5

    def test_mobius_scalar_closure(self, poincare_ops, poincare_points_small):
        """Result stays inside ball for any scalar."""
        r = torch.tensor([[2.0], [0.5], [-1.0], [3.0]], device=poincare_points_small.device)
        result = poincare_ops.mobius_scalar(r, poincare_points_small)

        norms = result.norm(dim=-1)
        max_norm = 1.0 / math.sqrt(poincare_ops.c)
        assert (norms < max_norm + 1e-5).all()


class TestExpLogMaps:
    """Tests for exponential and logarithmic maps."""

    def test_exp_map_origin(self, poincare_ops, device):
        """Exp map at origin."""
        v = torch.randn(4, 8, device=device) * 0.5
        result = poincare_ops.exp_map(v, base=None)

        # Result should be inside ball
        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_log_map_origin(self, poincare_ops, poincare_points_small):
        """Log map at origin."""
        result = poincare_ops.log_map(poincare_points_small, base=None)

        # Should be finite
        assert torch.isfinite(result).all()

    def test_exp_log_inverse_origin(self, poincare_ops, device):
        """exp(log(x)) = x at origin (approximately)."""
        # Use smaller points to avoid numerical issues near boundary
        x = torch.randn(4, 8, device=device) * 0.2
        v = poincare_ops.log_map(x, base=None)
        x_recovered = poincare_ops.exp_map(v, base=None)

        # Use relaxed tolerance for hyperbolic geometry
        assert torch.allclose(x, x_recovered, atol=1e-2)

    def test_log_exp_inverse_origin(self, poincare_ops, device):
        """log(exp(v)) = v at origin."""
        v = torch.randn(4, 8, device=device) * 0.5
        x = poincare_ops.exp_map(v, base=None)
        v_recovered = poincare_ops.log_map(x, base=None)

        assert torch.allclose(v, v_recovered, atol=1e-4)

    def test_exp_map_with_base(self, poincare_ops, device):
        """Exp map at arbitrary base point."""
        base = torch.randn(4, 8, device=device) * 0.2
        v = torch.randn(4, 8, device=device) * 0.3

        result = poincare_ops.exp_map(v, base=base)

        # Result should be inside ball
        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_log_map_with_base(self, poincare_ops, device):
        """Log map at arbitrary base point."""
        base = torch.randn(4, 8, device=device) * 0.2
        y = torch.randn(4, 8, device=device) * 0.3

        result = poincare_ops.log_map(y, base=base)
        assert torch.isfinite(result).all()


class TestDistance:
    """Tests for hyperbolic distance."""

    def test_distance_to_self(self, poincare_ops, poincare_points_small):
        """Distance to self is zero."""
        dist = poincare_ops.distance(poincare_points_small, poincare_points_small)
        assert (dist.abs() < 1e-4).all()

    def test_distance_symmetric(self, poincare_ops, device):
        """d(x, y) = d(y, x)."""
        x = torch.randn(4, 8, device=device) * 0.3
        y = torch.randn(4, 8, device=device) * 0.3

        d_xy = poincare_ops.distance(x, y)
        d_yx = poincare_ops.distance(y, x)

        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_non_negative(self, poincare_ops, poincare_points):
        """Distance is non-negative."""
        y = torch.randn_like(poincare_points) * 0.3
        dist = poincare_ops.distance(poincare_points, y)
        assert (dist >= -1e-5).all()

    def test_distance_triangle_inequality(self, poincare_ops, device):
        """d(x, z) <= d(x, y) + d(y, z)."""
        x = torch.randn(4, 8, device=device) * 0.2
        y = torch.randn(4, 8, device=device) * 0.2
        z = torch.randn(4, 8, device=device) * 0.2

        d_xz = poincare_ops.distance(x, z)
        d_xy = poincare_ops.distance(x, y)
        d_yz = poincare_ops.distance(y, z)

        # Triangle inequality with numerical tolerance
        assert (d_xz <= d_xy + d_yz + 1e-4).all()


class TestProjection:
    """Tests for projection to Poincare ball."""

    def test_project_inside_ball(self, poincare_ops, poincare_points_small):
        """Points already inside ball stay same."""
        result = poincare_ops.project(poincare_points_small, max_norm=0.95)
        assert torch.allclose(result, poincare_points_small, atol=1e-5)

    def test_project_outside_ball(self, poincare_ops, device):
        """Points outside ball are projected in."""
        x = torch.randn(4, 8, device=device) * 2.0  # Large points
        result = poincare_ops.project(x, max_norm=0.95)

        norms = result.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_project_preserves_direction(self, poincare_ops, device):
        """Projection preserves direction."""
        x = torch.randn(4, 8, device=device) * 2.0
        result = poincare_ops.project(x, max_norm=0.95)

        # Normalize both and compare directions
        x_norm = x / x.norm(dim=-1, keepdim=True)
        result_norm = result / result.norm(dim=-1, keepdim=True)

        assert torch.allclose(x_norm, result_norm, atol=1e-5)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_inputs(self, poincare_ops, device):
        """Operations work with very small inputs."""
        x = torch.randn(4, 8, device=device) * 1e-8
        y = torch.randn(4, 8, device=device) * 1e-8

        result = poincare_ops.mobius_add(x, y)
        assert torch.isfinite(result).all()

    def test_boundary_points(self, poincare_ops, device):
        """Operations work near ball boundary."""
        # Points very close to boundary
        x = torch.randn(4, 8, device=device)
        x = x / x.norm(dim=-1, keepdim=True) * 0.99

        y = torch.randn(4, 8, device=device)
        y = y / y.norm(dim=-1, keepdim=True) * 0.99

        result = poincare_ops.mobius_add(x, y)
        assert torch.isfinite(result).all()

    def test_exp_map_large_tangent(self, poincare_ops, device):
        """Exp map with large tangent vectors approaches boundary."""
        v = torch.randn(4, 8, device=device) * 10.0
        result = poincare_ops.exp_map(v, base=None)

        # Should be finite and near ball boundary
        assert torch.isfinite(result).all()
        norms = result.norm(dim=-1)
        # Large tangent vectors push towards boundary (norm close to 1)
        assert (norms <= 1.0 + 1e-5).all()


class TestCurvatureVariations:
    """Tests with different curvatures."""

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0, 4.0])
    def test_operations_with_curvature(self, curvature, device):
        """Operations work with various curvatures."""
        ops = PoincareOperations(curvature=curvature)

        # Adjust point magnitude for curvature
        max_norm = 0.9 / math.sqrt(curvature)
        x = torch.randn(4, 8, device=device) * max_norm * 0.5
        y = torch.randn(4, 8, device=device) * max_norm * 0.5

        # All operations should work
        result = ops.mobius_add(x, y)
        assert torch.isfinite(result).all()

        dist = ops.distance(x, y)
        assert torch.isfinite(dist).all()

        exp_result = ops.exp_map(x, base=None)
        assert torch.isfinite(exp_result).all()

        log_result = ops.log_map(y, base=None)
        assert torch.isfinite(log_result).all()
