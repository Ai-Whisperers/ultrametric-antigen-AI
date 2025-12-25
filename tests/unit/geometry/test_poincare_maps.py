# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for exponential and logarithmic maps in src/geometry/poincare.py.

Tests exp_map_zero and log_map_zero operations.
"""

import pytest  # noqa: F401 - used for fixtures
import torch

from src.geometry.poincare import exp_map_zero, log_map_zero


class TestExpMapZeroBasic:
    """Basic exponential map tests."""

    def test_zero_vector_maps_to_origin(self, device):
        """exp_0(0) = 0."""
        v = torch.zeros(5, 4, device=device)
        z = exp_map_zero(v)
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)

    def test_result_on_manifold(self, device):
        """Result should be on the Poincare ball (norm < 1)."""
        v = torch.randn(10, 4, device=device)
        z = exp_map_zero(v)
        norms = z.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_result_shape(self, device):
        """Output shape should match input shape."""
        v = torch.randn(8, 6, device=device)
        z = exp_map_zero(v)
        assert z.shape == v.shape


class TestExpMapZeroCurvature:
    """Tests for curvature effects on exp map."""

    def test_different_curvature(self, device):
        """Test with different curvature values."""
        v = torch.randn(5, 3, device=device) * 0.5
        z1 = exp_map_zero(v, c=1.0)
        z2 = exp_map_zero(v, c=2.0)

        # Results should differ
        assert not torch.allclose(z1, z2)

    def test_various_curvatures(self, device):
        """Test exp map at various curvatures produces finite results."""
        v = torch.randn(5, 4, device=device) * 0.3  # Small vectors

        for c in [0.5, 1.0, 2.0]:
            z = exp_map_zero(v, c=c)
            # All results should be finite
            assert torch.isfinite(z).all()
            assert z.shape == v.shape


class TestLogMapZeroBasic:
    """Basic logarithmic map tests."""

    def test_origin_maps_to_zero(self, device):
        """log_0(0) = 0."""
        z = torch.zeros(5, 4, device=device)
        v = log_map_zero(z)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)

    def test_result_shape(self, device):
        """Output shape should match input shape."""
        from src.geometry.poincare import project_to_poincare

        z = project_to_poincare(torch.randn(8, 6, device=device) * 0.5)
        v = log_map_zero(z)
        assert v.shape == z.shape


class TestExpLogInverse:
    """Tests for exp/log inverse relationship."""

    def test_inverse_of_exp_map(self, device):
        """log_0(exp_0(v)) should approximately recover v for small v."""
        v = torch.randn(5, 4, device=device) * 0.3
        z = exp_map_zero(v)
        v_recovered = log_map_zero(z)

        # Should approximately recover original
        assert torch.allclose(v, v_recovered, atol=1e-4)

    def test_inverse_for_various_magnitudes(self, device):
        """Test inverse property at various magnitudes."""
        for scale in [0.1, 0.3, 0.5]:
            v = torch.randn(10, 4, device=device) * scale
            z = exp_map_zero(v)
            v_recovered = log_map_zero(z)

            assert torch.allclose(v, v_recovered, atol=1e-3)

    def test_exp_of_log(self, device):
        """exp_0(log_0(z)) should approximately recover z."""
        from src.geometry.poincare import project_to_poincare

        z = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)

        v = log_map_zero(z)
        z_recovered = exp_map_zero(v)

        assert torch.allclose(z, z_recovered, atol=1e-4)


class TestExpMapGradient:
    """Gradient flow tests for exp map."""

    def test_gradient_flows(self, device):
        """Gradient should flow through exp_map_zero."""
        v = torch.randn(10, 4, device=device, requires_grad=True)

        z = exp_map_zero(v)
        loss = z.sum()
        loss.backward()

        assert v.grad is not None
        assert torch.isfinite(v.grad).all()


class TestLogMapGradient:
    """Gradient flow tests for log map."""

    def test_gradient_flows(self, device):
        """Gradient should flow through log_map_zero."""
        from src.geometry.poincare import project_to_poincare

        z = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        z.requires_grad_(True)

        v = log_map_zero(z)
        loss = v.sum()
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()


class TestExpLogEdgeCases:
    """Edge case tests for exp and log maps."""

    def test_exp_small_vector(self, device):
        """Exp map of very small vector."""
        v = torch.randn(5, 4, device=device) * 1e-6
        z = exp_map_zero(v)

        # Should be close to input for very small vectors
        assert torch.allclose(z, v, atol=1e-5)

    def test_exp_large_vector(self, device):
        """Exp map of large vector should still be on ball."""
        v = torch.randn(5, 4, device=device) * 10
        z = exp_map_zero(v)

        norms = z.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_single_point(self, device):
        """Test with single point."""
        v = torch.randn(1, 4, device=device) * 0.5
        z = exp_map_zero(v)

        assert z.shape == (1, 4)
        assert z.norm() < 1.0

    def test_high_dimensional(self, device):
        """Test in high dimensions."""
        v = torch.randn(5, 128, device=device) * 0.5
        z = exp_map_zero(v)

        norms = z.norm(dim=-1)
        assert (norms < 1.0).all()
