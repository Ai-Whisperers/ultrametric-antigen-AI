# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Poincare ball operations in src/geometry/poincare.py.

Tests Mobius addition, conformal factor, and parallel transport.
"""

import pytest
import torch
from src.geometry.poincare import (
    mobius_add,
    lambda_x,
    parallel_transport,
    project_to_poincare,
)


# =============================================================================
# Mobius Addition Tests
# =============================================================================


class TestMobiusAddBasic:
    """Basic Mobius addition tests."""

    def test_add_zero_right(self, device):
        """x (+) 0 = x."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        zero = torch.zeros_like(x)
        result = mobius_add(x, zero)
        assert torch.allclose(result, x, atol=1e-5)

    def test_zero_add_x(self, device):
        """0 (+) x = x."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        zero = torch.zeros_like(x)
        result = mobius_add(zero, x)
        assert torch.allclose(result, x, atol=1e-5)

    def test_result_on_manifold(self, device):
        """Result should be on the Poincare ball."""
        x = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        result = mobius_add(x, y)

        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_result_shape(self, device):
        """Output shape should match input shapes."""
        x = project_to_poincare(torch.randn(8, 6, device=device) * 0.5)
        y = project_to_poincare(torch.randn(8, 6, device=device) * 0.5)
        result = mobius_add(x, y)
        assert result.shape == x.shape


class TestMobiusAddGradient:
    """Gradient flow tests for Mobius addition."""

    def test_gradient_flows(self, device):
        """Gradient should flow through Mobius addition."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        x.requires_grad_(True)
        y.requires_grad_(True)

        result = mobius_add(x, y)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()


# =============================================================================
# Conformal Factor Tests
# =============================================================================


class TestLambdaXBasic:
    """Basic conformal factor tests."""

    def test_origin_conformal_factor(self, device):
        """lambda(0) = 2 / (1 - 0) = 2 for c=1."""
        x = torch.zeros(5, 4, device=device)
        lam = lambda_x(x, c=1.0)
        expected = torch.full((5, 1), 2.0, device=device)
        assert torch.allclose(lam, expected, atol=1e-5)

    def test_conformal_factor_positive(self, device):
        """Conformal factor should always be positive."""
        x = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        lam = lambda_x(x)
        assert (lam > 0).all()

    def test_conformal_factor_increases_near_boundary(self, device):
        """Conformal factor increases as we approach boundary."""
        x_small = project_to_poincare(torch.randn(5, 4, device=device) * 0.1)
        x_large = project_to_poincare(torch.randn(5, 4, device=device) * 0.9)

        lam_small = lambda_x(x_small).mean()
        lam_large = lambda_x(x_large).mean()

        assert lam_large > lam_small


class TestLambdaXShape:
    """Tests for conformal factor output shape."""

    def test_keepdim_true(self, device):
        """Test keepdim=True parameter."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        lam = lambda_x(x, keepdim=True)
        assert lam.shape == (5, 1)

    def test_keepdim_false(self, device):
        """Test keepdim=False parameter."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        lam = lambda_x(x, keepdim=False)
        assert lam.shape == (5,)


class TestLambdaXGradient:
    """Gradient flow tests for conformal factor."""

    def test_gradient_flows(self, device):
        """Gradient should flow through lambda_x."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        x.requires_grad_(True)

        lam = lambda_x(x)
        loss = lam.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# =============================================================================
# Parallel Transport Tests
# =============================================================================


class TestParallelTransportBasic:
    """Basic parallel transport tests."""

    def test_transport_to_same_point(self, device):
        """Transport from x to x should preserve the vector."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        v = torch.randn(5, 4, device=device) * 0.1

        v_transported = parallel_transport(x, x, v)
        assert torch.allclose(v_transported, v, atol=1e-4)

    def test_transport_returns_valid_tensor(self, device):
        """Parallel transport returns a valid tensor with correct shape."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        v = torch.randn(5, 4, device=device) * 0.1

        v_transported = parallel_transport(x, y, v)

        assert v_transported.shape == v.shape
        assert torch.isfinite(v_transported).all()

    def test_transport_nonzero_for_nonzero_input(self, device):
        """Transported norms should be non-zero for non-zero input."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        v = torch.randn(5, 4, device=device) * 0.1

        v_transported = parallel_transport(x, y, v)
        trans_norms = v_transported.norm(dim=-1)
        assert (trans_norms > 0).all()


class TestParallelTransportOutput:
    """Output property tests for parallel transport."""

    def test_output_shape_matches_input(self, device):
        """Transport output should match input shape."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        v = torch.randn(5, 4, device=device) * 0.1

        v_transported = parallel_transport(x, y, v)

        assert v_transported.shape == v.shape
        assert torch.isfinite(v_transported).all()


# =============================================================================
# Edge Cases
# =============================================================================


class TestOperationsEdgeCases:
    """Edge case tests for Poincare operations."""

    def test_mobius_add_near_boundary(self, device):
        """Mobius addition near boundary should remain on ball."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.9)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.9)
        result = mobius_add(x, y)

        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_lambda_near_boundary(self, device):
        """Conformal factor near boundary should be large but finite."""
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.99)
        lam = lambda_x(x)

        assert torch.isfinite(lam).all()
        assert (lam > 2).all()  # Should be larger than at origin

    def test_single_point_operations(self, device):
        """Test all operations with single point."""
        x = project_to_poincare(torch.randn(1, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(1, 4, device=device) * 0.5)
        v = torch.randn(1, 4, device=device) * 0.1

        # All should work
        result = mobius_add(x, y)
        assert result.shape == (1, 4)

        lam = lambda_x(x)
        assert lam.shape == (1, 1)

        v_trans = parallel_transport(x, y, v)
        assert v_trans.shape == (1, 4)
