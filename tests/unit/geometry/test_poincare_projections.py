# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Poincare ball projection in src/geometry/poincare.py.

Tests projection onto the Poincare ball manifold.
"""

import pytest  # noqa: F401 - used for fixtures
import torch

from src.geometry.poincare import project_to_poincare


class TestProjectToPoincareBasic:
    """Basic projection tests."""

    def test_projection_constrains_norm(self, device):
        """Projected points should have norm <= max_norm."""
        x = torch.randn(20, 5, device=device) * 10  # Large norms
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_projection_preserves_small_norms(self, device):
        """Small norm points should be mostly preserved."""
        x = torch.randn(10, 4, device=device) * 0.1  # Small norms
        x_proj = project_to_poincare(x, max_norm=0.95)

        # Should be close to original
        assert torch.allclose(x, x_proj, atol=0.1)

    def test_projection_returns_same_shape(self, device):
        """Projection should preserve shape."""
        for shape in [(5, 4), (10, 8), (20, 16)]:
            x = torch.randn(*shape, device=device) * 2
            x_proj = project_to_poincare(x)
            assert x_proj.shape == x.shape


class TestProjectToPoincareMaxNorm:
    """Tests for different max_norm values."""

    def test_projection_different_max_norm(self, device):
        """Different max_norm values."""
        x = torch.randn(10, 4, device=device) * 2

        x_50 = project_to_poincare(x, max_norm=0.50)
        x_90 = project_to_poincare(x, max_norm=0.90)

        assert (x_50.norm(dim=-1) <= 0.50 + 1e-5).all()
        assert (x_90.norm(dim=-1) <= 0.90 + 1e-5).all()

    def test_projection_strict_constraint(self, device):
        """All points should strictly satisfy the constraint."""
        x = torch.randn(100, 8, device=device) * 10

        for max_norm in [0.5, 0.9, 0.95, 0.99]:
            x_proj = project_to_poincare(x, max_norm=max_norm)
            norms = x_proj.norm(dim=-1)
            assert (norms <= max_norm + 1e-5).all()


class TestProjectToPoincareEdgeCases:
    """Edge case tests for projection."""

    def test_projection_of_zero(self, device):
        """Zero vector should remain zero."""
        x = torch.zeros(5, 4, device=device)
        x_proj = project_to_poincare(x)
        assert torch.allclose(x_proj, x)

    def test_projection_of_very_large(self, device):
        """Very large vectors should be projected."""
        x = torch.randn(5, 4, device=device) * 1000
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_projection_single_point(self, device):
        """Single point projection."""
        x = torch.randn(1, 4, device=device) * 2
        x_proj = project_to_poincare(x)

        assert x_proj.shape == (1, 4)
        assert x_proj.norm() < 1.0

    def test_projection_preserves_direction(self, device):
        """Projection should preserve direction for large vectors."""
        x = torch.randn(5, 4, device=device) * 10
        x_proj = project_to_poincare(x, max_norm=0.9)

        # Normalize both and compare directions
        x_dir = x / x.norm(dim=-1, keepdim=True)
        x_proj_dir = x_proj / x_proj.norm(dim=-1, keepdim=True)

        # Directions should be the same
        assert torch.allclose(x_dir, x_proj_dir, atol=1e-5)


class TestProjectToPoincareOutput:
    """Output property tests for projection."""

    def test_projection_result_is_detached(self, device):
        """Projection may detach gradients for numerical stability."""
        x = torch.randn(10, 4, device=device, requires_grad=True) * 2

        x_proj = project_to_poincare(x)

        # Result should be valid and on manifold
        assert x_proj.shape == x.shape
        assert (x_proj.norm(dim=-1) < 1.0).all()

    def test_projection_with_small_norm_preserves_values(self, device):
        """Small norm points should be mostly preserved."""
        x = torch.randn(10, 4, device=device, requires_grad=True) * 0.1

        x_proj = project_to_poincare(x, max_norm=0.95)

        # Values should be close for small norms
        assert torch.allclose(x.detach(), x_proj.detach(), atol=0.1)


class TestProjectToPoincareDimensions:
    """Tests for various dimensions."""

    def test_high_dimensional(self, device):
        """Test projection in high dimensions."""
        x = torch.randn(10, 128, device=device) * 2
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_low_dimensional(self, device):
        """Test projection in low dimensions."""
        x = torch.randn(10, 2, device=device) * 2
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_single_dimension(self, device):
        """Test projection in 1D."""
        x = torch.randn(10, 1, device=device) * 2
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()
