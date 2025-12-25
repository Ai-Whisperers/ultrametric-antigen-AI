# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PoincareModule base class in src/geometry/poincare.py.

Tests the PyTorch module for Poincare ball geometry.
"""

import pytest
import torch
from src.geometry.poincare import PoincareModule, project_to_poincare


class TestPoincareModuleInit:
    """Initialization tests for PoincareModule."""

    def test_initialization_default(self):
        """Test default module initialization."""
        module = PoincareModule()
        assert module.c == 1.0
        assert module.max_norm == 0.95
        assert module.manifold is not None

    def test_initialization_custom(self):
        """Test custom module initialization."""
        module = PoincareModule(c=1.5, max_norm=0.9)
        assert module.c == 1.5
        assert module.max_norm == 0.9
        assert module.manifold is not None

    def test_various_parameters(self):
        """Test various parameter combinations."""
        for c in [0.5, 1.0, 2.0]:
            for max_norm in [0.8, 0.9, 0.95]:
                module = PoincareModule(c=c, max_norm=max_norm)
                assert module.c == c
                assert module.max_norm == max_norm


class TestPoincareModuleDistMethod:
    """Tests for PoincareModule.dist method."""

    def test_dist_method(self, device):
        """Test dist method."""
        module = PoincareModule(c=1.0)
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)

        d = module.dist(x, y)
        assert d.shape == (5,)

    def test_dist_self_zero(self, device):
        """Distance to self should be zero."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)

        d = module.dist(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)

    def test_dist_symmetry(self, device):
        """Distance should be symmetric."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)

        d_xy = module.dist(x, y)
        d_yx = module.dist(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)


class TestPoincareModuleProjMethod:
    """Tests for PoincareModule.proj method."""

    def test_proj_method(self, device):
        """Test proj method."""
        module = PoincareModule(c=1.0, max_norm=0.9)
        x = torch.randn(5, 4, device=device) * 10

        x_proj = module.proj(x)
        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.9 + 1e-5).all()

    def test_proj_preserves_small(self, device):
        """Projection preserves small norm vectors."""
        module = PoincareModule(max_norm=0.95)
        x = torch.randn(5, 4, device=device) * 0.1

        x_proj = module.proj(x)
        assert torch.allclose(x, x_proj, atol=0.1)


class TestPoincareModuleExpLogMethods:
    """Tests for PoincareModule exp/log methods."""

    def test_expmap0_method(self, device):
        """Test expmap0 method."""
        module = PoincareModule()
        v = torch.randn(5, 4, device=device) * 0.5
        z = module.expmap0(v)
        assert z.shape == v.shape
        assert (z.norm(dim=-1) < 1.0).all()

    def test_logmap0_method(self, device):
        """Test logmap0 method."""
        module = PoincareModule()
        z = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        v = module.logmap0(z)
        assert v.shape == z.shape

    def test_exp_log_inverse(self, device):
        """Test exp/log inverse property."""
        module = PoincareModule()
        v = torch.randn(5, 4, device=device) * 0.3

        z = module.expmap0(v)
        v_recovered = module.logmap0(z)

        assert torch.allclose(v, v_recovered, atol=1e-4)


class TestPoincareModuleAddMethod:
    """Tests for PoincareModule.add method."""

    def test_add_method(self, device):
        """Test add (Mobius addition) method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        result = module.add(x, y)
        assert result.shape == x.shape

    def test_add_zero(self, device):
        """Adding zero should preserve input."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        zero = torch.zeros_like(x)
        result = module.add(x, zero)
        assert torch.allclose(result, x, atol=1e-5)

    def test_add_stays_on_ball(self, device):
        """Result should stay on ball."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        y = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        result = module.add(x, y)
        assert (result.norm(dim=-1) < 1.0).all()


class TestPoincareModuleConformalMethod:
    """Tests for PoincareModule.conformal method."""

    def test_conformal_method(self, device):
        """Test conformal factor method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.5)
        lam = module.conformal(x)
        assert lam.shape == (5, 1)

    def test_conformal_at_origin(self, device):
        """Conformal factor at origin should be 2."""
        module = PoincareModule(c=1.0)
        x = torch.zeros(5, 4, device=device)
        lam = module.conformal(x)
        expected = torch.full((5, 1), 2.0, device=device)
        assert torch.allclose(lam, expected, atol=1e-5)

    def test_conformal_positive(self, device):
        """Conformal factor should always be positive."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(10, 4, device=device) * 0.5)
        lam = module.conformal(x)
        assert (lam > 0).all()


class TestPoincareModuleTransportMethod:
    """Tests for PoincareModule.transport method."""

    def test_transport_method(self, device):
        """Test parallel transport method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        y = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        v = torch.randn(5, 4, device=device) * 0.1
        result = module.transport(x, y, v)
        assert result.shape == v.shape

    def test_transport_to_same(self, device):
        """Transport to same point preserves vector."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4, device=device) * 0.3)
        v = torch.randn(5, 4, device=device) * 0.1
        result = module.transport(x, x, v)
        assert torch.allclose(result, v, atol=1e-4)


class TestPoincareModuleAsModule:
    """Tests for PoincareModule as PyTorch module."""

    def test_is_nn_module(self):
        """Should be a valid nn.Module."""
        import torch.nn as nn
        module = PoincareModule()
        assert isinstance(module, nn.Module)

    def test_has_minimal_params(self):
        """Module should have minimal parameters (curvature only)."""
        module = PoincareModule()
        params = list(module.parameters())
        # Module may have curvature as a parameter
        assert len(params) <= 1

    def test_to_device(self, device):
        """Module should be movable to device."""
        module = PoincareModule()
        module.to(device)
        # Should not error
