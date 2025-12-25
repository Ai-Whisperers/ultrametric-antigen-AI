# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for manifold parameters and optimizers in src/geometry/poincare.py.

Tests ManifoldParameter creation and Riemannian optimizer factory.
"""

import pytest
import torch
from geoopt import ManifoldParameter, ManifoldTensor
from src.geometry.poincare import (
    create_manifold_parameter,
    create_manifold_tensor,
    get_riemannian_optimizer,
)


# =============================================================================
# ManifoldParameter Tests
# =============================================================================


class TestCreateManifoldParameter:
    """Tests for create_manifold_parameter function."""

    def test_creates_manifold_parameter(self, device):
        """Test creating a learnable manifold parameter."""
        data = torch.randn(10, 4, device=device)
        param = create_manifold_parameter(data, c=1.0, requires_grad=True)

        assert isinstance(param, ManifoldParameter)
        assert param.requires_grad

    def test_data_projected_onto_manifold(self, device):
        """Data should be projected onto manifold."""
        data = torch.randn(10, 4, device=device) * 2  # Potentially outside ball
        param = create_manifold_parameter(data, c=1.0, requires_grad=True)

        # All norms should be < 1
        assert (param.data.norm(dim=-1) < 1.0).all()

    def test_create_parameter_no_grad(self, device):
        """Test creating parameter without gradients."""
        data = torch.randn(10, 4, device=device)
        param = create_manifold_parameter(data, c=1.0, requires_grad=False)

        assert not param.requires_grad

    def test_various_shapes(self, device):
        """Test with various shapes."""
        for shape in [(5, 4), (10, 8), (20, 16)]:
            data = torch.randn(*shape, device=device)
            param = create_manifold_parameter(data, c=1.0)
            assert param.shape == shape

    def test_different_curvatures(self, device):
        """Test with different curvatures."""
        # Use small data that's already on the ball
        data = torch.randn(10, 4, device=device) * 0.3

        for c in [0.5, 1.0, 2.0]:
            param = create_manifold_parameter(data, c=c)
            # Parameter should be a ManifoldParameter
            assert isinstance(param, ManifoldParameter)


class TestCreateManifoldTensor:
    """Tests for create_manifold_tensor function."""

    def test_creates_manifold_tensor(self, device):
        """Test creating a non-learnable manifold tensor."""
        data = torch.randn(10, 4, device=device)
        tensor = create_manifold_tensor(data, c=1.0)

        assert isinstance(tensor, ManifoldTensor)

    def test_data_projected(self, device):
        """Data should be projected onto manifold."""
        data = torch.randn(10, 4, device=device) * 2
        tensor = create_manifold_tensor(data, c=1.0)

        assert (tensor.data.norm(dim=-1) < 1.0).all()

    def test_various_shapes(self, device):
        """Test with various shapes."""
        for shape in [(5, 4), (10, 8), (20, 16)]:
            data = torch.randn(*shape, device=device)
            tensor = create_manifold_tensor(data, c=1.0)
            assert tensor.shape == shape


# =============================================================================
# Riemannian Optimizer Tests
# =============================================================================


class TestGetRiemannianOptimizer:
    """Tests for get_riemannian_optimizer function."""

    def test_create_adam_optimizer(self):
        """Test creating RiemannianAdam optimizer."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]
        opt = get_riemannian_optimizer(params, lr=1e-3, optimizer_type="adam")

        assert opt is not None
        assert opt.defaults["lr"] == 1e-3

    def test_create_sgd_optimizer(self):
        """Test creating RiemannianSGD optimizer."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]
        opt = get_riemannian_optimizer(params, lr=1e-2, optimizer_type="sgd")

        assert opt is not None
        assert opt.defaults["lr"] == 1e-2

    def test_invalid_optimizer_type(self):
        """Invalid optimizer type should raise error."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]

        with pytest.raises(ValueError):
            get_riemannian_optimizer(params, optimizer_type="invalid")

    def test_various_learning_rates(self):
        """Test with various learning rates."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]

        for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
            opt = get_riemannian_optimizer(params, lr=lr, optimizer_type="adam")
            assert opt.defaults["lr"] == lr

    def test_multiple_parameter_groups(self):
        """Test with multiple parameter groups."""
        params = [
            torch.nn.Parameter(torch.randn(10, 4)),
            torch.nn.Parameter(torch.randn(5, 8)),
        ]
        opt = get_riemannian_optimizer(params, lr=1e-3, optimizer_type="adam")

        assert opt is not None
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 2


class TestRiemannianOptimizerStep:
    """Tests for Riemannian optimizer stepping."""

    def test_adam_step(self, device):
        """Test RiemannianAdam can perform step."""
        param = torch.nn.Parameter(torch.randn(10, 4, device=device) * 0.5)
        opt = get_riemannian_optimizer([param], lr=1e-2, optimizer_type="adam")

        # Create fake gradients
        param.grad = torch.randn_like(param) * 0.1

        # Should be able to step
        opt.step()

    def test_sgd_step(self, device):
        """Test RiemannianSGD can perform step."""
        param = torch.nn.Parameter(torch.randn(10, 4, device=device) * 0.5)
        opt = get_riemannian_optimizer([param], lr=1e-2, optimizer_type="sgd")

        param.grad = torch.randn_like(param) * 0.1

        opt.step()


# =============================================================================
# Integration Tests
# =============================================================================


class TestManifoldParameterOptimization:
    """Integration tests for manifold parameter optimization."""

    def test_parameter_stays_on_manifold(self, device):
        """Parameter should stay on manifold during optimization."""
        data = torch.randn(10, 4, device=device) * 0.5
        param = create_manifold_parameter(data, c=1.0, requires_grad=True)
        opt = get_riemannian_optimizer([param], lr=0.1, optimizer_type="adam")

        # Run a few optimization steps
        for _ in range(10):
            param.grad = torch.randn_like(param) * 0.1
            opt.step()

            # Should still be on manifold
            norms = param.data.norm(dim=-1)
            assert (norms < 1.0).all()

    def test_optimization_changes_parameter(self, device):
        """Optimization should change parameter values."""
        data = torch.randn(10, 4, device=device) * 0.5
        param = create_manifold_parameter(data, c=1.0, requires_grad=True)
        initial_value = param.data.clone()

        opt = get_riemannian_optimizer([param], lr=0.1, optimizer_type="adam")

        param.grad = torch.randn_like(param)
        opt.step()

        # Values should have changed
        assert not torch.allclose(param.data, initial_value)
