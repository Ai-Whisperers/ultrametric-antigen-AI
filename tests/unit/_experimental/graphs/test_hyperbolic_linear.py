# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HyperbolicLinear layer.

Tests cover:
- Forward pass
- Output shape
- Output stays in Poincare ball
- Gradient flow
- Different configurations
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.graphs import HyperbolicLinear


class TestHyperbolicLinearInit:
    """Tests for HyperbolicLinear initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = HyperbolicLinear(in_features=16, out_features=8)

        assert layer.in_features == 16
        assert layer.out_features == 8
        assert layer.curvature == 1.0

    def test_custom_curvature(self):
        """Test initialization with custom curvature."""
        layer = HyperbolicLinear(in_features=16, out_features=8, curvature=2.0)
        assert layer.curvature == 2.0

    def test_no_bias(self):
        """Test initialization without bias."""
        layer = HyperbolicLinear(in_features=16, out_features=8, bias=False)
        assert layer.linear.bias is None

    def test_with_bias(self):
        """Test initialization with bias."""
        layer = HyperbolicLinear(in_features=16, out_features=8, bias=True)
        assert layer.linear.bias is not None


class TestHyperbolicLinearForward:
    """Tests for HyperbolicLinear forward pass."""

    def test_output_shape(self, hyperbolic_linear, poincare_points):
        """Test output has correct shape."""
        # hyperbolic_linear is 16 -> 8
        output = hyperbolic_linear(poincare_points)

        assert output.shape == (32, 8)

    def test_output_in_ball(self, hyperbolic_linear, poincare_points):
        """Test output stays inside Poincare ball."""
        output = hyperbolic_linear(poincare_points)

        norms = output.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_single_sample(self, hyperbolic_linear, device):
        """Test with single sample."""
        x = torch.randn(1, 16, device=device) * 0.3
        output = hyperbolic_linear.to(device)(x)

        assert output.shape == (1, 8)
        assert output.norm() < 1.0

    def test_batch_processing(self, device):
        """Test batch processing."""
        layer = HyperbolicLinear(32, 16).to(device)
        x = torch.randn(64, 32, device=device) * 0.2

        output = layer(x)

        assert output.shape == (64, 16)

    def test_output_finite(self, hyperbolic_linear, poincare_points):
        """Test output is always finite."""
        output = hyperbolic_linear(poincare_points)
        assert torch.isfinite(output).all()


class TestHyperbolicLinearGradients:
    """Tests for gradient flow through HyperbolicLinear."""

    def test_gradients_flow(self, device):
        """Test gradients flow through the layer."""
        layer = HyperbolicLinear(16, 8).to(device)
        # Create leaf tensor with requires_grad
        x = (torch.randn(4, 16, device=device) * 0.2).requires_grad_(True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_parameter_gradients(self, device):
        """Test parameters receive gradients."""
        layer = HyperbolicLinear(16, 8).to(device)
        x = torch.randn(4, 16, device=device) * 0.2

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check weight gradients
        assert layer.linear.weight.grad is not None
        assert torch.isfinite(layer.linear.weight.grad).all()

    def test_multiple_forward_backward(self, device):
        """Test multiple forward-backward passes."""
        layer = HyperbolicLinear(16, 8).to(device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        for _ in range(5):
            x = torch.randn(4, 16, device=device) * 0.2
            optimizer.zero_grad()

            output = layer(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()

            # Output should still be valid
            with torch.no_grad():
                test_output = layer(x)
                assert torch.isfinite(test_output).all()
                assert (test_output.norm(dim=-1) < 1.0).all()


class TestHyperbolicLinearEdgeCases:
    """Tests for edge cases."""

    def test_small_input(self, hyperbolic_linear, device):
        """Test with very small input."""
        x = torch.randn(4, 16, device=device) * 1e-8
        output = hyperbolic_linear.to(device)(x)

        assert torch.isfinite(output).all()

    def test_boundary_input(self, device):
        """Test with input near ball boundary."""
        layer = HyperbolicLinear(8, 4).to(device)

        x = torch.randn(4, 8, device=device)
        x = x / x.norm(dim=-1, keepdim=True) * 0.95

        output = layer(x)

        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) < 1.0).all()

    def test_zero_input(self, hyperbolic_linear, device):
        """Test with zero input (origin)."""
        x = torch.zeros(4, 16, device=device)
        output = hyperbolic_linear.to(device)(x)

        # Origin maps through zero tangent vector
        assert torch.isfinite(output).all()


class TestHyperbolicLinearConfigurations:
    """Tests for different layer configurations."""

    @pytest.mark.parametrize("in_features,out_features", [
        (8, 8),    # Same dimension
        (16, 8),   # Compression
        (8, 16),   # Expansion
        (64, 4),   # Large compression
    ])
    def test_dimension_configurations(self, in_features, out_features, device):
        """Test various input/output dimension configurations."""
        layer = HyperbolicLinear(in_features, out_features).to(device)
        x = torch.randn(4, in_features, device=device) * 0.2

        output = layer(x)

        assert output.shape == (4, out_features)
        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) < 1.0).all()

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0, 4.0])
    def test_curvature_configurations(self, curvature, device):
        """Test with different curvatures."""
        layer = HyperbolicLinear(16, 8, curvature=curvature).to(device)
        x = torch.randn(4, 16, device=device) * 0.2

        output = layer(x)

        assert torch.isfinite(output).all()
        # Check output is inside ball for given curvature
        import math
        max_norm = 0.95 / math.sqrt(curvature)
        assert (output.norm(dim=-1) < max_norm + 0.1).all()


class TestHyperbolicLinearIntegration:
    """Integration tests with other layers."""

    def test_stacked_layers(self, device):
        """Test stacking multiple HyperbolicLinear layers."""
        layers = nn.Sequential(
            HyperbolicLinear(32, 16),
            HyperbolicLinear(16, 8),
            HyperbolicLinear(8, 4),
        ).to(device)

        x = torch.randn(4, 32, device=device) * 0.2
        output = layers(x)

        assert output.shape == (4, 4)
        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) < 1.0).all()

    def test_with_euclidean_layers(self, device):
        """Test mixing with Euclidean layers."""
        class MixedNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.euclidean = nn.Linear(32, 16)
                self.hyperbolic = HyperbolicLinear(16, 8)
                self.output = nn.Linear(8, 4)

            def forward(self, x):
                x = torch.relu(self.euclidean(x))
                x = x * 0.1  # Scale down to stay in ball
                x = self.hyperbolic(x)
                return self.output(x)

        model = MixedNetwork().to(device)
        x = torch.randn(4, 32, device=device)
        output = model(x)

        assert output.shape == (4, 4)
        assert torch.isfinite(output).all()
