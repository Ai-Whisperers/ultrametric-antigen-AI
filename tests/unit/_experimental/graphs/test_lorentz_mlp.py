# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for LorentzMLP class.

Tests cover:
- Forward pass
- Output on hyperboloid
- Gradient flow
- Different configurations
"""

from __future__ import annotations

import pytest
import torch

from src.graphs import LorentzMLP, LorentzOperations


class TestLorentzMLPInit:
    """Tests for LorentzMLP initialization."""

    def test_default_init(self):
        """Test default initialization."""
        mlp = LorentzMLP(
            in_features=16,
            hidden_features=32,
            out_features=8,
        )

        assert mlp.in_features == 16
        assert mlp.out_features == 8
        assert mlp.curvature == 1.0

    def test_custom_layers(self):
        """Test initialization with custom layer count."""
        mlp = LorentzMLP(
            in_features=16,
            hidden_features=32,
            out_features=8,
            n_layers=4,
        )

        # Should have 4 linear layers
        linear_count = sum(1 for layer in mlp.layers if isinstance(layer, torch.nn.Linear))
        assert linear_count == 4

    def test_custom_curvature(self):
        """Test initialization with custom curvature."""
        mlp = LorentzMLP(
            in_features=16,
            hidden_features=32,
            out_features=8,
            curvature=2.0,
        )

        assert mlp.curvature == 2.0


class TestLorentzMLPForward:
    """Tests for LorentzMLP forward pass."""

    def test_output_shape(self, lorentz_mlp, lorentz_points):
        """Test output has correct shape."""
        mlp = lorentz_mlp.to(lorentz_points.device)
        output = mlp(lorentz_points)

        # out_features=8, so output dim = 8 + 1 = 9
        assert output.shape == (32, 9)

    def test_output_on_hyperboloid(self, lorentz_mlp, lorentz_points):
        """Test output lies on hyperboloid."""
        mlp = lorentz_mlp.to(lorentz_points.device)
        lorentz_ops = LorentzOperations(curvature=1.0)

        output = mlp(lorentz_points)

        # Check <x, x>_L = -1/c
        inner = lorentz_ops.minkowski_inner(output, output)
        expected = -1.0 / mlp.curvature

        assert torch.allclose(inner.squeeze(), torch.full((32,), expected, device=output.device), atol=1e-3)

    def test_output_positive_time(self, lorentz_mlp, lorentz_points):
        """Test output has positive time component."""
        mlp = lorentz_mlp.to(lorentz_points.device)
        output = mlp(lorentz_points)

        assert (output[..., 0] > 0).all()

    def test_output_finite(self, lorentz_mlp, lorentz_points):
        """Test output is always finite."""
        mlp = lorentz_mlp.to(lorentz_points.device)
        output = mlp(lorentz_points)

        assert torch.isfinite(output).all()


class TestLorentzMLPGradients:
    """Tests for gradient flow through LorentzMLP."""

    def test_gradients_flow(self, lorentz_ops, device):
        """Test gradients flow through the MLP."""
        mlp = LorentzMLP(8, 16, 4, n_layers=2).to(device)

        x = torch.randn(4, 9, device=device, requires_grad=True)
        x = lorentz_ops.project_to_hyperboloid(x)

        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check linear layer gradients
        for layer in mlp.layers:
            if isinstance(layer, torch.nn.Linear):
                assert layer.weight.grad is not None
                assert torch.isfinite(layer.weight.grad).all()

    def test_training_step(self, lorentz_ops, device):
        """Test complete training step."""
        mlp = LorentzMLP(8, 16, 4, n_layers=2).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

        for _ in range(3):
            x = torch.randn(4, 9, device=device)
            x = lorentz_ops.project_to_hyperboloid(x)

            optimizer.zero_grad()
            output = mlp(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Output should still be valid
        with torch.no_grad():
            test_x = torch.randn(2, 9, device=device)
            test_x = lorentz_ops.project_to_hyperboloid(test_x)
            test_out = mlp(test_x)
            assert torch.isfinite(test_out).all()


class TestLorentzMLPConfigurations:
    """Tests for different configurations."""

    @pytest.mark.parametrize("in_features,hidden,out_features", [
        (8, 16, 4),
        (16, 32, 8),
        (32, 64, 16),
        (4, 8, 4),
    ])
    def test_dimension_configurations(self, in_features, hidden, out_features, device):
        """Test various dimension configurations."""
        lorentz_ops = LorentzOperations()
        mlp = LorentzMLP(in_features, hidden, out_features).to(device)

        x = torch.randn(4, in_features + 1, device=device)
        x = lorentz_ops.project_to_hyperboloid(x)

        output = mlp(x)

        assert output.shape == (4, out_features + 1)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_layer_configurations(self, n_layers, device):
        """Test different layer depths."""
        lorentz_ops = LorentzOperations()
        mlp = LorentzMLP(8, 16, 4, n_layers=n_layers).to(device)

        x = torch.randn(4, 9, device=device)
        x = lorentz_ops.project_to_hyperboloid(x)

        output = mlp(x)

        assert output.shape == (4, 5)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3])
    def test_dropout_configurations(self, dropout, device):
        """Test different dropout rates."""
        lorentz_ops = LorentzOperations()
        mlp = LorentzMLP(8, 16, 4, dropout=dropout).to(device)

        x = torch.randn(4, 9, device=device)
        x = lorentz_ops.project_to_hyperboloid(x)

        # Training mode
        mlp.train()
        output_train = mlp(x)
        assert torch.isfinite(output_train).all()

        # Eval mode
        mlp.eval()
        output_eval = mlp(x)
        assert torch.isfinite(output_eval).all()


class TestLorentzMLPEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self, lorentz_ops, device):
        """Test with single sample."""
        mlp = LorentzMLP(8, 16, 4).to(device)

        x = torch.randn(1, 9, device=device)
        x = lorentz_ops.project_to_hyperboloid(x)

        output = mlp(x)

        assert output.shape == (1, 5)
        assert torch.isfinite(output).all()

    def test_large_batch(self, lorentz_ops, device):
        """Test with large batch."""
        mlp = LorentzMLP(8, 16, 4).to(device)

        x = torch.randn(256, 9, device=device)
        x = lorentz_ops.project_to_hyperboloid(x)

        output = mlp(x)

        assert output.shape == (256, 5)
        assert torch.isfinite(output).all()
