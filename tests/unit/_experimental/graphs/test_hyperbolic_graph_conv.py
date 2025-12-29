# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HyperbolicGraphConv layer.

Tests cover:
- Forward pass
- Output shape
- Attention mechanism
- Message aggregation
- Gradient flow
- Edge cases
"""

from __future__ import annotations

import pytest
import torch

from src.graphs import HyperbolicGraphConv


class TestHyperbolicGraphConvInit:
    """Tests for HyperbolicGraphConv initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = HyperbolicGraphConv(in_channels=16, out_channels=8)

        assert layer.in_channels == 16
        assert layer.out_channels == 8
        assert layer.curvature == 1.0
        assert layer.use_attention is False

    def test_with_attention(self):
        """Test initialization with attention."""
        layer = HyperbolicGraphConv(in_channels=16, out_channels=8, use_attention=True, heads=4)

        assert layer.use_attention is True
        assert layer.heads == 4
        assert layer.att_linear is not None

    def test_custom_curvature(self):
        """Test initialization with custom curvature."""
        layer = HyperbolicGraphConv(in_channels=16, out_channels=8, curvature=2.0)
        assert layer.curvature == 2.0


class TestHyperbolicGraphConvForward:
    """Tests for HyperbolicGraphConv forward pass."""

    def test_output_shape(self, hyperbolic_graph_conv, small_graph):
        """Test output has correct shape."""
        layer = hyperbolic_graph_conv.to(small_graph["x"].device)
        output = layer(small_graph["x"], small_graph["edge_index"])

        assert output.shape == (small_graph["n_nodes"], 16)

    def test_output_in_ball(self, hyperbolic_graph_conv, small_graph):
        """Test output stays inside Poincare ball."""
        layer = hyperbolic_graph_conv.to(small_graph["x"].device)
        output = layer(small_graph["x"], small_graph["edge_index"])

        norms = output.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_output_finite(self, hyperbolic_graph_conv, small_graph):
        """Test output is always finite."""
        layer = hyperbolic_graph_conv.to(small_graph["x"].device)
        output = layer(small_graph["x"], small_graph["edge_index"])

        assert torch.isfinite(output).all()

    def test_with_attention(self, hyperbolic_graph_conv_attention, small_graph):
        """Test forward pass with attention."""
        layer = hyperbolic_graph_conv_attention.to(small_graph["x"].device)
        output = layer(small_graph["x"], small_graph["edge_index"])

        assert output.shape == (small_graph["n_nodes"], 16)
        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) < 1.0).all()

    def test_medium_graph(self, hyperbolic_graph_conv, medium_graph):
        """Test with medium-sized graph."""
        layer = hyperbolic_graph_conv.to(medium_graph["x"].device)
        # Resize layer for different input channels
        layer = HyperbolicGraphConv(32, 32).to(medium_graph["x"].device)
        output = layer(medium_graph["x"], medium_graph["edge_index"])

        assert output.shape == (medium_graph["n_nodes"], 32)
        assert torch.isfinite(output).all()


class TestHyperbolicGraphConvGradients:
    """Tests for gradient flow through HyperbolicGraphConv."""

    def test_gradients_flow_to_input(self, small_graph):
        """Test gradients flow to input features."""
        layer = HyperbolicGraphConv(16, 16).to(small_graph["x"].device)

        x = small_graph["x"].clone().requires_grad_(True)
        output = layer(x, small_graph["edge_index"])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_parameter_gradients(self, small_graph):
        """Test parameters receive gradients."""
        layer = HyperbolicGraphConv(16, 16).to(small_graph["x"].device)

        output = layer(small_graph["x"], small_graph["edge_index"])
        loss = output.sum()
        loss.backward()

        # Check msg_linear gradients
        assert layer.msg_linear.linear.weight.grad is not None
        assert torch.isfinite(layer.msg_linear.linear.weight.grad).all()

    def test_attention_gradients(self, small_graph):
        """Test attention parameters receive gradients."""
        layer = HyperbolicGraphConv(16, 16, use_attention=True).to(small_graph["x"].device)

        output = layer(small_graph["x"], small_graph["edge_index"])
        loss = output.sum()
        loss.backward()

        # Check attention layer gradients
        assert layer.att_linear.weight.grad is not None

    def test_training_step(self, small_graph):
        """Test a complete training step."""
        layer = HyperbolicGraphConv(16, 16).to(small_graph["x"].device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        for _ in range(3):
            optimizer.zero_grad()
            output = layer(small_graph["x"], small_graph["edge_index"])
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Output should still be valid after training
        with torch.no_grad():
            output = layer(small_graph["x"], small_graph["edge_index"])
            assert torch.isfinite(output).all()


class TestHyperbolicGraphConvEdgeCases:
    """Tests for edge cases."""

    def test_isolated_node(self, device):
        """Test graph with isolated node."""
        layer = HyperbolicGraphConv(8, 8).to(device)

        # 5 nodes, but node 4 is isolated
        x = torch.randn(5, 8, device=device) * 0.2
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], device=device)

        output = layer(x, edge_index)

        assert output.shape == (5, 8)
        assert torch.isfinite(output).all()

    def test_single_node(self, device):
        """Test single-node graph."""
        layer = HyperbolicGraphConv(8, 8).to(device)

        x = torch.randn(1, 8, device=device) * 0.2
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        output = layer(x, edge_index)

        assert output.shape == (1, 8)
        assert torch.isfinite(output).all()

    def test_self_loops(self, device):
        """Test graph with self-loops."""
        layer = HyperbolicGraphConv(8, 8).to(device)

        x = torch.randn(4, 8, device=device) * 0.2
        edge_index = torch.tensor([[0, 1, 2, 0, 1, 2, 3],
                                   [1, 2, 3, 0, 1, 2, 3]], device=device)  # Self-loops

        output = layer(x, edge_index)

        assert output.shape == (4, 8)
        assert torch.isfinite(output).all()

    def test_dense_graph(self, device):
        """Test densely connected graph."""
        layer = HyperbolicGraphConv(8, 8).to(device)

        n_nodes = 10
        x = torch.randn(n_nodes, 8, device=device) * 0.2

        # Fully connected
        src = []
        dst = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], device=device)

        output = layer(x, edge_index)

        assert output.shape == (n_nodes, 8)
        assert torch.isfinite(output).all()


class TestHyperbolicGraphConvConfigurations:
    """Tests for different layer configurations."""

    @pytest.mark.parametrize("in_channels,out_channels", [
        (8, 8),
        (16, 8),
        (8, 16),
        (32, 4),
    ])
    def test_channel_configurations(self, in_channels, out_channels, device):
        """Test various channel configurations."""
        layer = HyperbolicGraphConv(in_channels, out_channels).to(device)

        n_nodes = 10
        x = torch.randn(n_nodes, in_channels, device=device) * 0.2
        edge_index = torch.randint(0, n_nodes, (2, 30), device=device)

        output = layer(x, edge_index)

        assert output.shape == (n_nodes, out_channels)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("heads", [1, 2, 4, 8])
    def test_attention_head_configurations(self, heads, device):
        """Test with different numbers of attention heads."""
        layer = HyperbolicGraphConv(16, 16, use_attention=True, heads=heads).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = layer(x, edge_index)

        assert output.shape == (10, 16)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout_configurations(self, dropout, device):
        """Test with different dropout rates."""
        layer = HyperbolicGraphConv(16, 16, dropout=dropout).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        # Test in training mode
        layer.train()
        output_train = layer(x, edge_index)
        assert torch.isfinite(output_train).all()

        # Test in eval mode
        layer.eval()
        output_eval = layer(x, edge_index)
        assert torch.isfinite(output_eval).all()


class TestHyperbolicGraphConvStacking:
    """Tests for stacking multiple graph conv layers."""

    def test_two_layer_stack(self, small_graph):
        """Test two stacked layers."""
        device = small_graph["x"].device
        layer1 = HyperbolicGraphConv(16, 32).to(device)
        layer2 = HyperbolicGraphConv(32, 16).to(device)

        h = layer1(small_graph["x"], small_graph["edge_index"])
        output = layer2(h, small_graph["edge_index"])

        assert output.shape == (small_graph["n_nodes"], 16)
        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) < 1.0).all()

    def test_deep_stack(self, small_graph):
        """Test deeper stack of layers."""
        device = small_graph["x"].device

        layers = [HyperbolicGraphConv(16, 16).to(device) for _ in range(5)]

        h = small_graph["x"]
        for layer in layers:
            h = layer(h, small_graph["edge_index"])

        assert h.shape == (small_graph["n_nodes"], 16)
        assert torch.isfinite(h).all()
        assert (h.norm(dim=-1) < 1.0).all()
