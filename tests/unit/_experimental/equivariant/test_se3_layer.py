# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for SE(3)-equivariant layers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.equivariant import (
    EGNN,
    EGNNLayer,
    SE3Layer,
    SE3Linear,
    SE3MessagePassing,
    SE3Transformer,
)


class TestSE3Linear:
    """Tests for SE3Linear layer."""

    def test_init_scalar_only(self):
        """Test initialization with scalar features only."""
        layer = SE3Linear(scalar_in=32, scalar_out=64)
        assert layer.scalar_in == 32
        assert layer.scalar_out == 64

    def test_init_with_vectors(self):
        """Test initialization with vector features."""
        layer = SE3Linear(scalar_in=32, scalar_out=64, vector_in=8, vector_out=16)
        assert layer.vector_in == 8
        assert layer.vector_out == 16

    def test_forward_scalar_only(self, device):
        """Test forward with scalar features only."""
        layer = SE3Linear(scalar_in=32, scalar_out=64)
        layer = layer.to(device)

        scalars = torch.randn(10, 32, device=device)
        out_scalars, out_vectors = layer(scalars)

        assert out_scalars.shape == (10, 64)
        assert out_vectors is None

    def test_forward_with_vectors(self, device):
        """Test forward with both scalar and vector features."""
        layer = SE3Linear(scalar_in=16, scalar_out=32, vector_in=4, vector_out=8)
        layer = layer.to(device)

        scalars = torch.randn(10, 16, device=device)
        vectors = torch.randn(10, 4, 3, device=device)

        out_scalars, out_vectors = layer(scalars, vectors)

        assert out_scalars.shape == (10, 32)
        assert out_vectors.shape == (10, 8, 3)

    def test_batched_forward(self, device):
        """Test batched forward pass."""
        layer = SE3Linear(scalar_in=16, scalar_out=32, vector_in=4, vector_out=8)
        layer = layer.to(device)

        scalars = torch.randn(4, 10, 16, device=device)
        vectors = torch.randn(4, 10, 4, 3, device=device)

        out_scalars, out_vectors = layer(scalars, vectors)

        assert out_scalars.shape == (4, 10, 32)
        assert out_vectors.shape == (4, 10, 8, 3)


class TestSE3MessagePassing:
    """Tests for SE3MessagePassing layer."""

    def test_init(self):
        """Test initialization."""
        layer = SE3MessagePassing(hidden_dim=64)
        assert layer.hidden_dim == 64

    def test_init_with_coord_update(self):
        """Test initialization with coordinate update."""
        layer = SE3MessagePassing(hidden_dim=64, update_coords=True)
        assert layer.update_coords is True

    def test_forward_shape(self, device, simple_edge_index):
        """Test forward pass shape."""
        layer = SE3MessagePassing(hidden_dim=32)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 32, device=device)
        pos = torch.randn(n_nodes, 3, device=device)

        h_new, pos_new = layer(h, pos, simple_edge_index)

        assert h_new.shape == (n_nodes, 32)
        assert pos_new.shape == (n_nodes, 3)

    def test_coords_unchanged_without_update(self, device, simple_edge_index):
        """Test coordinates unchanged when update_coords=False."""
        layer = SE3MessagePassing(hidden_dim=32, update_coords=False)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 32, device=device)
        pos = torch.randn(n_nodes, 3, device=device)

        _, pos_new = layer(h, pos, simple_edge_index)

        assert torch.allclose(pos, pos_new)


class TestSE3Layer:
    """Tests for full SE3Layer."""

    def test_init(self):
        """Test initialization."""
        layer = SE3Layer(hidden_dim=64)
        assert layer.hidden_dim == 64

    def test_forward_shape(self, device, simple_edge_index):
        """Test forward pass shape."""
        layer = SE3Layer(hidden_dim=32)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 32, device=device)
        pos = torch.randn(n_nodes, 3, device=device)

        h_new, pos_new = layer(h, pos, simple_edge_index)

        assert h_new.shape == (n_nodes, 32)
        assert pos_new.shape == (n_nodes, 3)

    def test_gradient_flow(self, device, simple_edge_index):
        """Test gradient flows through layer."""
        layer = SE3Layer(hidden_dim=16)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 16, device=device, requires_grad=True)
        pos = torch.randn(n_nodes, 3, device=device)

        h_new, _ = layer(h, pos, simple_edge_index)
        loss = h_new.sum()
        loss.backward()

        assert h.grad is not None
        assert h.grad.shape == h.shape


class TestSE3Transformer:
    """Tests for SE3Transformer model."""

    def test_init(self):
        """Test initialization."""
        model = SE3Transformer(
            in_features=16,
            hidden_dim=64,
            out_features=10,
            n_layers=3,
        )
        assert len(model.layers) == 3

    def test_forward_shape(self, graph_data, device):
        """Test forward pass shape."""
        model = SE3Transformer(
            in_features=16,
            hidden_dim=32,
            out_features=5,
            n_layers=2,
        )
        model = model.to(device)

        x = graph_data["x"]
        pos = graph_data["pos"]
        edge_index = graph_data["edge_index"]

        out, pos_final = model(x, pos, edge_index)

        assert out.shape == (graph_data["n_nodes"], 5)
        assert pos_final.shape == (graph_data["n_nodes"], 3)

    def test_with_coord_update(self, graph_data, device):
        """Test with coordinate updates enabled."""
        model = SE3Transformer(
            in_features=16,
            hidden_dim=32,
            out_features=5,
            n_layers=2,
            update_coords=True,
        )
        model = model.to(device)

        x = graph_data["x"]
        pos = graph_data["pos"].clone()
        edge_index = graph_data["edge_index"]

        _, pos_final = model(x, pos, edge_index)

        # Positions should be updated
        # (May or may not be different depending on weights)
        assert pos_final.shape == pos.shape


class TestEGNNLayer:
    """Tests for single EGNN layer."""

    def test_init(self):
        """Test initialization."""
        layer = EGNNLayer(hidden_dim=64)
        assert layer.hidden_dim == 64

    def test_forward_shape(self, device, simple_edge_index):
        """Test forward pass shape."""
        layer = EGNNLayer(hidden_dim=32)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 32, device=device)
        pos = torch.randn(n_nodes, 3, device=device)

        h_new, pos_new = layer(h, pos, simple_edge_index)

        assert h_new.shape == (n_nodes, 32)
        assert pos_new.shape == (n_nodes, 3)

    def test_without_coord_update(self, device, simple_edge_index):
        """Test layer without coordinate update."""
        layer = EGNNLayer(hidden_dim=32, update_coords=False)
        layer = layer.to(device)

        n_nodes = 4
        h = torch.randn(n_nodes, 32, device=device)
        pos = torch.randn(n_nodes, 3, device=device)

        _, pos_new = layer(h, pos, simple_edge_index)

        assert torch.allclose(pos, pos_new)


class TestEGNN:
    """Tests for full EGNN model."""

    def test_init(self):
        """Test initialization."""
        model = EGNN(
            in_features=16,
            hidden_dim=64,
            out_features=10,
            n_layers=3,
        )
        assert len(model.layers) == 3

    def test_forward_shape(self, graph_data, device):
        """Test forward pass shape."""
        model = EGNN(
            in_features=16,
            hidden_dim=32,
            out_features=5,
            n_layers=2,
        )
        model = model.to(device)

        x = graph_data["x"]
        pos = graph_data["pos"]
        edge_index = graph_data["edge_index"]

        out, pos_final = model(x, pos, edge_index)

        assert out.shape == (graph_data["n_nodes"], 5)
        assert pos_final.shape == (graph_data["n_nodes"], 3)

    def test_translation_equivariance(self, graph_data, device):
        """Test that EGNN output is translation equivariant."""
        model = EGNN(
            in_features=16,
            hidden_dim=32,
            out_features=5,
            n_layers=2,
            update_coords=True,
        )
        model = model.to(device)
        model.eval()

        x = graph_data["x"]
        pos = graph_data["pos"]
        edge_index = graph_data["edge_index"]

        # Random translation
        translation = torch.randn(1, 3, device=device)

        with torch.no_grad():
            out1, pos1 = model(x, pos, edge_index)
            out2, pos2 = model(x, pos + translation, edge_index)

        # Output features should be the same
        assert torch.allclose(out1, out2, atol=1e-5)

        # Positions should differ by translation
        assert torch.allclose(pos2 - pos1, translation.expand_as(pos1), atol=1e-4)

    def test_without_residual(self, graph_data, device):
        """Test EGNN without residual connections."""
        model = EGNN(
            in_features=16,
            hidden_dim=32,
            out_features=5,
            n_layers=2,
            residual=False,
        )
        model = model.to(device)

        out, _ = model(graph_data["x"], graph_data["pos"], graph_data["edge_index"])
        assert out.shape == (graph_data["n_nodes"], 5)
