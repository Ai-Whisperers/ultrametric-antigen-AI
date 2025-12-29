# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for SO(3)-equivariant layers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.equivariant import (
    RadialBasisFunctions,
    SmoothCutoff,
    SO3Convolution,
    SO3GNN,
    SO3Layer,
    SO3Linear,
)


class TestRadialBasisFunctions:
    """Tests for RadialBasisFunctions."""

    def test_init_gaussian(self):
        """Test Gaussian RBF initialization."""
        rbf = RadialBasisFunctions(n_rbf=16, cutoff=5.0, rbf_type="gaussian")
        assert rbf.n_rbf == 16
        assert rbf.cutoff == 5.0

    def test_init_bessel(self):
        """Test Bessel RBF initialization."""
        rbf = RadialBasisFunctions(n_rbf=8, cutoff=4.0, rbf_type="bessel")
        assert rbf.n_rbf == 8

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        rbf = RadialBasisFunctions(n_rbf=16)
        distances = torch.rand(100, device=device) * 5.0
        result = rbf(distances)
        assert result.shape == (100, 16)

    def test_forward_batched(self, device):
        """Test forward with batched input."""
        rbf = RadialBasisFunctions(n_rbf=8)
        distances = torch.rand(4, 20, device=device) * 5.0
        result = rbf(distances)
        assert result.shape == (4, 20, 8)

    def test_gaussian_rbf_properties(self, device):
        """Test Gaussian RBF values are in valid range."""
        rbf = RadialBasisFunctions(n_rbf=16, rbf_type="gaussian")
        distances = torch.rand(50, device=device) * 5.0
        result = rbf(distances)

        # Gaussian RBF should be in [0, 1] (can be very close to 0)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestSmoothCutoff:
    """Tests for SmoothCutoff."""

    def test_init(self):
        """Test initialization."""
        cutoff = SmoothCutoff(cutoff=5.0)
        assert cutoff.cutoff == 5.0

    def test_within_cutoff(self, device):
        """Test values within cutoff."""
        cutoff = SmoothCutoff(cutoff=5.0)
        distances = torch.tensor([0.0, 1.0, 2.0], device=device)
        result = cutoff(distances)

        # Should be > 0 within cutoff
        assert torch.all(result > 0)
        # At distance 0, should be 1
        assert torch.isclose(result[0], torch.tensor(1.0, device=device))

    def test_beyond_cutoff(self, device):
        """Test values beyond cutoff."""
        cutoff = SmoothCutoff(cutoff=5.0)
        distances = torch.tensor([6.0, 7.0, 10.0], device=device)
        result = cutoff(distances)

        # Should be 0 beyond cutoff
        assert torch.allclose(result, torch.zeros_like(result))

    def test_smooth_transition(self, device):
        """Test smooth transition near cutoff."""
        cutoff = SmoothCutoff(cutoff=5.0)
        distances = torch.linspace(0, 5, 50, device=device)
        result = cutoff(distances)

        # Should be monotonically decreasing
        diffs = result[1:] - result[:-1]
        assert torch.all(diffs <= 0)


class TestSO3Linear:
    """Tests for SO3Linear layer."""

    def test_init(self):
        """Test initialization."""
        layer = SO3Linear(in_features=32, out_features=64, lmax_in=2, lmax_out=2)
        assert layer.in_features == 32
        assert layer.out_features == 64

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        layer = SO3Linear(in_features=32, out_features=64, lmax_in=2, lmax_out=2)
        layer = layer.to(device)

        n_harmonics = 9  # (2+1)^2
        x = torch.randn(4, 10, 32, n_harmonics, device=device)
        result = layer(x)

        assert result.shape == (4, 10, 64, n_harmonics)

    def test_forward_without_bias(self, device):
        """Test forward without bias."""
        layer = SO3Linear(in_features=16, out_features=32, bias=False)
        layer = layer.to(device)

        n_harmonics = 9
        x = torch.randn(2, 5, 16, n_harmonics, device=device)
        result = layer(x)

        assert result.shape == (2, 5, 32, n_harmonics)
        assert layer.bias is None


class TestSO3Convolution:
    """Tests for SO3Convolution layer."""

    def test_init(self):
        """Test initialization."""
        conv = SO3Convolution(in_features=32, out_features=64, lmax=2)
        assert conv.in_features == 32
        assert conv.out_features == 64
        assert conv.lmax == 2

    def test_forward_shape(self, device, simple_edge_index):
        """Test forward pass shape."""
        conv = SO3Convolution(in_features=16, out_features=32, lmax=2)
        conv = conv.to(device)

        n_nodes = 4
        n_harmonics = 9
        x = torch.randn(n_nodes, 16, n_harmonics, device=device)
        edge_vec = torch.randn(simple_edge_index.shape[1], 3, device=device)

        result = conv(x, simple_edge_index, edge_vec)
        assert result.shape == (n_nodes, 32, n_harmonics)


class TestSO3Layer:
    """Tests for full SO3Layer."""

    def test_init(self):
        """Test initialization."""
        layer = SO3Layer(in_features=32, out_features=64, lmax=2)
        assert layer.in_features == 32
        assert layer.out_features == 64
        assert layer.lmax == 2

    def test_forward_scalar_input(self, device, simple_edge_index):
        """Test forward with scalar-only input."""
        layer = SO3Layer(in_features=16, out_features=32, lmax=2)
        layer = layer.to(device)

        n_nodes = 4
        x = torch.randn(n_nodes, 16, device=device)  # Scalar features only
        edge_vec = torch.randn(simple_edge_index.shape[1], 3, device=device)

        result = layer(x, simple_edge_index, edge_vec)
        n_harmonics = 9
        assert result.shape == (n_nodes, 32, n_harmonics)

    def test_forward_spherical_input(self, device, simple_edge_index):
        """Test forward with spherical tensor input."""
        layer = SO3Layer(in_features=16, out_features=32, lmax=2)
        layer = layer.to(device)

        n_nodes = 4
        n_harmonics = 9
        x = torch.randn(n_nodes, 16, n_harmonics, device=device)
        edge_vec = torch.randn(simple_edge_index.shape[1], 3, device=device)

        result = layer(x, simple_edge_index, edge_vec)
        assert result.shape == (n_nodes, 32, n_harmonics)

    def test_without_self_interaction(self, device, simple_edge_index):
        """Test layer without self-interaction."""
        layer = SO3Layer(in_features=16, out_features=32, use_self_interaction=False)
        layer = layer.to(device)

        n_nodes = 4
        x = torch.randn(n_nodes, 16, device=device)
        edge_vec = torch.randn(simple_edge_index.shape[1], 3, device=device)

        result = layer(x, simple_edge_index, edge_vec)
        assert result.shape[0] == n_nodes


class TestSO3GNN:
    """Tests for full SO3GNN model."""

    def test_init(self):
        """Test initialization."""
        model = SO3GNN(
            in_features=16,
            hidden_features=64,
            out_features=10,
            n_layers=3,
        )
        assert len(model.layers) == 3

    def test_forward_shape(self, graph_data, device):
        """Test forward pass shape."""
        model = SO3GNN(
            in_features=16,
            hidden_features=32,
            out_features=5,
            n_layers=2,
            lmax=1,  # Smaller for faster test
        )
        model = model.to(device)

        x = graph_data["x"]
        pos = graph_data["pos"]
        edge_index = graph_data["edge_index"]

        result = model(x, pos, edge_index)
        assert result.shape == (1, 5)  # Single graph

    def test_forward_with_batch(self, device):
        """Test forward with multiple graphs."""
        model = SO3GNN(
            in_features=8,
            hidden_features=16,
            out_features=3,
            n_layers=2,
            lmax=1,
        )
        model = model.to(device)

        # Create batched input
        n_nodes = 20
        n_edges = 40
        n_graphs = 4

        x = torch.randn(n_nodes, 8, device=device)
        pos = torch.randn(n_nodes, 3, device=device)
        edge_index = torch.stack([
            torch.randint(0, n_nodes, (n_edges,), device=device),
            torch.randint(0, n_nodes, (n_edges,), device=device),
        ])
        batch = torch.repeat_interleave(torch.arange(n_graphs, device=device), n_nodes // n_graphs)

        result = model(x, pos, edge_index, batch)
        assert result.shape == (n_graphs, 3)

    def test_different_pooling(self, graph_data, device):
        """Test different pooling methods."""
        for pool in ["sum", "mean", "max"]:
            model = SO3GNN(
                in_features=16,
                hidden_features=16,
                out_features=3,
                n_layers=1,
                pool=pool,
            )
            model = model.to(device)

            result = model(graph_data["x"], graph_data["pos"], graph_data["edge_index"])
            assert result.shape == (1, 3)
