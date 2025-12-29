# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for SpectralWavelet class.

Tests cover:
- Multi-scale decomposition
- Laplacian computation
- Wavelet coefficient properties
- Edge cases
"""

from __future__ import annotations

import math

import pytest
import torch

from src.graphs import SpectralWavelet


class TestSpectralWaveletInit:
    """Tests for SpectralWavelet initialization."""

    def test_default_init(self):
        """Test default initialization."""
        wavelet = SpectralWavelet()

        assert wavelet.n_scales == 4
        assert wavelet.scales.shape == (4,)

    def test_custom_scales(self):
        """Test custom number of scales."""
        wavelet = SpectralWavelet(n_scales=8)

        assert wavelet.n_scales == 8
        assert wavelet.scales.shape == (8,)

    def test_scale_range(self):
        """Test custom scale range."""
        wavelet = SpectralWavelet(n_scales=4, min_scale=0.1, max_scale=10.0)

        # Check scales are log-spaced
        assert wavelet.scales[0] == pytest.approx(0.1, rel=0.1)
        assert wavelet.scales[-1] == pytest.approx(10.0, rel=0.1)

    def test_scales_increasing(self):
        """Test that scales are monotonically increasing."""
        wavelet = SpectralWavelet(n_scales=6)

        for i in range(len(wavelet.scales) - 1):
            assert wavelet.scales[i] < wavelet.scales[i + 1]


class TestSpectralWaveletForward:
    """Tests for SpectralWavelet forward pass."""

    def test_output_is_list(self, spectral_wavelet, small_graph):
        """Test output is a list of tensors."""
        wavelet = spectral_wavelet.to(small_graph["x"].device)
        output = wavelet(small_graph["x"], small_graph["edge_index"])

        assert isinstance(output, list)
        assert len(output) == spectral_wavelet.n_scales

    def test_output_shapes(self, spectral_wavelet, small_graph):
        """Test each scale output has correct shape."""
        wavelet = spectral_wavelet.to(small_graph["x"].device)
        output = wavelet(small_graph["x"], small_graph["edge_index"])

        for coeff in output:
            assert coeff.shape == small_graph["x"].shape

    def test_output_finite(self, spectral_wavelet, small_graph):
        """Test all outputs are finite."""
        wavelet = spectral_wavelet.to(small_graph["x"].device)
        output = wavelet(small_graph["x"], small_graph["edge_index"])

        for coeff in output:
            assert torch.isfinite(coeff).all()

    def test_different_scales_differ(self, spectral_wavelet, small_graph):
        """Test that different scales produce different outputs."""
        wavelet = spectral_wavelet.to(small_graph["x"].device)
        output = wavelet(small_graph["x"], small_graph["edge_index"])

        # At least some scales should differ
        differs = False
        for i in range(len(output) - 1):
            if not torch.allclose(output[i], output[i + 1], atol=1e-3):
                differs = True
                break

        assert differs, "Different scales should produce different outputs"


class TestLaplacianComputation:
    """Tests for Laplacian computation."""

    def test_laplacian_symmetric(self, spectral_wavelet, device):
        """Test computed Laplacian is symmetric."""
        wavelet = spectral_wavelet.to(device)

        n_nodes = 10
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], device=device)

        L = wavelet._compute_laplacian(edge_index, n_nodes)

        assert torch.allclose(L, L.t(), atol=1e-5)

    def test_laplacian_symmetric_random(self, spectral_wavelet, device):
        """Test Laplacian is symmetric for random edges."""
        wavelet = spectral_wavelet.to(device)

        n_nodes = 8
        edge_index = torch.randint(0, n_nodes, (2, 20), device=device)

        L = wavelet._compute_laplacian(edge_index, n_nodes)

        # For normalized Laplacian, check symmetry
        assert torch.allclose(L, L.t(), atol=1e-5)

    def test_laplacian_row_sum(self, spectral_wavelet, device):
        """Test normalized Laplacian properties."""
        wavelet = spectral_wavelet.to(device)

        n_nodes = 6
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], device=device)

        L = wavelet._compute_laplacian(edge_index, n_nodes)

        # For normalized Laplacian, diagonal should be 1
        diag = torch.diag(L)
        assert torch.allclose(diag, torch.ones(n_nodes, device=device), atol=1e-5)


class TestSpectralWaveletEdgeCases:
    """Tests for edge cases."""

    def test_disconnected_graph(self, spectral_wavelet, device):
        """Test with disconnected graph."""
        wavelet = spectral_wavelet.to(device)

        # Two disconnected components
        x = torch.randn(6, 16, device=device)
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 4, 3, 5, 4]], device=device)

        output = wavelet(x, edge_index)

        assert len(output) == wavelet.n_scales
        for coeff in output:
            assert torch.isfinite(coeff).all()

    def test_single_node(self, spectral_wavelet, device):
        """Test with single-node graph."""
        wavelet = spectral_wavelet.to(device)

        x = torch.randn(1, 16, device=device)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        output = wavelet(x, edge_index)

        assert len(output) == wavelet.n_scales

    def test_complete_graph(self, spectral_wavelet, device):
        """Test with complete graph."""
        wavelet = spectral_wavelet.to(device)

        n_nodes = 8
        x = torch.randn(n_nodes, 16, device=device)

        # Complete graph edges
        src, dst = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], device=device)

        output = wavelet(x, edge_index)

        assert len(output) == wavelet.n_scales
        for coeff in output:
            assert torch.isfinite(coeff).all()

    def test_large_graph_approximation(self, device):
        """Test large graph uses approximation."""
        wavelet = SpectralWavelet(n_scales=3).to(device)

        # Graph larger than 500 nodes triggers approximation
        n_nodes = 600
        x = torch.randn(n_nodes, 8, device=device)
        edge_index = torch.randint(0, n_nodes, (2, 2000), device=device)

        output = wavelet(x, edge_index)

        # Should still work (with approximation)
        assert len(output) == 3
        for coeff in output:
            assert coeff.shape == x.shape


class TestSpectralWaveletConfigurations:
    """Tests for different configurations."""

    @pytest.mark.parametrize("n_scales", [1, 2, 4, 8])
    def test_scale_count(self, n_scales, device):
        """Test different numbers of scales."""
        wavelet = SpectralWavelet(n_scales=n_scales).to(device)

        x = torch.randn(10, 16, device=device)
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = wavelet(x, edge_index)

        assert len(output) == n_scales

    @pytest.mark.parametrize("feature_dim", [4, 8, 16, 32, 64])
    def test_feature_dimensions(self, feature_dim, device):
        """Test different feature dimensions."""
        wavelet = SpectralWavelet(n_scales=3).to(device)

        x = torch.randn(10, feature_dim, device=device)
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = wavelet(x, edge_index)

        for coeff in output:
            assert coeff.shape == (10, feature_dim)


class TestSpectralWaveletGradients:
    """Tests for gradient flow."""

    def test_gradients_flow(self, spectral_wavelet, small_graph):
        """Test gradients flow through wavelet decomposition."""
        wavelet = spectral_wavelet.to(small_graph["x"].device)

        x = small_graph["x"].clone().requires_grad_(True)
        output = wavelet(x, small_graph["edge_index"])

        # Sum all scales
        loss = sum(coeff.sum() for coeff in output)
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
