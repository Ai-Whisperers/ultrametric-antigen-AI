# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for DiffusionMapEncoder module."""

import pytest
import torch

from src.encoders.diffusion_encoder import (
    DiffusionMapEncoder,
    DiffusionPseudotime,
    KernelBuilder,
    MultiscaleDiffusion,
)


class TestKernelBuilder:
    """Tests for KernelBuilder."""

    def test_creation(self):
        """Test kernel builder creation."""
        kernel = KernelBuilder(kernel_type="gaussian")
        assert kernel.kernel_type == "gaussian"

    def test_pairwise_distances(self):
        """Test pairwise distance computation."""
        kernel = KernelBuilder()
        x = torch.randn(2, 10, 8)
        distances = kernel.compute_pairwise_distances(x)

        assert distances.shape == (2, 10, 10)
        # Diagonal should be approximately zero (numerical precision)
        diag = distances[:, range(10), range(10)]
        assert diag.max() < 0.01

    def test_gaussian_kernel(self):
        """Test Gaussian kernel computation."""
        kernel = KernelBuilder(kernel_type="gaussian", adaptive_bandwidth=False)
        x = torch.randn(2, 10, 8)
        K = kernel(x)

        assert K.shape == (2, 10, 10)
        # Kernel values should be non-negative
        assert (K >= 0).all()
        # Kernel should be symmetric
        assert torch.allclose(K, K.transpose(-1, -2), atol=1e-5)

    def test_cosine_kernel(self):
        """Test cosine kernel computation."""
        kernel = KernelBuilder(kernel_type="cosine")
        x = torch.randn(2, 10, 8)
        K = kernel(x)

        assert K.shape == (2, 10, 10)
        # Kernel should be symmetric
        assert torch.allclose(K, K.transpose(-1, -2), atol=1e-5)

    def test_padic_kernel(self):
        """Test p-adic kernel computation."""
        kernel = KernelBuilder(kernel_type="padic")
        x = torch.randn(2, 10, 8)
        K = kernel(x)

        assert K.shape == (2, 10, 10)
        assert (K >= 0).all()


class TestDiffusionMapEncoder:
    """Tests for DiffusionMapEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = DiffusionMapEncoder(input_dim=32, n_components=8)
        assert encoder.n_components == 8

    def test_forward(self):
        """Test forward pass."""
        encoder = DiffusionMapEncoder(input_dim=32, n_components=8)
        x = torch.randn(2, 20, 32)

        result = encoder(x)

        assert "z" in result
        assert "coordinates" in result
        assert "eigenvalues" in result
        assert result["z"].shape == (2, 20, 8)

    def test_transition_matrix(self):
        """Test transition matrix is row-stochastic."""
        encoder = DiffusionMapEncoder()
        x = torch.randn(2, 10, 64)

        kernel = encoder.kernel(encoder.feature_proj(x))
        P = encoder.compute_transition_matrix(kernel)

        # Rows should sum to 1
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_diffusion_distance(self):
        """Test diffusion distance computation."""
        encoder = DiffusionMapEncoder()
        x = torch.randn(2, 10, 64)

        result = encoder(x)
        dist = result["diffusion_distances"]

        assert dist.shape == (2, 10, 10)
        # Distance to self should be ~0
        assert torch.allclose(dist[:, range(10), range(10)], torch.zeros(2, 10), atol=1e-3)

    def test_nystrom_extension(self):
        """Test Nystrom extension for new points."""
        encoder = DiffusionMapEncoder(input_dim=32, n_components=8)

        x_train = torch.randn(2, 20, 32)
        x_new = torch.randn(2, 5, 32)

        result_train = encoder(x_train)
        train_coords = result_train["coordinates"]

        new_coords = encoder.embed_new_points(x_new, x_train, train_coords)

        # Should have correct shape
        assert new_coords.shape[0] == 2
        assert new_coords.shape[1] == 5


class TestMultiscaleDiffusion:
    """Tests for MultiscaleDiffusion."""

    def test_creation(self):
        """Test multi-scale encoder creation."""
        encoder = MultiscaleDiffusion(input_dim=32, output_dim=16, n_scales=3)
        assert encoder.n_scales == 3

    def test_forward(self):
        """Test forward pass."""
        encoder = MultiscaleDiffusion(input_dim=32, output_dim=16, n_scales=3)
        x = torch.randn(2, 15, 32)

        result = encoder(x)

        assert "z" in result
        assert "scale_embeddings" in result
        assert result["z"].shape == (2, 15, 16)
        assert len(result["scale_embeddings"]) == 3

    def test_different_scales(self):
        """Test that different scales produce different results."""
        encoder = MultiscaleDiffusion(input_dim=32, output_dim=16, n_scales=3)
        x = torch.randn(2, 15, 32)

        result = encoder(x)
        scale_embs = result["scale_embeddings"]

        # Different scales should give different embeddings
        assert not torch.allclose(scale_embs[0], scale_embs[1], atol=1e-3)


class TestDiffusionPseudotime:
    """Tests for DiffusionPseudotime."""

    def test_creation(self):
        """Test pseudotime module creation."""
        pseudotime = DiffusionPseudotime(input_dim=32)
        assert pseudotime.input_dim == 32

    def test_forward(self):
        """Test forward pass."""
        pseudotime = DiffusionPseudotime(input_dim=32)
        x = torch.randn(2, 20, 32)

        result = pseudotime(x)

        assert "pseudotime" in result
        assert "diffusion_coordinates" in result
        assert result["pseudotime"].shape == (2, 20)
        # Pseudotime should be in [0, 1]
        assert (result["pseudotime"] >= 0).all() and (result["pseudotime"] <= 1).all()

    def test_with_root(self):
        """Test pseudotime with root specification."""
        pseudotime = DiffusionPseudotime(input_dim=32)
        x = torch.randn(2, 20, 32)

        result = pseudotime(x, root_idx=0)

        # Root should have pseudotime 0
        assert result["pseudotime"][:, 0].abs().max() < 1e-5

    def test_order_by_pseudotime(self):
        """Test ordering points by pseudotime."""
        pseudotime = DiffusionPseudotime(input_dim=32)
        x = torch.randn(2, 20, 32)

        indices, pt_values = pseudotime.order_by_pseudotime(x)

        assert indices.shape == (2, 20)
        # Pseudotime should be monotonically increasing along ordered indices
        for b in range(2):
            ordered_pt = torch.gather(pt_values[b], 0, indices[b])
            assert (ordered_pt[1:] >= ordered_pt[:-1] - 1e-5).all()
