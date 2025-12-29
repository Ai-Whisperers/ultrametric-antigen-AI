# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ProteinTopologyEncoder class.

Tests cover:
- Forward pass
- Output shape
- Gradient flow
- With and without p-adic features
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topology import ProteinTopologyEncoder


class TestProteinTopologyEncoderInit:
    """Tests for ProteinTopologyEncoder initialization."""

    def test_default_init(self):
        """Test default initialization."""
        encoder = ProteinTopologyEncoder()
        assert encoder.output_dim == 128
        assert encoder.max_dimension == 1
        assert encoder.use_padic is False

    def test_custom_params(self):
        """Test custom parameters."""
        encoder = ProteinTopologyEncoder(
            output_dim=64,
            hidden_dims=[256],
            max_dimension=2,
        )
        assert encoder.output_dim == 64
        assert encoder.max_dimension == 2

    def test_with_padic(self):
        """Test initialization with p-adic."""
        encoder = ProteinTopologyEncoder(use_padic=True, prime=5)
        assert encoder.use_padic is True
        assert encoder.padic.prime == 5


class TestProteinTopologyEncoderForward:
    """Tests for forward pass."""

    def test_output_shape(self, topology_encoder, batch_coordinates):
        """Test output shape."""
        encoder = topology_encoder.to(batch_coordinates.device)
        output = encoder(batch_coordinates)
        assert output.shape == (4, 64)

    def test_output_dtype(self, topology_encoder, batch_coordinates):
        """Test output dtype."""
        encoder = topology_encoder.to(batch_coordinates.device)
        output = encoder(batch_coordinates)
        assert output.dtype == torch.float32

    def test_output_finite(self, topology_encoder, batch_coordinates):
        """Test output is finite."""
        encoder = topology_encoder.to(batch_coordinates.device)
        output = encoder(batch_coordinates)
        assert torch.isfinite(output).all()

    def test_single_batch(self, topology_encoder, device):
        """Test with single sample."""
        encoder = topology_encoder.to(device)
        coords = torch.randn(1, 10, 3, device=device)
        output = encoder(coords)
        assert output.shape == (1, 64)


class TestProteinTopologyEncoderPAadic:
    """Tests for p-adic features."""

    def test_forward_with_padic(self, topology_encoder_padic, batch_coordinates, device):
        """Test forward with p-adic indices."""
        encoder = topology_encoder_padic.to(device)
        batch_size = batch_coordinates.shape[0]
        indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 10)

        output = encoder(batch_coordinates, indices=indices)
        assert output.shape == (batch_size, 64)

    def test_padic_requires_indices(self, topology_encoder_padic, batch_coordinates, device):
        """Test that p-adic encoder requires indices."""
        encoder = topology_encoder_padic.to(device)
        batch_size = batch_coordinates.shape[0]
        n_points = batch_coordinates.shape[1]

        # Must provide indices when use_padic=True
        indices = torch.arange(n_points, device=device).unsqueeze(0).expand(batch_size, -1)
        output = encoder(batch_coordinates, indices=indices)
        assert output.shape == (batch_size, 64)


class TestProteinTopologyEncoderGradients:
    """Tests for gradient flow."""

    def test_gradients_flow(self, device):
        """Test gradients flow through MLP."""
        encoder = ProteinTopologyEncoder(output_dim=32, hidden_dims=[64]).to(device)
        coords = torch.randn(2, 10, 3, device=device)

        output = encoder(coords)
        loss = output.sum()
        loss.backward()

        # Check MLP gradients
        for name, param in encoder.mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_training_step(self, device):
        """Test complete training step."""
        encoder = ProteinTopologyEncoder(output_dim=32, hidden_dims=[64]).to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

        for _ in range(2):
            coords = torch.randn(2, 8, 3, device=device)
            optimizer.zero_grad()
            output = encoder(coords)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Should still work
        with torch.no_grad():
            test_coords = torch.randn(1, 8, 3, device=device)
            test_output = encoder(test_coords)
            assert torch.isfinite(test_output).all()


class TestProteinTopologyEncoderComputeFingerprint:
    """Tests for compute_fingerprint method."""

    def test_compute_fingerprint(self, topology_encoder):
        """Test compute_fingerprint method."""
        coords = np.random.randn(20, 3).astype(np.float32)
        fingerprint = topology_encoder.compute_fingerprint(coords)

        from src.topology import TopologicalFingerprint
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_fingerprint_metadata(self, topology_encoder):
        """Test fingerprint has metadata."""
        coords = np.random.randn(15, 3).astype(np.float32)
        fingerprint = topology_encoder.compute_fingerprint(coords)
        assert fingerprint.metadata is not None


class TestProteinTopologyEncoderConfigurations:
    """Tests for different configurations."""

    @pytest.mark.parametrize("vectorization", ["statistics", "landscape", "image"])
    def test_vectorization_methods(self, vectorization, device):
        """Test different vectorization methods."""
        encoder = ProteinTopologyEncoder(
            output_dim=32,
            vectorization=vectorization,
            resolution=10,
        ).to(device)
        coords = torch.randn(2, 10, 3, device=device)
        output = encoder(coords)
        assert output.shape == (2, 32)

    @pytest.mark.parametrize("max_dim", [0, 1, 2])
    def test_max_dimensions(self, max_dim, device):
        """Test different max dimensions."""
        encoder = ProteinTopologyEncoder(
            output_dim=32,
            max_dimension=max_dim,
        ).to(device)
        coords = torch.randn(2, 10, 3, device=device)
        output = encoder(coords)
        assert output.shape == (2, 32)
