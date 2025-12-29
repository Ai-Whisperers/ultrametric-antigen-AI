# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for SimCLREncoder class."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.contrastive import SimCLREncoder


class SimpleEncoder(nn.Module):
    """Simple encoder for testing."""

    def __init__(self, input_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestSimCLREncoderInit:
    """Tests for SimCLREncoder initialization."""

    def test_default_init(self, device):
        """Test default initialization."""
        base = SimpleEncoder()
        encoder = SimCLREncoder(base, representation_dim=64).to(device)
        assert encoder.representation_dim == 64
        assert encoder.projection_dim == 128

    def test_custom_projection_dim(self, device):
        """Test custom projection dimension."""
        base = SimpleEncoder()
        encoder = SimCLREncoder(base, representation_dim=64, projection_dim=32)
        assert encoder.projection_dim == 32

    def test_with_bn(self, device):
        """Test with batch normalization."""
        base = SimpleEncoder()
        encoder = SimCLREncoder(base, representation_dim=64, use_bn=True)
        # Check BN layer exists
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in encoder.projection_head.modules())
        assert has_bn

    def test_without_bn(self, device):
        """Test without batch normalization."""
        base = SimpleEncoder()
        encoder = SimCLREncoder(base, representation_dim=64, use_bn=False)
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in encoder.projection_head.modules())
        assert not has_bn


class TestSimCLREncoderForward:
    """Tests for forward pass."""

    def test_output_shape(self, simclr_encoder, device):
        """Test output shape."""
        x = torch.randn(8, 32, device=device)
        output = simclr_encoder(x)
        assert output.shape == (8, 32)  # projection_dim = 32

    def test_return_representation(self, simclr_encoder, device):
        """Test returning representation."""
        x = torch.randn(8, 32, device=device)
        projection, representation = simclr_encoder(x, return_representation=True)
        assert projection.shape == (8, 32)
        assert representation.shape == (8, 64)

    def test_output_finite(self, simclr_encoder, device):
        """Test output is finite."""
        x = torch.randn(8, 32, device=device)
        output = simclr_encoder(x)
        assert torch.isfinite(output).all()

    def test_gradient_flows(self, simclr_encoder, device):
        """Test gradients flow through encoder."""
        x = torch.randn(8, 32, device=device, requires_grad=True)
        output = simclr_encoder(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestSimCLREncoderTraining:
    """Tests for training behavior."""

    def test_parameters_trainable(self, simclr_encoder):
        """Test all parameters are trainable."""
        for name, param in simclr_encoder.named_parameters():
            assert param.requires_grad, f"{name} is not trainable"

    def test_training_step(self, simclr_encoder, device):
        """Test complete training step."""
        optimizer = torch.optim.Adam(simclr_encoder.parameters(), lr=0.01)
        x = torch.randn(8, 32, device=device)

        # Forward
        output = simclr_encoder(x)
        loss = output.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should still work
        with torch.no_grad():
            output2 = simclr_encoder(x)
            assert torch.isfinite(output2).all()
