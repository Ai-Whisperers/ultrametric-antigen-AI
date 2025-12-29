# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for MomentumContrastEncoder class."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.contrastive import MomentumContrastEncoder


class SimpleEncoder(nn.Module):
    """Simple encoder for testing."""

    def __init__(self, input_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestMoCoEncoderInit:
    """Tests for MomentumContrastEncoder initialization."""

    def test_default_init(self, device):
        """Test default initialization."""
        base = SimpleEncoder()
        encoder = MomentumContrastEncoder(base, dim=64, queue_size=64).to(device)
        assert encoder.momentum == 0.999
        assert encoder.temperature == 0.07
        assert encoder.queue_size == 64

    def test_queue_initialized(self, moco_encoder):
        """Test queue is initialized."""
        assert moco_encoder.queue.shape == (64, 64)
        assert moco_encoder.queue_indices.shape == (64,)
        assert moco_encoder.queue_ptr.shape == (1,)

    def test_key_encoder_frozen(self, moco_encoder):
        """Test key encoder has no gradients."""
        for param in moco_encoder.encoder_k.parameters():
            assert not param.requires_grad


class TestMoCoEncoderForward:
    """Tests for forward pass."""

    def test_output_shapes(self, moco_encoder, device):
        """Test output shapes."""
        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        logits, labels = moco_encoder(x_q, x_k, indices)

        # Logits: (batch, 1 + queue_size)
        assert logits.shape == (8, 1 + 64)
        # Labels: (batch,)
        assert labels.shape == (8,)

    def test_labels_zero(self, moco_encoder, device):
        """Test labels are all zeros (positives at index 0)."""
        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        _, labels = moco_encoder(x_q, x_k, indices)
        assert (labels == 0).all()

    def test_output_finite(self, moco_encoder, device):
        """Test output is finite."""
        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        logits, _ = moco_encoder(x_q, x_k, indices)
        assert torch.isfinite(logits).all()


class TestMoCoMomentumUpdate:
    """Tests for momentum update."""

    def test_momentum_update_changes_key_encoder(self, moco_encoder, device):
        """Test momentum update modifies key encoder."""
        # Store original key encoder params
        original_params = {
            name: param.clone()
            for name, param in moco_encoder.encoder_k.named_parameters()
        }

        # Modify query encoder
        for param in moco_encoder.encoder_q.parameters():
            param.data += 0.1

        # Trigger momentum update
        moco_encoder._momentum_update()

        # Key encoder should have changed
        for name, param in moco_encoder.encoder_k.named_parameters():
            assert not torch.allclose(param, original_params[name])

    def test_momentum_preserves_structure(self, moco_encoder, device):
        """Test momentum update preserves tensor shapes."""
        for param in moco_encoder.encoder_q.parameters():
            param.data += 0.1

        original_shapes = {
            name: param.shape
            for name, param in moco_encoder.encoder_k.named_parameters()
        }

        moco_encoder._momentum_update()

        for name, param in moco_encoder.encoder_k.named_parameters():
            assert param.shape == original_shapes[name]


class TestMoCoQueue:
    """Tests for queue management."""

    def test_queue_updates(self, moco_encoder, device):
        """Test queue is updated after forward pass."""
        original_queue = moco_encoder.queue.clone()

        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        moco_encoder(x_q, x_k, indices)

        # Queue should have changed
        assert not torch.allclose(moco_encoder.queue, original_queue)

    def test_queue_ptr_increments(self, moco_encoder, device):
        """Test queue pointer increments."""
        original_ptr = moco_encoder.queue_ptr.item()

        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        moco_encoder(x_q, x_k, indices)

        expected_ptr = (original_ptr + 8) % moco_encoder.queue_size
        assert moco_encoder.queue_ptr.item() == expected_ptr

    def test_queue_wraps_around(self, moco_encoder, device):
        """Test queue pointer wraps around."""
        # Fill queue with multiple batches
        for _ in range(10):
            x_q = torch.randn(8, 32, device=device)
            x_k = torch.randn(8, 32, device=device)
            indices = torch.arange(8, device=device)
            moco_encoder(x_q, x_k, indices)

        # Pointer should be < queue_size
        assert moco_encoder.queue_ptr.item() < moco_encoder.queue_size


class TestMoCoTraining:
    """Tests for training behavior."""

    def test_gradient_flows_query(self, moco_encoder, device):
        """Test gradients flow through query encoder."""
        x_q = torch.randn(8, 32, device=device, requires_grad=True)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        logits, labels = moco_encoder(x_q, x_k, indices)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        assert x_q.grad is not None

    def test_key_encoder_no_gradient(self, moco_encoder, device):
        """Test key encoder has no gradient."""
        x_q = torch.randn(8, 32, device=device)
        x_k = torch.randn(8, 32, device=device)
        indices = torch.arange(8, device=device)

        logits, labels = moco_encoder(x_q, x_k, indices)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        for param in moco_encoder.encoder_k.parameters():
            assert param.grad is None
