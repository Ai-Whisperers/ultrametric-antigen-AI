# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ReconstructionLoss from src/losses/dual_vae_loss.py.

Tests cross-entropy reconstruction loss for ternary operations.
"""

import pytest
import torch

from src.losses.dual_vae_loss import ReconstructionLoss


class TestReconstructionLossBasic:
    """Basic tests for ReconstructionLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create a ReconstructionLoss instance."""
        return ReconstructionLoss()

    @pytest.fixture
    def sample_logits(self, device):
        """Sample logits from model (batch, 9, 3)."""
        return torch.randn(32, 9, 3, device=device)

    @pytest.fixture
    def sample_input(self, device):
        """Sample input tensor with ternary values {-1, 0, 1}."""
        return torch.randint(-1, 2, (32, 9), device=device).float()

    def test_forward_returns_scalar(self, loss_fn, sample_logits, sample_input):
        """Forward should return a scalar loss."""
        loss = loss_fn(sample_logits, sample_input)

        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_forward_is_positive(self, loss_fn, sample_logits, sample_input):
        """Reconstruction loss should be non-negative."""
        loss = loss_fn(sample_logits, sample_input)
        assert loss >= 0

    def test_random_logits_have_reasonable_loss(self, loss_fn, sample_logits, sample_input):
        """Random logits should have reasonable loss."""
        loss = loss_fn(sample_logits, sample_input)

        # Cross-entropy with 3 classes and random prediction is about log(3) per position
        # With 9 positions, expect roughly 9-10
        assert 0 < loss < 20


class TestReconstructionLossPerfect:
    """Tests for perfect and near-perfect reconstruction."""

    @pytest.fixture
    def loss_fn(self):
        return ReconstructionLoss()

    def test_perfect_prediction_low_loss(self, loss_fn, device):
        """Perfect prediction should have very low loss."""
        batch_size = 4

        # Create targets
        x = torch.randint(-1, 2, (batch_size, 9), device=device).float()
        x_classes = (x + 1).long()  # {0, 1, 2}

        # Create logits that strongly predict correct class
        logits = torch.zeros(batch_size, 9, 3, device=device)
        for b in range(batch_size):
            for d in range(9):
                logits[b, d, x_classes[b, d]] = 10.0

        loss = loss_fn(logits, x)
        assert loss < 0.1


class TestReconstructionLossEdgeCases:
    """Edge case tests for ReconstructionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return ReconstructionLoss()

    def test_handles_all_negative_one(self, loss_fn, device):
        """Should handle input with all -1 values."""
        batch_size = 8
        x = torch.full((batch_size, 9), -1.0, device=device)
        logits = torch.randn(batch_size, 9, 3, device=device)

        loss = loss_fn(logits, x)
        assert torch.isfinite(loss)

    def test_handles_all_positive_one(self, loss_fn, device):
        """Should handle input with all +1 values."""
        batch_size = 8
        x = torch.full((batch_size, 9), 1.0, device=device)
        logits = torch.randn(batch_size, 9, 3, device=device)

        loss = loss_fn(logits, x)
        assert torch.isfinite(loss)

    def test_handles_all_zeros(self, loss_fn, device):
        """Should handle input with all 0 values."""
        batch_size = 8
        x = torch.zeros(batch_size, 9, device=device)
        logits = torch.randn(batch_size, 9, 3, device=device)

        loss = loss_fn(logits, x)
        assert torch.isfinite(loss)

    def test_handles_single_sample(self, loss_fn, device):
        """Should handle batch size of 1."""
        x = torch.randint(-1, 2, (1, 9), device=device).float()
        logits = torch.randn(1, 9, 3, device=device)

        loss = loss_fn(logits, x)
        assert torch.isfinite(loss)


class TestReconstructionLossGradient:
    """Gradient flow tests for ReconstructionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return ReconstructionLoss()

    def test_gradient_flows(self, loss_fn, device):
        """Gradients should flow through the loss."""
        x = torch.randint(-1, 2, (32, 9), device=device).float()
        logits = torch.randn(32, 9, 3, device=device, requires_grad=True)

        loss = loss_fn(logits, x)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert torch.isfinite(logits.grad).all()

    def test_gradient_magnitude(self, loss_fn, device):
        """Gradients should have reasonable magnitude."""
        x = torch.randint(-1, 2, (16, 9), device=device).float()
        logits = torch.randn(16, 9, 3, device=device, requires_grad=True)

        loss = loss_fn(logits, x)
        loss.backward()

        grad_norm = logits.grad.norm()
        assert grad_norm > 0
        assert grad_norm < 100  # Reasonable upper bound
